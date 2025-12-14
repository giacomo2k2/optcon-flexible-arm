import numpy as np
from scipy.interpolate import CubicSpline
from model import DoublePendulumFlexible, PARAM_SET_2
import matplotlib.pyplot as plt

from solver import solver_step
from plots import plot_trajectory
from animation import animate

import control
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams['font.family'] = 'serif'


# ----------------------------------------------------
# Trajectory generator (unchanged)
# ----------------------------------------------------
def trajectory_generator(
    T0: float = 0.0,
    Tend: float = 1.0,
    setpoints: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0]]),
    vel_setpoints: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0]]),
    dt: float = 0.01,
    method: str = "stepwise",
    poly_degree: int = 3,
):
    t = np.arange(T0, Tend, dt)
    traj = np.zeros((len(t), setpoints.shape[1]))

    t_setpoints = np.linspace(T0, Tend, setpoints.shape[0])

    if method == "stepwise":
        if setpoints.shape[0] != 2:
            raise ValueError("Stepwise method requires exactly two setpoints.")
        for i, ti in enumerate(t):
            if ti < (Tend / 2):
                traj[i, :] = setpoints[0, :]
            else:
                traj[i, :] = setpoints[1, :]

    elif method == "polynomial":
        if setpoints.shape[0] < poly_degree + 1:
            raise ValueError(
                f"Number of setpoints must be at least {poly_degree + 1} "
                f"for a {poly_degree}° order polynomial."
            )
        for dim in range(setpoints.shape[1]):
            coeffs = np.polyfit(t_setpoints, setpoints[:, dim], poly_degree)
            traj[:, dim] = np.polyval(coeffs, t)

    elif method == "cubic_spline":
        vel0 = vel_setpoints[0, :]
        vel_end = vel_setpoints[-1, :]
        for dim in range(setpoints.shape[1]):
            cs = CubicSpline(
                t_setpoints,
                setpoints[:, dim],
                bc_type=((1, vel0[dim]), (1, vel_end[dim])),
            )
            traj[:, dim] = cs(t)

    else:
        raise ValueError(f"Trajectory type '{method}' not recognized.")

    return t, traj


# ----------------------------------------------------
# Cost utilities (TRUE tracking cost for nonlinear system)
# ----------------------------------------------------
def compute_cost(x_traj, u_traj, x_ref, u_ref, Q, R, Qf):
    """
    True nonlinear tracking cost:

      sum_{t=0}^{N-2} 1/2 (x_t-x_ref_t)^T Q (x_t-x_ref_t)
                       + 1/2 (u_t-u_ref_t)^T R (u_t-u_ref_t)
      + 1/2 (x_{N-1}-x_ref_{N-1})^T Qf (x_{N-1}-x_ref_{N-1})
    """
    N = x_traj.shape[0]
    J = 0.0
    for t in range(N - 1):
        dx = x_traj[t] - x_ref[t]
        du = u_traj[t] - u_ref[t]
        J += 0.5 * dx @ Q @ dx + 0.5 * du @ R @ du
    dxT = x_traj[-1] - x_ref[-1]
    J += 0.5 * dxT @ Qf @ dxT
    return float(J)


# ----------------------------------------------------
# iLQR backward pass
# ----------------------------------------------------
def ilqr_backward(A_seq, B_seq, x_traj, u_traj, x_ref, u_ref, Q, R, Qf,
                  reg=1e-6):
    """
    Standard discrete-time iLQR backward pass for tracking.

    A_seq: (N-1, nx, nx) Jacobians
    B_seq: (N-1, nx, nu)
    x_traj: (N, nx) current nominal trajectory
    u_traj: (N, nu) current nominal controls (last entry unused)
    x_ref, u_ref: same shapes as x_traj, u_traj
    """
    N, nx = x_traj.shape
    nu = u_traj.shape[1]

    # Storage
    k_seq = np.zeros((N - 1, nu))
    K_seq = np.zeros((N - 1, nu, nx))
    V_x = np.zeros((N, nx))
    V_xx = np.zeros((N, nx, nx))

    # Terminal value
    dxT = x_traj[-1] - x_ref[-1]
    V_x[-1] = Qf @ dxT
    V_xx[-1] = Qf

    dV1 = 0.0  # linear term in predicted cost change
    dV2 = 0.0  # quadratic term

    Iu = np.eye(nu)

    for t in reversed(range(N - 1)):
        xt = x_traj[t]
        ut = u_traj[t]
        xrt = x_ref[t]
        urt = u_ref[t]
        At = A_seq[t]
        Bt = B_seq[t]

        dx = xt - xrt
        du = ut - urt

        # Stage cost expansion around (xt, ut)
        l_x = Q @ dx
        l_u = R @ du
        l_xx = Q
        l_uu = R
        l_ux = np.zeros((nu, nx))

        Vx_next = V_x[t + 1]
        Vxx_next = V_xx[t + 1]

        # Q-functions
        Q_x = l_x + At.T @ Vx_next
        Q_u = l_u + Bt.T @ Vx_next
        Q_xx = l_xx + At.T @ Vxx_next @ At
        Q_ux = l_ux + Bt.T @ Vxx_next @ At
        Q_uu = l_uu + Bt.T @ Vxx_next @ Bt

        # Regularize Q_uu to ensure positive definiteness
        Q_uu_reg = Q_uu + reg * Iu

        # Solve for gains
        k = -np.linalg.solve(Q_uu_reg, Q_u)      # feedforward
        K = -np.linalg.solve(Q_uu_reg, Q_ux)     # feedback

        k_seq[t] = k
        K_seq[t] = K

        # Expected decrease contribution
        dV1_t = float(Q_u.T @ k)
        dV2_t = 0.5 * float(k.T @ Q_uu @ k)
        dV1 += dV1_t
        dV2 += dV2_t

        # Update value function
        V_x[t] = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
        V_xx[t] = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K

        # Symmetrize V_xx for numerical stability
        V_xx[t] = 0.5 * (V_xx[t] + V_xx[t].T)

    return k_seq, K_seq, V_x, V_xx, dV1, dV2


# ----------------------------------------------------
# Line search forward pass (true nonlinear dynamics & cost)
# ----------------------------------------------------
def ilqr_line_search(model, x0, x_traj, u_traj,
                     k_seq, K_seq,
                     x_ref, u_ref,
                     Q, R, Qf,
                     dt,
                     dV1, dV2,
                     alpha_init=1.0,
                     beta=0.5,
                     c=1e-4,
                     max_ls_iters=10):
    """
    Forward rollout with line search on TRUE nonlinear cost.
    """
    N = x_traj.shape[0]
    nx = x_traj.shape[1]
    nu = u_traj.shape[1]

    J_old = compute_cost(x_traj, u_traj, x_ref, u_ref, Q, R, Qf)

    alpha = alpha_init
    for _ in range(max_ls_iters):
        x_new = np.zeros((N, nx))
        u_new = np.zeros((N, nu))
        x_new[0] = x0

        for t in range(N - 1):
            dx = x_new[t] - x_traj[t]
            u_new[t] = (
                u_traj[t]
                + alpha * k_seq[t]
                + K_seq[t] @ dx
            )
            x_new[t + 1] = solver_step(model, x_new[t], dt, u_new[t], "fe")

        J_new = compute_cost(x_new, u_new, x_ref, u_ref, Q, R, Qf)

        # Predicted decrease for step size alpha
        dJ_pred = alpha * dV1 + alpha**2 * dV2

        if J_new <= J_old + c * dJ_pred:
            # Accept step
            return alpha, x_new, u_new, J_new

        alpha *= beta

    # If all steps rejected, return original
    print("WARNING: line search failed, keeping previous trajectory.")
    return 0.0, x_traj, u_traj, J_old


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    # ------------- Parameters -------------
    use_armijo = True          # <- enable iLQR line search
    stepsize_fixed = 1e-2      # used only if use_armijo=False
    armijo_alpha0 = 1.0
    armijo_beta = 0.5
    armijo_c = 1e-4
    armijo_maxiters = 10

    term_cond = 1e-6
    max_iters = 200

    params = PARAM_SET_2
    model = DoublePendulumFlexible(**params)
    nx = model.ns
    nu = model.ni

    # ------------- Reference trajectory -------------
    T0 = 0.0
    Tend = 10.0
    dt = 0.01

    x_eq = np.array([
        [0.0,       0.0,       0.0, 0.0],
        [-np.pi/2, -np.pi/2,  0.0, 0.0],
        [0.0,       np.pi,     0.0, 0.0],
        [np.pi,     np.pi,     0.0, 0.0],
    ])

    x_setpoints = np.array([
        [0.0, 0.0, 0.0, 0.0],
        x_eq[1, :]
    ])

    u_setpoints = []
    for i in range(x_setpoints.shape[0]):
        u_setpoints.append(model.equilibrium_input(x_eq[i, :]))
    u_setpoints = np.stack(u_setpoints)

    method = "cubic_spline"  # 'polynomial' or 'cubic_spline'
    time, x_ref = trajectory_generator(T0, Tend, dt=dt,
                                       setpoints=x_setpoints,
                                       method=method)
    _, u_ref = trajectory_generator(T0, Tend, dt=dt,
                                    setpoints=u_setpoints,
                                    method=method)

    N = time.shape[0]
    x0 = x_ref[0].copy()

    # ------------- Cost weights -------------
    Q = np.diag([100.0, 10.0, 0.1, 0.1])
    # Terminal weight: solve DARE at equilibrium to get "good" Qf
    A0, B0 = model.discrete_jacobian(x0, u_ref[0, :], dt)
    Qf = control.dare(A0, B0, Q, np.eye(nu))[0]
    R = 0.1 * np.eye(nu)

    # ------------- Initial guess -------------
    x_traj = np.zeros((max_iters, N, nx))
    u_traj = np.zeros((max_iters, N, nu))

    # Start from following the reference input once
    x_traj[0, 0] = x0
    for t in range(N - 1):
        u_traj[0, t] = u_ref[t]
        x_traj[0, t + 1] = solver_step(model, x_traj[0, t], dt,
                                       u_traj[0, t], "fe")
    # last control just copy previous for plotting
    u_traj[0, -1] = u_traj[0, -2]

    # ------------- iLQR loop -------------
    J = np.zeros(max_iters)
    descent_pred = np.zeros(max_iters)

    # Pre-allocate Jacobians
    A_seq = np.zeros((N - 1, nx, nx))
    B_seq = np.zeros((N - 1, nx, nu))

    for k in range(max_iters - 1):
        xk = x_traj[k]
        uk = u_traj[k]

        # Current cost
        J[k] = compute_cost(xk, uk, x_ref, u_ref, Q, R, Qf)
        print(f"Iter {k:03d} | J = {J[k]:.6e}")

        if J[k] < term_cond:
            print("Converged by cost threshold.")
            max_iters = k + 1
            break

        # Linearize dynamics along current trajectory
        for t in range(N - 1):
            A_seq[t], B_seq[t] = model.discrete_jacobian(
                xk[t, :], uk[t, :], dt
            )

        # Backward pass (iLQR)
        k_seq, K_seq, _, _, dV1, dV2 = ilqr_backward(
            A_seq, B_seq, xk, uk, x_ref, u_ref, Q, R, Qf,
            reg=1e-6
        )
        descent_pred[k] = dV1 + dV2

        # If predicted improvement is not descent, bail out
        if descent_pred[k] >= 0:
            print("WARNING: predicted improvement is non-negative; "
                  "stopping iLQR.")
            max_iters = k + 1
            break

        # Forward pass + line search
        if use_armijo:
            alpha, x_new, u_new, J_new = ilqr_line_search(
                model, x0, xk, uk,
                k_seq, K_seq,
                x_ref, u_ref,
                Q, R, Qf,
                dt,
                dV1, dV2,
                alpha_init=armijo_alpha0,
                beta=armijo_beta,
                c=armijo_c,
                max_ls_iters=armijo_maxiters,
            )
        else:
            # Single full step with fixed step size
            alpha = stepsize_fixed
            N = xk.shape[0]
            x_new = np.zeros_like(xk)
            u_new = np.zeros_like(uk)
            x_new[0] = x0
            for t in range(N - 1):
                dx = x_new[t] - xk[t]
                u_new[t] = uk[t] + alpha * k_seq[t] + K_seq[t] @ dx
                x_new[t + 1] = solver_step(model, x_new[t], dt,
                                           u_new[t], "fe")
            J_new = compute_cost(x_new, u_new, x_ref, u_ref, Q, R, Qf)

        x_traj[k + 1] = x_new
        u_traj[k + 1] = u_new

        # Early break if line search returns same trajectory
        if alpha == 0.0:
            print("No improving step found; stopping iLQR.")
            max_iters = k + 1
            break

    # ------------- Extract best trajectory -------------
    best_idx = np.argmin(J[:max_iters])
    x_star = x_traj[best_idx]
    u_star = u_traj[best_idx]
    u_star[-1, :] = u_star[-2, :]  # for plotting continuity

    # ------------- Plots -------------
    plt.figure("Predicted descent (dV)")
    plt.plot(np.arange(max_iters), descent_pred[:max_iters])
    plt.grid(True)
    plt.yscale("linear")
    plt.xlabel("Iteration")
    plt.ylabel("Predicted ΔJ")
    plt.show(block=False)

    plt.figure("Cost")
    plt.plot(np.arange(max_iters), J[:max_iters])
    plt.grid(True)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("J")
    plt.show(block=False)

    fig, ax = plt.subplots(3, 2, sharex=True)
    ax = ax.flatten()

    ax[0].plot(time, x_star[:, 0], linewidth=2)
    ax[0].plot(time, x_ref[:, 0], "g--", linewidth=2)
    ax[0].grid(True)

    ax[1].plot(time, x_star[:, 1], linewidth=2)
    ax[1].plot(time, x_ref[:, 1], "g--", linewidth=2)
    ax[1].grid(True)

    ax[2].plot(time, x_star[:, 2], linewidth=2)
    ax[2].plot(time, x_ref[:, 2], "g--", linewidth=2)
    ax[2].grid(True)

    ax[3].plot(time, x_star[:, 3], linewidth=2)
    ax[3].plot(time, x_ref[:, 3], "g--", linewidth=2)
    ax[3].grid(True)

    ax[4].plot(time, u_star[:, 0], "r", linewidth=2)
    ax[4].plot(time, u_ref[:, 0], "r--", linewidth=2)
    ax[4].grid(True)

    plt.show()

    # Animate optimal trajectory
    animate(model, x_star, dt, target_fps=30)


if __name__ == "__main__":
    main()
