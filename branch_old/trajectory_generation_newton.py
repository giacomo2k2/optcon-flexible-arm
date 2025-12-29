import numpy as np
from scipy.interpolate import CubicSpline
from model import DoublePendulumFlexible, PARAM_SET_2
import scipy as sp
import matplotlib.pyplot as plt

from solver import solver_step
from plots import plot_reference_traj

from animation import animate

import control

# Q1: corrispondenza fisica del dv1, dv2
# Q2: can we use the hand computed jacobians?
# Q3: regularization strategies for LQR

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Matplotlib settings
plt.rcParams['font.family'] = 'serif'

def trajectory_generator(
    T0: int = 0,
    Tend: float = 1.0,
    setpoints: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    vel_setpoints: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    
    dt: float = 0.01,
    method: str = 'stepwise',
    poly_degree: int = 3,
):
    
    t = np.arange(T0, Tend, dt)
    traj = np.zeros((len(t), setpoints.shape[1]))
    

    t_setpoints = np.linspace(T0, Tend, setpoints.shape[0])

    if method == 'stepwise':
        if setpoints.shape[0] != 2:
            raise ValueError("Stepwise method requires exactly two setpoints.")

        for i, ti in enumerate(t):
            if ti < (Tend / 2):
                traj[i, :] = setpoints[0, :]
            else:
                traj[i, :] = setpoints[1, :]
                

    elif method == 'polynomial':
        if setpoints.shape[0] < poly_degree + 1:
            raise ValueError(f"Number of setpoints must be at least {poly_degree + 1} for a {poly_degree}° order polynomial.")

        for dim in range(setpoints.shape[1]):
            coeffs = np.polyfit(t_setpoints, setpoints[:, dim], poly_degree)
            traj[:, dim] = np.polyval(coeffs, t)

    elif method == 'cubic_spline':
        vel0 = vel_setpoints[0, :]
        vel_end = vel_setpoints[-1, :]

        for dim in range(setpoints.shape[1]):
            cs = CubicSpline(
                t_setpoints, 
                setpoints[:, dim], 
                bc_type=((1, vel0[dim]), (1, vel_end[dim])) 
            )
            traj[:, dim] = cs(t)

    else:
        raise ValueError(f"Trajectory type '{method}' not recognized.")

    return t, traj


def select_stepsize(model, stepsize_0, armijo_maxiters, c, beta,
                    x_ref, u_ref, x0, dt,
                    xk, uk, dV1, dV2, Q, R, QT, K_vec, sigma_vec):
    
    
    alpha = stepsize_0
    steps = xk.shape[0]
    ns = xk.shape[1]
    ni = uk.shape[1]
    
    J_old = 0.0

    for t in range(steps-1):
        temp_cost = stagecost(xk[t, :], uk[t, :], x_ref[t, :], u_ref[t, :], Q, R)[0]
        J_old += temp_cost
    
    temp_cost = termcost(xk[-1, :], x_ref[-1, :], QT)[0]
    J_old += temp_cost

 
    # Armijo backtracking line search
    for i in range(armijo_maxiters):
        x_cand = np.zeros((steps, ns))
        u_cand = np.zeros((steps, ni))
        x_cand[0, :] = x0
        J_cand = 0
        
        
        
        for t in range(steps-1):
            dx = x_cand[t, :] - xk[t, :]
            u_cand[t, :] = uk[t, :] + alpha * sigma_vec[t, :] + K_vec[t, :, :] @ dx
            
            # Forward Integrate
            x_cand[t+1, :] = solver_step(model, x_cand[t, :], dt, u_cand[t, :], 'fe')
            
        for t in range(steps-1):
            J_cand += stagecost(x_cand[t, :], u_cand[t, :], x_ref[t, :], u_ref[t, :], Q, R)[0]
            
        J_cand += termcost(x_cand[-1, :], x_ref[-1, :], QT)[0]
        
        
        dJ_cand = alpha * dV1 + 0.5*(alpha **2) * dV2
        
        if J_cand <= J_old + c * dJ_cand:
            print(f'Armijo stepsize = {alpha:.3e} | Cost: {J_old:.4e} -> {J_cand:.4e}')
            return alpha, x_cand, u_cand, J_cand
    
        # If we are here, the step failed; reduce alpha and retry
        alpha *= beta
        
    print("WARNING: Armijo failed to find a step. Returning small step.")
    return 0.0, xk, uk, J_old



def ltv_LQR(Ain, Bin, xk, uk, x_ref, u_ref, Qt, Rt, QT, reg=1e-6):

    # Steps and number of states/inputs
    steps, ns = xk.shape
    ni = uk.shape[1]

    sigma_vec = np.zeros((steps-1, ni))
    K_vec = np.zeros((steps-1, ni, ns))
    
    p_vec = np.zeros((steps, ns))
    P_vec = np.zeros((steps, ns, ns))

    p_vec[-1, :], P_vec[-1, :, :] = termcost(xk[-1, :], x_ref[-1, :], QT)[1:]

    # TODO: check this dV1, dV2 calculation
    dV1 = 0.0
    dV2 = 0.0

    # (basically a scalar)
    Iu = np.eye(ni)

    
    # Solve Riccati equation
    for t in reversed(range(steps-1)):
        xt = xk[t, :][:, None]
        ut = uk[t, :][:, None]
        xrt = x_ref[t, :][:, None]
        urt = u_ref[t, :][:, None]
        At = Ain[t, :, :]
        Bt = Bin[t, :, :]
 
        q, r, Q, R = stagecost(xt, ut, xrt, urt, Qt, Rt)[1:5]
        S = np.zeros((ni, ns))

        Ptp = P_vec[t+1]
        ptp = p_vec[t+1]

        
        # P derivatives calculated "on the fly"
        P_x = q.squeeze() + At.T @ ptp
        P_u = r.squeeze() + Bt.T @ ptp
        P_xx = Q + At.T @ Ptp @ At
        P_ux = S + Bt.T @ Ptp @ At
        P_uu = R + Bt.T @ Ptp @ Bt 

        

        P_uu_reg = P_uu + reg * Iu
        
        K = -np.linalg.solve(P_uu_reg, P_ux)
        sigma = -np.linalg.solve(P_uu_reg, P_u) # if ct!=0 there is a "+ B.t @ Pt+1 @ ct"

        K_vec[t] = K
        sigma_vec[t] = sigma

        dV1_t = float(P_u @ sigma)
        dV2_t = 0.5*float(sigma.T @ P_uu @ sigma)

        dV1 += dV1_t
        dV2 += dV2_t

        
    
        pt = P_x - K.T @ P_uu @ sigma  #TODO che significa la aggiunta di K.T @ Q_u + Q_ux.T @ k
        Pt = P_xx - K.T @ P_uu @ K

        # pt = P_x + K.T @ P_uu @ sigma + K.T @ P_u + P_ux.T @ sigma
        # Pt = P_xx + K.T @ P_uu @ K + K.T @ P_ux + P_ux.T @ K
        

        P_vec[t, :, :] = Pt
        p_vec[t, :] = pt.squeeze()

        # symmetrize
        P_vec[t, :, :] = 0.5 * (P_vec[t, :, :] + P_vec[t, :, :].T)


    return sigma_vec, K_vec, p_vec, P_vec, dV1, dV2
        


def stagecost(x, u, x_ref, u_ref, Qt=np.eye(4), Rt=np.eye(1)):
    # Assuming inputs are 1D arrays (ns,) and (ni,)
    
    dx = x - x_ref
    du = u - u_ref

    l = 0.5 * dx.T @ Qt @ dx + 0.5 * du.T @ Rt @ du
    lx = Qt @ dx
    lu = Rt @ du
    
    return l, lx, lu, Qt, Rt

def termcost(xT, xT_ref, QT = np.eye(4)):
    dxT = xT - xT_ref

    lT = 0.5 * dxT.T @ QT @ dxT
    lx = QT @ dxT
  
    return lT, lx, QT



def main():

    #######################################
    # Parameters
    #######################################

    Armijo = True
    stepsize_0 = 1.0
    armijo_maxiters = 20
    c = 1e-5 # TODO: considerare c = 0.5
    beta = 0.7
    fixed_stepsize = 1e-2
    visu_descent_plot = False
    term_cond = 1e-6
    max_iters = 200
    
    params = PARAM_SET_2
    model = DoublePendulumFlexible(**params)
    
    ns = model.ns 
    ni = model.ni 

    ######################################
    # Reference curve
    ######################################

    T0 = 0.0
    Tend = 16.0
    dt = 0.01

    x_eq1, u_eq1 = model.find_equilibrium(fix_type='q2', value=np.pi/2)
    model.plot_equilibrium_point(x_eq1, u_eq1, title="1")
    x_eq2, u_eq2 = model.find_equilibrium(fix_type='q2', value=-3*np.pi/4)
    model.plot_equilibrium_point(x_eq2, u_eq2, title="2")

    # Select some equilibria as setpoints
    x_setpoints = np.array([
                            x_eq1,
                            x_eq2
                        ])
    
    u_setpoints = np.array([
                            u_eq1,
                            u_eq2
                        ]).reshape(-1,1)

    method = 'stepwise'  # 'polynomial' 'stepwise' 'cubic_spline'
    time, x_ref = trajectory_generator(T0, Tend, dt=dt, setpoints=x_setpoints, method=method)
    time, u_ref = trajectory_generator(T0, Tend, dt=dt, setpoints=u_setpoints, method=method)

    plot_reference_traj(time, x_ref, u_ref, method)

    steps = time.shape[0]

    
    x0 = x_ref[0,:].copy()
    Q = np.diag([10.0, 10.0, 0.1, 0.1])
    A0, B0 = model.discrete_jacobian(x0, u_ref[0,:], dt)
    QT = control.dare(A0, B0, Q, np.eye(ni))[0]
    R = 0.01*np.eye(ni)
   
    
    # ============ Initial trajectory guess ==============
    x_traj = np.zeros((max_iters, steps, ns))   # state seq.
    u_traj = np.zeros((max_iters, steps, ni))  # input seq.
    x_traj[0, 0] = x0

    for t in range(steps - 1):
        u_traj[0, t, :] = u_ref[t]
        x_traj[0, t+1, :] = solver_step(model, x_traj[0, t], dt, u_traj[0, t, 0], 'fe')

    u_traj[0, -1] = u_traj[0, -2]  # last input equal to previous for plotting purposes

  

    J = np.zeros(max_iters)      # collect cost
    descent = np.zeros(max_iters) # collect descent direction
    

    A_vec = np.zeros((steps -1, ns, ns))
    B_vec =  np.zeros((steps -1, ns, ni))

    ##############################
    # Main LQR loop
    ##############################
    for iter in range(max_iters-1):
        
        # Compute cost
        for t in range(steps-1):
            temp_cost = stagecost(x_traj[iter, t, :], u_traj[iter, t, :], x_ref[t, :], u_ref[t, :], Q, R)[0]
            J[iter] += temp_cost
        temp_cost = termcost(x_traj[iter, steps-1, :], x_ref[steps-1, :], QT)[0]
        J[iter] += temp_cost

        # Convergence check
        if J[iter] <= term_cond:
            print(f"Converged at iteration {iter} with cost {J[iter]:.3e}")
            max_iters = iter + 1
            break

        # Compute linearized dynamics along trajectory
        for t in range(steps - 1):
            A, B = model.discrete_jacobian(x_traj[iter, t, :], u_traj[iter, t, :], dt)
            A_vec[t, :, :] = A
            B_vec[t, :, :] = B
        
        ###################################
        # LTV-LQR to compute feedback law
        ###################################
        sigma_vec, K_vec, _,  _, dV1, dV2 = ltv_LQR(Ain=A_vec, Bin=B_vec, 
                                                     xk=x_traj[iter, :, :], uk=u_traj[iter, :, :], 
                                                     x_ref=x_ref, u_ref=u_ref, 
                                                     Qt=Q, Rt=R, QT=QT, reg=1e-6)

        descent[iter] = dV1 + dV2

        if descent[iter] >= 0:
            print("WARNING: predicted improvement is non-negative; "
                  "stopping.")
            max_iters = iter + 1
            break

        ##################################
        # Stepsize selection - ARMIJO
        ##################################
        
        if Armijo:
            stepsize, x_new, u_new, J_new = select_stepsize(
                model, stepsize_0, armijo_maxiters, c, beta,
                x_ref, u_ref, x0, dt,
                x_traj[iter, :, :], u_traj[iter, :, :], dV1, dV2,
                Q, R, QT, K_vec, sigma_vec
                )
        else:
            stepsize = fixed_stepsize

            x_new = np.zeros((steps, ns))
            u_new = np.zeros((steps, ni))

            x_new[0, :] = x0

            for t in range(steps-1): # TODO : question here about last addition to utemp
                dx = x_new[t, :] - x_traj[iter, t, :]
                u_new[t,:] = u_traj[iter,t,:] + stepsize * sigma_vec[t, :]  + K_vec[t, :] @ dx
                x_new[t+1, :] = solver_step(model, x_new[t, :], dt, u_new[t, :], 'fe')

        # update trajectory       
        x_traj[iter+1,:,:] = x_new
        u_traj[iter+1,:,:] = u_new


        if stepsize == 0.0:
            print("no improvement found, stopping.")
            max_iters = iter + 1
            break
        
        print(f"Iter {iter}: Cost = {J[iter]:.4e}, Stepsize = {stepsize:.3e}, Predicted descent = {descent[iter]:.4e}")

    # could implemnt early stopping based on cost improvement here
    
    x_star = x_traj[-1, :, :]
    u_star = u_traj[-1, :, :]
    u_star[-1, :] = u_star[-2, :] # for plotting purposes

    ############################
    # Plots
    ############################

    # cost and descent

    plt.figure('descent direction')
    plt.plot(np.arange(max_iters), -descent[:max_iters])
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)


    plt.figure('cost')
    plt.plot(np.arange(max_iters), J[:max_iters])
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

    # optimal trajectory

    

    fig, ax = plt.subplots(3, 2, sharex='all')
    ax = ax.flatten()

    ax[0].plot(time, x_star[:, 0], linewidth=2)
    ax[0].plot(time, x_ref[:, 0], 'g--', linewidth=2)
    ax[0].grid()

    ax[1].plot(time, x_star[:, 1], linewidth=2)
    ax[1].plot(time, x_ref[:, 1], 'g--', linewidth=2)
    ax[1].grid()

    ax[2].plot(time, x_star[:, 2], linewidth=2)
    ax[2].plot(time, x_ref[:, 2], 'g--', linewidth=2)
    ax[2].grid()
    
    ax[3].plot(time, x_star[:, 3], linewidth=2)
    ax[3].plot(time, x_ref[:, 3], 'g--', linewidth=2)
    ax[3].grid()

    ax[4].plot(time, u_star[:, 0],'r', linewidth=2)
    ax[4].plot(time, u_ref[:, 0], 'r--', linewidth=2)
    ax[4].grid()

    plt.show()


    animate(model, x_star, dt, target_fps=30)




if __name__ == "__main__":
    main()






