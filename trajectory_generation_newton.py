import numpy as np
from scipy.interpolate import CubicSpline
from model import DoublePendulumFlexible, PARAM_SET_2
import scipy as sp
import matplotlib.pyplot as plt

from solver import solver_step
from plots import plot_trajectory

from animation import animate

import control

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
                    x_traj, u_traj, J_old, descent_slope, 
                    Qt, Rt, K, sigma, q, r, qT):
    
    alpha = stepsize_0
    steps = x_traj.shape[0]
    ns = x_traj.shape[1]
    ni = u_traj.shape[1]
    
    # Check if descent slope is valid (should be negative for descent)
    if descent_slope >= 0:
        print(f"Warning: Descent slope is positive ({descent_slope:.3e}). Check gradients.")
        return 0.0, x_traj, u_traj, J_old
    
    # Armijo backtracking line search
    for i in range(armijo_maxiters):
        x_cand = np.zeros((steps, ns))
        u_cand = np.zeros((steps, ni))
        x_cand[0, :] = x0
        J_cand = 0
        
        feasible = True
        
        for t in range(steps-1):
            # --- CRITICAL FIX: Closed-Loop Update ---
            # We calculate deviation from the *nominal* trajectory (x_traj)
            # to the *current candidate* trajectory (x_cand)
            dx = x_cand[t, :] - x_traj[t, :]
            
            # Apply feedforward (scaled by alpha) + feedback (stabilization)
            u_cand[t, :] = u_traj[t, :] + alpha * sigma[t, :] + K[t, :, :] @ dx
            
            # Forward Integrate
            x_cand[t+1, :] = solver_step(model, x_cand[t, :], dt, u_cand[t, :], 'fe')
            
            # Accumulate Cost
            dx = x_cand[t] - x_ref[t]
            du = u_cand[t] - u_ref[t]
            J_cand += 0.5 * dx @ Qt @ dx + 0.5 * du @ Rt @ du + q[t, :] @ dx + r[t, :] @ du
            
            # Safety check: if cost explodes, stop this alpha early
            if J_cand > 5 * J_old: 
                feasible = False
                break
        
        if feasible:
            dxT = x_cand[-1, :] - x_ref[-1, :]
            J_cand += 0.5 * dxT @ Qt @ dxT + qT @ dxT
            
            # --- ARMIJO CONDITION ---
            # Check if actual improvement is at least a fraction (c) of expected improvement
            expected_decrease = c * alpha * descent_slope
            
            if J_cand <= J_old + expected_decrease:
                print(f'Armijo stepsize = {alpha:.3e} | Cost: {J_old:.4e} -> {J_cand:.4e}')
                return alpha, x_cand, u_cand, J_cand
        
        # If we are here, the step failed; reduce alpha and retry
        alpha *= beta
        
    print("WARNING: Armijo failed to find a step. Returning small step.")
    return 0.0, x_traj, u_traj, J_old



def ltv_LQR(Ain, Bin, Qin, Rin, Sin, Qfin, T, x0, qin = None, rin = None, qfin = None):

	
    try:
        # check if matrix is (.. x .. x TT) - 3 dimensional array 
        lA, ns, _ = Ain.shape
    except:
        # if not 3 dimensional array, make it (.. x .. x 1)
        Ain = Ain[None, :,:]
        lA, ns, _ = Ain.shape

    try:  
        lB, _, ni= Bin.shape
    except:
        Bin = Bin[None,:,:]
        lB, _, ni = Bin.shape

    try:
        lQ, nQ, _ = Qin.shape
    except:
        Qin = Qin[None,:,:]
        lQ, nQ, _ = Qin.shape

    try:
        lR, nR, _ = Rin.shape
    except:
        Rin = Rin[None,:,:]
        lR, nR, _ = Rin.shape

    try:
        lS, nSi, nSs,  = Sin.shape
    except:
        Sin = Sin[None,:,:]
        lS, nSi, nSs  = Sin.shape

    # Check dimensions consistency -- safety
    if nQ != ns:
        print("Matrix Q does not match number of states")
        exit()
    if nR != ni:
        print("Matrix R does not match number of inputs")
        exit()
    if nSs != ns:
        print("Matrix S does not match number of states")
        exit()
    if nSi != ni:
        print("Matrix S does not match number of inputs")
        exit()


    if lA < T:
        Ain = Ain.repeat(T, axis=0)
    if lB < T:
        Bin = Bin.repeat(T, axis=0)
    if lQ < T:
        Qin = Qin.repeat(T, axis=0)
    if lR < T:
        Rin = Rin.repeat(T, axis=0)
    if lS < T:
        Sin = Sin.repeat(T, axis=0)

    # Check for affine terms

    augmented = False

    if Qin is not None or Rin is not None or Qfin is not None:
        augmented = True
        print("Augmented term!")

    K = np.zeros((T, ni, ns))
    sigma = np.zeros((T, ni))
    P = np.zeros((T, ns, ns))
    p = np.zeros((T, ns))

    Q = Qin
    R = Rin
    S = Sin
    QT = Qfin
    
    q = qin
    r = rin

    qT = qfin

    A = Ain
    B = Bin

    x = np.zeros((T, ns))
    u = np.zeros((T, ni))

    
    x[0,:] = x0
    
    P[-1,:,:] = QT
    p[-1,:] = qT

    delta_V = 0.0
    
    # Solve Riccati equation
    for t in reversed(range(T-1)):
        Qt = Q[t, :, :]
        qt = q[t, :][:,None]
        Rt = R[t, :, :]
        rt = r[t, :][:,None]
        At = A[t, :, :]
        Bt = B[t, :, :]
        St = S[t,:,:]
        Ptp = P[t+1,:,:]
        ptp = p[t+1, :][:,None]

        Mt_inv = np.linalg.inv(Rt + Bt.T @ Ptp @ Bt)
        mt = rt + Bt.T @ ptp
        
        Pt = At.T @ Ptp @ At - (Bt.T @ Ptp @ At + St).T @ Mt_inv @ (Bt.T @ Ptp @ At + St) + Qt
        pt = At.T @ ptp - (Bt.T @ Ptp @ At + St).T @ Mt_inv @ mt + qt

        P[t,:,:] = Pt
        p[t,:] = pt.squeeze()   


    # Evaluate KK
    
    for t in range(T-1):
        Qt = Q[t,:,:]
        qt = q[t,:][:,None]
        Rt = R[t,:,:]
        rt = r[t,:][:,None]
        At = A[t,:,:]
        Bt = B[t,:,:]
        St = S[t,:,:]

        Ptp = P[t+1,:,:]
        ptp = p[t+1,:][:,None]

        # Check positive definiteness

        Mt_inv = np.linalg.inv(Rt + Bt.T @ Ptp @ Bt)
        mt = rt + Bt.T @ ptp

        # for other purposes we could add a regularization step here...

        K[t,:,:] = -Mt_inv@(Bt.T@Ptp@At + St)
        sigma_t = -Mt_inv@mt

        sigma[t,:] = sigma_t.squeeze()

        delta_V += (sigma[t, :].T @ mt).item()

    for t in range(T - 1):
        # Trajectory

        u[t, :] = K[t,:,:]@x[t, :] + sigma[t,:]
        x_p = A[t,:,:]@x[t, :] + B[t,:,:]@u[t, :]

        x[t+1, :] = x_p

        xout = x
        uout = u

    return K, sigma, P, delta_V
        



# TODO: sistemare trasposti
def stagecost(x, u, x_ref, u_ref, Qt=np.eye(4), Rt=np.eye(1), q=None, r=None):
    # Assuming inputs are 1D arrays (ns,) and (ni,)
    
    l = 0.5 * x @ Qt @ x + 0.5 * u @ Rt @ u + (q @ x if q is not None else 0)+ (r @ u if r is not None else 0)
    lx = Qt @ x + (q.T if q is not None else 0)
    lu = Rt @ u + (r.T if r is not None else 0)
    
    return l, lx, lu

def termcost(xT, xT_ref, QT = np.eye(4), Tf : float = 0.0, qT = None):
  lT = 0.5* xT @ QT @ xT + (qT @ xT if qT is not None else 0)
  lx = QT @ xT + (qT.T if qT is not None else 0)
  
  return lT.squeeze(), lx



def main():

    #######################################
    # Parameters
    #######################################

    Armijo = False
    stepsize_0 = 1.0
    armijo_maxiters = 20
    c = 0.5
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
    Tend = 4.0
    dt = 0.01


    x_eq = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [np.pi/2, -np.pi/2, 0.0, 0.0],
        [0.0, np.pi, 0.0, 0.0],
        [np.pi, np.pi, 0.0, 0.0]
    ])

    x_setpoints = np.array([
                            [0.0, 0.0, 0.0, 0.0],
                            x_eq[1,:]
                        ])

    u_setpoints = []

    for i in range(x_setpoints.shape[0]):
        
        u_setpoints.append(model.equilibrium_input(x_eq[i,:]))
    
    
    u_setpoints = np.stack(u_setpoints)

    method = 'stepwise'  # 'polynomial' 'stepwise' 'cubic_spline'
    time, x_ref = trajectory_generator(T0, Tend, dt=dt, setpoints=x_setpoints, method=method)
    time, u_ref = trajectory_generator(T0, Tend, dt=dt, setpoints=u_setpoints, method=method)

    steps = time.shape[0]
    
    #plot_trajectory(time, x_ref, u_ref, method)
    
    x0 = x_ref[0,:]

    Q = np.diag([10.0, 10.0, 1.0, 1.0])

    A, B = model.discrete_jacobian(x0, u_ref[0,:], dt)
    QT = control.dare(A, B, Q, np.eye(ni))[0]
    r_coeff = 0.5

    R = r_coeff*np.eye(ni)
    S = np.zeros((ni,ns))

    q = np.zeros((steps, ns))
    r = np.zeros((steps, ni))


    for t in range(steps):
        q[t, :] = (-Q @ x_ref[t, :]).squeeze()
        r[t, :] = (-R @ u_ref[t, :]).squeeze()

    qT = -QT @ x_ref[-1,:]


    ######################################
    # Initial guess
    ######################################

    x_init = np.zeros((steps, ns))
    u_init = np.zeros((steps, ni))

    ######################################
    # Arrays to store data
    ######################################

    x_traj = np.zeros((max_iters, steps, ns))   # state seq.
    u_traj = np.zeros((max_iters, steps, ni))   # input seq.


    J = np.zeros(max_iters)      # collect cost
    descent = np.zeros(max_iters) # collect descent direction
    
    

    ######################################
    # Main
    ######################################

    
    iter = 0
    
    x_traj[0, :, :] = x_init
    u_traj[0, :, :] = u_init
    x_traj[0, 0, :] = x0
    
    Q_in = Q
    R_in = R
    S_in = S
    q_in = q
    r_in = r
    qT_in = qT

    A_vec = np.zeros((steps, ns, ns))
    B_vec =  np.zeros((steps, ns, ni))

    for t in range(steps-1):
        x_traj[0, t+1, :] = solver_step(model, x_traj[0, t, :], dt, u_traj[0, t, :], 'fe')
    
    for iter in range(max_iters-1):

        # step 2: Compute new state_input trajectory
        # calculate cost
        for t in range(steps-1):
            temp_cost = stagecost(x_traj[iter, t, :], u_traj[iter, t, :], x_ref[t, :], u_ref[t, :], Q_in, R_in, q_in[t, :], r_in[t, :])[0]
            J[iter] += temp_cost
        
        temp_cost = termcost(x_traj[iter, steps-1, :], x_ref[steps-1, :], Q_in, qT_in)[0]
        J[iter] += temp_cost

     
        for t in range(steps):
            
            A, B = model.discrete_jacobian(x_traj[iter, t, :], u_traj[iter, t, :], dt)
            A_vec[t, :, :] = A
            B_vec[t, :, :] = B
            

        ###################################
        # LTV-LQR to compute feedback law
        ###################################


        K, sigma, _, dV = ltv_LQR(A_vec, B_vec, Q_in, R_in, S_in, QT, steps, x0, q_in, r_in, qT_in)

        
        ##################################
        # Stepsize selection - ARMIJO
        ##################################
        
        if Armijo:
            stepsize, _, _, _ = select_stepsize(
                model, stepsize_0, armijo_maxiters, c, beta,
                x_ref, u_ref, x0, dt,
                x_traj[iter, :, :], u_traj[iter, :, :], J[iter], dV,
                Q_in, R_in, K, sigma, q_in, r_in, qT_in
            )
        else:
            stepsize = fixed_stepsize


        # update trajectory

        x_temp = np.zeros((steps, ns))
        u_temp = np.zeros((steps, ni))

        x_temp[0, :] = x0

        for t in range(steps-1): # TODO : question here about last addition to utemp
            
            u_temp[t,:] = u_traj[iter,t,:] + stepsize * sigma[t, :]  + K[t, :] @ (x_temp[t, :] - x_traj[iter, t, :])
            x_temp[t+1, :] = solver_step(model, x_temp[t, :], dt, u_temp[t, :], 'fe')

        x_traj[iter+1,:,:] = x_temp
        u_traj[iter+1,:,:] = u_temp

        

        ############################
        # Termination condition
        ############################

        print('Iter = {}\t Cost = {:.3e}'.format(iter, J[iter]))

        if J[iter] <= term_cond:

            max_iters = iter


    x_star = x_traj[max_iters-1, :, :]
    u_star = u_traj[max_iters-1, :, :]
    u_star[-1, :] = u_star[-2, :] # for plotting purposes

    ############################
    # Plots
    ############################

    # cost and descent

    plt.figure('descent direction')
    plt.plot(np.arange(max_iters), descent[:max_iters])
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






