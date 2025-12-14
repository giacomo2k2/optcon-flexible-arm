import numpy as np
from scipy.interpolate import CubicSpline
from model import DoublePendulumFlexible, PARAM_SET_2
import scipy as sp
import matplotlib.pyplot as plt

from solver import solver_step
from plots import plot_trajectory


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



def select_stepsize(model, stepsize_0, armijo_maxiters, c, beta, deltau, x_ref, u_ref,  x0, u, J, descent_arm, QT, plot = False):


    T = u.shape[1]


    stepsizes = []  # list of stepsizes
    costs_armijo = []

    stepsize = stepsize_0

    ns = 4
    ni = 1

    # #compute initial cost
    # JJ_armijo = 0
    # #first dynamic integration
    # x = np.zeros((ns,TT))
    # x[:,0] = x0
    # for tt in range(TT-1):
    #       x[:,tt+1] = dyn.dynamics(x[:,tt], u[:,tt])[0]
    #       temp_cost = cst.stagecost(x[:,tt], u[:,tt], x_ref[:,tt], u_ref[:,tt])[0]
    #       JJ_armijo += temp_cost
    # temp_cost = cst.termcost(x[:,-1], x_ref[:,-1])[0]
    # JJ_armijo += temp_cost

    for i in range(armijo_maxiters):

        # temp solution update

        x_temp = np.zeros((T, ns))
        u_temp = np.zeros((T, ni))

        x_temp[0, :] = x0

        for t in range(T-1):
                u_temp[t,:] = u[t, :] + stepsize*deltau[t, :]
                x_temp[t+1, :] = model.dynamics(x_temp[t, :], u_temp[t, :])[0]

        # temp cost calculation
        J_temp = 0

        for t in range(T-1):
            temp_cost = stagecost(x_temp[t, :], u_temp[t, :], x_ref[t,:], u_ref[t, :])[0]
            J_temp += temp_cost

        #print(f'x_temp shape: {x_temp[:,-1].shape}, x_ref shape: {x_temp[:,-1].shape}, QQT shape: {QQT.shape}')
        temp_cost = termcost(x_temp[-1,:], x_ref[-1,:],QT)[0]
        J_temp += temp_cost

        stepsizes.append(stepsize)      # save the stepsize
        costs_armijo.append(J_temp)    # save the cost associated to the stepsize

        if J_temp > J + c*stepsize*descent_arm:
                # update the stepsize
                stepsize = beta*stepsize

        else:
                print('Armijo stepsize = {:.3e}'.format(stepsize))
                break
        
        if i == armijo_maxiters -1:
                print("WARNING: no stepsize was found with armijo rule!")

    return stepsize

# TODO: sistemare trasposti
def stagecost(x,u, x_ref, u_ref, Qt = np.eye(4), Rt = np.eye(1)):
    x = x[None, :]
    u = u[None, :]

    x_ref = x_ref[None, :]
    u_ref = u_ref[None, :]

    ll = 0.5*(x - x_ref) @ Qt @(x - x_ref).T + 0.5*(u - u_ref) @ Rt @(u - u_ref).T
 
    lx = Qt @ (x - x_ref).T
    lu = Rt @ (u - u_ref).T

    return ll.squeeze(), lx, lu


def termcost(xT, xT_ref, QT = np.eye(4), Tf : float = 0.0):
  llT = 0.5*(xT - xT_ref) @ QT @(xT - xT_ref).T
  lTx = QT @ (xT - xT_ref).T
  
  return llT.squeeze(), lTx




def main():

    Armijo = True
    stepsize_0 = 1.0
    armijo_maxiters = 20
    cc = 1e-4
    beta = 0.5
    fixed_stepsize = 1e-3
    visu_descent_plot = False
    term_cond = 1e-6
    max_iters = 50
    tf = 10.0
    
    params = PARAM_SET_2
    model = DoublePendulumFlexible(**params)
    
    ns = model.ns 
    ni = model.ni 


    Qt = np.diag([10.0, 1.0, 10.0, 1.0])
    Rt = 0.01*np.eye(ni)
    # Reference curve
    ######################################

    T0 = 0.0
    Tend = 10.0
    dt = 0.01


    x_eq = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [np.pi, 0.0, 0.0, 0.0],
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

    T = time.shape[0]
    
    #plot_trajectory(time, x_ref, u_ref, method)
    
    x0 = x_ref[0,:]

    ######################################
    # Initial guess
    ######################################

    x_init = np.zeros((T, ns))
    u_init = np.zeros((T, ni))

    ######################################
    # Arrays to store data
    ######################################

    x_traj = np.zeros((max_iters, T, ns))   # state seq.
    u_traj = np.zeros((max_iters, T, ni))   # input seq.

    lmbd = np.zeros((max_iters, T, ns)) # lambdas - costate seq.

    deltau = np.zeros((max_iters, T, ni)) # Du - descent direction
    dJ = np.zeros((max_iters, T, ni))     # DJ - gradient of J wrt u

    J = np.zeros(max_iters)      # collect cost
    descent = np.zeros(max_iters) # collect descent direction
    descent_arm = np.zeros(max_iters) # collect descent direction

    

    ######################################
    # Main
    ######################################

    print('-*-*-*-*-*-')

    iter = 0

    # Order: Iteration index kk, time index tt, state/input index
    x_traj[0, :, :] = x_init
    u_traj[0, :, :] = u_init
    x_traj[0, 0, :] = x0

    for t in range(T-1):
        x_traj[iter, t+1, :] = solver_step(model, x_traj[iter, t, :], u_traj[iter, t, :])
    
    
    # calculate cost
    for t in range(T-1):
        temp_cost = stagecost(x_traj[iter, t, :], u_traj[iter, t, :], x_ref[t, :], u_ref[t, :])[0]
        J[iter] += temp_cost
    
    temp_cost = termcost(x_traj[iter, T-1, :], x_ref[T-1, :])[0]
    J[iter] += temp_cost


    ##################################
    # Descent direction calculation
    ##################################

    lmbd_temp = termcost(x_traj[iter, T-1, :], x_ref[T-1, :])[1]
    lmbd[iter, T-1, :] = lmbd_temp.squeeze()

    for t in reversed(range(T-1)):  # integration backward in time

        qt, rt = stagecost(x_traj[iter,t, :], u_traj[iter,t, :], x_ref[t, :], u_ref[t, :])[1:]
        

        #print(f'dfx shape: {dfx.shape}, dfu shape: {dfu.shape}')

        At, Bt = model.jacobian(x_traj[iter, t, :], u_traj[iter, t, :][0])

        #print(f'Bt shape: {Bt.shape}, lmbd shape: {lmbd[:,tt+1,kk].shape}, rt shape: {rt.shape}')

        # TODO: trasposto in A?
        lmbd_temp = At @ lmbd[iter, t+1, :].T + qt.squeeze()    # costate equation

        
        dJ_temp = Bt.T @ lmbd[iter, t+1, :].T + rt.squeeze()    # gradient of J wrt u
        deltau_temp = - dJ_temp

        lmbd[iter,t, :] = lmbd_temp.squeeze()
        dJ[iter,t,:] = dJ_temp.squeeze()
        deltau[iter,t,:] = deltau_temp.squeeze()

        descent[iter] += deltau[iter,t,:] @ deltau[iter,t,:].T
        descent_arm[iter] += dJ[iter,t,:] @ deltau[iter,t,:].T

    ##################################
    # Stepsize selection - ARMIJO
    ##################################

    if Armijo:
        stepsize = select_stepsize(model, stepsize_0, armijo_maxiters, cc, beta,
                                    deltau[iter, :, :], x_ref, u_ref, x0, 
                                    u_traj[iter, :, :], J[iter], descent_arm[iter], Qt, visu_descent_plot)
    else:
        stepsize = fixed_stepsize

        
    ############################
    # Update the current solution
    ############################

    x_temp = np.zeros((T, ns))
    u_temp = np.zeros((T, ni))

    x_temp[0, :] = x0


    
    
    for t in range(T-1):
        u_temp[t,:] = u_traj[iter,t,:] + stepsize * deltau[iter,t,:]
        x_temp[t+1, :] = model.dynamics(x_temp[t, :], u_temp[t, :][0])

    x_traj[iter+1,:,:] = x_temp
    u_traj[iter+1,:,:] = u_temp

    ############################
    # Termination condition
    ############################

    print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(iter, descent[iter], J[iter]))

    if descent[iter] <= term_cond:

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
    #plt.xlabel('$k$')
    #plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)


    plt.figure('cost')
    plt.plot(np.arange(max_iters), J[:max_iters])
    #plt.xlabel('$k$')
    #plt.ylabel('$J(\\mathbf{u}^k)$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

    # optimal trajectory

    t_hor = np.linspace(0,tf,T)

    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    axs[0].plot(t_hor, x_star[:, 0], linewidth=2)
    axs[0].plot(t_hor, x_ref[:, 0], 'g--', linewidth=2)
    axs[0].grid()
    #axs[0].set_ylabel('$x_1$')

    axs[1].plot(t_hor, x_star[:, 1], linewidth=2)
    axs[1].plot(t_hor, x_ref[:, 1], 'g--', linewidth=2)
    axs[1].grid()
    #axs[1].set_ylabel('$x_2$')

    axs[2].plot(t_hor, u_star[:, 0],'r', linewidth=2)
    axs[2].plot(t_hor, u_ref[:, 0], 'r--', linewidth=2)
    axs[2].grid()
    #axs[2].set_ylabel('$u$')
    #axs[2].set_xlabel('time')
    

    plt.show()




if __name__ == "__main__":
    main()






