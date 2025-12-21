import numpy as np
from scipy.interpolate import CubicSpline
from model import DoublePendulumFlexible, PARAM_SET_2
import scipy as sp
import matplotlib.pyplot as plt

from solver import solver_step
from plots import plot_reference_traj

from animation import animate

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
    print(setpoints.shape)
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



def select_stepsize(model, stepsize_0, armijo_maxiters, c, beta, deltau, x_ref, u_ref,  x0, dt, u, J, descent_arm, Qt, Rt, steps, plot = False):


    stepsizes = []      # List of stepsizes
    costs_armijo = []   # List of costs associated to the stepsizes

    stepsize = stepsize_0 # initial stepsize

    ns = model.ns # number of states
    ni = model.ni # number of inputs


    ##############################
    # Main Armijo loop
    ##############################
    for i in range(armijo_maxiters):

        # Temp solution
        x_temp = np.zeros((steps, ns))
        u_temp = np.zeros((steps, ni))
        x_temp[0, :] = x0

        # Forward simulation with perturbed input
        for t in range(steps-1):
            u_temp[t,:] = u[t, :] + stepsize * deltau[t, :] # TODO: check if this is correct
            x_temp[t+1, :] = solver_step(model, x_temp[t, :], dt, u_temp[t, :], 'fe')

        # Compute cost
        J_temp = 0.0
        for t in range(steps-1):
            temp_cost = stagecost(x_temp[t, :], u_temp[t, :], x_ref[t,:], u_ref[t, :], Qt, Rt)[0]
            J_temp += temp_cost
        temp_cost = termcost(x_temp[-1,:], x_ref[-1,:], Qt)[0]
        J_temp += temp_cost

        # Save data for plotting
        stepsizes.append(stepsize)     # Save the stepsize
        costs_armijo.append(J_temp)    # Save the cost associated to the stepsize

        # Armijo condition
        if J_temp > J + c * stepsize * descent_arm:
            # Update the stepsize
            stepsize = beta*stepsize

        else:
            print('Armijo stepsize = {:.3e}'.format(stepsize))
            break
        
        if i == armijo_maxiters -1:
            print("WARNING: no stepsize was found with armijo rule!")

    return stepsize



def stagecost(x, u, x_ref, u_ref, Qt=np.eye(4), Rt=np.eye(1)):
    # Assuming inputs are 1D arrays (ns,) and (ni,)
    dx = x - x_ref
    du = u - u_ref
    
    l = 0.5 * dx @ Qt @ dx + 0.5 * du @ Rt @ du
    lx = Qt @ dx
    lu = Rt @ du
    
    return l, lx, lu

def termcost(xT, xT_ref, QT = np.eye(4), Tf : float = 0.0):
  lT = 0.5*(xT - xT_ref) @ QT @(xT - xT_ref).T
  lx = QT @ (xT - xT_ref).T
  
  return lT.squeeze(), lx




def main():

    #######################################
    # Parameters
    #######################################

    Armijo = True
    stepsize_0 = 1.0
    armijo_maxiters = 20
    c = 0.5
    beta = 0.7
    fixed_stepsize = 1e-2
    visu_descent_plot = False
    term_cond = 1e-6
    max_iters = 200
    
    model = DoublePendulumFlexible(**PARAM_SET_2)
    
    ns = model.ns 
    ni = model.ni 


    Qt = np.diag([10.0, 10.0, 1.0, 1.0])
    QT = 100.0*np.diag([10.0, 10.0, 1.0, 1.0])
    Rt = 0.01*np.eye(ni)

    ######################################
    # Reference curve
    ######################################

    T0 = 0.0
    Tend = 15.0
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

    steps = time.shape[0]
    
    plot_reference_traj(time, x_ref, u_ref, method)
    
    x0 = x_ref[0,:]

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

    lmbd = np.zeros((max_iters, steps, ns)) # lambdas - costate seq.

    deltau = np.zeros((max_iters, steps, ni)) # Du - descent direction
    dJ = np.zeros((max_iters, steps, ni))     # DJ - gradient of J wrt u

    J = np.zeros(max_iters)      # collect cost
    descent = np.zeros(max_iters) # collect descent direction
    descent_arm = np.zeros(max_iters) # collect descent direction

    

    ######################################
    # Main
    ######################################

    
    iter = 0

    
    x_traj[0, :, :] = x_init
    u_traj[0, :, :] = u_init
    x_traj[0, 0, :] = x0

    for t in range(steps-1):
        x_traj[0, t+1, :] = solver_step(model, x_traj[0, t, :], dt, u_traj[0, t, :], 'fe')
    
    # Loop over iteration k
    for iter in range(max_iters-1):
        
        # Compute cost at current iteration k
        for t in range(steps-1):
            temp_cost = stagecost(x_traj[iter, t, :], u_traj[iter, t, :], x_ref[t, :], u_ref[t, :], Qt, Rt)[0]
            J[iter] += temp_cost
        temp_cost = termcost(x_traj[iter, steps-1, :], x_ref[steps-1, :], QT)[0]
        J[iter] += temp_cost

        

        ##################################
        # Descent direction calculation
        ##################################

        # Final costate condition
        lmbd_temp = termcost(x_traj[iter, steps-1, :], x_ref[steps-1, :], QT)[1]
        lmbd[iter, steps-1, :] = lmbd_temp.squeeze()

        
        # Backward integration of costate equation
        for t in reversed(range(steps-1)):

            qt, rt = stagecost(x_traj[iter,t, :], u_traj[iter,t, :], x_ref[t, :], u_ref[t, :], Qt, Rt)[1:]
            
            At, Bt = model.discrete_jacobian(x_traj[iter, t, :], u_traj[iter, t, :], dt)


            lmbd_temp = At.T @ lmbd[iter, t+1, :].T + qt.squeeze()    # costate equation
            dJ_temp = Bt.T @ lmbd[iter, t+1, :].T + rt.squeeze()    # gradient of J wrt 
            
            deltau_temp = - dJ_temp

            lmbd[iter,t, :] = lmbd_temp.squeeze()
            dJ[iter,t,:] = dJ_temp.squeeze()
            deltau[iter,t,:] = deltau_temp.squeeze()

            # TODO: verifica cosa rappresentano loro
            descent[iter] += deltau[iter,t,:] @ deltau[iter,t,:].T
            descent_arm[iter] += dJ[iter,t,:] @ deltau[iter,t,:].T

        ##################################
        # Stepsize selection - ARMIJO
        ##################################

        if Armijo:
            stepsize = select_stepsize(model, stepsize_0, armijo_maxiters, c, beta,
                                        deltau[iter, :, :], x_ref, u_ref, x0, dt,
                                        u_traj[iter, :, :], J[iter], descent_arm[iter], Qt, Rt,  steps, visu_descent_plot)
        else:
            stepsize = fixed_stepsize

            
        ############################
        # Update the current solution
        ############################

        x_temp = np.zeros((steps, ns))
        u_temp = np.zeros((steps, ni))

        x_temp[0, :] = x0

        for t in range(steps-1):
            u_temp[t,:] = u_traj[iter,t,:] + stepsize * deltau[iter,t,:]
            x_temp[t+1, :] = solver_step(model, x_temp[t, :], dt, u_temp[t, :], 'fe')

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






