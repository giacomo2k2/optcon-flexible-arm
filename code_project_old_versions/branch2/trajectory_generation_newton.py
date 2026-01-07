#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         main_newton_method.py
# Description:  Trajectory generation for a non-linear flexible link via Newton method.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         December 2025
#########################################################################################

#######################################
# Imports
#######################################

# Python libriaries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import signal

plt.rcParams['font.family'] = 'serif'
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Model
from model import DoublePendulumFlexible, PARAM_SET_2

# Costs
import costs as cst

# Armijo
import armijo as armijo

# Trajectories
import trajectory as traj

# Solver
import solver as solver

# Animation & plots
import animation as anim
import plots as plots

# Control
import control as ctrl

#######################################
# Model
#######################################

model = DoublePendulumFlexible(**PARAM_SET_2)

ns = model.ns  # number of states
ni = model.ni  # number of inputs


def generate_optimal_trajectory(task, max_iters=300, n_mid=20, method='poly5', discr_method='fe', plot_results=True, plot_armijo=False, debug_print=True, initial_as_PDg=False) -> tuple[np.ndarray, np.ndarray, tuple, tuple, np.ndarray]:
    """
    Generate optimal trajectory using Newton method.

    Parameters
    ----------
    task : str
        Task to perform (choose from keys of traj.tasks).
    max_iters : int, optional
        Maximum number of iterations for the Newton method (default is 300).
    n_mid : int, optional
        Number of intermediate trajectories to plot (default is 20).
    method : str, optional
        Trajectory generation method ('poly5', 'poly7', 'stepwise') (default is 'poly5').
    discr_method : str, optional
        Discretization method ('fe' for Forward Euler, 'be' for Backward Euler, 'tr' for Trapezoidal) (default is 'fe').
    plot_results : bool, optional
        Whether to plot the results (default is True).
    plot_armijo : bool, optional
        Whether to plot the Armijo line search results (default is False).
    debug_print : bool, optional
        Whether to print debug information (default is True).
    initial_as_PDg : bool, optional
        Whether to use initial trajectory as PD+g controller (default is False).
    """
    #######################################
    # Algorithm parameters
    #######################################

    # General parameters
    term_cond = 1e-6
    
    # Amijo parameters
    stepsize_0=1.0
    c=0.5
    beta=0.7
    armijo_maxiters=20

    # Discretization method
    model.discr_method = discr_method

    # Base Cost matrix
    QQt_base = np.diag([10.0, 10.0, 1e-1, 1e-1])    # State cost
    RRt = 1e-2 * np.eye(1)                          # Input cost

    #######################################
    # Task and Equilibrium points
    #######################################

    selected_task = task  # choose from keys of equilibrium_traj

    data_dict = model.compute_equilibrium_manifold()

    selected_equilibrium = traj.tasks[selected_task]
    eq_pair1, eq_pair2 = selected_equilibrium['start'], selected_equilibrium['end']
    x_eq1, u_eq1 = eq_pair1
    x_eq2, u_eq2 = eq_pair2

    Tend = selected_equilibrium['time']
    angle_mult = selected_equilibrium['angle_mult']
    vel_mult = selected_equilibrium['vel_mult']


    if plot_results:
        plots.equilibrium_manifold(data_dict, eq_pair1, eq_pair2, print_results=False)

        x_eq = np.array([x_eq1, x_eq2]) # Equilibrium states (2,4)
        u_eq = np.array([u_eq1, u_eq2]) # Equilibrium inputs (2,)
        model.draw_pendulums(x_eq, u_eq, task=selected_task)


    # Select setpoints as equilibria
    x_setpoints = np.array([x_eq1, x_eq2])
    u_setpoints = np.array([u_eq1, u_eq2]).reshape(-1,1)


    #######################################
    # Time parameters
    #######################################

    T0 = 0.0        # Initial absolute time
    ti = 0.3        # Start time for trajectory segment (relative)
    tf = 0.7        # End time for trajectory segment (relative)
    dt = model.dt   # Time step
    t = np.arange(T0, Tend, dt) # Time vector
    TT = len(t)                 # Number of time steps

    ######################################
    # Reference curve
    ######################################

    # Generate reference trajectory for states and inputs
    time, xx_ref = traj.trajectory_generator(T0, Tend, ti=ti, tf=tf, dt=dt, setpoints=x_setpoints, method=method)
    time, uu_ref = traj.trajectory_generator(T0, Tend, ti=ti, tf=tf, dt=dt, setpoints=u_setpoints, method=method)

    # Plot reference trajectory
    if plot_results:
        plots.show_traj(time, xx_ref, uu_ref, eq_pair1, eq_pair2, "Reference " + method)

    #############################################
    # Compute QQT as solution of the DARE
    ##############################################

    # scale cost matrices 
    scale_QQt = np.diag([angle_mult, angle_mult, vel_mult, vel_mult])

    QQt = QQt_base @ scale_QQt

    # Compute QQT as solution of dare
    AA_ref, BB_ref = solver.jacobian_analytic(xx_ref[:,TT-1], uu_ref[:,TT-1], model)
    QQT = ctrl.dare(AA_ref, BB_ref, QQt, RRt)[0]


    ######################################
    # Arrays to store data
    ######################################

    xx = np.zeros((ns, TT, max_iters))          # State seq.
    uu = np.zeros((ni, TT, max_iters))          # Input seq.

    lmbd = np.zeros((ns, TT, max_iters))        # Lambdas - costate seq.

    deltau = np.zeros((ni,TT, max_iters))       # Du - descent direction
    dJ = np.zeros((ni,TT, max_iters))           # DJ - gradient of J wrt u

    JJ = np.zeros(max_iters)                    # Collect cost
    descent = np.zeros(max_iters)               # Collect descent direction norm
    descent_arm = np.zeros(max_iters)           # Collect descent condition value
    gradient_error = np.zeros((max_iters,))

    PP = np.zeros((ns, ns, TT, max_iters))      # P matrices for affine LQR
    pp = np.zeros((ns, TT, max_iters))          # p vectors for affine LQR
    KK = np.zeros((ni, ns, TT, max_iters))      # K gain matrices for affine LQR
    ss = np.zeros((ni, TT, max_iters))          # s vectors for affine LQR

    AA = np.zeros((ns, ns, TT, max_iters))
    BB = np.zeros((ns, ni, TT, max_iters))
    deltax = np.zeros((ns, TT, max_iters))

    ######################################
    # Initial Trajectory Guess via PD controller + gravity compensation
    ######################################

    x0 = xx_ref[:, 0]  # initial state

    uu_init = np.zeros((ni, TT))
    xx_init = np.zeros((ns, TT))

    if initial_as_PDg: 
        xx_init[:, 0] = x0
        for tt in range(TT-1):
            uu_init[:, tt] = solver.PD_controller_gravity_comp(model, xx_init[:, tt], xx_ref[:, tt], uu_ref[0, tt])
            xx_init[:, tt+1] = model.discrete_dynamics(xx_init[:, tt], uu_init[:, tt])
    else:
        pass

    uu_init[:,-1] = uu_init[:,-2]

    if plot_results:
        plots.trajectory_comparison(time, xx_ref, uu_ref, xx_init, uu_init, title="Initial Guess vs Reference")


    xx[:,:,0] = xx_init
    uu[:,:,0] = uu_init


    ############################
    # Iteration loop
    ############################

    # Iteration counter
    kk = 0

    for kk in range(max_iters-1):
        
        ############################
        # 0. Compute cost
        ############################
        
        JJ[kk] = cst.total_cost(xx[:,:,kk], uu[:,:,kk], xx_ref, uu_ref, QQt, RRt, QQT)
        
        ############################
        # 1. Compute descent direction
        ############################

        lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
        PP[:,:,TT-1,kk] = QQT
        pp[:,TT-1,kk] = lmbd[:,TT-1,kk].squeeze()

        for tt in reversed(range(TT-1)):
            qt, rt, QQt, RRt, SSt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:]
            At, Bt = solver.jacobian_analytic(xx[:, tt, kk], uu[:, tt, kk], model)
                
            AA[:, :, tt, kk] = At
            BB[:, :, tt, kk] = Bt

            # Regularization for 1st order approx.
            Qtk = QQt
            Rtk = RRt + np.eye(ni)*1e-6  # regularization
            Stk = SSt 

            # Costate equation
            lmbd_temp = At.T @ lmbd[:,tt+1,kk][:,None] + qt
            lmbd[:,tt,kk] = lmbd_temp.squeeze()


            # Single step LQR
            KK[:,:,tt,kk], ss[:,tt,kk], PP[:,:,tt,kk], pp[:,tt,kk] = solver.ltv_affine_LQR_step(
                At, Bt, Qtk, Rtk, Stk, PP[:,:,tt+1,kk], pp[:,tt+1,kk], qt, rt
            )

            # Gradient of cost w.r.t. input from LQR quantities
            dJ_temp = rt + Bt.T @ lmbd[:,tt+1,kk][:,None]
            dJ[:,tt,kk] = dJ_temp.squeeze()

        
        
        # Compute descent direction norm after backward pass
        for tt in range(TT-1):
            deltau[:,tt,kk] = ss[:,tt,kk] +  KK[:,:,tt,kk] @  deltax[:, tt, kk]   
            deltax[:, tt+1, kk] = AA[:, :, tt, kk] @ deltax[:, tt, kk] + BB[:, :, tt, kk] @ deltau[:,tt,kk] 
            
            descent[kk] += deltau[:,tt,kk].T @ deltau[:,tt,kk]  
            descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk]  

        
        ############################
        # 2. Stepsize selection - ARMIJO
        ############################
        
        stepsize, grad_error_temp = armijo.select_stepsize_newton(model, stepsize_0, armijo_maxiters, c, beta, 
                                            ss[:,:,kk], KK[:,:,:,kk], xx_ref, uu_ref, 
                                            x0, xx[:,:,kk], uu[:,:,kk], JJ[kk], descent_arm[kk], 
                                            QQt, RRt, QQT, kk, plot=plot_armijo)
        
        gradient_error[kk] = grad_error_temp
     
        ############################
        # 3. Forward closed-loop integration
        ############################
        
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT)) 
        xx_temp[:,0] = x0

        
        for tt in range(TT-1):
            dx = xx_temp[:,tt] - xx[:,tt,kk]
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize * ss[:,tt,kk] + KK[:,:,tt,kk] @ dx
            xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])
            
            
        uu_temp[:,-1] = uu_temp[:,-2]

        # Store trajectories
        xx[:,:,kk+1] = xx_temp.copy()
        uu[:,:,kk+1] = uu_temp.copy()

        
        ############################
        # 4. Visualization
        ############################
        
        if debug_print:
            log_str = (f"Iter: {kk} | "
                f"Cost: {JJ[kk]:.6e} | "
                f"Stepsize: {stepsize:.3e} | "
                f"Descent norm: {descent[kk]:.6e} | "
                f"Descent cond: {descent_arm[kk]:.6e}")
            print(f"\r{log_str}".ljust(120), end="", flush=True)
            
        else:
            dots = "." * (kk % 4)
            print(f"\rComputing optimal trajectory{dots:3}", end="", flush=True)

        ############################
        # 5. Termination condition
        ############################

        if descent[kk] < term_cond:
            print(f"\n\nConvergence reached at iteration {kk}!")
            max_iters = kk
            break

        if np.isnan(descent[kk]) or np.isnan(descent_arm[kk]):
            print(f"\n\nNaN detected at iteration {kk}, time {tt}! Stopping computation.")
            max_iters = kk
            break

    ##################################
    # Results
    ##################################
    xx_star = xx[:,:,max_iters-1]
    uu_star = uu[:,:,max_iters-1]
    uu_star[:,-1] = uu_star[:,-2] 
    descent[max_iters-1] = descent[ max_iters-2]

    if plot_results:
        # Descent Direction
        plots.descent_direction(descent, max_iters)

        # Cost
        plots.cost(JJ, max_iters)

        # Optimal vs Reference Trajectories
        plots.trajectory_comparison(time, xx_ref, uu_ref, xx_star, uu_star, title="Optimal vs Reference")

        # Intermediate Trajectories
        plots.trajectory_comparison_mid(time, xx, uu, n_mid, max_iters, title="Trajectory Comparison with Intermediate Trajectories")

        # Gradient error
        plots.gradient_error(gradient_error, max_iters)

    # Animation of the optimal trajectory
    if plot_results:
        anim.animate_comparison(model, xx_star, xx_ref, (x_eq1, u_eq1), (x_eq2, u_eq2), target_fps=30)

    return xx_star, uu_star, eq_pair1, eq_pair2, time



if __name__ == "__main__":
    generate_optimal_trajectory('swing_up', max_iters=300, n_mid=20, method='stepwise', discr_method='fe', plot_results=True, plot_armijo=False, debug_print=True, initial_as_PDg=True)