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

# Standard libriaries
import numpy as np
import matplotlib.pyplot as plt 
import signal
import control as ctrl

# Custom libraries
# - Costs
import costs as cst
# - Armijo
import armijo as armijo
# - Solver
import solver as solver
# - Animation
import animation as anim
# - Plots
import plots as plots
# - Model
from model import DoublePendulumFlexible

# set font family
plt.rcParams['font.family'] = 'serif'
# set interrupt signal to default to allow Ctrl+C interruption
signal.signal(signal.SIGINT, signal.SIG_DFL)


######################################
# Functions
######################################

def generate_optimal_trajectory(model: DoublePendulumFlexible, optimal_gen_params:dict, reference_trajectory_params:dict, armijo_params: dict, debug_params:dict, folder: str = None) -> tuple[np.ndarray, np.ndarray, tuple, tuple, np.ndarray]:
    """
    Generate optimal trajectory via Newton method.

    Parameters
    ----------
    model : DoublePendulumFlexible
        - ns : int
            Number of states.
        - ni : int
            Number of inputs.
        - dt : float
            Discretization time step.
    optimal_gen_params : dict
        - max_iters : int
            Maximum number of iterations.
        - n_mid : int
            Number of intermediate trajectories to plot.
        - descent_threshold : float
            Stopping criterion for descent norm.
        - QQt : np.ndarray
            State cost matrix.
        - RRt : np.ndarray
            Input cost matrix.
    reference_trajectory_params : dict
        - time : np.ndarray
            Time vector.
        - xx_ref : np.ndarray
            Reference state trajectory.
        - uu_ref : np.ndarray
            Reference input trajectory.
        - eq_pair1 : tuple
            Initial equilibrium point (x_eq1, u_eq1).
        - eq_pair2 : tuple
            Final equilibrium point (x_eq2, u_eq2).
    armijo_params : dict
        - stepsize_0 : float
            Initial stepsize for Armijo search.
        - c : float
            Armijo condition constant.
        - beta : float
            Stepsize reduction factor.
        - max_iters : int
            Maximum number of iterations for Armijo search.
    debug_params : dict
        - plot_flag : bool
            Whether to plot the results.
        - debug_print : bool
            Whether to print debug information.
        - plot_armijo : bool
            Whether to plot Armijo descent.
        - save_armijo : bool
            Whether to save Armijo descent plots.
        - save_armijo_iters : int
            Number of Armijo iterations to save.
    folder : str, optional
        Folder to save plots, by default None.

    Returns
    -------
    xx_star : np.ndarray
        Optimal state trajectory.
    uu_star : np.ndarray
        Optimal input trajectory.
    time : np.ndarray
        Time vector.
    """

    ######################################
    # Algorithm parameters
    ######################################

    # Compute equilibrium manifold
    ns = model.ns                                                       # Number of states
    ni = model.ni                                                       # Number of inputs
    dt = model.dt                                                       # Discretization time step

    # Optimal Generation parameters
    max_iters = optimal_gen_params['max_iters']                         # Maximum number of iterations  
    n_mid = optimal_gen_params['n_mid']                                 # Number of intermediate trajectories to plot
    descent_threshold = optimal_gen_params['descent_threshold']         # Stopping criterion for descent norm
    QQt = optimal_gen_params['QQt']                                     # State cost matrix
    RRt = optimal_gen_params['RRt']                                     # Input cost matrix

    # Plot and debug parameters
    plot_flag = debug_params['plot_flag']                               # Whether to plot the results
    debug_print = debug_params['debug_print']                           # Whether to print debug information
    plot_armijo = debug_params['plot_armijo']                           # Whether to plot Armijo descent
    save_armijo = debug_params['save_armijo']                           # Whether to save Armijo descent plots
    save_armijo_iters = debug_params['save_armijo_iters']               # Number of Armijo iterations to save
    
    # Armijo parameters
    stepsize_0=armijo_params['stepsize_0']                              # Initial stepsize for Armijo search   
    c=armijo_params['c']                                                # Armijo condition constant
    beta=armijo_params['beta']                                          # Stepsize reduction factor
    armijo_maxiters=armijo_params['max_iters']                          # Maximum number of iterations for Armijo search

    ######################################
    # Reference trajectory unpacking
    ######################################

    time = reference_trajectory_params['time']                          # Time vector   
    xx_ref = reference_trajectory_params['xx_ref']                      # Reference state trajectory 
    uu_ref = reference_trajectory_params['uu_ref']                      # Reference input trajectory
    eq_pair1 = reference_trajectory_params['eq_pair1']                  # Initial equilibrium point (x_eq1, u_eq1)
    eq_pair2 = reference_trajectory_params['eq_pair2']                  # Final equilibrium point (x_eq2, u_eq2)

    TT = time.shape[0]                                                  # Number of time steps

    ############################################################
    # Compute terminal cost matrix and check PBH stabilizability
    ############################################################

    x_eq1, u_eq1 = eq_pair1
    x_eq2, u_eq2 = eq_pair2
    result, message, A_final, B_final = model.check_pbh_stabilizability(x_eq2, u_eq2)   # Check PBH stabilizability at final equilibrium
    
    if not result:
        print(f"WARNING: The system is not locally stabilizable at the final equilibrium point. {message}")
    else:
        print(f"The system is locally stabilizable at the final equilibrium point. {message}")
        
    QQT, _, _ = ctrl.dare(A_final, B_final, QQt, RRt)                   # Terminal state cost matrix from discrete-time algebraic Riccati equation    

    ######################################
    # Arrays to store data
    ######################################

    xx = np.zeros((ns, TT, max_iters))                                  # State seq.
    uu = np.zeros((ni, TT, max_iters))                                  # Input seq.

    lmbd = np.zeros((ns, TT, max_iters))                                # Lambdas - costate seq.

    deltau = np.zeros((ni,TT, max_iters))                               # Du - descent direction
    dJ = np.zeros((ni,TT, max_iters))                                   # DJ - gradient of J wrt u

    JJ = np.zeros(max_iters)                                            # Collect cost
    descent = np.zeros(max_iters)                                       # Collect descent direction norm
    descent_arm = np.zeros(max_iters)                                   # Collect descent condition value
    gradient_error = np.zeros((max_iters,))                     

    PP = np.zeros((ns, ns, TT, max_iters))                              # P matrices for affine LQR
    pp = np.zeros((ns, TT, max_iters))                                  # p vectors for affine LQR
    KK = np.zeros((ni, ns, TT, max_iters))                              # K gain matrices for affine LQR
    ss = np.zeros((ni, TT, max_iters))                                  # s vectors for affine LQR

    AA = np.zeros((ns, ns, TT, max_iters))                              # State matrices    
    BB = np.zeros((ns, ni, TT, max_iters))                              # Input matrices
    deltax = np.zeros((ns, TT, max_iters))                              # Dx - state deviation

    
    ######################################
    # Initial Trajectory Guess
    ######################################

    # Set velocity of second link to zero in reference trajectory
    xx_ref[3,:] = 0.0

    x0 = xx_ref[:, 0]                   # Initial state

    uu_init = np.zeros((ni, TT))        # Initial input trajectory
    xx_init = np.zeros((ns, TT))        # Initial state trajectory

    xx_init[:, :] = x0[:, None]

    # Simulate guess
    for tt in range(TT-1):
        # if tt=0 give a little impulse in the direction of the reference
        if tt == 0:
            uu_init[:, tt] = (x_eq2[0] - x_eq1[0])*0.1
        xx_init[:, tt+1] = model.discrete_dynamics(xx_init[:, tt], uu_init[:, tt])
        
    uu_init[:,-1] = uu_init[:,-2]
    uu_init[:,-1] = uu_init[:,-2]

    # Store initial guess in iterations array
    xx[:,:,0] = xx_init
    uu[:,:,0] = uu_init


    ######################################
    # [PLOTS] Initial guess vs Reference trajectory comparison
    ######################################

    plots.trajectory_comparison(time, xx_ref, uu_ref, xx_init, uu_init, title="Initial Guess vs Reference", plot_flag=plot_flag, folder=folder)

    ######################################
    # [MAIN] Iteration loop
    ######################################

    # Iteration counter
    kk = 0

    # Loop over iterations
    for kk in range(max_iters-1):

        ############################
        # 0. Compute cost
        ############################
        
        JJ[kk] = cst.total_cost(xx[:,:,kk], uu[:,:,kk], xx_ref, uu_ref, QQt, RRt, QQT)
        
        ############################
        # 1. Compute descent direction
        ############################
        
        # Terminal condition
        lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
        PP[:,:,TT-1,kk] = QQT
        pp[:,TT-1,kk] = lmbd[:,TT-1,kk].squeeze()

    
        # Backward recursion
        for tt in reversed(range(TT-1)):
            qt, rt, QQt, RRt, SSt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:]
            At, Bt = model.discrete_jacobian(xx[:, tt, kk], uu[:, tt, kk])

            # Discretized jacobians
            AA[:, :, tt, kk] = At
            BB[:, :, tt, kk] = Bt
        
            # Cost matrices at time k
            Qtk = QQt
            Rtk = RRt
            Stk = SSt 

            # Costate equation
            lmbd_temp = At.T @ lmbd[:,tt+1,kk][:,None] + qt
            lmbd[:,tt,kk] = lmbd_temp.squeeze()

            # Single step LQR
            KK[:,:,tt,kk], ss[:,tt,kk], PP[:,:,tt,kk], pp[:,tt,kk] = solver.ltv_affine_LQR_step(            # calculate K, s, P, p according to affine lqr equations
                At, Bt, Qtk, Rtk, Stk, PP[:,:,tt+1,kk], pp[:,tt+1,kk], qt, rt                               
            )

            # Gradient of cost w.r.t. input from LQR quantities
            dJ_temp = rt + Bt.T @ lmbd[:,tt+1,kk][:,None]                                                   # calculate dJ according to costate equation for armijo
            dJ[:,tt,kk] = dJ_temp.squeeze()                                                                 # store dJ

        
        # Compute descent direction norm after backward pass
        for tt in range(TT-1):
            deltau[:,tt,kk] = ss[:,tt,kk] +  KK[:,:,tt,kk] @  deltax[:, tt, kk]                             # closed loop update for deltau
            deltax[:, tt+1, kk] = AA[:, :, tt, kk] @ deltax[:, tt, kk] + BB[:, :, tt, kk] @ deltau[:,tt,kk] # state deviation update : useless
            
            descent[kk] += deltau[:,tt,kk].T @ deltau[:,tt,kk]  
            descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk]  

        ############################
        # 2. Stepsize selection - ARMIJO
        ############################
        
        stepsize, grad_err = armijo.select_stepsize_newton(model, stepsize_0, armijo_maxiters, c, beta, 
                                                ss[:,:,kk], KK[:,:,:,kk], xx_ref, uu_ref, 
                                                x0, xx[:,:,kk], uu[:,:,kk], JJ[kk], descent_arm[kk], 
                                                QQt, RRt, QQT, kk, plot=plot_armijo, save_armijo=save_armijo, save_armijo_iters=save_armijo_iters, folder=folder)
        
        gradient_error[kk] = grad_err

        ############################
        # 4. Termination condition
        ############################

        if abs(descent[kk]) < descent_threshold:                                                            # Check convergence
            print(f"\n\nConvergence reached at iteration {kk}!")
            max_iters = kk
            break

        if np.isnan(descent[kk]) or np.isnan(descent_arm[kk]):                                              # Check for NaN values                                   
            print(f"\n\nNaN detected at iteration {kk}, time {tt}! Stopping computation.")
            max_iters = kk
            break
                
        ############################
        # 3. Forward closed-loop integration
        ############################
        
        # Temporary trajectories
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT)) 
        xx_temp[:,0] = x0

        # Forward closed loop integration with updated control
        for tt in range(TT-1):
            dx = xx_temp[:,tt] - xx[:,tt,kk]                                                                # state deviation at time tt: closed loop                       
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize * ss[:,tt,kk] + KK[:,:,tt,kk] @ dx
            xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])                         # system evolution : rollout  
        
        uu_temp[:,-1] = uu_temp[:,-2]

        # Store trajectories
        xx[:,:,kk+1] = xx_temp.copy()
        uu[:,:,kk+1] = uu_temp.copy()

        ############################
        # 5. Visualization
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

    ######################################
    # [RESULTS AND PLOTS] Final results and plots
    ######################################
    
    # Optimal trajectories
    xx_star = xx[:,:,max_iters-1]
    uu_star = uu[:,:,max_iters-1]
    uu_star[:,-1] = uu_star[:,-2] 
    descent[max_iters-1] = descent[ max_iters-2]

    # Descent Direction
    plots.descent_direction(descent, max_iters, plot_flag=plot_flag, folder=folder)

    # Cost
    plots.cost(JJ, max_iters, plot_flag=plot_flag, folder=folder)

    # Optimal vs Reference Trajectories
    plots.trajectory_comparison(time, xx_ref, uu_ref, xx_star, uu_star, title="Optimal vs Reference", plot_flag=plot_flag, folder=folder)

    # Intermediate Trajectories
    plots.trajectory_comparison_mid(time, xx, uu, n_mid, max_iters=max_iters, eq_pair1=eq_pair1, eq_pair2=eq_pair2, title="Trajectory Comparison with Intermediate Trajectories", plot_flag=plot_flag, folder=folder)

    # Gradient error
    plots.gradient_error(gradient_error, max_iters, plot_flag=plot_flag, folder=folder)

    # Animation of the optimal trajectory
    anim.animate_comparison(model, xx_star, xx_ref, (x_eq1, u_eq1), (x_eq2, u_eq2), dt = dt, folder=folder)

    # Close all plots if opened
    plt.close('all')

    return xx_star, uu_star, time
