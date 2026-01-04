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
import os   

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

######################################
# Model
######################################

# Model 
model = DoublePendulumFlexible(**PARAM_SET_2)

ns = model.ns  # Number of states
ni = model.ni  # Number of inputs

def generate_optimal_trajectory(task: str, max_iters: int = 300, n_mid: int = 20, method: str = 'poly5', plot_flag: bool = False, plot_armijo: bool = False, debug_print: bool = True, folder: str = None) -> tuple[np.ndarray, np.ndarray, tuple, tuple, np.ndarray]:
    """
    Generate optimal trajectory via Newton method.

    Parameters
    ----------
    task : str
        Task name (choose from keys of traj.tasks).
    max_iters : int, optional
        Maximum number of iterations. Default is 300.
    n_mid : int, optional
        Number of intermediate trajectories to plot. Default is 20.
    method : str, optional
        Trajectory generation method ('poly5', 'spline', ...). Default is 'poly5'.
    plot_flag : bool, optional
        Flag to enable/disable plotting. Default is True.
    plot_armijo : bool, optional
        Flag to enable/disable Armijo stepsize plotting. Default is False.
    debug_print : bool, optional
        Flag to enable/disable debug printing. Default is True.
    folder : str or None, optional
        Folder to save plots. If None, plots are not saved. Default is None.

    Returns
    -------
    xx_star : np.ndarray
        Optimal state trajectory.
    uu_star : np.ndarray
        Optimal input trajectory.
    eq_pair1 : tuple
        Initial equilibrium point (x_eq1, u_eq1).
    eq_pair2 : tuple
        Final equilibrium point (x_eq2, u_eq2).
    time : np.ndarray
        Time vector.
    """

    ######################################
    # Log header
    ######################################

    print("\n" + "="*80)
    print(f"[Trajectory Generation] - Newton Method".center(80))
    print("="*80)

    ######################################
    # Task and Equilibrium points
    ######################################
    
    # Choose from keys of equilibrium_traj
    selected_task = task  

    # Compute equilibrium manifold
    data_dict = model.compute_equilibrium_manifold()

    # Exrtact selected equilibrium
    selected_equilibrium = traj.tasks[selected_task]
    eq_pair1, eq_pair2 = selected_equilibrium['start'], selected_equilibrium['end']
    x_eq1, u_eq1 = eq_pair1
    x_eq2, u_eq2 = eq_pair2

    # Extract task parameters
    Tend = selected_equilibrium['time']
    angle_mult1 = selected_equilibrium['angle1_mult']
    vel_mult1 = selected_equilibrium['vel1_mult']
    angle_mult2 = selected_equilibrium['angle2_mult']
    vel_mult2 = selected_equilibrium['vel2_mult']
    termcost_mult = selected_equilibrium['termcost_mult']
    guess = selected_equilibrium['guess']
    discr_method = selected_equilibrium['discr_method']

    ######################################
    # Algorithm parameters
    ######################################
    
    # Amijo parameters
    stepsize_0=1.0
    c=0.5
    beta=0.7
    armijo_maxiters=20

    # Stopping criteria over descent norm
    descent_threshold = 1e-6

    # Discretization method
    model.discr_method = discr_method

    # Base Cost matrix
    QQt_base = np.diag([10.0, 10.0, 1.0, 1.0])
    RRt = 1e-2 * np.eye(ni)

    ######################################
    # [PLOTS] Equilibrium manifold and pendulum equilibria
    ######################################

    plots.equilibrium_manifold(data_dict, eq_pair1, eq_pair2, print_results=False, plot_flag=plot_flag, folder=folder)
    model.draw_pendulums(eq_pair1, eq_pair2, plot_flag=plot_flag, folder=folder)

    ######################################
    # Time parameters
    ######################################

    T0 = 0.0                        # Initial absolute time
    ti = 0.3                        # Start time for trajectory segment (relative)
    tf = 0.7                        # End time for trajectory segment (relative)
    dt = model.dt                   # Time step
    t = np.arange(T0, Tend, dt)     # Time vector
    TT = len(t)                     # Number of time steps

    ######################################
    # Reference curve
    ######################################

    # Generate reference trajectory for states and inputs
    time, xx_ref= traj.trajectory_generator(start = x_eq1, end = x_eq2, T0=T0, Tend=Tend, ti=ti, tf=tf, dt=dt, method=method, input_type='state')
    time, uu_ref= traj.trajectory_generator(start = u_eq1, end = u_eq2, T0=T0, Tend=Tend, ti=ti, tf=tf, dt=dt, method=method, input_type='input')
    
   
    ######################################
    # [PLOTS] Reference trajectory
    ######################################

    plots.show_traj(time, xx_ref, uu_ref, eq_pair1, eq_pair2, "Reference " + method, plot_flag=plot_flag, folder=folder)

    ######################################
    # Compute weighting matrices
    ######################################

    # Scale cost matrices for the given task
    scale_QQt = np.diag([angle_mult1, angle_mult2, vel_mult1, vel_mult2])

    QQt = QQt_base @ scale_QQt

    QQT = termcost_mult * np.array([1e3, 0, 0, 0,
                                0, 1e3, 0, 0,
                                0, 0, 1e3, 0,
                                0, 0, 0, 1e3]).reshape((ns,ns))

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

    xx_total = np.zeros((ns, TT, max_iters))    # all state trajectories
    uu_total = np.zeros((ni, TT, max_iters))    # all input trajectories

    ######################################
    # Initial Trajectory Guess
    ######################################

    x0 = xx_ref[:, 0]                   # Initial state

    uu_init = np.zeros((ni, TT))        # Initial input trajectory
    xx_init = np.zeros((ns, TT))        # Initial state trajectory


    xx_init[:, :] = x0[:, None]

    # Controller for initial guess
    match (guess):
        case 'PD_gc':
            f = solver.PD_controller_gravity_comp   # PD + gravity compensation
        case 'PD':
            f = solver.PD_controller                # PD only
        case 'zero':
            f = solver.zero_controller              # Zero input
        case 'gc':
            f = solver.gravity_compensation_only    # Gravity compensation only
        case 'D_gc':
            f = solver.initial_guess_dissipative    # Dissipative + gravity compensation
        case 'P':
            f = solver.proportional_controller      # Proportional only
        case _:
            raise ValueError("Invalid initial guess type specified.")
    
    # Generate initial guess trajectory
    for tt in range(TT-1):
        uu_init[:, tt] = f(model, xx_init[:, tt], xx_ref[:, tt], uu_ref[0, tt])     # Compute input evolution
        xx_init[:, tt+1] = model.discrete_dynamics(xx_init[:, tt], uu_init[:, tt])  # Compute state evolution

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

        lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
        PP[:,:,TT-1,kk] = QQT
        pp[:,TT-1,kk] = lmbd[:,TT-1,kk].squeeze()

        # Backward recursion
        for tt in reversed(range(TT-1)):
            qt, rt, QQt, RRt, SSt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:]
            At, Bt = solver.jacobian_analytic(xx[:, tt, kk], uu[:, tt, kk], model)
            
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
                                            QQt, RRt, QQT, kk, plot=plot_armijo, folder=folder)
        
        gradient_error[kk] = grad_error_temp

        ############################
        # 3. Termination condition
        ############################

        if abs(descent[kk]) < descent_threshold:
            print(f"\n\nConvergence reached at iteration {kk}!")
            max_iters = kk
            break

        if np.isnan(descent[kk]) or np.isnan(descent_arm[kk]):
            print(f"\n\nNaN detected at iteration {kk}, time {tt}! Stopping computation.")
            max_iters = kk
            break
                
        ############################
        # 4. Forward closed-loop integration
        ############################
        
        # Temporary trajectories
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT)) 
        xx_temp[:,0] = x0

        # Forward closed loop integration with updated control
        for tt in range(TT-1):
            dx = xx_temp[:,tt] - xx[:,tt,kk]
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize * ss[:,tt,kk] + KK[:,:,tt,kk] @ dx
            xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])
            
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
    plots.trajectory_comparison_mid(time, xx, uu, n_mid, max_iters, title="Trajectory Comparison with Intermediate Trajectories", plot_flag=plot_flag, folder=folder)

    # Gradient error
    plots.gradient_error(gradient_error, max_iters, plot_flag=plot_flag, folder=folder)

    # Animation of the optimal trajectory
    anim.animate_comparison(model, xx_star, xx_ref, (x_eq1, u_eq1), (x_eq2, u_eq2), dt = dt, plot_flag=plot_flag, folder=folder)

    # Close all plots if opened
    plt.close('all')

    return xx_star, uu_star, eq_pair1, eq_pair2, time

######################################
# MAIN
######################################

def main():

    ######################################
    # Algorithm parameters
    ######################################

    max_iters = 300                      # Maximum number of iterations
    n_mid = 20                           # Number of intermediate trajectories to plot
    method = 'poly5'                     # Trajectory generation method ('poly5', 'spline', ...)
    task = 'half_middle_dive'           # Choose from keys of equilibrium_traj

    ######################################
    # Algorithm flags
    ######################################

    save_flag = False                       # Save flag
    plot_flag = True                        # Plot flag
    plot_flag_armijo = False                # Plot Armijo stepsize flag
    debug_flag = True                       # Debug print flag

    ######################################
    # Environment setup
    ######################################

    # Check task name correctness 
    try:
        if task not in traj.tasks:
            raise ValueError(f"Task '{task}' not found in predefined tasks. Please choose a valid task.")
    except ValueError as e:
        print(f"\n{e}\nAvailable tasks are: {list(traj.tasks.keys())}\n")
        exit(1)
    
    # Create folder for plots
    if save_flag:
        folder = f'report/figs/traj_gen_newton/{task}/'

        # Create folder if it does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)
    else:
        folder = None
        
    ######################################
    # Generate optimal trajectory
    ######################################

    generate_optimal_trajectory(
        task=task, max_iters=max_iters, n_mid=n_mid, method=method, plot_flag=plot_flag, plot_armijo=plot_flag_armijo, debug_print=debug_flag, folder=folder)

if __name__ == "__main__":
    main()

# Easy trajectories: lateral_swing, diagonal_swing, overhead_wipe, half_middle_dive
# Mid trajectories: swing_up, topside_pump, half_swing
# Hard trajectories: the_dive, half_dive

# SELECTED
# easy: diagonal_swing, lateral_swing, half_middle_dive
# mid: swing_up, topside_pump
# hard: half_dive, the_dive