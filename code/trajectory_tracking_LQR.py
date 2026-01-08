#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         trajectory_tracking_LQR.py
# Description:  Trajectory tracking via LQR.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         January 2026
#########################################################################################

#######################################
# Imports
#######################################

# Standart libriaries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import signal
import os
import control as ctrl

# Custom libraries
# - Model
from model import DoublePendulumFlexible, PARAM_SET_2
# - Trajectories
import trajectory_reference_generation as traj
import trajectory_generation_newton as traj_gen_newton
# - Solver
import solver as solver
# - Animation 
import animation as anim
# - Plots
import plots as plots
# - Model
from model import DoublePendulumFlexible

# Set font family
plt.rcParams['font.family'] = 'serif'
# Set interrupt signal to default to allow Ctrl+C interruption
signal.signal(signal.SIGINT, signal.SIG_DFL)

def trajectory_tracking_LQR(model:DoublePendulumFlexible, optimal_trajectory_params: dict, tracking_params: dict, debug_params: dict, folder: str = None):
    """
    Trajectory tracking via LQR.

    Parameters
    ----------
    model : DoublePendulumFlexible
        The model to control.
        - ns : int
            Number of states.
        - ni : int
            Number of inputs.
        - dt : float
            Time step.
    optimal_trajectory_params : dict
        Dictionary containing optimal trajectory parameters:
        - xx_star : np.ndarray
            Optimal state trajectory.
        - uu_star : np.ndarray
            Optimal input trajectory.
        - eq_pair1 : tuple
            Initial equilibrium points.
        - eq_pair2 : tuple
            Final equilibrium points.
        - time : np.ndarray
            Time vector.
    tracking_params : dict
        Dictionary containing tracking parameters:
        - runs : int
            Number of tracking runs.
        - perturb : bool
            Whether to apply initial state perturbation.
        - perturbation_scale : float
            Scale of the initial state perturbation.
        - extend_time : float
            Time to extend the trajectory for tracking.
        - QQt : np.ndarray
            State cost matrix.
        - RRt : np.ndarray
            Input cost matrix.
    debug_params : dict
        Dictionary containing debug parameters:
        - plot_flag : bool
            Whether to plot results.
        - debug_print : bool
            Whether to print debug information.
    folder : str, optional
        Folder to save results, by default None.

    Returns
    -------
    None
    """
    #######################################
    # Algorithm parameters
    ######################################

    # Compute equilibrium manifold
    ns = model.ns                                                                       # Number of states
    ni = model.ni                                                                       # Number of inputs
    dt = model.dt                                                                       # Time step

    ######################################
    # Unpack parameters
    ######################################

    # Optimal trajectory parameters
    xx_star = optimal_trajectory_params['xx_star']                                      # Optimal state trajectory.
    uu_star = optimal_trajectory_params['uu_star']                                      # Optimal input trajectory.
    eq_pair1 = optimal_trajectory_params['eq_pair1']                                    # Initial equilibrium points.
    eq_pair2 = optimal_trajectory_params['eq_pair2']                                    # Final equilibrium points.
    time = optimal_trajectory_params['time']                                            # Time vector.
    
    # Tracking parameters
    runs = tracking_params['runs']                                                      # Number of runs with different initial perturbations                          
    perturb = tracking_params['perturb']                                                # Whether to perturb the initial state         
    perturbation_scale = tracking_params['perturbation_scale']                          # Scale of the initial perturbation    
    extend_time = tracking_params['extend_time']                                        # Time extension after reaching the target state
    QQt = tracking_params['QQt']                                                        # State cost matrix.              
    RRt = tracking_params['RRt']                                                        # Control cost matrix.

    # Debug parameters
    plot_flag = debug_params['plot_flag']                                               # Whether to plot results.  
    debug_print = debug_params['debug_print']                                           # Whether to print debug information.
    
    ##############################################################
    # Extend optimal trajectory to test final equilibrium tracking
    ##############################################################

    x_eq, u_eq = eq_pair2

    TT_ext_traj = int(extend_time / dt)

    xx_ext = np.tile(xx_star[:, -1][:, np.newaxis], (1, TT_ext_traj))
    uu_ext = np.tile(uu_star[:, -1][:, np.newaxis], (1, TT_ext_traj))

    # Extend optimal trajectories by using final equilibrium
    
    uu_star = np.hstack((uu_star, uu_ext)) # Extend input trajectory discontinuity here
    xx_star = np.hstack((xx_star, xx_ext)) # Extend state trajectory no discontinuity here
    
    TT_total = xx_star.shape[1]
    
    time_ext = np.arange(0, TT_total * dt, dt)

    ######################################
    # [PLOTS] Optimal trajectory
    ######################################

    plots.show_traj(time_ext, xx_star, uu_star, eq_pair1, eq_pair2, title="Optimal trajectory", plot_flag=plot_flag)

    ######################################
    # Arrays to store data
    ######################################

    xx_lqr = np.zeros((ns, TT_total, runs))       # State seq.
    uu_lqr = np.zeros((ni, TT_total, runs))       # Input seq.
    At = np.zeros((ns, ns, TT_total, runs))       # State matrices
    Bt = np.zeros((ns, ni, TT_total, runs))       # Input matrices
    x0 = np.zeros((ns, runs))                     # Initial states

    ######################################
    # [TRACKING PROCEDURE] 1. Linearize system along optimal trajectory
    ######################################

    # Linearization along optimal trajectory
    for rr in range(runs):
        for tt in range(TT_total):
            At[:, :, tt, rr], Bt[:, :, tt, rr] = model.discrete_jacobian(xx_star[:,tt], uu_star[:,tt])              # Linearized state and input matrices for control design
            

    #######################################
    # [TRACKING PROCEDURE] 2. Calculate terminal cost matrix
    #######################################

    # Terminal matrices
    A_final, B_final = model.discrete_jacobian(x_eq, u_eq)                                                          # Linearized state and input matrices at final equilibrium point                    
    
    # Terminal cost weight matrix
    QQT, _, _ = ctrl.dare(A_final, B_final, QQt, RRt)  
    
    
    ######################################
    # [TRACKING PROCEDURE] 3. Track the optimal trajectory with LQR controller
    ######################################

    # Initial state perturbation
    if perturb:
        epsilon = np.random.uniform(-0.05, 0.05, size=(4,))  # Initial state perturbation
    else:
        epsilon = np.zeros(4)

    # LQR tracking for each run
    for rr in range(runs):
        if debug_print:
            print(f"\nRun {rr+1}/{runs}:")

        # Perturbation epsilon slightly increase for each run
        epsilon_scaled = epsilon * perturbation_scale * rr

        x0[:,rr] = xx_star[:,0] + epsilon_scaled # Initial state perturbation
        
        # LQR tracking: u update with time-varying feedback gain and rollout
        xx_lqr[:,:,rr], uu_lqr[:,:,rr] = solver.ltv_LQR(At=At[:,:,:,rr], Bt=Bt[:,:,:,rr], QQt=QQt, RRt=RRt, QQT=QQT, x0=x0[:, rr], xx_star=xx_star, uu_star=uu_star, model=model)   

    ######################################
    # [RESULTS AND PLOTS] Final results and plots
    ######################################

    if debug_print:
        for rr in range(runs):
            final_error = np.linalg.norm(xx_lqr[:, -1, rr] - xx_star[:, -1])
            print(f"Final tracking error (run {rr+1}): {final_error:.4f}")

    # Trajectory 
    plots.trajectory_comparison(time_ext, xx_star, uu_star, xx_lqr, uu_lqr, title="LQR Trajectory Tracking", plot_flag=plot_flag, folder=folder)
    # Tracking comparison
    plots.tracking_error(time_ext, xx_star, uu_star, xx_lqr, uu_lqr, title="LQR Tracking Error", plot_flag=plot_flag, folder=folder)

    # Animation of the tracking for the first run
    anim.animate_comparison(model, xx_lqr[:,:,0], xx_star, eq_pair1, eq_pair2, dt, folder=folder)



