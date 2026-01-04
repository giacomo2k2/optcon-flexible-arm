#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         trajectory_tracking_LQR.py
# Description:  Trajectory tracking via LQR.
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

# Trajectories
import trajectory as traj
import trajectory_generation_newton as traj_gen_newton

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

model = DoublePendulumFlexible(**PARAM_SET_2)

ns = model.ns  # Number of states
ni = model.ni  # Number of inputs
dt = model.dt  # Time step

def trajectory_tracking_LQR(eq_pair1: tuple, eq_pair2: tuple, xx_star: np.ndarray, uu_star: np.ndarray, time: np.ndarray, extend_time: float, runs: int = 1, perturb: bool = True, perturbation_scale: float = 2, plot_flag: bool = True, debug_print: bool = False, folder: str = None):
    """
    Trajectory tracking via LQR.

    Parameters
    ----------
    eq_pair1 : tuple
        Initial equilibrium points.
    eq_pair2 : tuple
        Final equilibrium points.
    xx_star : np.ndarray
        Optimal state trajectory.
    uu_star : np.ndarray
        Optimal input trajectory.
    time : np.ndarray
        Time vector.
    extend_time : float # TODO WHAT IS THIS HELL
        Time to extend the trajectory for tracking.
    runs : int, optional
        Number of runs with different initial perturbations. Default is 1.
    perturb : bool, optional
        Whether to apply initial state perturbations. Default is True.
    perturbation_scale : float, optional
        Scale factor for the initial perturbations. Default is 2.
    plot_flag : bool, optional
        Whether to plot the trajectories. Default is True.
    debug_print : bool, optional
        Whether to print debug information. Default is False.
    """

    ######################################
    # Log header
    ######################################

    print("\n" + "="*80)
    print(f"[Trajectory Tracking] - LQR Method".center(80))
    print("="*80)

    #######################################
    # Optimal trajectory
    #######################################

    # Extract time horizon
    TT = len(time)

    x_eq, u_eq = eq_pair2

    TT_ext = int(extend_time / dt)

    
    xx_eq_ext = np.tile(x_eq[:, np.newaxis], (1, TT_ext))
    uu_eq_ext = np.tile(u_eq[:, np.newaxis], (1, TT_ext))

    xx_traj_ext = np.tile(xx_star[:, -1][:, np.newaxis], (1, TT_ext))
    uu_traj_ext = np.tile(uu_star[:, -1][:, np.newaxis], (1, TT_ext))

    # Extend optimal trajectories by using final equilibrium
    
    uu_star_eq = np.hstack((uu_star, uu_eq_ext)) # Extend input trajectory discontinuity here
    xx_star_eq = np.hstack((xx_star, xx_eq_ext)) # Extend state trajectory no discontinuity here
    
    uu_star_traj = np.hstack((uu_star, uu_traj_ext)) # Extend input trajectory discontinuity here
    xx_star_traj = np.hstack((xx_star, xx_traj_ext)) #


    TT_total = xx_star_eq.shape[1]
    
    time_ext = np.arange(0, TT_total * dt, dt)

    ######################################
    # [PLOTS] Optimal trajectory
    ######################################

    plots.show_traj(time_ext, xx_star_eq, uu_star_eq, eq_pair1, eq_pair2, title="Optimal trajectory", plot_flag=plot_flag)

    ######################################
    # Arrays to store data
    ######################################

    xx_lqr = np.zeros((ns, TT_total, runs))       # State seq.
    uu_lqr = np.zeros((ni, TT_total, runs))       # Input seq.
    At = np.zeros((ns, ns, TT_total, runs))       # State matrices
    Bt = np.zeros((ns, ni, TT_total, runs))       # Input matrices
    x0 = np.zeros((ns, runs))                      # Initial states

    ######################################
    # [TRACKING PROCEDURE] 1. Linearize system along optimal trajectory
    ######################################

    # Linearization along optimal trajectory
    for rr in range(runs):
        for tt in range(TT_total):
            At[:, :, tt, rr], Bt[:, :, tt, rr] = solver.jacobian_analytic(xx_star_traj[:,tt], uu_star_traj[:,tt], model) #no jumps here 
            

    #######################################
    # [TRACKING PROCEDURE] 2. Solve time-varying LQR problem over finite horizon
    #######################################

    # Cost matrices
    QQt = np.diag([10, 10, 1, 1])
    RRt = 0.01 * np.eye(ni)


    # Terminal matrices
    A_final, B_final = solver.jacobian_analytic(x_eq, u_eq, model) #no jumps here 

    # Terminal cost weight matrix
    QQT,_, _ = ctrl.dare(A_final, B_final, QQt, RRt)  
    
    
    
    ######################################
    # [TRACKING PROCEDURE] 3. Track the optimal trajectory with LQR controller
    ######################################

    # Initial state perturbation
    if perturb:
        epsilon = np.random.uniform(-0.1, 0.1, size=(4,))  # Initial state perturbation
    else:
        epsilon = np.zeros(4)

    # LQR tracking for each run
    for rr in range(runs):
        if debug_print:
            print(f"\nRun {rr+1}/{runs}:")

        # Perturbation epsilon slightly increase for each run
        epsilon_scaled = epsilon * perturbation_scale * rr

        x0[:,rr] = xx_star_eq[:,0] + epsilon_scaled # Initial state
        
        
        xx_lqr[:,:,rr], uu_lqr[:,:,rr] = solver.ltv_LQR(At[:,:,:,rr], Bt[:,:,:,rr], QQt, RRt, QQT, x0[:, rr], xx_star_eq, uu_star_eq, model)

    ######################################
    # [RESULTS AND PLOTS] Final results and plots
    ######################################

    if debug_print:
        for rr in range(runs):
            final_error = np.linalg.norm(xx_lqr[:, -1, rr] - xx_star_eq[:, -1])
            print(f"Final tracking error (run {rr+1}): {final_error:.4f}")

    # Trajectory 
    plots.trajectory_comparison(time_ext, xx_star_eq, uu_star_eq, xx_lqr, uu_lqr, title="LQR Trajectory Tracking", plot_flag=plot_flag, folder=folder)

    # Tracking comparison
    plots.tracking_error(time_ext, xx_star_eq, uu_star_eq, xx_lqr, uu_lqr, title="LQR Tracking Error", plot_flag=plot_flag, folder=folder)

    # Animation of the tracking for the first run
    anim.animate_comparison(model, xx_lqr[:,:,0], xx_star_eq, eq_pair1, eq_pair2, dt, plot_flag=plot_flag, folder=folder)

######################################
# MAIN
######################################

def main():

    ######################################
    # Algorithm Parameters
    ######################################

    task = 'swing_up'   # Choose from keys of equilibrium_traj
    max_iters = 300          # Max Newton iterations for trajectory generation
    n_mid = 20                # Number of mid points for trajectory generation
    method = 'poly5'         # Trajectory generation method: 'poly3' or
    runs = 5                    # Number of runs with different initial perturbations
    perturb_scale = 2          # Scale of the initial perturbation

    ######################################
    # Algorithm flags
    ######################################
    
    save_flag = True            # Save flag
    plot_flag = True            # Plot flag
    debug_flag = True           # Debug print flag

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
        folder = f'report/figs/traj_track_LQR/{task}/'

        # Create folder if it does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)
    else:
        folder = None

    ######################################
    # Generate optimal trajectory
    ######################################

    xx_star, uu_star, eq_pair1, eq_pair2, time = traj_gen_newton.generate_optimal_trajectory(
        task=task, max_iters=max_iters, n_mid=n_mid, method=method, plot_armijo=False, plot_flag=False, debug_print=debug_flag)
    
    ######################################
    # Track optimal trajectory via LQR
    ######################################

    trajectory_tracking_LQR(
        eq_pair1, eq_pair2, xx_star, uu_star, time, extend_time=5, runs=runs, 
        perturb=True, perturbation_scale=perturb_scale, plot_flag=plot_flag, debug_print=debug_flag, folder=folder)


if __name__ == "__main__":
    main()