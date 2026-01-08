#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         trajectory_tracking_MPC.py
# Description:  Trajectory tracking via MPC.
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
import trajectory as traj

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
import trajectory_generation_newton as traj_gen_newton

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


def trajectory_tracking_MPC(eq_pair1, eq_pair2, xx_star, uu_star, time, umax, umin, T_pred = 100, runs = 1, perturb=True, perturbation_scale=2, plot_flag=True, print_debug=False, folder=None):
    """
    Trajectory tracking via MPC.
    """
    
    if print_debug:
        print("\n--- Trajectory Tracking via MPC ---\n")


    #######################################
    # Algorithm parameters
    #######################################

    dict_ioptions = {
        "print_time": True,                 # Show computation time
        "ipopt.print_level": 1,             # Level of detail (0 to 12)
        "ipopt.sb": "no",                   # Remove Ipopt banner at start
        "ipopt.derivative_test": "first-order", # Utility to check if gradients are correct
    }

    
    #######################################
    # Optimal trajectory
    #######################################


    TT = len(time)

    if plot_flag:
        plots.show_traj(time, xx_star, uu_star, eq_pair1, eq_pair2, title="Optimal trajectory")

    ######################################
    # Arrays to store data
    ######################################

    At = np.zeros((ns, ns, TT))      # State matrices
    Bt = np.zeros((ns, ni, TT))      # Input matrices
    PP = np.zeros((ns, ns, TT))      # P matrices for LQR
    KK = np.zeros((ni, ns, TT))      # Feedback gain matrices
    x0 = np.zeros((ns, runs))

    #######################################
    # 1. Linearize system along optimal trajectory
    #######################################


    # Linearization along optimal trajectory

    for tt in range(TT):
        At[:, :, tt], Bt[:, :, tt] = solver.jacobian_analytic(xx_star[:,tt], uu_star[:,tt], model)


    #######################################
    # 2. Setup problem over finite horizon
    #######################################

    # Cost matrices
    QQt = np.diag([100, 100, 100, 100])
    RRt = 0.01 * np.eye(ni)

    # Terminal matrices
    A_final, B_final = At[:, :, -1], Bt[:, :, -1]

    QQT = ctrl.dare(A_final, B_final, QQt, RRt)[0]  # Terminal cost weight matrix

    #######################################
    # 3. Track the optimal trajcetory with controllers
    #######################################

    # Initial state
    if perturb:
        epsilon = np.random.uniform(-0.1, 0.1, size=(4,))  # initial state perturbation
    else:
        epsilon = np.zeros(4)


    #######################################
    # Model Predictive Control
    #######################################

    xx_mpc = np.zeros((ns,TT,runs))     # State seq.
    uu_mpc = np.zeros((ni,TT,runs))     # Input seq.

    for rr in range(runs):
        
        epsilon_scaled = epsilon * perturbation_scale * rr

        x0[:,rr] = xx_star[:,0] + epsilon_scaled # initial state
        xx_mpc[:,0, rr] = x0[:, rr]

        for tt in range(TT - 1):
            x_curr = xx_mpc[:, tt, rr] # Measure initial condition

            end_idx = tt + T_pred
            
            if end_idx <= TT:
                # Normal time window
                At_slice = At[:, :, tt : end_idx]
                Bt_slice = Bt[:, :, tt : end_idx]
                x_ref_slice = xx_star[:, tt : end_idx]
                u_ref_slice = uu_star[:, tt : end_idx]
            else:
                # Last time window
                available_steps = TT - tt
                padding_len = T_pred - available_steps
                
                # Took the slices of available data
                At_slice = At[:, :, tt:]
                Bt_slice = Bt[:, :, tt:]
                x_ref_slice = xx_star[:, tt:]
                u_ref_slice = uu_star[:, tt:]
                
                # Pad the last available data
                At_pad = np.repeat(At[:, :, -1:], padding_len, axis=2)
                Bt_pad = np.repeat(Bt[:, :, -1:], padding_len, axis=2)
                x_ref_pad = np.repeat(xx_star[:, -1:], padding_len, axis=1)
                u_ref_pad = np.repeat(uu_star[:, -1:], padding_len, axis=1)
                
                # Concatenate real data with padded data
                At_slice = np.concatenate([At_slice, At_pad], axis=2)
                Bt_slice = np.concatenate([Bt_slice, Bt_pad], axis=2)
                x_ref_slice = np.concatenate([x_ref_slice, x_ref_pad], axis=1)
                u_ref_slice = np.concatenate([u_ref_slice, u_ref_pad], axis=1)

            dict_ioptions = None

            res = solver.ltv_mpc(At_slice, Bt_slice, QQt, RRt, QQT, x_curr, x_ref_slice, u_ref_slice,
                umax=umax, umin=umin, T_pred=T_pred, ipopt_opts=dict_ioptions
            )
            
            if res[0] is not None:
                du0 = res[0] # Questa è la deviazione ottima calcolata
                
                # CONTROLLO REALE = Riferimento + Deviazione MPC
                uu_mpc[:, tt, rr] = uu_star[:, tt] + du0
                
                # Evoluzione sistema
                xx_mpc[:, tt+1, rr] = model.discrete_dynamics(xx_mpc[:, tt, rr], uu_mpc[:, tt, rr])
            else:
                print(f"MPC failed at step {tt} at run {rr}")
                break
        
        uu_mpc[:, -1, rr] = uu_mpc[:, -2, rr]  # Last control input equal to previous one
        if print_debug:
            print(f"Completed MPC run {rr+1}/{runs}")

    #######################################
    # Results
    #######################################

    # Final plots
    if plot_flag:
        plots.trajectory_comparison(time, xx_star, uu_star, xx_mpc, uu_mpc, title="MPC Trajectory Tracking", plot_flag=plot_flag, folder=folder)
        plots.tracking_error(time, xx_star, uu_star, xx_mpc, uu_mpc, title="MPC Tracking Error", plot_flag=plot_flag, folder=folder)

    # Animation of the optimal trajectory
    if plot_flag:
        anim.animate_comparison(model, xx_mpc[:,:,0], xx_star, eq_pair1, eq_pair2, target_fps=30)


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
        folder = f'report/figs/branch4/traj_track_MPC/{task}/'

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
    # Trajectory tracking via MPC
    ######################################

    trajectory_tracking_MPC(eq_pair1, eq_pair2, xx_star, uu_star, time, umax=100, umin=-100, T_pred=100, runs=runs, perturb=True, perturbation_scale=perturb_scale, plot_flag=plot_flag, print_debug=True, folder=folder)


if __name__ == "__main__":
    main()