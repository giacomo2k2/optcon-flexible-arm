#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         trajectory_tracking_MPC.py
# Description:  Trajectory tracking via MPC.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         January 2026
#########################################################################################

#######################################
# Imports
#######################################

# Standard libriaries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import signal
import os
import control as ctrl

# Custom libraries
# - Model
from model import DoublePendulumFlexible
# - Armijo
import armijo as armijo
# - Solver
import solver as solver
# - Animation 
import animation as anim
# - Plots
import plots as plots


# Set font family
plt.rcParams['font.family'] = 'serif'
# Set interrupt signal to default to allow Ctrl+C interruption
signal.signal(signal.SIGINT, signal.SIG_DFL)

def trajectory_tracking_MPC(model:DoublePendulumFlexible, optimal_trajectory_params: dict, tracking_params: dict, debug_params: dict, folder: str = None):
    """
    Trajectory tracking via MPC.

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
    
    ######################################
    # Log header
    ######################################

    print("\n" + "="*80)
    print(f"[Trajectory Tracking] - MPC".center(80))
    print("="*80)

    #######################################
    # Unpack parameters
    #######################################

    ns = model.ns  # number of states
    ni = model.ni  # number of inputs
    TT = len(time)

    # Optimal trajectory parameters
    xx_star = optimal_trajectory_params['xx_star']
    uu_star = optimal_trajectory_params['uu_star']
    eq_pair1 = optimal_trajectory_params['eq_pair1']
    eq_pair2 = optimal_trajectory_params['eq_pair2']
    time = optimal_trajectory_params['time']
    
    # Tracking parameters
    runs = tracking_params['runs']
    perturb = tracking_params['perturb']
    perturbation_scale = tracking_params['perturbation_scale']
    T_pred = tracking_params['T_pred']
    QQt = tracking_params['QQt']
    RRt = tracking_params['RRt']
    u_max = np.max(np.abs(uu_star))
    u_min = -u_max


    # Debug parameters
    plot_flag = debug_params['plot_flag']
    debug_print = debug_params['debug_print']


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
        At[:, :, tt], Bt[:, :, tt] = model.discrete_jacobian(xx_star[:,tt], uu_star[:,tt])


    #######################################
    # 2. Setup problem over finite horizon
    #######################################



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
                umax=u_max, umin=u_min, T_pred=T_pred, ipopt_opts=dict_ioptions
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
        if debug_print:
            print(f"Completed MPC run {rr+1}/{runs}")

    #######################################
    # Results
    #######################################

    plots.trajectory_comparison(time, xx_star, uu_star, xx_mpc, uu_mpc, u_min=u_min, u_max=u_max, title="MPC Trajectory Tracking", plot_flag=plot_flag, folder=folder)

    plots.tracking_error(time, xx_star, uu_star, xx_mpc, uu_mpc, title="MPC Tracking Error", plot_flag=plot_flag, folder=folder)

    anim.animate_comparison(model, xx_mpc[:,:,0], xx_star, eq_pair1, eq_pair2, model.dt, folder=folder)
