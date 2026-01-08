#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         main.py
# Description:  Main script to generate and track optimal trajectories
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         January 2026
#########################################################################################

#######################################
# Imports
#######################################

# Standard Libraries
import numpy as np
import os

# Custom Libraries
from trajectory_generation_newton import generate_optimal_trajectory
from trajectory_generation_newton_extended import generate_optimal_trajectory_extended
from trajectory_tracking_LQR import trajectory_tracking_LQR
from trajectory_tracking_MPC import trajectory_tracking_MPC
import trajectory_reference_generation as traj
from model import DoublePendulumFlexible, PARAM_SET_2
from plots import print_header



GENERATOR = "Newton"            # Newton | Newton-Extended
TRACKER = "LQR"                 # LQR | MPC
GENERATION_ONLY = False
# choose task from {'swing_up', 'elbow_up', 'swing_down', 'complex_maneuver', 'topside_pump', 'diagonal_swing', 'top_to_elbow_up', 'low_diagonal_hold', 'half_swing_up'}
TASKS = ['swing_up', 'elbow_up', 'swing_down', 'complex_maneuver']

if __name__ == "__main__":

    ######################################
    # General parameters
    #######################################

    save_flag = True            # Save flag
    plot_flag = False           # Plot flag
    debug_flag = True           # Debug print flag
    plot_armijo = False         # Plot Armijo flag
    save_armijo = True         # Plot Armijo flag
    save_armijo_iters = 5      # Save Armijo every n iterations

    # definition of the model
    discr_method = "fe"                # fe | rk4  
    dt = 0.01      
    model = DoublePendulumFlexible(**PARAM_SET_2, method=discr_method, dt=dt)                                

    ######################################
    # Parameters for trajectory generation and tracking
    ######################################

    # Reference Trajectory generation parameters
    method = 'poly5'                # Trajectory generation method: 'poly3' | 'poly5' | 'stepwise'
    ti = 0.0                        # Start time for trajectory segment (relative)
    tf = 0.7                        # End time for trajectory segment (relative)
    tf_ext = 1.0                    # End time for extended trajectory segment (relative)
    
    # Optimal Trajectory generation parameters
    max_iters = 300             # Max Newton iterations for trajectory generation
    n_mid = 20                  # Number of mid points for trajectory generation
    descent_threshold = 1e-6    # Threshold for stopping criterion based on descent
    max_time = 20.0             # Max time for trajectory generation
    extension_duration = 0.5    # Time extension for Newton-Extended method
    terminal_error_tol = 1e-2    # Tolerance for terminal time condition
    QQt_gen = np.diag([1.0, 1.0, 1.0, 1.0])
    RRt_gen = 1e-3 * np.eye(model.ni)

    # Armijo parameters
    stepsize_0=1.0
    c=0.5
    beta=0.7
    armijo_maxiters=20
    
    # Trajectory tracking parameters
    runs = 5                    # Number of runs with different initial perturbations
    perturb = True              # Whether to perturb the initial state
    perturb_scale = 2           # Scale of the initial perturbation
    time_extension = 5          # Time extension after reaching the target state
    T_pred = 100                # Prediction horizon for MPC
    QQt_track = np.diag([1.0, 1.0, 1.0, 1.0])
    RRt_track = 1e-4 * np.eye(model.ni)
    
    #######################################
    # Pack parameters into dictionaries
    ########################################


    debug_params = {'plot_flag': plot_flag, 'debug_print': debug_flag, 'save_flag': save_flag, 'save_armijo': save_armijo, 'plot_armijo': plot_armijo, 'save_armijo_iters': save_armijo_iters}
    
    optimal_gen_params = {'max_iters': max_iters, 'n_mid': n_mid, 'descent_threshold': descent_threshold, 'max_time': max_time, 'extension_duration': extension_duration, 'terminal_error_tol': terminal_error_tol, 'QQt': QQt_gen, 'RRt': RRt_gen}
    
    armijo_params = {'stepsize_0': stepsize_0, 'c':c, 'beta':beta, 'max_iters':armijo_maxiters}

    tracking_params = {'runs': runs, 'perturb': perturb, 'perturbation_scale': perturb_scale, 'extend_time': time_extension, 'T_pred': T_pred, 'QQt': QQt_track, 'RRt': RRt_track}

    if GENERATOR == "Newton-Extended":
        reference_params = {'ti': ti, 'tf': tf_ext, 'method': method}
    else:
        reference_params = {'ti': ti, 'tf': tf, 'method': method}

    ########################################
    # Initialize tasks and configurations
    ########################################
    
    traj.init_trajectories(model)

    ######################################
    # Main loop over tasks
    ######################################

    folder_generation = None
    folder_tracker = None

    for task in TASKS:    
        # Check task name correctness 
        try:
            if task not in traj.tasks.keys():
                raise ValueError(f"Task '{task}' not found in predefined tasks. Please choose a valid task.")
        except ValueError as e:
            print(f"\n{e}\nAvailable tasks are: {list(traj.tasks.keys())}\n")
            exit(1)
        
        # Print header
        print_header(f'[Task Selected] - {task}', rank=0, size=80, char='=')

        # Create folders for saving results
        if save_flag:
            if GENERATOR == "Newton-Extended":
                folder_generation = f'report/figs/code/traj_gen_newton_extended/{task}'
                
                if TRACKER == "LQR":
                    folder_tracker = f'report/figs/code/traj_track_LQR_extended/{task}'
                elif TRACKER == "MPC":
                    folder_tracker = f'report/figs/code/traj_track_MPC_extended/{task}'
                else:
                    raise ValueError(f"Unknown TRACKER '{TRACKER}'. Please choose 'LQR' or 'MPC'.")
            elif GENERATOR == "Newton":
                folder_generation = f'report/figs/code/traj_gen_newton/{task}'
                if TRACKER == "LQR":
                    folder_tracker = f'report/figs/code/traj_track_LQR/{task}'
                elif TRACKER == "MPC":
                    folder_tracker = f'report/figs/code/traj_track_MPC/{task}'
                else:
                    raise ValueError(f"Unknown TRACKER '{TRACKER}'. Please choose 'LQR' or 'MPC'.")
            else:
                raise ValueError(f"Unknown GENERATOR '{GENERATOR}'. Please choose 'Newton' or 'Newton-Extended'.")
        
            # Create folder if it does not exist    
            if not os.path.exists(folder_generation):
                    os.makedirs(folder_generation)
            
            if not GENERATION_ONLY:
                if not os.path.exists(folder_tracker):
                    os.makedirs(folder_tracker)
            else:
                folder_tracker = None

        ######################################
        # Generate reference trajectory
        ######################################

        # Print header
        print_header(f'[Trajectory Reference] - {method}', rank=1, size=80, char='=')

        # Generate reference trajectory
        time, xx_ref, uu_ref, eq_pair1, eq_pair2 = traj.reference_trajectory_generator(model=model, task=task, reference_params=reference_params, debug_params=debug_params, folder=folder_generation, generator=GENERATOR)

        print(f"Reference trajectory for task '{task}' successfully generated.\n")
        
        # Pack reference trajectory parameters
        reference_trajectory_params = {'time': time, 'xx_ref': xx_ref, 'uu_ref': uu_ref, 'eq_pair1': eq_pair1, 'eq_pair2': eq_pair2}

        ######################################
        # Generate optimal trajectory
        ######################################

        # Print header
        print_header(f'[Trajectory Generation] - {"Newton Method" if GENERATOR=="Newton" else "Newton Method - (Dynamic Horizon Extension)"}', rank=1, size=80, char='=')

        # Generate optimal trajectory 
        if GENERATOR == "Newton-Extended":
            xx_star, uu_star, time = generate_optimal_trajectory_extended(
                model=model, armijo_params=armijo_params, optimal_gen_params=optimal_gen_params, reference_trajectory_params=reference_trajectory_params, debug_params=debug_params, folder=folder_generation)
        elif GENERATOR == "Newton":
            xx_star, uu_star, time = generate_optimal_trajectory(
                model = model, optimal_gen_params=optimal_gen_params, reference_trajectory_params=reference_trajectory_params, armijo_params=armijo_params, debug_params=debug_params, folder=folder_generation)
        else:
            raise ValueError(f"Unknown GENERATOR '{GENERATOR}'. Please choose 'Newton' or 'Newton-Extended'.")
        
        # Pack optimal trajectory parameters
        optimal_trajectory_params = {'xx_star': xx_star, 'uu_star': uu_star, 'eq_pair1': eq_pair1, 'eq_pair2': eq_pair2, 'time': time}
        
        ######################################
        # Track optimal trajectory via LQR
        ######################################

        # Print header
        print_header(f'[Trajectory Tracking] - {TRACKER} Method', rank=1, size=80, char='=', capitalize=False)

        if GENERATION_ONLY:
            print(f"Generation only flag is set. Skipping trajectory tracking for task '{task}'.")
            continue
        if TRACKER == "MPC":
            trajectory_tracking_MPC(
            model=model, optimal_trajectory_params=optimal_trajectory_params, tracking_params=tracking_params,  debug_params=debug_params, folder=folder_tracker)
        elif TRACKER == "LQR":
            trajectory_tracking_LQR(
                model=model, optimal_trajectory_params=optimal_trajectory_params, tracking_params=tracking_params,  debug_params=debug_params, folder=folder_tracker)
        else:
            raise ValueError(f"Unknown TRACKER '{TRACKER}'. Please choose 'LQR' or 'MPC'.")
        
        ######################################
        # End of single task loop
        ######################################
        
        print(f"Completed task '{task}'.\n")
    
    ######################################
        # End of task loop
    ######################################

    print("All tasks completed.")
