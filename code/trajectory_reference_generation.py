#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         trajectories.py
# Description:  This module contains functions for generating reference trajectories for the flexible robotic link.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         January 2025
#########################################################################################

#######################################
# Imports
#######################################

# Standard Libraries
import numpy as np

# Custom Libraries
from model import DoublePendulumFlexible, PARAM_SET_2
import plots


#######################################
# Configurations
#######################################

# Model Initialization
model = DoublePendulumFlexible(**PARAM_SET_2)


configurations = {
    'D-D': model.find_equilibrium(fix_type='q1', value_fixed=0.0, up=False), 
    'D-U': model.find_equilibrium(fix_type='q1', value_fixed=0.0, up=True),
    'U-U': model.find_equilibrium(fix_type='q1', value_fixed=np.pi, up=False), 
    'U-D': model.find_equilibrium(fix_type='q1', value_fixed=np.pi, up=True),
    'R-D': model.find_equilibrium(fix_type='q1', value_fixed=np.pi/2, up=False),
    'R-U': model.find_equilibrium(fix_type='q1', value_fixed=np.pi/2, up=True),
    'L-D': model.find_equilibrium(fix_type='q1', value_fixed=-np.pi/2, up=False),
    'L-U': model.find_equilibrium(fix_type='q1', value_fixed=-np.pi/2, up=True),
    'RD-D': model.find_equilibrium(fix_type='q1', value_fixed=np.pi/4, up=False),
    'RD-U': model.find_equilibrium(fix_type='q1', value_fixed=np.pi/4, up=True),
    'LD-D': model.find_equilibrium(fix_type='q1', value_fixed=-np.pi/4, up=False),
    'LD-U': model.find_equilibrium(fix_type='q1', value_fixed=-np.pi/4, up=True),
    'RU-D': model.find_equilibrium(fix_type='q1', value_fixed=np.pi/4*3, up=False),
    'RU-U': model.find_equilibrium(fix_type='q1', value_fixed=np.pi/4*3, up=True),
    'LU-D': model.find_equilibrium(fix_type='q1', value_fixed=-np.pi/4*3, up=False),
    'LU-U': model.find_equilibrium(fix_type='q1', value_fixed=-np.pi/4*3, up=True),
}

#######################################
# Tasks
#######################################

tasks = {
    'swing_up': { 
        'start': configurations['D-D'],
        'end': configurations['U-U'],
        'time': 10.0,
        'time_ext': 2.0,
        'desc': 'Swing up from Down-Down to Up-Up'
    },
    'topside_pump': { 
        'start': configurations['U-D'],
        'end': configurations['U-U'],
        'time': 6.0,
        'time_ext': 2.0,
        'desc': 'Topside pump from Up-Down to Up-Up'
    },
    'diagonal_swing': { 
       'start': configurations['LD-D'],
       'end': configurations['RU-U'],
       'time': 6.0,
       'time_ext': 2.0,
        'desc': 'Diagonal swing from Left-Down to Right-Up'
    },
    'elbow_up': { 
        'start': configurations['D-D'],
        'end': configurations['R-U'],
        'time': 10.0,
        'time_ext': 2.0,
        'desc': 'Elbow up from Down-Down to Right-Up'
    },
    'top_to_elbow_up': { 
        'start': configurations['U-U'],
        'end': configurations['R-U'],
        'time': 8.0,
        'time_ext': 2.0,
        'desc': 'Top to Elbow up from Up-Up to Right-Up'
    },
    'low_diagonal_hold': {
        'start': configurations['D-D'],
        'end': configurations['RD-D'],
        'time':6.0,
        'time_ext':2.0,
        'desc': 'Low diagonal hold from Down-Down to Right-Diagonal Down'
    },
    'half_swing_up': { 
        'start': configurations['D-U'],
        'end': configurations['U-D'],
        'time': 10.0,
        'time_ext': 4.0,
        'desc': 'Half swing up from Down-Up to Up-Down'
    },
    'swing_down': {
        'start': configurations['U-U'],
        'end': configurations['D-D'],
        'time': 10.0,
        'time_ext': 4.0,
        'desc': 'Swing down from Up-Up to Down-Down'
    },
    'complex_maneuver': { 
        'start': configurations['R-D'],
        'end': configurations['D-U'],
        'time': 10.0,
        'time_ext': 2.0,
        'desc': 'Complex maneuver from Right-Down to Down-Up'
    },
}


#######################################
# Functions
#######################################

def trajectory_interpolation(
    start: np.ndarray,
    end: np.ndarray,
    T0: int,
    Tend: float,
    ti: float,
    tf: float,
    dt: float,
    method: str = 'stepwise',
    input_type:str = 'state'
 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a reference trajectory between start and end states/inputs using specified interpolation method.
    Parameters:
        start: Starting state/input vector.
        end: Ending state/input vector.
        T0: Initial time.
        Tend: End time.
        ti: Start time for transition phase (relative to Tend).
        tf: End time for transition phase (relative to Tend).
        dt: Time step for trajectory generation.
        method: Interpolation method ('stepwise', 'poly3', 'poly5', 'poly7').
        input_type: Type of input ('state' or 'input').
    Returns:
        t: Time vector 
        traj: Generated trajectory array.
    """    

    # Initialization
    t = np.arange(T0, Tend, dt)
    TT = len(t)
    NN = start.shape[0]

    traj = np.zeros((NN, TT))  # Shape (N, T)
    
    # Absolute times for ti and tf
    ti_abs = ti * Tend
    tf_abs = tf * Tend

    # Duration of transition phase
    DT = tf_abs - ti_abs
    
    # Time masks for the three phases
    initial_mask = t < ti_abs                           # Before ti             (starting equilibrium)
    transition_mask = (t >= ti_abs) & (t <= tf_abs)     # Between ti and tf     (transition phase)
    post_mask = t > tf_abs                              # After tf              (final equilibrium)
    
    # Relative time for the transition (normalized to [0, 1] within transition phase)
    tau = (t[transition_mask] - ti_abs) / DT

    # Extract positions and velocities based on input type
    if input_type == 'state':
        ndof = NN // 2
        pos_start = start[0:ndof]
        vel_start = start[ndof:]*DT 
        pos_end = end[0:ndof]
        vel_end = end[ndof:]*DT
    else:
        ndof = NN
        pos_start = start
        vel_start = np.zeros(ndof)
        pos_end = end
        vel_end = np.zeros(ndof)

    # Setpoints for higher-order derivatives (assumed zero)
    acc_start, acc_end = np.zeros(ndof), np.zeros(ndof)
    jerk_start, jerk_end = np.zeros(ndof), np.zeros(ndof)
    
    # Initial and Final Phases Masks
    traj[:, initial_mask] = start[:, None]
    traj[:, post_mask]    = end[:, None]

    # Stepwise Interpolation
    if method == 'stepwise':
        t_mid = (ti_abs + tf_abs) / 2
        traj[:, (t >= ti_abs) & (t <= t_mid)] = start[:, None]
        traj[:, (t > t_mid) & (t <= tf_abs)]  = end[:, None]

    # Polynomial Interpolation
    elif method in ['poly3', 'poly5', 'poly7']:
      
        # 3rd order polynomial
        if method == 'poly3':
            # Constraints: [q_start, q_end, dq_start, dq_end]
            A = np.array([
                [1, 0, 0, 0],
                [1, 1, 1, 1], 
                [0, 1, 0, 0],
                [0, 1, 2, 3] 
            ])
            b = np.vstack([pos_start, pos_end, vel_start, vel_end])
        # 5th order polynomial
        elif method == 'poly5':
            # Constraints: [q_start, q_end, dq_start, dq_end, acc_start, acc_end]
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 2, 6, 12, 20]
            ])
            b = np.vstack([pos_start, pos_end, vel_start, vel_end, acc_start, acc_end])
        # 7th order polynomial
        elif method == 'poly7':
            # Constraints: [+ jerk_start, jerk_end]
            A = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 2, 6, 12, 20, 30, 42],
                [0, 0, 0, 6, 0, 0, 0, 0],
                [0, 0, 0, 6, 24, 60, 120, 210]
            ])
            b = np.vstack([pos_start, pos_end, vel_start, vel_end, acc_start, acc_end, jerk_start, jerk_end])
        
        coeffs = np.linalg.solve(A, b)
        coeffs_rev = coeffs[::-1] # Reverse coefficients for np.polyval
        
        q_traj = np.zeros((ndof, len(tau)))
        
        for i in range(ndof):
            q_traj[i, :] = np.polyval(coeffs_rev[:, i], tau)
        
        # Assign positions to trajectory
        traj[:ndof, transition_mask] = q_traj

        # If input type is 'state', compute and assign velocities
        if input_type == 'state':
            dq_traj = np.zeros((ndof, len(tau)))
            for i in range(ndof):
                der_coeffs = np.polyder(coeffs_rev[:, i])
                dq_traj[i, :] = np.polyval(der_coeffs, tau) / DT

            traj[ndof:,  transition_mask] = dq_traj
    else:
        raise ValueError(f"Method '{method}' not recognized.")

    return t, traj


def reference_trajectory_generator(model: DoublePendulumFlexible, task, reference_params: dict, debug_params: dict,  folder:'str'=None,) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, tuple]:
    """
    Generate reference trajectory for a specified task using the given model and parameters.
    Parameters:
        - model: The flexible double pendulum model.
        - task: The task identifier for which to generate the trajectory.
        - reference_params: Parameters for trajectory generation (ti, tf, method).
        - debug_params: Debugging parameters (plot_flag).
        - folder: Folder path to save plots. Defaults to None.
    Returns:
        - time: Time vector for the reference trajectory.
        - xx_ref: Reference state trajectory.
        - uu_ref: Reference input trajectory.
        - eq_pair1: Starting equilibrium state and input.
        - eq_pair2: Ending equilibrium state and input.
    """

    #######################################
    # Unpack parameters
    #######################################
    
    # debug parameters 
    plot_flag = debug_params['plot_flag']

    # reference parameters
    ti = reference_params['ti']                         # Start time for trajectory segment (relative)
    tf = reference_params['tf']                         # End time for trajectory segment (relative)
    method = reference_params['method']                 # Interpolation method
    dt = model.dt                                       # Time step
    Tend = selected_equilibrium['time']

    #######################################
    # Equilibrium manifold and selected equilibria
    #######################################

    data_dict = model.compute_equilibrium_manifold()
    selected_equilibrium = tasks[task]
    eq_pair1, eq_pair2 = selected_equilibrium['start'], selected_equilibrium['end']
    x_eq1, u_eq1 = eq_pair1
    x_eq2, u_eq2 = eq_pair2

    plots.equilibrium_manifold(data_dict, eq_pair1, eq_pair2, plot_flag=plot_flag, folder=folder)
    model.draw_pendulums(eq_pair1, eq_pair2, title=task, plot_flag=plot_flag, folder=folder)

    ######################################
    # Reference curve
    ######################################

    # Generate reference trajectory for states and inputs
    time, xx_ref = trajectory_interpolation(start = x_eq1, end = x_eq2, T0=0.0, Tend=Tend, ti=ti, tf=tf, dt=dt, method=method, input_type='state')
    time, uu_ref = trajectory_interpolation(start = u_eq1, end = u_eq2, T0=0.0, Tend=Tend, ti=ti, tf=tf, dt=dt, method=method, input_type='input')
    
    # Plot pendulums envelope along reference trajectory
    model.draw_pendulum_envelope(eq_pair1, eq_pair2, xx_ref, title=task, plot_flag=plot_flag, folder=folder)

    plots.show_traj(time, xx_ref, uu_ref, eq_pair1, eq_pair2, "Reference " + method, plot_flag=plot_flag, folder=folder)

    return time, xx_ref, uu_ref, eq_pair1, eq_pair2