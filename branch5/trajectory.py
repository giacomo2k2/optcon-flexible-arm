#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         trajectories.py
# Description:  Trajectory generation functions.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         December 2025
#########################################################################################

#######################################
# Imports
#######################################
import numpy as np
from scipy.interpolate import CubicSpline
from model import DoublePendulumFlexible, PARAM_SET_2


#######################################
# Functions
#######################################


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

tasks = {
    
    'swing_up': { 
        'start': configurations['D-D'],
        'end': configurations['U-U'],
        'time': 2.0,
    },
    'topside_pump': { 
        'start': configurations['U-D'],
        'end': configurations['U-U'],
        'time': 2.0,
    },
    #'overhead_wipe': {
    #    'start': configurations['LU-U'],
    #    'end': configurations['RU-U'],
    #    'time': 2.0,
    #},
    #'diagonal_swing': { 
    #    'start': configurations['LD-D'],
    #    'end': configurations['RU-U'],
    #    'time': 2.0,
    #},
    'elbow_up': { 
        'start': configurations['D-D'],
        'end': configurations['R-U'],
        'time': 2.0,
    },
    #'top_to_elbow_up': { 
    #    'start': configurations['U-U'],
    #    'end': configurations['R-U'],
    #    'time': 2.0,
    #},
    'low_diagonal_hold': {
        'start': configurations['D-D'],
        'end': configurations['RD-D'],
        'time':2.0,
    },
    'lateral_swing': {
        'start': configurations['L-U'],
        'end': configurations['R-U'],
        'time': 2.0,
    },
    #'half_swing':{ 
    #    'start': configurations['D-D'],
    #    'end': configurations['U-D'],
    #    'time': 4.0,
    #},
    'inverted_hand_off': { 
        'start': configurations['D-U'],
        'end': configurations['U-D'],
        'time': 6.0,
    },
    'the_dive': {
        'start': configurations['U-U'],
        'end': configurations['D-D'],
        'time': 4.0,
    },
    #'half_dive': { 
    #    'start': configurations['U-U'],
    #    'end': configurations['U-D'],
    #    'time': 4.0,
    #},
    #'top_to_elbow_down': { 
    #    'start': configurations['U-U'],
    #    'end': configurations['R-D'],
    #    'time': 4.0,
    #},
    #'custom1': { 
    #    'start': configurations['LD-U'],
    #    'end': configurations['RU-U'],
    #    'time': 2.0,
    #},
    #'custom2': { 
    #    'start': configurations['RU-U'],
    #    'end': configurations['LD-U'],
    #    'time': 2.0,
    #},
    #'custom3': { 
    #    'start': configurations['RU-D'],
    #    'end': configurations['LD-U'],
    #    'time': 2.0,
    #},
    #'custom4': { 
    #    'start': configurations['RU-U'],
    #    'end': configurations['D-U'],
    #    'time': 2.0,
    #},
    'custom5': { 
        'start': configurations['R-D'],
        'end': configurations['D-U'],
        'time': 2.0,
    },
    #'custom6': { 
    #    'start': configurations['R-D'],
    #    'end': configurations['L-U'],
    #    'time': 2.0,
    #},
    #'custom7': { 
    #    'start': configurations['R-D'],
    #    'end': configurations['R-U'],
    #    'time': 4.0,
    #},
}



def trajectory_generator(
    start: np.ndarray,
    end: np.ndarray,
    T0: int = 0,
    Tend: float = 1.0,
    ti: float = 0.3,
    tf: float = 0.7,
    dt: float = 0.01,
    method: str = 'stepwise',
    input_type:str = 'state'
 ) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Generates a trajectory for a Double Pendulum system based on specified setpoints and method.

    Parameters
    ----------
    start : np.ndarray
        Starting state of the system.
    end : np.ndarray
        Ending state of the system.
    T0 : int
        Initial time.
    Tend : float
        End time.
    ti : float
        Start time for trajectory segment.
    tf : float
        End time for trajectory segment.
    setpoints : np.ndarray
        Array of setpoints for the trajectory (shape: [M, 4]).
    vel_setpoints : np.ndarray
        Array of velocity setpoints for the trajectory (shape: [M, 4]).
    dt : float
        Time step for the trajectory.
    method : str
        Method for trajectory generation ('stepwise', 'poly3', 'poly5', 'poly7', 'cubic_spline').
    input_type : str
        Type of input provided ('state' or 'position').
    q2_linear_interpolation : bool
        If True, applies linear interpolation to the second joint angle.

    Returns
    -------
    t : np.ndarray
        Array of time points.
    traj : np.ndarray
        Generated trajectory (shape: [4, N]).
    """
    
    
    # Initialization
    t = np.arange(T0, Tend, dt)
    TT = len(t)
    NN = start.shape[0]

    traj = np.zeros((NN, TT))  # Shape (N, T)
    
    
    # Time boundaries (absolute)
    ti_abs = ti * Tend
    tf_abs = tf * Tend
    DT = tf_abs - ti_abs
    
    # Time masks for the three phases
    initial_mask = t < ti_abs                           # Before ti             (starting equilibrium)
    transition_mask = (t >= ti_abs) & (t <= tf_abs)     # Between ti and tf     (transition phase)
    post_mask = t > tf_abs                              # After tf              (final equilibrium)
    
    # Relative time for the transition (normalized to [0, 1] within transition phase)
    tau = (t[transition_mask] - ti_abs) / DT

    
    if input_type == 'state':
        ndof = NN // 2
        pos_start = start[0:ndof]
        vel_start = start[ndof:]*DT # scale velocities by DT
        pos_end = end[0:ndof]
        vel_end = end[ndof:]*DT # scale velocities by DT

    else:
        ndof = NN
        pos_start = start
        vel_start = np.zeros(ndof)
        pos_end = end
        vel_end = np.zeros(ndof)


    acc_start, acc_end = np.zeros(ndof), np.zeros(ndof)
    jerk_start, jerk_end = np.zeros(ndof), np.zeros(ndof)
    
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
        


        traj[:ndof, transition_mask] = q_traj

        if input_type == 'state':
            dq_traj = np.zeros((ndof, len(tau)))
            for i in range(ndof):
                # polyder computes analytical derivative of coefficients
                der_coeffs = np.polyder(coeffs_rev[:, i])
                dq_traj[i, :] = np.polyval(der_coeffs, tau) / DT


            traj[ndof:,  transition_mask] = dq_traj
    else:
        raise ValueError(f"Method '{method}' not recognized.")

    return t, traj
