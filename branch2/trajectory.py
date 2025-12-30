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
    'U-D': model.find_equilibrium(fix_type='q1', value_fixed=np.pi, up=False),
    'U-U': model.find_equilibrium(fix_type='q1', value_fixed=np.pi, up=True),
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
    'elbow_up': {'start': configurations['D-D'], 'end': configurations['D-U']},
    'swing_up': {'start': configurations['D-D'], 'end': configurations['U-D']},
    'half_swing':{'start': configurations['D-D'], 'end': configurations['U-D']},
    'top_to_elbow_up': {'start': configurations['U-U'], 'end': configurations['R-U']},
    'top_to_elbow_down': {'start': configurations['U-U'], 'end': configurations['L-D']},
    'inverted_hand_off': {'start': configurations['D-U'], 'end': configurations['U-D']},
    'topside_pump': {'start': configurations['U-D'], 'end': configurations['U-U']},
    'lateral_swing': {'start': configurations['L-U'], 'end': configurations['R-U']},
    'diagonal_swing': {'start': configurations['LD-D'], 'end': configurations['RU-U']},
    'the_dive': {'start': configurations['U-U'], 'end': configurations['D-D']},
    'overhead_wipe': {'start': configurations['LU-U'], 'end': configurations['RU-U']},
    'low_diagonal_hold': {'start': configurations['D-D'], 'end': configurations['RD-D']},
}



def trajectory_generator(
    T0: int = 0,
    Tend: float = 1.0,
    ti: float = 0.3,
    tf: float = 0.7,
    setpoints: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    vel_setpoints: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    acc_setpoints: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    jerk_setpoints: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    dt: float = 0.01,
    method: str = 'stepwise',
 ) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Generates a trajectory for a Double Pendulum system based on specified setpoints and method.

    Parameters
    ----------
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

    Returns
    -------
    t : np.ndarray
        Array of time points.
    traj : np.ndarray
        Generated trajectory (shape: [4, N]).
    """
    
    # Initialization
    t = np.arange(T0, Tend, dt)
    num_steps = len(t)
    num_states = setpoints.shape[1]
    traj = np.zeros((num_states, num_steps))  # Shape (4, N)
    
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

    p0, p1 = setpoints[0, :], setpoints[-1, :]
    v0, v1 = vel_setpoints[0, :], vel_setpoints[-1, :] * DT
    a0, a1 = acc_setpoints[0, :], acc_setpoints[-1, :] * (DT**2)
    j0, j1 = jerk_setpoints[0, :], jerk_setpoints[-1, :] * (DT**3)
    
    # Stepwise Interpolation
    if method == 'stepwise':
        t_mid = (ti_abs + tf_abs) / 2
        traj[:, (t <= t_mid)] = p0[:, None]
        traj[:, (t > t_mid)] = p1[:, None]

    # Polynomial Interpolation
    elif method in ['poly3', 'poly5', 'poly7']:
        traj[:, initial_mask] = p0[:, None]
        traj[:, post_mask] = p1[:, None]
        
        for dim in range(num_states):
            # 3rd order polynomial
            if method == 'poly3':
                A = np.array([
                    [1, 0, 0, 0], [1, 1, 1, 1], # Pos
                    [0, 1, 0, 0], [0, 1, 2, 3] # Vel
                    ])
                b = np.array([p0[dim], p1[dim], v0[dim], v1[dim]])
            # 5th order polynomial
            elif method == 'poly5':
                A = np.array([
                    [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], # Pos
                    [0, 1, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], # Vel
                    [0, 0, 2, 0, 0, 0], [0, 0, 2, 6, 12, 20] # Acc
                    ])
                b = np.array([p0[dim], p1[dim], v0[dim], v1[dim], a0[dim], a1[dim]])
            # 7th order polynomial
            elif method == 'poly7':
                A = np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], # Pos
                    [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7], # Vel
                    [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 2, 6, 12, 20, 30, 42], # Acc
                    [0, 0, 0, 6, 0, 0, 0, 0], [0, 0, 0, 6, 24, 60, 120, 210] # Jerk
                ])
                b = np.array([p0[dim], p1[dim], v0[dim], v1[dim], a0[dim], a1[dim], j0[dim], j1[dim]])
            coeffs = np.linalg.solve(A, b)
            traj[dim, transition_mask] = np.polyval(coeffs[::-1], tau)

    # Cubic Spline Interpolation
    elif method == 'cubic_spline':
        traj[:, initial_mask] = p0[:, None]
        traj[:, post_mask] = p1[:, None]
        t_pts_rel = np.linspace(0, 1, setpoints.shape[0])
        for dim in range(num_states):
            bc = ((1, vel_setpoints[0, dim] * DT), (1, vel_setpoints[-1, dim] * DT))
            cs = CubicSpline(t_pts_rel, setpoints[:, dim], bc_type=bc)
            traj[dim, transition_mask] = cs(tau)
    else:
        raise ValueError(f"Method '{method}' not recognized.")

    return t, traj
