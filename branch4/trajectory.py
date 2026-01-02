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

# TODO: sistemare U-D e U-D
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
    'elbow_up': {
        'start': configurations['D-D'],
        'end': configurations['R-U'],
        'time': 4.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'

    },
    'swing_up': {
        'start': configurations['D-D'],
        'end': configurations['U-U'],
        'time': 4.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'
    },
    'half_swing':{ # difficult
        'start': configurations['D-D'],
        'end': configurations['U-D'],
        'time': 6.5,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'rk4'
    },
    'top_to_elbow_up': {
        'start': configurations['U-U'],
        'end': configurations['R-U'],
        'time': 3.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'
    },
    'top_to_elbow_down': { # difficult
        'start': configurations['U-U'],
        'end': configurations['L-D'],
        'time': 8.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD',
        'discr_method': 'rk4'
    },
    'inverted_hand_off': {
        'start': configurations['D-U'],
        'end': configurations['U-D'],
        'time': 3.0,
        'angle1_mult': 1000,
        'vel1_mult': 1000,
        'angle2_mult': 1000,
        'vel2_mult': 1000,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'
    },
    'topside_pump': { # difficult
        'start': configurations['U-D'],
        'end': configurations['U-U'],
        'time': 4.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'D_gc',
        'discr_method': 'rk4'
    },
    'lateral_swing': {
        'start': configurations['L-U'],
        'end': configurations['R-U'],
        'time': 4.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'
    },
    'diagonal_swing': {
        'start': configurations['LD-D'],
        'end': configurations['RU-U'],
        'time': 6.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'
    },
    'the_dive': { # difficult
        'start': configurations['U-U'],
        'end': configurations['D-D'],
        'time': 8.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'zero',
        'discr_method': 'rk4'
    },
    'overhead_wipe': {
        'start': configurations['LU-U'],
        'end': configurations['RU-U'],
        'time': 4.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'
    },
    'low_diagonal_hold': {
        'start': configurations['D-D'],
        'end': configurations['RD-D'],
        'time':6.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'
    },
    'half_dive': { # difficult
        'start': configurations['U-U'],
        'end': configurations['U-D'],
        'time': 8.0,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'D_gc',
        'discr_method': 'rk4'
    },
    'half_middle_dive': { 
        'start': configurations['RU-U'],
        'end': configurations['R-U'],
        'time': 3,
        'angle1_mult': 1,
        'vel1_mult': 1,
        'angle2_mult': 1,
        'vel2_mult': 1,
        'termcost_mult': 1,
        'guess': 'PD_gc',
        'discr_method': 'fe'
    },
    
#the_dive
#QQt_base = np.diag([10.0, 10.0, 1.0, 1.0])
#RRt = 1e-2 * np.eye(1)
#zero controller initial guess
#time 8
#rk4
#2 e3

#top_to_elbow_down
#QQt_base = np.diag([10.0, 10.0, 1.0, 1.0])
#RRt = 1e-2 * np.eye(1)
#PD 
#time 8
#rk4
#5 e3
#REGULARIZATION rho = 0.1 in ltv_affine_LQR_step OR IT EXPLODES

#half_swing
#QQt_base = np.diag([10.0, 10.0, 1.0, 1.0])
#RRt = 1e-2 * np.eye(1)
#PD + gravity compensation initial guess or dissipative
#time 6
#rk4
#4.75 e3

#topside_pump
#QQt_base = np.diag([10.0, 10.0, 1.0, 1.0])
#RRt = 1e-2 * np.eye(1)
#PD + gravity compensation initial guess or dissipative
#time 4
#rk4
#2.95 e3

#half_dive
#QQt_base = np.diag([10.0, 10.0, 1.0, 1.0])
#RRt = 1e-2 * np.eye(1)
#PD + gravity compensation initial guess or dissipative
#time 8
#rk4
#1.42 e3

#TODO check the regularization rho in ltv_affine_LQR_step
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
    input_type:str = 'state',
    q2_linear_interpolation: bool = True
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

    acc_traj = np.zeros((ndof, TT))  # Shape (N/2, T) for accelerations

    acc_start, acc_end = np.zeros(ndof), np.zeros(ndof)
    jerk_start, jerk_end = np.zeros(ndof), np.zeros(ndof)
    
    traj[:, initial_mask] = start[:, None]
    traj[:, post_mask]    = end[:, None]
    acc_traj[:, initial_mask] = acc_start[:, None]
    acc_traj[:, post_mask]    = acc_end[:, None]

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


        if q2_linear_interpolation:
            # apply a simple linear interpolation to theta2
            q2_start = pos_start[1]
            q2_end = pos_end[1]
            q_traj[1, :] = q2_start + (q2_end - q2_start) * tau
        


        traj[:ndof, transition_mask] = q_traj

        if input_type == 'state':
            dq_traj = np.zeros((ndof, len(tau)))
            for i in range(ndof):
                # polyder computes analytical derivative of coefficients
                der_coeffs = np.polyder(coeffs_rev[:, i])
                dq_traj[i, :] = np.polyval(der_coeffs, tau) / DT
        
            #if q2_linear_interpolation:
                #dq = (q2_end - q2_start)/DT
                #dq_traj[1, :] = dq

            #dq_traj[1, :] = np.zeros(len(tau))
            #dq_traj[0, :] = np.zeros(len(tau))


            traj[ndof:,  transition_mask] = dq_traj

        # compute acceleration
        ddq_traj = np.zeros((ndof, len(tau)))
        for i in range(ndof):
            der2_coeffs = np.polyder(coeffs_rev[:, i], 2)
            ddq_traj[i, :] = np.polyval(der2_coeffs, tau) / (DT**2)
        
        acc_traj[:, transition_mask] = ddq_traj
    else:
        raise ValueError(f"Method '{method}' not recognized.")

    return t, traj, acc_traj
