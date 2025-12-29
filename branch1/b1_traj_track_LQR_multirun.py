#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         trajectory_tracking_LQR.py
# Description:  Trajectory tracking via LQR.
# TODO: questo file dovrà sostituire il b1_traj_track_LQR.py
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

plt.rcParams['font.family'] = 'serif'
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Model
from b1_model import DoublePendulumFlexible, PARAM_SET_2
from b1_model_symbolic import DoublePendulumFlexibleSymbolic

# Costs
import b1_costs as cst

# Armijo
import b1_armijo as armijo

# Trajectories
import b1_trajectories as traj
import b1_main_newton_method as traj_gen_newton

# Solver
import b1_solver as solver

# Animation & plots
import b1_animation as anim
import b1_plots as plots

# Control
import control as ctrl

#######################################
# Model
#######################################

model = DoublePendulumFlexible(**PARAM_SET_2)
model_sym = DoublePendulumFlexibleSymbolic(**PARAM_SET_2)

ns = model.ns  # number of states
ni = model.ni  # number of inputs

#######################################
# Algorithm parameters
#######################################

# Visualization parameters
initial_optimal_plot = True    # plot optimal trajectory
multiple_runs = True         # perform multiple runs with perturbed initial conditions
perturb_initial_state = True   # perturb initial state for LQR tracking
final_plots = True              # plot final optimal trajectory
visu_animation = True          # animate final optimal trajectory

runs = 5                       # number of runs with perturbed initial conditions

#######################################
# Optimal trajectory
#######################################

eq_pair1, eq_pair2 = traj_gen_newton.get_equilibrium_pairs()
time, xx_star, uu_star = traj_gen_newton.get_optimal_trajectory()

TT = len(time)

if initial_optimal_plot:
    plots.show_traj(time, xx_star, uu_star, eq_pair1, eq_pair2, title="Optimal trajectory")

######################################
# Arrays to store data
######################################

if multiple_runs:
    xx_lqr = np.zeros((ns, TT, runs))      # State seq.
    uu_lqr = np.zeros((ni, TT, runs))      # Input seq.
    At = np.zeros((ns, ns, TT, runs))      # State matrices
    Bt = np.zeros((ns, ni, TT, runs))      # Input matrices
    PP = np.zeros((ns, ns, TT, runs))      # P matrices for LQR
    KK = np.zeros((ni, ns, TT, runs))      # Feedback gain matrices
    x0 = np.zeros((ns, runs))
else:
    xx_lqr = np.zeros((ns, TT))      # State seq.
    uu_lqr = np.zeros((ni, TT))      # Input seq.
    At = np.zeros((ns, ns, TT))      # State matrices
    Bt = np.zeros((ns, ni, TT))      # Input matrices
    PP = np.zeros((ns, ns, TT))      # P matrices for LQR
    KK = np.zeros((ni, ns, TT))      # Feedback gain matrices

#######################################
# 1. Linearize system along optimal trajectory
#######################################

# Symbolic jacobian
AA_sym, BB_sym = model_sym.symbolic_jacobian() 

# Linearization along optimal trajectory
if multiple_runs:
    for rr in range(runs):
        for tt in range(TT):
            At[:, :, tt, rr], Bt[:, :, tt, rr] = solver.jacobian_solver(AA_sym, BB_sym, xx_star[:,tt], uu_star[:,tt], model)
else:
    for tt in range(TT):
        At[:, :, tt], Bt[:, :, tt] = solver.jacobian_solver(AA_sym, BB_sym, xx_star[:,tt], uu_star[:,tt], model)

#######################################
# 2. Solve time-varying LQR problem over finite horizon
#######################################

# Cost matrices
QQt = np.diag([10, 10, 1, 1]) * 1e-3
RRt = 0.01 * np.eye(ni)

A_final, B_final = At[:, :, -1, 0], Bt[:, :, -1, 0] if multiple_runs else At[:, :, -1], Bt[:, :, -1] 

QQT = ctrl.dare(A_final, B_final, QQt, RRt)[0]  # Terminal cost weight matrix

# Solve Riccati difference equation backward in time
if multiple_runs:
    for rr in range(runs):
        PP[:,:,TT-1,rr] = QQT
        for tt in reversed(range(TT-1)):
            KK[:,:,tt,rr], PP[:,:,tt,rr] = solver.ltv_LQR_step(At[:,:,tt,rr], Bt[:,:,tt,rr], QQt, RRt, PP[:,:,tt+1,rr])
else:
    PP[:,:,TT-1] = QQT
    for tt in reversed(range(TT-1)):
        KK[:,:,tt], PP[:,:,tt] = solver.ltv_LQR_step(At[:,:,tt], Bt[:,:,tt], QQt, RRt, PP[:,:,tt+1])

#######################################
# 3. Track the optimal trajcetory with LQR controller
#######################################

# Initial state
if multiple_runs:
    for rr in range(runs):
        if perturb_initial_state:
            epsilon = np.random.uniform(-0.1, 0.1, size=(4,))  # initial state perturbation
        else:
            epsilon = np.zeros(4)
        
        x0[:,rr] = xx_star[:,0] + epsilon # initial state
        xx_lqr[:,0,rr] = x0[:,rr]

        for tt in range(TT-1):
            uu_lqr[:,tt,rr] = uu_star[:,tt] + KK[:,:,tt,rr] @ (xx_lqr[:,tt,rr] - xx_star[:,tt])
            xx_lqr[:,tt+1,rr] = model.discrete_dynamics(xx_lqr[:,tt,rr], uu_lqr[:,tt,rr])
        
    uu_lqr[:,-1,:] = uu_lqr[:,-2,:] # for plotting purposes
    
else:
    if perturb_initial_state:
        epsilon = np.array([0.1, -0.1, 0.0, 0.0])  # initial state perturbation
    else:
        epsilon = np.zeros(4)

    x0 = xx_star[:,0] + epsilon  # initial state
    xx_lqr[:,0] = x0

    for tt in range(TT-1):
        uu_lqr[:,tt] = uu_star[:,tt] + KK[:,:,tt] @ (xx_lqr[:,tt] - xx_star[:,tt])
        xx_lqr[:,tt+1] = model.discrete_dynamics(xx_lqr[:,tt], uu_lqr[:,tt])

    uu_lqr[:,-1] = uu_lqr[:,-2] # for plotting purposes

#######################################
# Results
#######################################

# Final plots
if final_plots:
    if multiple_runs:
        # Optimal traj vs LQR traj for multiple perturbed initial conditions
        plots.trajectory_comparison_multiple_runs(time, xx_star, uu_star, xx_lqr, uu_lqr, runs, title="LQR Trajectory Tracking - Multiple Runs")
    else:
        # Optimal traj vs LQR traj for single perturbed initial condition
        plots.trajectory_comparison(time, xx_star, uu_star, xx_lqr, uu_lqr, title="LQR Trajectory Tracking")
    

    # TODO: tracking error for different initial conditions

# Animation of the optimal trajectory
if visu_animation:
    anim.animate_comparison(model, xx_lqr, xx_star, eq_pair1, eq_pair2, target_fps=30)