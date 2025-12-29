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
perturb_initial_state = True   # perturb initial state for LQR tracking
final_plots = True              # plot final optimal trajectory
visu_animation = True          # animate final optimal trajectory

runs = 5                       # number of runs with perturbed initial conditions

#######################################
# Optimal trajectory
#######################################

print("\n" + "="*80)
print("Trajectory Tracking - LQR".center(80))
print("="*80)

eq_pair1, eq_pair2 = traj_gen_newton.get_equilibrium_pairs()
time, xx_star, uu_star = traj_gen_newton.get_optimal_trajectory()

TT = len(time)

if initial_optimal_plot:
    plots.show_traj(time, xx_star, uu_star, eq_pair1, eq_pair2, title="Optimal trajectory")

######################################
# Arrays to store data
######################################

xx_lqr = np.zeros((ns, TT, runs))      # State seq.
uu_lqr = np.zeros((ni, TT, runs))      # Input seq.
At = np.zeros((ns, ns, TT, runs))      # State matrices
Bt = np.zeros((ns, ni, TT, runs))      # Input matrices
PP = np.zeros((ns, ns, TT, runs))      # P matrices for LQR
KK = np.zeros((ni, ns, TT, runs))      # Feedback gain matrices
x0 = np.zeros((ns, runs))

#######################################
# 1. Linearize system along optimal trajectory
#######################################

# Symbolic jacobian
AA_sym, BB_sym = model_sym.symbolic_jacobian() 

# Linearization along optimal trajectory
for rr in range(runs):
    for tt in range(TT):
        At[:, :, tt, rr], Bt[:, :, tt, rr] = solver.jacobian_solver(AA_sym, BB_sym, xx_star[:,tt], uu_star[:,tt], model)


#######################################
# 2. Solve time-varying LQR problem over finite horizon
# NOTE: the solution is directly computed in the solver.ltv_LQR() function in step 3
#######################################

# Cost matrices
QQt = np.diag([10, 10, 10, 10])
RRt = 0.01 * np.eye(ni)

# Terminal matrices
A_final, B_final = At[:, :, -1, 0], Bt[:, :, -1, 0]

QQT = ctrl.dare(A_final, B_final, QQt, RRt)[0]  # Terminal cost weight matrix

#######################################
# 3. Track the optimal trajcetory with LQR controller
#######################################

# Initial state
if perturb_initial_state:
    epsilon = np.random.uniform(-0.1, 0.1, size=(4,))  # initial state perturbation
else:
    epsilon = np.zeros(4)

for rr in range(runs):
    # Perturbation epsilon slightly increase for each run
    epsilon_scaled = epsilon * (rr**3 + 1) / runs # Select a scaling factor

    x0[:,rr] = xx_star[:,0] + epsilon_scaled # initial state
    
    xx_lqr[:,:,rr], uu_lqr[:,:,rr] = solver.ltv_LQR(At[:,:,:,rr], Bt[:,:,:,rr], QQt, RRt, QQT, x0[:, rr], xx_star, uu_star, model)

#######################################
# Results
#######################################

# Final plots
if final_plots:
    plots.trajectory_comparison(time, xx_star, uu_star, xx_lqr, uu_lqr, title="LQR Trajectory Tracking - Multiple Runs")
    plots.tracking_error(time, xx_star, uu_star, xx_lqr, uu_lqr, title="LQR Tracking Error - Multiple Runs")

# Animation of the optimal trajectory
if visu_animation:
    anim.animate_comparison(model, xx_lqr[:,:,0], xx_star, eq_pair1, eq_pair2, target_fps=30)
