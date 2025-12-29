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
perturb_initial_state = True    # perturb initial state
final_plots = True              # plot final optimal trajectory
visu_animation = True          # animate final optimal trajectory

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

At = np.zeros((ns, ns, TT))      # State matrices
Bt = np.zeros((ns, ni, TT))      # Input matrices

#######################################
# 1. Linearize system along optimal trajectory
#######################################

# Symbolic jacobian
AA_sym, BB_sym = model_sym.symbolic_jacobian() 

# Linearization along optimal trajectory
for tt in range(TT):
    At[:, :, tt], Bt[:, :, tt] = solver.jacobian_solver(AA_sym, BB_sym, xx_star[:,tt], uu_star[:,tt], model)

#######################################
# 2. Solve time-varying LQR problem over finite horizon
#######################################

# Cost matrices
QQt = np.diag([10, 10, 10, 10]) * 1
RRt = 0.01 * np.eye(ni)
QQT = ctrl.dare(At[:, :, -1], Bt[:, :, -1], QQt, RRt)[0]  # Terminal cost weight matrix


#######################################
# 3. Track the optimal trajcetory with LQR controller
#######################################

# Initial state
if perturb_initial_state:
    epsilon = np.array([0.1, -0.1, 0.0, 0.0])  # initial state perturbation
else:
    epsilon = np.zeros(4)

x0 = xx_star[:,0] + epsilon  # initial state



xx_lqr, uu_lqr = solver.ltv_LQR(At, Bt, QQt, RRt, QQT, x0, xx_star, uu_star, model)


#######################################
# Results
#######################################

if final_plots:
    # TODO: fix label in plots (ref, guess --> star, lqr)
    plots.trajectory_comparison(time, xx_star, uu_star, xx_lqr, uu_lqr, title="LQR Trajectory Tracking")
    # TODO: plot system traj and desidred over perturbed initial condition
    # TODO: tracking error for different initial conditions

# Animation of the optimal trajectory
if visu_animation:
    anim.animate_comparison(model, xx_lqr, xx_star, eq_pair1, eq_pair2, target_fps=30)