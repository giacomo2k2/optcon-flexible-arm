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

plt.rcParams['font.family'] = 'serif'
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Model
from branch2.model import DoublePendulumFlexible, PARAM_SET_2
from b1_model_symbolic import DoublePendulumFlexibleSymbolic

# Costs
import branch2.costs as cst

# Armijo
import branch2.armijo as armijo

# Trajectories
import branch2.trajectory_generation as traj
import branch2.trajectory_generation_newton as traj_gen_newton

# Solver
import branch2.solver as solver

# Animation & plots
import branch2.animation as anim
import branch2.plots as plots

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

dict_ioptions = {
    "print_time": True,                 # Show computation time
    "ipopt.print_level": 1,             # Level of detail (0 to 12)
    "ipopt.sb": "no",                   # Remove Ipopt banner at start
    "ipopt.derivative_test": "first-order", # Utility to check if gradients are correct
}

# MPC parameters
T_pred = 100                 # MPC Prediction horizon

# Input constraints
umax = 48
umin = -umax

# State constraints
x1max = np.pi
x1min = -x1max
x2max = np.pi
x2min = -x2max
x3max = 5
x3min = -x3max
x4max = 5
x4min = -x4max

#######################################
# Visualization parameters
initial_optimal_plot = True    # plot optimal trajectory
perturb_initial_state = True   # perturb initial state for LQR tracking
final_plots = True              # plot final optimal trajectory
visu_animation = True          # animate final optimal trajectory

runs = 2                       # number of runs with perturbed initial conditions

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

xx_lqr = np.zeros((ns, TT, runs))      # State seq.
uu_lqr = np.zeros((ni, TT, runs))      # Input seq.
At = np.zeros((ns, ns, TT))      # State matrices
Bt = np.zeros((ns, ni, TT))      # Input matrices
PP = np.zeros((ns, ns, TT))      # P matrices for LQR
KK = np.zeros((ni, ns, TT))      # Feedback gain matrices
x0 = np.zeros((ns, runs))

#######################################
# 1. Linearize system along optimal trajectory
#######################################

# Symbolic jacobian
AA_sym, BB_sym = model_sym.symbolic_jacobian() 

# Linearization along optimal trajectory

for tt in range(TT):
    At[:, :, tt], Bt[:, :, tt] = solver.jacobian_solver(AA_sym, BB_sym, xx_star[:,tt], uu_star[:,tt], model)


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
if perturb_initial_state:
    epsilon = np.random.uniform(-0.1, 0.1, size=(4,))  # initial state perturbation
else:
    epsilon = np.zeros(4)

#######################################
# LQR Controller
#######################################

#for rr in range(runs):
#    # Perturbation epsilon slightly increase for each run
#    epsilon_scaled = epsilon * (rr**3 + 1) / runs # Select a scaling factor
#
#    x0[:,rr] = xx_star[:,0] + epsilon_scaled # initial state
#    
#    xx_lqr[:,:,rr], uu_lqr[:,:,rr] = solver.ltv_LQR(At, Bt, QQt, RRt, QQT, x0[:, rr], xx_star, uu_star, model)


#######################################
# Model Predictive Control
#######################################

print("\n" + "="*80)
print("Trajectory Tracking - MPC".center(80))
print("="*80)

xx_mpc = np.zeros((ns,TT,runs))     # State seq.
uu_mpc = np.zeros((ni,TT,runs))     # Input seq.

for rr in range(runs):
    
    epsilon_scaled = epsilon * (rr**3 + 1) / runs # Select a scaling factor

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
            umax=umax, umin=umin, x1_max=x1max, x1_min=x1min,
            x2_max=x2max, x2_min=x2min, x3_max=x3max, x3_min=x3min,
            x4_max=x4max, x4_min=x4min, T_pred=T_pred, ipopt_opts=dict_ioptions
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
    print(f"Completed MPC run {rr+1}/{runs}")

#######################################
# Results
#######################################

# Final plots
if final_plots:
    plots.trajectory_comparison(time, xx_star, uu_star, xx_mpc, uu_mpc, title="MPC Trajectory Tracking ")
    plots.tracking_error(time, xx_star, uu_star, xx_mpc, uu_mpc, title="MPC Tracking Error")

# Animation of the optimal trajectory
if visu_animation:
    anim.animate_comparison(model, xx_mpc[:,:,0], xx_star, eq_pair1, eq_pair2, target_fps=30)
