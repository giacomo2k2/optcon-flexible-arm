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

dict_ioptions = {
    "print_time": True,                 # Show computation time
    "ipopt.print_level": 1,             # Level of detail (0 to 12)
    "ipopt.sb": "no",                   # Remove Ipopt banner at start
    "ipopt.derivative_test": "first-order", # Utility to check if gradients are correct
}


# MPC parameters
T_pred = 300                  # MPC Prediction horizon

# Input constraints
umax = 40
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

# Visualization parameters
initial_optimal_plot = True    # plot optimal trajectory
perturb_initial_state = True    # perturb initial state
final_plots = True              # plot final optimal trajectory
visu_animation = True          # animate final optimal trajectory

#######################################
# Optimal trajectory
#######################################

print("\n" + "="*80)
print("Trajectory Tracking - MPC".center(80))
print("="*80)

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

######################################
# Unconstrained LQR solution
######################################

# Cost matrices
QQt = np.diag([10, 10, 10, 10]) * 1
RRt = 0.0 * np.eye(ni)
QQT = ctrl.dare(At[:, :, -1], Bt[:, :, -1], QQt, RRt)[0]  # Terminal cost weight matrix

# Initial state
if perturb_initial_state:
    epsilon = np.array([0.1, -0.1, 0.0, 0.0])  # initial state perturbation
else:
    epsilon = np.zeros(4)

x0 = xx_star[:,0] + epsilon  # initial state

# Solve unconstrained LTV LQR
xx_LQR, uu_LQR = solver.ltv_LQR(At, Bt, QQt, RRt, QQT, x0, xx_star, uu_star, model)

#######################################
# Model Predictive Control
#######################################

xx_mpc = np.zeros((ns,TT))
uu_mpc = np.zeros((ni,TT))
xx_mpc[:,0] = x0.squeeze()

for tt in range(TT - 1):
    x_curr = xx_mpc[:, tt] # Measure initial condition

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
        
        # Aggiungi (padda) l'ultimo valore disponibile per riempire T_pred
        # Usiamo np.repeat per mantenere l'ultima matrice/stato costante
        At_pad = np.repeat(At[:, :, -1:], padding_len, axis=2)
        Bt_pad = np.repeat(Bt[:, :, -1:], padding_len, axis=2)
        x_ref_pad = np.repeat(xx_star[:, -1:], padding_len, axis=1)
        u_ref_pad = np.repeat(uu_star[:, -1:], padding_len, axis=1)
        
        # Concatena i dati reali con il padding
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
        uu_mpc[:, tt] = uu_star[:, tt] + du0
        
        # Evoluzione sistema
        xx_mpc[:, tt+1] = model.discrete_dynamics(xx_mpc[:, tt], uu_mpc[:, tt])
    else:
        print(f"MPC failed at step {tt}")
        break


#######################################
# Results
#######################################

uu_mpc[:, -1] = uu_mpc[:, -2]  # for plotting purposes


fig, axs = plt.subplots(ns+ni, 1, figsize=(10, 8), sharex=True)
fig.suptitle('MPC vs LQR: states and input', fontsize=14)

# x1
axs[0].fill_between(time, x1min, x1max, color='tab:green', alpha=0.08, label='x1 constraints')
axs[0].plot(time, xx_mpc[0, :], color='tab:blue', lw=2, label='MPC')
axs[0].plot(time, xx_LQR[0, :], '--', color='tab:red', lw=2, label='LQR')
axs[0].plot(time, xx_star[0, :], ':', color='tab:orange', lw=2, label='Optimal traj.')
axs[0].grid(which='both', linestyle='--', alpha=0.5)
axs[0].set_ylabel('x1', fontsize=12)
axs[0].legend(loc='upper right')

# x2
axs[1].fill_between(time, x2min, x2max, color='tab:green', alpha=0.08, label='x2 constraints')
axs[1].plot(time, xx_mpc[1, :], color='tab:blue', lw=2, label='MPC')
axs[1].plot(time, xx_LQR[1, :], '--', color='tab:red', lw=2, label='LQR')
axs[1].plot(time, xx_star[1, :], ':', color='tab:orange', lw=2, label='Optimal traj.')
axs[1].grid(which='both', linestyle='--', alpha=0.5)
axs[1].set_ylabel('x2', fontsize=12)
axs[1].legend(loc='upper right')

# x3
axs[2].fill_between(time, x3min, x3max, color='tab:green', alpha=0.08, label='x3 constraints')
axs[2].plot(time, xx_mpc[2, :], color='tab:blue', lw=2, label='MPC')
axs[2].plot(time, xx_LQR[2, :], '--', color='tab:red', lw=2, label='LQR')
axs[2].plot(time, xx_star[2, :], ':', color='tab:orange', lw=2, label='Optimal traj.')
axs[2].grid(which='both', linestyle='--', alpha=0.5)
axs[2].set_ylabel('x3', fontsize=12)
axs[2].legend(loc='upper right')

# x4
axs[3].fill_between(time, x4min, x4max, color='tab:green', alpha=0.08, label='x4 constraints')
axs[3].plot(time, xx_mpc[3, :], color='tab:blue', lw=2, label='MPC')
axs[3].plot(time, xx_LQR[3, :], '--', color='tab:red', lw=2, label='LQR')
axs[3].plot(time, xx_star[3, :], ':', color='tab:orange', lw=2, label='Optimal traj.')
axs[3].grid(which='both', linestyle='--', alpha=0.5)
axs[3].set_ylabel('x4', fontsize=12)
axs[3].legend(loc='upper right')

# u --> i am using step to show the piecewise constant input, if you put "plot" it connects the dots with lines
axs[4].fill_between(time, umin, umax, color='tab:green', alpha=0.06, label='u constraints')
axs[4].step(time, uu_mpc[0, :], where='post', color='tab:blue', lw=2, label='MPC')
axs[4].step(time, uu_LQR[0, :], where='post', linestyle='--', color='tab:red', lw=2, label='LQR')
axs[4].plot(time, uu_star[0, :], ':', color='tab:orange', lw=2, label='Optimal traj.')
axs[4].grid(which='both', linestyle='--', alpha=0.5)
axs[4].set_ylabel('u', fontsize=12)
axs[4].set_xlabel('time (steps)', fontsize=12)
axs[4].legend(loc='upper right')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.align_ylabels(axs)
plt.show()

# Animation of the optimal trajectory
if visu_animation:
    anim.animate_comparison(model, xx_mpc, xx_star, eq_pair1, eq_pair2, target_fps=30)