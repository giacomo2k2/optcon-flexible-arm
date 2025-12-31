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
from model import DoublePendulumFlexible, PARAM_SET_2

# Trajectories
import trajectory_generation_newton as traj_gen_newton

# Solver
import solver as solver

# Animation & plots
import animation as anim
import plots as plots

# Control
import control as ctrl

#######################################
# Model
#######################################

model = DoublePendulumFlexible(**PARAM_SET_2)

ns = model.ns  # number of states
ni = model.ni  # number of inputs


def trajectory_tracking_LQR(eq_pair1, eq_pair2, xx_star, uu_star, time, runs = 1, perturb=True, perturbation_scale=2, plot_results=True, debug_print=False):
    """
    Trajectory tracking via LQR.
    """

    if debug_print:
        print("\n--- Trajectory Tracking via LQR ---\n")


    #######################################
    # Optimal trajectory
    #######################################


    # Extract time horizon
    TT = len(time)

    # Plot optimal trajectory
    if plot_results:
        plots.show_traj(time, xx_star, uu_star, eq_pair1, eq_pair2, title="Optimal trajectory")

    ######################################
    # Arrays to store data
    ######################################

    xx_lqr = np.zeros((ns, TT, runs))       # State seq.
    uu_lqr = np.zeros((ni, TT, runs))       # Input seq.
    At = np.zeros((ns, ns, TT, runs))       # State matrices
    Bt = np.zeros((ns, ni, TT, runs))       # Input matrices
    PP = np.zeros((ns, ns, TT, runs))       # P matrices for LQR
    KK = np.zeros((ni, ns, TT, runs))       # Feedback gain matrices
    x0 = np.zeros((ns, runs))               # Initial states

    #######################################
    # 1. Linearize system along optimal trajectory
    #######################################

    # Linearization along optimal trajectory
    for rr in range(runs):
        for tt in range(TT):
            At[:, :, tt, rr], Bt[:, :, tt, rr] = solver.jacobian_analytic(xx_star[:,tt], uu_star[:,tt], model)

    #######################################
    # 2. Solve time-varying LQR problem over finite horizon
    #######################################

    # Cost matrices
    QQt = np.diag([10, 10, 10, 10])
    RRt = 0.01 * np.eye(ni)

    # Terminal matrices
    A_final, B_final = At[:, :, -1, 0], Bt[:, :, -1, 0]

    # Terminal cost weight matrix
    QQT = ctrl.dare(A_final, B_final, QQt, RRt)[0]  

    #######################################
    # 3. Track the optimal trajcetory with LQR controller
    #######################################

    # Initial state
    if perturb:
        epsilon = np.random.uniform(-0.1, 0.1, size=(4,))  # initial state perturbation
    else:
        epsilon = np.zeros(4)

    for rr in range(runs):

        if debug_print:
            print(f"\nRun {rr+1}/{runs}:")

        # Perturbation epsilon slightly increase for each run
        epsilon_scaled = epsilon * perturbation_scale * rr

        x0[:,rr] = xx_star[:,0] + epsilon_scaled # initial state
        
        xx_lqr[:,:,rr], uu_lqr[:,:,rr] = solver.ltv_LQR(At[:,:,:,rr], Bt[:,:,:,rr], QQt, RRt, QQT, x0[:, rr], xx_star, uu_star, model)

    #######################################
    # Results
    #######################################

    if debug_print:
        for rr in range(runs):
            final_error = np.linalg.norm(xx_lqr[:, -1, rr] - xx_star[:, -1])
            print(f"Final tracking error (run {rr+1}): {final_error:.4f}")

    # Final plots
    if plot_results:
        plots.trajectory_comparison(time, xx_star, uu_star, xx_lqr, uu_lqr, title="LQR Trajectory Tracking")
        plots.tracking_error(time, xx_star, uu_star, xx_lqr, uu_lqr, title="LQR Tracking Error")

    # Animation of the optimal trajectory
    if plot_results:
        anim.animate_comparison(model, xx_lqr[:,:,0], xx_star, eq_pair1, eq_pair2, target_fps=30)


if __name__ == "__main__":
    # Generate optimal trajectory
    xx_star, uu_star, eq_pair1, eq_pair2, time = traj_gen_newton.generate_optimal_trajectory('swing_up', max_iters=300, n_mid=20, method='stepwise', discr_method='fe', plot_results=False, plot_armijo=False, debug_print=False)

    # Trajectory tracking via LQR
    trajectory_tracking_LQR(eq_pair1, eq_pair2, xx_star, uu_star, time, runs=5, perturb=True, perturbation_scale=2, plot_results=True, debug_print=True)