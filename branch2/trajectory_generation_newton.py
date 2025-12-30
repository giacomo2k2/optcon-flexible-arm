#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         main_newton_method.py
# Description:  Trajectory generation for a non-linear flexible link via Newton method.
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

# Costs
import costs as cst

# Armijo
import armijo as armijo

# Trajectories
import trajectory as traj

# Solver
import solver as solver

# Animation & plots
import animation as anim
import plots as plots

# Control
import control as ctrl

import equilibrium as eq

#######################################
# Model
#######################################

model = DoublePendulumFlexible(**PARAM_SET_2)

ns = model.ns  # number of states
ni = model.ni  # number of inputs

#######################################
# Algorithm parameters
#######################################

# General parameters
max_iters = 300
n_mid = 20                  # number of intermediate trajectories to plot

# Armijo parameters
stepsize_0 = 1.0
c = 0.5
beta = 0.7
armijo_maxiters = 20

# Visualization parameters
initial_reference_plot = False  # plot reference trajectory at beginning
equilibrium_plot = True        # plot equilibrium manifold and points
armijo_plot = False             # plot armijo descent direction
iteration_traj_plot = False     # plot trajectory at each iteration
animation = False          # animate final optimal trajectory
debug_print = True             # print debug information
final_plots = True             # plot final optimal trajectory

#######################################
# Trajectory parameters
#######################################

T0 = 0.0        # Initial absolute time
ti = 0.3        # Start time for trajectory segment (relative)
tf = 0.7        # End time for trajectory segment (relative)
Tend = 4.0      # End absolute time
dt = model.dt   # Time step
t = np.arange(T0, Tend, dt) # Time vector
TT = len(t)                 # Number of time steps


# Trajectory generation technique
method = 'poly5'  # 'poly3', 'poly5', 'poly7', 'stepwise', 'cubic_spline'
model.discr_method = 'rk4'



#######################################
# Equilibrium points
#######################################

selected_equilibrium = 'elbow_up'  # choose from keys of equilibrium_traj

data_dict = model.compute_equilibrium_manifold()

trajectory_selected = traj.configurations[selected_equilibrium]
(x_eq1, u_eq1), (x_eq2, u_eq2) = trajectory_selected['start'], trajectory_selected['end']

if equilibrium_plot:
    plots.equilibrium_manifold(data_dict, print_results=False)
    model.plot_equilibrium_point(x_eq1, u_eq1, title="1")
    model.plot_equilibrium_point(x_eq2, u_eq2, title="2")

# Select setpoints as equilibria
x_setpoints = np.array([x_eq1, x_eq2])
u_setpoints = np.array([u_eq1, u_eq2]).reshape(-1,1)

######################################
# Reference curve
######################################

# Generate reference trajectory for states and inputs
time, xx_ref = traj.trajectory_generator(T0, Tend, ti=ti, tf=tf, dt=dt, setpoints=x_setpoints, method=method)
time, uu_ref = traj.trajectory_generator(T0, Tend, ti=ti, tf=tf, dt=dt, setpoints=u_setpoints, method=method)

# Plot reference trajectory
if initial_reference_plot:
    plots.show_traj(time, xx_ref, uu_ref, eq_pair1, eq_pair2, "Reference " + method)

#############################################
# Compute QQT as solution of the DARE
##############################################

# Get cost matrices
QQt, RRt = cst.get_cost_matrices(selected_equilibrium)

# Compute QQT as solution of dare
AA_ref, BB_ref = solver.jacobian_analytic(xx_ref[:,TT-1], uu_ref[:,TT-1], model)
QQT = ctrl.dare(AA_ref, BB_ref, QQt, RRt)[0]


######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))          # State seq.
uu = np.zeros((ni, TT, max_iters))          # Input seq.

lmbd = np.zeros((ns, TT, max_iters))        # Lambdas - costate seq.

deltau = np.zeros((ni,TT, max_iters))       # Du - descent direction
dJ = np.zeros((ni,TT, max_iters))           # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)                    # Collect cost
descent = np.zeros(max_iters)               # Collect descent direction norm
descent_arm = np.zeros(max_iters)           # Collect descent condition value
gradient_error = np.zeros((max_iters,))

PP = np.zeros((ns, ns, TT, max_iters))      # P matrices for affine LQR
pp = np.zeros((ns, TT, max_iters))          # p vectors for affine LQR
KK = np.zeros((ni, ns, TT, max_iters))      # K gain matrices for affine LQR
ss = np.zeros((ni, TT, max_iters))          # s vectors for affine LQR

AA = np.zeros((ns, ns, TT, max_iters))
BB = np.zeros((ns, ni, TT, max_iters))
deltax = np.zeros((ns, TT, max_iters))

xx_total = np.zeros((ns, TT, max_iters))    # all state trajectories
uu_total = np.zeros((ni, TT, max_iters))    # all input trajectories


######################################
# Initial Trajectory Guess
######################################

x0 = xx_ref[:, 0]  # initial state

# uu_init = np.zeros((ni, TT))
# xx_init = np.zeros((ns, TT))

uu_init = uu_ref.copy()
xx_init = np.zeros((ns, TT))
# Rollout to make the state trajectory dynamically feasible consistent with x0
xx_init[:, 0] = x0
for t in range(TT - 1):
    xx_init[:, t+1] = model.discrete_dynamics(xx_init[:, t], uu_init[:, t])


if initial_reference_plot:
    plots.trajectory_comparison(time, xx_ref, uu_ref, xx_init, uu_init, title="Initial Guess vs Reference")


xx[:,:,0] = xx_init
uu[:,:,0] = uu_init


############################
# Iteration loop
############################

# Iteration counter
kk = 0

for kk in range(max_iters-1):
    
    ############################
    # 0. Compute cost
    ############################
    
    JJ[kk] = cst.total_cost(xx[:,:,kk], uu[:,:,kk], xx_ref, uu_ref, QQt, RRt, QQT)
    
    ############################
    # 1. Compute descent direction
    ############################

    lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
    lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
    PP[:,:,TT-1,kk] = QQT
    pp[:,TT-1,kk] = lmbd[:,TT-1,kk].squeeze()

    for tt in reversed(range(TT-1)):
        qt, rt, QQt, RRt, SSt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:]
        At, Bt = solver.jacobian_analytic(xx[:, tt, kk], uu[:, tt, kk], model)
            
        AA[:, :, tt, kk] = At
        BB[:, :, tt, kk] = Bt

        # Regularization for 1st order approx.
        Qtk = QQt
        Rtk = RRt + np.eye(ni)*1e-6  # regularization
        Stk = SSt 

        # Costate equation
        lmbd_temp = At.T @ lmbd[:,tt+1,kk][:,None] + qt
        lmbd[:,tt,kk] = lmbd_temp.squeeze()


        # Single step LQR
        KK[:,:,tt,kk], ss[:,tt,kk], PP[:,:,tt,kk], pp[:,tt,kk] = solver.ltv_affine_LQR_step(
            At, Bt, Qtk, Rtk, Stk, PP[:,:,tt+1,kk], pp[:,tt+1,kk], qt, rt
        )

        # Gradient of cost w.r.t. input from LQR quantities
        dJ_temp = rt + Bt.T @ lmbd[:,tt+1,kk][:,None]
        dJ[:,tt,kk] = dJ_temp.squeeze()

    
    
    # Compute descent direction norm after backward pass
    for tt in range(TT-1):
        deltau[:,tt,kk] = ss[:,tt,kk] +  KK[:,:,tt,kk] @  deltax[:, tt, kk]   
        deltax[:, tt+1, kk] = AA[:, :, tt, kk] @ deltax[:, tt, kk] + BB[:, :, tt, kk] @ deltau[:,tt,kk] 
        
        descent[kk] += deltau[:,tt,kk].T @ deltau[:,tt,kk]  
        descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk]  

    
    ############################
    # 2. Stepsize selection - ARMIJO
    ############################
    
    stepsize, grad_error_temp = armijo.select_stepsize_newton(model, stepsize_0, armijo_maxiters, c, beta, 
                                        ss[:,:,kk], KK[:,:,:,kk], xx_ref, uu_ref, 
                                        x0, xx[:,:,kk], uu[:,:,kk], JJ[kk], descent_arm[kk], 
                                        QQt, RRt, QQT, kk, plot=armijo_plot)
    
    gradient_error[kk] = grad_error_temp


    ############################
    # 3. Termination condition
    ############################

    if stepsize == 0.0:
        print(f"\n\nConvergence reached at iteration {kk}!")
        max_iters = kk
        break

    if np.isnan(descent[kk]) or np.isnan(descent_arm[kk]):
        print(f"\n\nNaN detected at iteration {kk}, time {tt}! Stopping computation.")
        max_iters = kk
        break
            
    ############################
    # 4. Forward closed-loop integration
    ############################
    
    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT)) 
    xx_temp[:,0] = x0

    
    for tt in range(TT-1):
        dx = xx_temp[:,tt] - xx[:,tt,kk]
        uu_temp[:,tt] = uu[:,tt,kk] + stepsize * ss[:,tt,kk] + KK[:,:,tt,kk] @ dx
        xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])
        
        
    uu_temp[:,-1] = uu_temp[:,-2]

    # Store trajectories
    xx_total[:,:,kk+1] = xx_temp.copy()
    uu_total[:,:,kk+1] = uu_temp.copy()

    xx[:,:,kk+1] = xx_temp.copy()
    uu[:,:,kk+1] = uu_temp.copy()

    
    ############################
    # 5. Visualization
    ############################


    if iteration_traj_plot and kk % 50 == 0:
        plots.trajectory_comparison(time, xx_ref, uu_ref, xx[:,:,kk+1], uu[:,:,kk+1], 
                                        title=f"Iteration {kk}")
    
    if debug_print:
        log_str = (f"Iter: {kk} | "
               f"Cost: {JJ[kk]:.6e} | "
               f"Stepsize: {stepsize:.3e} | "
               f"Descent norm: {descent[kk]:.6e} | "
               f"Descent cond: {descent_arm[kk]:.6e}")
        print(f"\r{log_str}".ljust(120), end="", flush=True)
        
    else:
        dots = "." * (kk % 4)
        print(f"\rComputing trajectory{dots:3}", end="", flush=True)

    

##################################
# Results
##################################
xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] 
descent[max_iters-1] = descent[ max_iters-2]


plt.figure()
plt.plot(abs(gradient_error[:max_iters-1]))
plt.show()

if final_plots:
    # Descent Direction
    plots.descent_direction(descent, max_iters)

    # Cost
    plots.cost(JJ, max_iters)

    # Optimal vs Reference Trajectories
    plots.trajectory_comparison(time, xx_ref, uu_ref, xx_star, uu_star, title="Optimal vs Reference")

    # Intermediate Trajectories
    plots.trajectory_comparison_mid(time, xx, uu, n_mid, max_iters, title="Trajectory Comparison with Intermediate Trajectories")

# Animation of the optimal trajectory
if animation:
    anim.animate_comparison(model, xx_star, xx_ref, eq_pair1, eq_pair2, target_fps=30)

##################################
# Export functions
##################################

def get_optimal_trajectory():
    """
    Returns the optimal trajectory computed by the Newton method.

    Returns
    -------
    time : np.ndarray
    Time vector (T,).   
    xx_star : np.ndarray
        Optimal state trajectory (ns, T).
    uu_star : np.ndarray
        Optimal input trajectory (ni, T).
    """
    return time, xx_star, uu_star

def get_equilibrium_pairs():
    """
    Returns the equilibrium pairs used for trajectory generation.

    Returns
    -------
    eq_pair1 : tuple
        First equilibrium pair (x_eq1, u_eq1).
    eq_pair2 : tuple
        Second equilibrium pair (x_eq2, u_eq2).
    """
    return eq_pair1, eq_pair2

