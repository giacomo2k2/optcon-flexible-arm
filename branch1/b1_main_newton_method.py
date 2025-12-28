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
from b1_model import DoublePendulumFlexible, PARAM_SET_2
from b1_model_symbolic import DoublePendulumFlexibleSymbolic

# Costs
import b1_costs as cst

# Armijo
import b1_armijo as armijo

# Trajectories
import b1_trajectories as traj

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

# General parameters
max_iters = 300
fixed_stepsize = 1*1e-2
term_cond = 1e-5

# Armijo parameters
Armijo = True
stepsize_0 = 1.0
c = 0.5
beta = 0.7
armijo_maxiters = 10
approx_1st_order = True

# Visualization parameters
initial_reference_plot = False    # plot reference trajectory at beginning
equilibrium_plot = False        # plot equilibrium manifold and points
armijo_plot = False             # plot armijo descent direction
iteration_traj_plot = False     # plot trajectory at each iteration
visu_animation = False          # animate final optimal trajectory
debug_print = False             # print debug information
final_plots = False             # plot final optimal trajectory

#######################################
# Trajectory parameters
#######################################

# General setting: T0, ti, tf, Tend, dt = (0.0, 0.2, 0.5, 4.0, 0.01)

T0 = 0.0        # Initial absolute time
ti = 0.3        # Start time for trajectory segment (relative)
tf = 0.5        # End time for trajectory segment (relative)
Tend = 4.0      # End absolute time
dt = model.dt   # Time step
t = np.arange(T0, Tend, dt) # Time vector
TT = len(t)                 # Number of time steps

# Trajectory generation technique
method = 'poly7'  # 'poly3', 'poly5', 'poly7', 'stepwise', 'cubic_spline'
model.discr_method = 'fe'

# Equilibrium go-to configurations
equilibrium_traj = {
    'elbow_up': (
        model.find_equilibrium(fix_type='q1', value=0),
        model.find_equilibrium(fix_type='q2', value=np.pi/2)
    ),

    'swing_up': (
        model.find_equilibrium(fix_type='q1', value=0),
        model.find_equilibrium(fix_type='q1', value=np.pi - 0.5)
    ),

    'left_to_right_elbow_up': (
        tuple(-val for val in model.find_equilibrium(fix_type='q2', value=np.pi/2)),
        model.find_equilibrium(fix_type='q2', value=np.pi/2)
    ),

    'top_to_elbow_up': (
        model.find_equilibrium(fix_type='q1', value=np.pi - 0.5),
        model.find_equilibrium(fix_type='q2', value=np.pi/2)
    ),

    'top_to_elbow_down': (
        model.find_equilibrium(fix_type='q1', value=np.pi - 0.5),
        tuple(-val for val in model.find_equilibrium(fix_type='q1', value=np.pi/2))
    ),

    'custom': (
        model.find_equilibrium(fix_type='q1', value=0.0),
        model.find_equilibrium(fix_type='q1', value=np.pi/4)
    )
}

#######################################
# Equilibrium points
#######################################

data_dict = model.compute_equilibrium_manifold()
# TODO: top_to_elbow_up da problemi con rk4, esplosione sin da subito
# Select equilibrium trajectory (potentially ti=0.3, tf=0.5)
# - elbow_up (it may finish at tf = 0.4)
# - swing_up
# - left_to_right_elbow_up
# - top_to_elbow_up (con fe need tf = 0.5, ti = 0.3)
# - top_to_elbow_down
selected_equilibrium = 'left_to_right_elbow_up'
eq_pair1, eq_pair2 = equilibrium_traj[selected_equilibrium]
(x_eq1, u_eq1), (x_eq2, u_eq2) = eq_pair1, eq_pair2

if equilibrium_plot:
    plots.equilibrium_manifold(data_dict)
    model.plot_equilibrium_point(x_eq1, u_eq1, title="1")
    model.plot_equilibrium_point(x_eq2, u_eq2, title="2")

# Select setpoints as equilibria
x_setpoints = np.array([
                        x_eq1,
                        x_eq2
                    ])
u_setpoints = np.array([
                        u_eq1,
                        u_eq2
                    ]).reshape(-1,1)

######################################
# Reference curve
######################################

time, xx_ref = traj.trajectory_generator(T0, Tend, ti=ti, tf=tf, dt=dt, setpoints=x_setpoints, method=method)
time, uu_ref = traj.trajectory_generator(T0, Tend, ti=ti, tf=tf, dt=dt, setpoints=u_setpoints, method=method)

if initial_reference_plot:
    plots.show_traj(time, xx_ref, uu_ref, eq_pair1, eq_pair2, "Reference " + method)

#############################################
# Compute QQT as solution of the DARE
##############################################

# Get cost matrices
QQt, RRt = cst.get_cost_matrices(selected_equilibrium)

# Symbolic jacobian
AA_sym, BB_sym = model_sym.symbolic_jacobian() 

# Compute QQT as solution of dare
AA_ref, BB_ref = solver.jacobian_solver(AA_sym, BB_sym, xx_ref[:,TT-1], uu_ref[:,TT-1], model)
QQT = ctrl.dare(AA_ref, BB_ref, QQt, RRt)[0]

# Symbolic hessian lambda product
if not approx_1st_order:
    Hxx_sym, Huu_sym, Hux_sym = model_sym.symbolic_hessians_lambda_product()

######################################
# Initial Trajectories Guesses
######################################

xx_init = np.zeros((ns, TT))  # initial state trajectory guess
uu_init = np.zeros((ni, TT))  # initial input trajectory guess

x0 = xx_ref[:, 0]  # initial state

if initial_reference_plot:
    plots.trajectory_comparison(time, xx_ref, uu_ref, xx_init, uu_init, title="Initial Guess vs Reference")

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))              # State seq.
uu = np.zeros((ni, TT, max_iters))              # Input seq.

lmbd = np.zeros((ns, TT, max_iters))            # Lambdas - costate seq.

deltau = np.zeros((ni,TT, max_iters))           # Du - descent direction
dJ = np.zeros((ni,TT, max_iters))               # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)                        # Collect cost
descent = np.zeros(max_iters)                   # Collect descent direction norm
descent_arm = np.zeros(max_iters)               # Collect descent condition value

PP = np.zeros((ns, ns, TT, max_iters))          # P matrices for affine LQR
pp = np.zeros((ns, TT, max_iters))              # p vectors for affine LQR
KK = np.zeros((ni, ns, TT, max_iters))          # K gain matrices for affine LQR
ss = np.zeros((ni, TT, max_iters))              # s vectors for affine LQR

######################################
# Trajectory Generation
######################################

print("=================== Trajectory generation - Newton method ===================")

# Iteration counter
kk = 0

# Initialize trajectories
xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

############################
# Iteration loop
############################
for kk in range(max_iters-1):
    
    ############################
    # 0. Compute cost
    ############################
    for tt in range(TT-1):
        stage_cost = cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
        JJ[kk] += stage_cost
    term_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1], QQT)[0]
    JJ[kk] += term_cost

    #JJ[kk] = cst.total_cost(xx[:,:,kk], uu[:,:,kk], xx_ref, uu_ref, QQt, RRt, QQT)

    ############################
    # 1. Compute descent direction
    ############################
    descent_norm_sq = 0.0
    descent_arm_val = 0.0
    lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
    lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
    PP[:,:,TT-1,kk] = QQT
    pp[:,TT-1,kk] = lmbd[:,TT-1,kk].squeeze()
    for tt in reversed(range(TT-1)):
        qt, rt, QQt, RRt, SSt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:]
        At, Bt = solver.jacobian_solver(AA_sym, BB_sym, xx[:, tt, kk], uu[:, tt, kk], model)

        # Compute modified cost matrices if 2nd order approximation
        if not approx_1st_order and kk > 50:
            Hxx_num, Huu_num, Hxu_num = solver.hessian_numerical_solver(
                Hxx_sym, Huu_sym, Hux_sym, xx[:,tt,kk], uu[:,tt,kk], lmbd[:,tt+1,kk], model, AA_sym, BB_sym
            )
            Qtk = QQt + Hxx_num
            Rtk = RRt + Huu_num + np.eye(ni)*10  # regularization
            Stk = SSt + Hxu_num.T
        else:
            Qtk = QQt
            Rtk = RRt + np.eye(ni)*1e-2  # regularization
            Stk = SSt 

        # Costate equation
        lmbd_temp = At.T @ lmbd[:,tt+1,kk][:,None] + qt
        lmbd[:,tt,kk] = lmbd_temp.squeeze()

        # Single step LQR
        KK[:,:,tt,kk], ss[:,tt,kk], PP[:,:,tt,kk], pp[:,tt,kk] = solver.ltv_affine_LQR_step(
            At, Bt, Qtk, Rtk, Stk, PP[:,:,tt+1,kk], pp[:,tt+1,kk], qt, rt
        )

        # Gradient of cost w.r.t. input from LQR quantities
        dJ_temp = rt + Bt.T @ pp[:,tt+1,kk][:,None]
        dJ[:,tt,kk] = dJ_temp.squeeze()

    # TODO: QUESTION - descent norm often increase rather than decrease, why?
    # Compute descent direction norm after backward pass
    for tt in range(TT-1):
        deltau[:,tt,kk] = ss[:,tt,kk]                       # Descent direction in Newton method
        descent_norm_sq += np.linalg.norm(ss[:,tt,kk])**2   # Norm squared of descent direction
        descent_arm_val += np.dot(dJ[:,tt,kk], ss[:,tt,kk]) # Descent condition value
    
    descent[kk] = np.sqrt(descent_norm_sq)                  # Store norm
    descent_arm[kk] = float(descent_arm_val)                # Store descent condition value

    ############################
    # 2. Stepsize selection - ARMIJO
    ############################
    
    if Armijo:
        stepsize = armijo.select_stepsize_newton(model, stepsize_0, armijo_maxiters, c, beta, 
                                         ss[:,:,kk], KK[:,:,:,kk], xx_ref, uu_ref, 
                                         x0, xx[:,:,kk], uu[:,:,kk], JJ[kk], descent_arm[kk], 
                                         QQt, RRt, QQT, kk, plot=armijo_plot)
        
    else:
        stepsize = fixed_stepsize
        if debug_print:
            print(f"Fixed stepsize: {stepsize:.3e}")

    ############################
    # 3. Forward closed-loop integration
    ############################
    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))

    xx_temp[:,0] = x0

    for tt in range(TT-1):
        delta_x = xx_temp[:,tt] - xx[:,tt,kk]
        uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt,kk] @ delta_x + stepsize * ss[:,tt,kk]
        xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])

    uu_temp[:,-1] = uu_temp[:,-2]

    xx[:,:,kk+1] = xx_temp.copy()
    uu[:,:,kk+1] = uu_temp.copy()

    ############################
    # 4. Visualization
    ############################
    if iteration_traj_plot and kk % 50 == 0:
        plots.trajectory_comparison(time, xx_ref, uu_ref, xx[:,:,kk+1], uu[:,:,kk+1], 
                                        title=f"Iteration {kk}")
    
    if debug_print:
        print(f"Iter: {kk} \tCost: {JJ[kk]:.6e} \tStepsize: {stepsize:.3e} \tDescent norm: {descent[kk]:.6e} \tDescent condition value: {descent_arm[kk]:.6e}")
    else:
        dots = "." * (kk % 4)
        print(f"\rComputing trajectory{dots:3}", end="", flush=True)

    ############################
    # 5. Termination condition
    ############################
    if descent[kk] <= term_cond:
        print(f"\nConvergence reached at iteration {kk}!")
        max_iters = kk
        break

##################################
# Results
##################################
xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

if final_plots:
    # Descent Direction
    plots.descent_direction(descent, max_iters)

    # Cost
    plots.cost(JJ, max_iters)

    # Optimal vs Reference Trajectories
    plots.trajectory_comparison(time, xx_ref, uu_ref, xx_star, uu_star, title="Optimal vs Reference")

    # TODO: comparison con qualche curva intermedia
    

# Animation of the optimal trajectory
if visu_animation:
    anim.animate_comparison(model, xx_star, xx_ref, eq_pair1, eq_pair2, target_fps=30)


def get_optimal_trajectory():
    return time, xx_star, uu_star

def get_equilibrium_pairs():
    return eq_pair1, eq_pair2

