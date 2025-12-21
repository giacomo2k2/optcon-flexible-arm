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
from b1_costs import QQt, RRt

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

# GENERAL PARAMETERS
max_iters = 300
fixed_stepsize = 1e-3
term_cond = 1e-4

# ARMIJO PARAMETERS
Armijo = True
stepsize_0 = 1
c = 0.5
beta = 0.7
armijo_maxiters = 20

# VISUALIZATION PARAMETERS
iteration_traj_plot = True     # plot trajectory at each iteration
armijo_plot = False             # plot armijo descent direction
visu_animation = False          # animate final optimal trajectory
equilibrium_plot = False        # plot equilibrium manifold and points
initial_reference_plot = True    # plot reference trajectory at beginning

#######################################
# Trajectory parameters
#######################################

T0 = 0.0        # Initial absolute time
ti = 0.3        # Start time for trajectory segment (relative)
tf = 0.7        # End time for trajectory segment (relative)
Tend = 15.0     # End absolute time
dt = model.dt   # Time step

t = np.arange(T0, Tend, dt)
TT = len(t)

method = 'stepwise'  # 'poly3', 'poly5', 'poly7', 'stepwise', 'cubic_spline'

#######################################
# Equilibrium points
#######################################
data_dict = model.compute_equilibrium_manifold()
x_eq1, u_eq1 = model.find_equilibrium(fix_type='q2', value=np.pi/2)
x_eq2, u_eq2 = model.find_equilibrium(fix_type='q2', value=-3*np.pi/4)

if equilibrium_plot:
    plots.plot_equilibrium_manifold(data_dict)
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
    plots.plot_reference_traj(time, xx_ref, uu_ref, method)

#############################################
# Compute QQT as solution of the DARE
##############################################
# Symbolic jacobian
AA_sym, BB_sym = model_sym.symbolic_jacobian() # TODO - Q1 - Qua tiriamo fuori gli jacobiani (A, B). Che devo pluggarli dentro la DARE, non devo fare il trasposto vero?

AA_ref, BB_ref = solver.jacobian_solver(AA_sym, BB_sym, xx_ref[:,TT-1], uu_ref[:,TT-1])

# TODO: maybe unceck this
# AA_ref = AA_ref.T
# BB_ref = BB_ref.T

QQT = ctrl.dare(AA_ref, BB_ref, QQt, RRt)[0]
#QQT = QQt

######################################
# Initial Trajectories Guesses
######################################

xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))

x0 = xx_init[:, 0]  # initial state

if initial_reference_plot:
    plots.plot_trajectory_comparison(time, xx_ref, uu_ref, xx_init, uu_init, title="Initial Guess vs Reference")

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))       # State seq.
uu = np.zeros((ni, TT, max_iters))       # Input seq.

lmbd = np.zeros((ns, TT, max_iters))     # Lambdas - costate seq.

deltau = np.zeros((ni,TT, max_iters))    # Du - descent direction
dJ = np.zeros((ni,TT, max_iters))        # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)                        # Collect cost
descent = np.zeros(max_iters)                   # Collect descent direction
descent_arm = np.zeros(max_iters)               # Collect descent direction

PP = np.zeros((ns, ns, TT, max_iters))  
pp = np.zeros((ns, TT, max_iters))
KK = np.zeros((ni, ns, TT, max_iters))
ss = np.zeros((ni, TT, max_iters))

######################################
# Main
######################################

print("======================= Gradient Method =======================")

kk = 0

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

##################################
# Iteration loop
##################################
for kk in range(max_iters-1):
    JJ[kk] = 0
    
    # =================== Step 0: COMPUTE COST ===================
    for tt in range(TT-1):
        stage_cost = cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ[kk] += stage_cost
    term_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1], QQT)[0]
    JJ[kk] += term_cost

    print(f"Cost at iteration {kk}: {JJ[kk]:.6e}")

    # =================== Step 1: Backward pass ===================
    PP[:, :, TT-1, kk] = QQT
    pp[:, TT-1, kk] = cst.termcost(xx[:, -1, kk], xx_ref[:, -1], QQT)[1]

    lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
    lmbd[:,TT-1,kk] = lmbd_temp.copy()
    for tt in reversed(range(TT - 1)):
        At, Bt = solver.jacobian_solver(AA_sym, BB_sym, xx[:, tt, kk], uu[:, tt, kk])
        qt, rt = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1:]
        
        # Hessians
        Qt, Rt, St = cst.QQt, cst.RRt, np.zeros((ni, ns))

        # Riccati equations
        # Inverse term (R + B^T P B)
        mat_to_inv = Rt + Bt.T @ PP[:, :, tt+1, kk] @ Bt
        inv_mat = np.linalg.inv(mat_to_inv)

        # Feedback and Feedforward gains
        KK[:, :, tt, kk] = -inv_mat @ (St + Bt.T @ PP[:, :, tt+1, kk] @ At)
        ss[:, tt, kk] = -inv_mat @ (rt.flatten() + Bt.T @ pp[:, tt+1, kk])

        # Descent direction components
        lmbd_temp = At.T@lmbd[:,tt+1,kk][:,None] + qt   # costate equation
        grad_u = rt.flatten() + Bt.T @ pp[:, tt+1, kk]     # gradient of J wrt u
        deltau_temp = - grad_u                         # descent direction 

        lmbd[:,tt,kk] = lmbd_temp.squeeze()
        dJ[:,tt,kk] = grad_u.squeeze()
        deltau[:,tt,kk] = deltau_temp.squeeze()

        descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk] # TODO: not sure about this two descent calculations
        descent_arm[kk] += dJ[:,tt,kk] @ ss[:, tt, kk]

        # Update P and p
        PP[:, :, tt, kk] = Qt + At.T @ PP[:, :, tt+1, kk] @ At - KK[:, :, tt, kk].T @ mat_to_inv @ KK[:, :, tt, kk]
        pp[:, tt, kk] = (qt.flatten() + At.T @ pp[:, tt+1, kk] - 
                         KK[:, :, tt, kk].T @ mat_to_inv @ ss[:, tt, kk])

    # =================== Step 2: Armijo ===================
    if Armijo:
        # Usiamo una versione specifica di Armijo per Newton che gestisce il feedback K
        stepsize, xx_temp, uu_temp = armijo.select_stepsize_newton(stepsize_0, armijo_maxiters, c, beta,
                                      ss[:,:,kk], KK[:,:,:,kk], xx_ref, uu_ref, x0, 
                            xx[:,:,kk], uu[:,:,kk], JJ[kk], descent_arm[kk], QQT, armijo_plot)
    else:
        stepsize = fixed_stepsize

    # =================== Step 3: Forward pass ===================
    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))

    for tt in range(TT - 1):
        delta_x = xx_temp[:, tt] - xx[:, tt, kk]
        uu_temp[:, tt] = uu[:, tt, kk] + KK[:, :, tt, kk] @ delta_x + stepsize * ss[:, tt, kk]
        xx_temp[:, tt+1] = model.discrete_dynamics(xx_temp[:, tt], uu_temp[:, tt])

    uu_temp[:, -1] = uu_temp[:, -2]

    xx[:, :, kk+1] = xx_temp
    uu[:, :, kk+1] = uu_temp

    ############################
    # Plot at each iteration
    ############################
    if iteration_traj_plot and kk % 50 == 0:
        plots.plot_trajectory_comparison(time, xx_ref, uu_ref, xx[:,:,kk+1], uu[:,:,kk+1], title=f"Iteration {kk}")

    ############################
    # Termination condition
    ############################
    descent[kk] = np.linalg.norm(ss[:, :, kk])
    if descent[kk] <= term_cond:
        max_iters = kk
        break

##################################
# Results
##################################

xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

# DESCENT DIRECTION PLOT
plt.figure('descent direction', figsize=(10, 6))
plt.plot(np.arange(max_iters), descent[:max_iters], linewidth=2, color='#1f77b4')
plt.xlabel('Iteration $k$', fontsize=14)
plt.ylabel(r'$\| \Delta u \| ^2$', fontsize=14)
plt.title('Descent', fontsize=16)
plt.yscale('log')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show(block=False)

# COST PLOT
plt.figure('cost', figsize=(10, 6))
plt.plot(np.arange(max_iters-1), JJ[:max_iters-1], linewidth=2, color='#ff7f0e')
plt.xlabel('Iteration $k$', fontsize=14)
plt.ylabel(r'$J(\mathbf{u}^k)$', fontsize=14)
plt.title('Cost Function', fontsize=16)
plt.yscale('log')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show(block=False)

# REFERENCE VS OPTIMAL TRAJECTORY PLOT
plots.plot_trajectory_comparison(time, xx_ref, uu_ref, xx_star, uu_star, title="Optimal vs Reference")