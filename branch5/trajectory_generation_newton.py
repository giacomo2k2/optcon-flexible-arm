#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         main_newton_method.py
# Description:  Trajectory generation for a non-linear flexible link via Newton method.
#               Includes DYNAMIC HORIZON EXTENSION and DARE-based Terminal Cost.
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
import os   

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

######################################
# Model
######################################

# Model 
model = DoublePendulumFlexible(**PARAM_SET_2)

ns = model.ns  # Number of states
ni = model.ni  # Number of inputs

######################################
# Helper: Array Extension
######################################

def extend_trajectory(xx_curr, uu_curr, xx_ref, uu_ref, x_eq, u_eq, dt, extension_time):
    """
    Extends trajectory arrays (solution and reference) by a specified time duration.
    Used for Dynamic Horizon Extension.
    """
    n_extra = int(np.ceil(extension_time / dt))
    
    if n_extra <= 0:
        return xx_curr, uu_curr, xx_ref, uu_ref

    # 1. Extend Reference (Pad with final equilibrium)
    xx_ref_ext = np.pad(xx_ref, ((0, 0), (0, n_extra)), mode='edge')
    xx_ref_ext[:, -n_extra:] = x_eq[:, None]

    uu_ref_ext = np.pad(uu_ref, ((0, 0), (0, n_extra)), mode='edge')
    uu_ref_ext[:, -n_extra:] = u_eq[:, None]

    # 2. Extend Current Solution (Warm Start for next iteration)
    uu_pad = np.tile(u_eq[:, None], (1, n_extra))
    uu_ext = np.hstack((uu_curr, uu_pad))

    T_old = xx_curr.shape[1]
    uu_ext[:, T_old - 1] = u_eq

    # For State: Simulate dynamics forward from the last known state
    xx_ext = np.pad(xx_curr, ((0, 0), (0, n_extra)), mode='edge') 
    
    curr_x = xx_curr[:, -1]
    for k in range(n_extra):
        curr_x = model.discrete_dynamics(curr_x, u_eq)
        xx_ext[:, -(n_extra) + k] = curr_x

    return xx_ext, uu_ext, xx_ref_ext, uu_ref_ext


def generate_optimal_trajectory(task: str, max_iters: int = 300, n_mid: int = 20, method: str = 'poly5', plot_flag: bool = False, plot_armijo: bool = False, debug_print: bool = True, folder: str = None) -> tuple[np.ndarray, np.ndarray, tuple, tuple, np.ndarray]:
    """
    Generate optimal trajectory via Newton method with Dynamic Horizon Extension.
    """

    ######################################
    # Log header
    ######################################

    print("\n" + "="*80)
    print(f"[Trajectory Generation] - Newton Method (Dynamic Extension)".center(80))
    print("="*80)

    ######################################
    # Task and Equilibrium points
    ######################################
    
    selected_task = task  
    data_dict = model.compute_equilibrium_manifold()
    selected_equilibrium = traj.tasks[selected_task]
    
    eq_pair1, eq_pair2 = selected_equilibrium['start'], selected_equilibrium['end']
    x_eq1, u_eq1 = eq_pair1
    x_eq2, u_eq2 = eq_pair2

    # Extract task parameters
    Tend = selected_equilibrium['time'] # Base duration
    angle_mult1 = selected_equilibrium['angle1_mult']
    vel_mult1 = selected_equilibrium['vel1_mult']
    angle_mult2 = selected_equilibrium['angle2_mult']
    vel_mult2 = selected_equilibrium['vel2_mult']
    input_mult = selected_equilibrium['input_mult']
    discr_method = selected_equilibrium['discr_method']

    ######################################
    # Algorithm parameters
    ######################################
    
    # Armijo parameters
    stepsize_0 = 1.0
    c = 0.5
    beta = 0.7
    armijo_maxiters = 20

    # Stopping criteria
    descent_threshold = 1e-6
    
    # Dynamic Extension Parameters
    max_time = 20          # Max number of seconds
    extension_duration = 0.5    # Seconds to add per extension
    terminal_error_tol = 1e-2   # Tolerance for ||x_final - x_eq||

    model.discr_method = discr_method

    # Cost matrices
    QQt_base = np.diag([1.0, 1.0, 1.0, 1.0])
    RRt = 1e-3 * np.eye(ni)*input_mult
    
    scale_QQt = np.diag([angle_mult1, angle_mult2, vel_mult1, vel_mult2])
    QQt = QQt_base @ scale_QQt

    # --- TERMINAL COST (DARE) ---
    A_final, B_final = solver.jacobian_analytic(x_eq2, u_eq2, model) 
    
    P_inf, _, _ = ctrl.dare(A_final, B_final, QQt, RRt)
    
    QQT = P_inf

    ######################################
    # [PLOTS] Equilibrium manifold
    ######################################

    plots.equilibrium_manifold(data_dict, eq_pair1, eq_pair2, print_results=False, plot_flag=plot_flag, folder=folder)
    model.draw_pendulums(eq_pair1, eq_pair2, plot_flag=plot_flag, folder=folder)

    ######################################
    # Initial Reference & Guess Generation
    ######################################

    T0 = 0.0
    ti = 0.0
    tf = 1.0
    dt = model.dt

    # Initial time vector
    time = np.arange(T0, Tend, dt)

    
    # Generate initial Reference
    _, xx_ref = traj.trajectory_generator(start=x_eq1, end=x_eq2, T0=T0, Tend=Tend, ti=ti, tf=tf, dt=dt, method=method, input_type='state')
    _, uu_ref = traj.trajectory_generator(start=u_eq1, end=u_eq2, T0=T0, Tend=Tend, ti=ti, tf=tf, dt=dt, method=method, input_type='input')


    # setting 
    xx_ref[2,:] = 0.0
    xx_ref[3,:] = 0.0

    # Generate initial Guess
    xx_init = np.zeros_like(xx_ref)
    uu_init = np.zeros_like(uu_ref)
    
    xx_init[:, :] = x_eq1[:, None]
    

    TT = len(time)

    # Simulate guess
    for tt in range(TT-1):
        # if tt=0 give a little impulse in the direction of the reference
        if tt == 0:
            uu_init[:, tt] = (x_eq2[0] - x_eq1[0])*0.1
            
        xx_init[:, tt+1] = model.discrete_dynamics(xx_init[:, tt], uu_init[:, tt])
    uu_init[:,-1] = uu_init[:,-2]

    plots.trajectory_comparison(time, xx_ref, uu_ref, xx_init, uu_init, title="Initial Guess vs Reference", plot_flag=plot_flag, folder=folder)

    ######################################
    # [OUTER LOOP] Dynamic Extension Strategy
    ######################################
    
    xx_star, uu_star = None, None
    final_kk = 0

    global_JJ = []
    global_descent = []
    phase_iters = []
    phase_history = []
    gradient_error = []
    phase_iters.append(0)  # First phase starts at iteration 0

    max_iters_ext = int(np.floor((max_time - Tend) / extension_duration))


    for jj in range(max_iters_ext):
        
        TT = len(time)
        print(f"\n--- Extension Phase {jj} (Horizon: {time[-1]:.2f}s, Steps: {TT}) ---")

        # ------------------------------------
        # Allocate Memory for CURRENT Phase 
        # ------------------------------------
        xx = np.zeros((ns, TT, max_iters))
        uu = np.zeros((ni, TT, max_iters))
        lmbd = np.zeros((ns, TT, max_iters))
        deltau = np.zeros((ni, TT, max_iters))
        dJ = np.zeros((ni, TT, max_iters))
        
        JJ = np.zeros(max_iters)
        descent = np.zeros(max_iters)
        descent_arm = np.zeros(max_iters)
        
        PP = np.zeros((ns, ns, TT, max_iters))
        pp = np.zeros((ns, TT, max_iters))
        KK = np.zeros((ni, ns, TT, max_iters))
        ss = np.zeros((ni, TT, max_iters))
        AA = np.zeros((ns, ns, TT, max_iters))
        BB = np.zeros((ns, ni, TT, max_iters))
        deltax = np.zeros((ns, TT, max_iters))

        # ------------------------------------
        # Set Initial Guess for this Phase
        # ------------------------------------
        xx[:,:,0] = xx_init
        uu[:,:,0] = uu_init
        x0 = xx_init[:, 0]

        # ------------------------------------
        # [INNER LOOP] Newton Solver
        # ------------------------------------
        converged = False
        
        for kk in range(max_iters-1):
            
            # 0. Cost
            JJ[kk] = cst.total_cost(xx[:,:,kk], uu[:,:,kk], xx_ref, uu_ref, QQt, RRt, QQT)
            
            # 1. Backward Pass
            lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
            lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
            PP[:,:,TT-1,kk] = QQT
            pp[:,TT-1,kk] = lmbd[:,TT-1,kk].squeeze()

            for tt in reversed(range(TT-1)):
                qt, rt, QQt, RRt, SSt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:]
                At, Bt = solver.jacobian_analytic(xx[:, tt, kk], uu[:, tt, kk], model)
                
                AA[:, :, tt, kk] = At
                BB[:, :, tt, kk] = Bt
                
                # Costate
                lmbd_temp = At.T @ lmbd[:,tt+1,kk][:,None] + qt
                lmbd[:,tt,kk] = lmbd_temp.squeeze()

                # LQR Step
                KK[:,:,tt,kk], ss[:,tt,kk], PP[:,:,tt,kk], pp[:,tt,kk] = solver.ltv_affine_LQR_step(
                    At, Bt, QQt, RRt, SSt, PP[:,:,tt+1,kk], pp[:,tt+1,kk], qt, rt
                )
                
                # Gradient
                dJ_temp = rt + Bt.T @ lmbd[:,tt+1,kk][:,None]
                dJ[:,tt,kk] = dJ_temp.squeeze()

            # Descent Norm
            for tt in range(TT-1):
                deltau[:,tt,kk] = ss[:,tt,kk] +  KK[:,:,tt,kk] @  deltax[:, tt, kk]   
                deltax[:, tt+1, kk] = AA[:, :, tt, kk] @ deltax[:, tt, kk] + BB[:, :, tt, kk] @ deltau[:,tt,kk] 
                descent[kk] += deltau[:,tt,kk].T @ deltau[:,tt,kk]  
                descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk]  

            # 2. Stepsize (Armijo)
            stepsize, grad_err = armijo.select_stepsize_newton(model, stepsize_0, armijo_maxiters, c, beta, 
                                                ss[:,:,kk], KK[:,:,:,kk], xx_ref, uu_ref, 
                                                x0, xx[:,:,kk], uu[:,:,kk], JJ[kk], descent_arm[kk], 
                                                QQt, RRt, QQT, kk, plot=plot_armijo, folder=folder)

            # --- SAVE METRICS TO GLOBAL HISTORY ---
            global_JJ.append(JJ[kk])
            global_descent.append(descent[kk]) 
            gradient_error.append(grad_err)

            # 3. Check Convergence
            if abs(descent[kk]) < descent_threshold or stepsize == 0.0:
                if debug_print: print(f"\nPhase converged at iter {kk}.")
                converged = True
                final_kk = kk
 
                phase_iters.append(phase_iters[jj]+final_kk+1)
                break
            
            if np.isnan(descent[kk]):
                print(f"\nNaN detected at iter {kk}. Stopping.")
                final_kk = kk
               
                phase_iters.append(phase_iters[jj]+final_kk+1)
                break

            # 4. Forward Update
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT)) 
            xx_temp[:,0] = x0

            for tt in range(TT-1):
                dx = xx_temp[:,tt] - xx[:,tt,kk]
                uu_temp[:,tt] = uu[:,tt,kk] + stepsize * ss[:,tt,kk] + KK[:,:,tt,kk] @ dx
                xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])
            
            uu_temp[:,-1] = uu_temp[:,-2]
            xx[:,:,kk+1] = xx_temp.copy()
            uu[:,:,kk+1] = uu_temp.copy()

            if debug_print:
                print(f"\rIter {kk} | Cost: {JJ[kk]:.4e} | Desc: {descent[kk]:.4e}".ljust(80), end="", flush=True)

        # ------------------------------------
        # End of Inner Loop
        # ------------------------------------
        
        # Extract best solution from this phase
        idx_best = final_kk if converged else max_iters-1
        xx_phase_opt = xx[:, :, idx_best]
        uu_phase_opt = uu[:, :, idx_best]

        phase_history.append({
            'time': time.copy(),
            'xx': xx[:, :, :idx_best+1].copy(),
            'uu': uu[:, :, :idx_best+1].copy(),
            'final_idx': idx_best,
            'phase_idx': jj
        })
        
        # Check Terminal Error
        term_error = np.linalg.norm(xx_phase_opt[:, -1] - x_eq2)
        print(f"\nPhase {jj} finished. Terminal Error: {term_error:.5f}")

        # Decide whether to stop or extend
        if term_error < terminal_error_tol:
            print(f"Terminal tolerance met (< {terminal_error_tol}). Stopping extensions.")
            xx_star = xx_phase_opt
            uu_star = uu_phase_opt
            break
        elif jj == max_iters_ext - 1:
            print("Max extensions reached. Returning current best.")
            xx_star = xx_phase_opt
            uu_star = uu_phase_opt
            break
        else:
            print(f"Error too high. Extending horizon by {extension_duration}s...")
            
            # --- EXTENSION LOGIC ---
            # 1. Extend Arrays
            xx_init, uu_init, xx_ref, uu_ref = extend_trajectory(
                xx_phase_opt, uu_phase_opt, xx_ref, uu_ref, x_eq2, u_eq2, dt, extension_duration
            )
            
            # 2. Extend Time Vector
            time_extra = np.arange(time[-1] + dt, time[-1] + extension_duration + dt/2, dt)
            time = np.concatenate((time, time_extra))

    ######################################
    # [RESULTS]
    ######################################

    global_descent_arr = np.array(global_descent)
    global_JJ_arr = np.array(global_JJ)
    gradient_error = np.array(gradient_error)
    total_iterations = len(global_JJ_arr)

    phase_iters = phase_iters[:-1]  # Remove initial 0 for plotting purposes

    print(f"\nTotal Iterations across all phases: {total_iterations}")

    plots.descent_direction(global_descent_arr, total_iterations, phase_iters=phase_iters, plot_flag=plot_flag, folder=folder)
    plots.cost(global_JJ_arr, total_iterations, phase_iters=phase_iters, plot_flag=plot_flag, folder=folder)
    
    plots.trajectory_comparison(time, xx_ref, uu_ref, xx_star, uu_star, title="Optimal vs Reference", plot_flag=plot_flag, folder=folder)
    plots.trajectory_comparison_mid(phase_history, n_mid=5, title="Evolution of Trajectories over Extension Phases", plot_flag=plot_flag, folder=folder)
    
    plots.gradient_error(gradient_error, total_iterations, plot_flag=plot_flag, folder=folder)

    anim.animate_comparison(model, xx_star, xx_ref, (x_eq1, u_eq1), (x_eq2, u_eq2), dt=dt, folder=folder)


    plt.close('all')

    return xx_star, uu_star, eq_pair1, eq_pair2, time

######################################
# MAIN
######################################

def main():

    ######################################
    # Algorithm parameters
    ######################################

    max_iters = 300                      
    n_mid = 20                           
    method = 'poly5'                     
            

    ######################################
    # Algorithm flags
    ######################################

    save_flag = True                       
    plot_flag = True                        
    plot_flag_armijo = False                
    debug_flag = True                       

    ######################################
    # Environment setup
    ######################################

    # for task in traj.tasks:
        
    task = 'half_middle_dive'

    print(f"\nSelected Task: {task}")
    
 

    try:
        if task not in traj.tasks:
            raise ValueError(f"Task '{task}' not found.")
    except ValueError as e:
        print(f"\n{e}\n")
        exit(1)
    
    if save_flag:
        folder = f'report/figs/traj_gen_newton/{task}'
        if not os.path.exists(folder):
            os.makedirs(folder)
    else:
        folder = None
        
    ######################################
    # Run
    ######################################

    

    generate_optimal_trajectory(
        task=task, max_iters=max_iters, n_mid=n_mid, method=method, 
        plot_flag=plot_flag, plot_armijo=plot_flag_armijo, 
        debug_print=debug_flag, folder=folder
    )

if __name__ == "__main__":
    main()