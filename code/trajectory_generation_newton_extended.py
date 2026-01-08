#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         trajectory_generation_newton_extended.py
# Description:  Module for generating optimal trajectories using a Newton-based method with
#               dynamic horizon extension strategy.      
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         January 2025
#########################################################################################

#######################################
# Imports
#######################################

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt 
import signal
import control as ctrl

plt.rcParams['font.family'] = 'serif'
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Custom libraries
import armijo as armijo
import solver as solver
import animation as anim
import plots as plots
from model import DoublePendulumFlexible
import costs as cst


def generate_optimal_trajectory_extended(model: DoublePendulumFlexible, optimal_gen_params:dict, reference_trajectory_params:dict, armijo_params: dict, debug_params:dict, folder: str = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an optimal trajectory for the given model using a Newton-based method with dynamic horizon extension.
    Parameters:
        - model: Instance of the DoublePendulumFlexible model.
        - optimal_gen_params: Dictionary containing parameters for the optimal trajectory generation.
        - reference_params: Dictionary containing reference trajectory parameters.
        - armijo_params: Dictionary containing Armijo line search parameters.
        - debug_params: Dictionary containing debugging and plotting options.
        - folder: Optional folder path to save plots and animations.
    Returns:
        - xx_star: Optimal state trajectory.
        - uu_star: Optimal input trajectory.
        - time: Time vector corresponding to the trajectories.
    """

    ######################################
    # Unpack parameters
    ######################################

    # Model parameters
    ns = model.ns  # Number of states
    ni = model.ni  # Number of inputs
    dt = model.dt  # Discretization time step

    # debug parameters
    plot_flag = debug_params['plot_flag']
    debug_print = debug_params['debug_print']
    plot_armijo = debug_params['plot_armijo']
    save_armijo = debug_params['save_armijo']
    save_armijo_iters = debug_params['save_armijo_iters']
    
    # Armijo parameters
    stepsize_0=armijo_params['stepsize_0']
    c=armijo_params['c']
    beta=armijo_params['beta']
    armijo_maxiters=armijo_params['max_iters']

    # Optimal generation parameters
    max_iters = optimal_gen_params['max_iters']
    descent_threshold = optimal_gen_params['descent_threshold']
    max_time = optimal_gen_params['max_time']
    extension_duration = optimal_gen_params['extension_duration']
    terminal_error_tol = optimal_gen_params['terminal_error_tol']
    QQt = optimal_gen_params['QQt']
    RRt = optimal_gen_params['RRt']
    
    # Reference parameters
    time = reference_trajectory_params['time']
    xx_ref = reference_trajectory_params['xx_ref']
    uu_ref = reference_trajectory_params['uu_ref']
    eq_pair1 = reference_trajectory_params['eq_pair1']
    eq_pair2 = reference_trajectory_params['eq_pair2']

    TT = time.shape[0]               # Number of time steps

    ######################################
    # Check Stabilizability at Final Equilibrium & DARE Terminal Cost
    ######################################
    
    x_eq1, u_eq1 = eq_pair1
    x_eq2, u_eq2 = eq_pair2
    result, message, A_final, B_final = model.check_pbh_stabilizability(x_eq2, u_eq2)
    
    if not result:
        print(f"WARNING: The system is not locally stabilizable at the final equilibrium point. {message}")
    else:
        print(f"The system is locally stabilizable at the final equilibrium point. {message}")
        
    QQT, _, _ = ctrl.dare(A_final, B_final, QQt, RRt)


    ######################################
    # Initial Trajectory and Guess Generation
    ######################################

    # setting reference velocity for second link to zero
    xx_ref[3,:] = 0.0

    # Generate initial Guess
    xx_init = np.zeros_like(xx_ref)
    uu_init = np.zeros_like(uu_ref)
    
    # Initialize at first equilibrium
    xx_init[:, :] = x_eq1[:, None]
    
    # evolve dynamics with zero input towards second equilibrium
    for tt in range(TT-1):
        # Provide a small initial kick at the first time step
        if tt == 0:
            uu_init[:, tt] = (x_eq2[0] - x_eq1[0])*0.1

        xx_init[:, tt+1] = model.discrete_dynamics(xx_init[:, tt], uu_init[:, tt])
    uu_init[:,-1] = uu_init[:,-2]

    ######################################
    # [PLOTS] Initial guess vs Reference trajectory comparison
    ######################################
    
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

    T_end = time[-1]

    max_iters_ext = int(np.floor((max_time - T_end) / extension_duration))

    for jj in range(max_iters_ext):
        
        TT = len(time)
        print('\n')
        print(f"--- Extension Phase {jj} (Horizon: {time[-1]:.2f}s, Steps: {TT}) ---".center(80))

        ######################################
        # Allocate Memory for this Phase
        ######################################

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

        ######################################
        # Initialization
        ######################################-----------------------------------

        xx[:,:,0] = xx_init
        uu[:,:,0] = uu_init
        x0 = xx_init[:, 0]

        ######################################
        # [INNER LOOP] Newton Method
        ######################################--------------------------------

        converged = False
        final_kk = max_iters - 2

        for kk in range(max_iters-1):
            
            ######################################
            # 1. Cost and Derivatives Computation
            ######################################

            JJ[kk] = cst.total_cost(xx[:,:,kk], uu[:,:,kk], xx_ref, uu_ref, QQt, RRt, QQT)
            
            ######################################
            # 2. Backward Pass
            ######################################

            lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
            lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
            PP[:,:,TT-1,kk] = QQT
            pp[:,TT-1,kk] = lmbd[:,TT-1,kk].squeeze()

            for tt in reversed(range(TT-1)):
                qt, rt, QQt, RRt, SSt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:]
                At, Bt = model.discrete_jacobian(xx[:, tt, kk], uu[:, tt, kk])
                
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

            
            ######################################
            # 3. Forward Pass with Line Search
            ######################################

            # compute descent direction
            for tt in range(TT-1):
                deltau[:,tt,kk] = ss[:,tt,kk] +  KK[:,:,tt,kk] @  deltax[:, tt, kk]   
                deltax[:, tt+1, kk] = AA[:, :, tt, kk] @ deltax[:, tt, kk] + BB[:, :, tt, kk] @ deltau[:,tt,kk] 
                descent[kk] += deltau[:,tt,kk].T @ deltau[:,tt,kk]  
                descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk]  

            # Line Search (Armijo)
            stepsize, grad_err = armijo.select_stepsize_newton(model, stepsize_0, armijo_maxiters, c, beta, 
                                                ss[:,:,kk], KK[:,:,:,kk], xx_ref, uu_ref, 
                                                x0, xx[:,:,kk], uu[:,:,kk], JJ[kk], descent_arm[kk], 
                                                QQt, RRt, QQT, kk, plot=plot_armijo, save_armijo=save_armijo, save_armijo_iters=save_armijo_iters, folder=folder)


            # Store Global History
            global_JJ.append(JJ[kk])
            global_descent.append(descent[kk]) 
            gradient_error.append(grad_err)

            ######################################
            # 4. Convergence Check 
            ######################################

            if abs(descent[kk]) < descent_threshold or stepsize == 0.0:
                if debug_print: print(f"\nPhase converged at iter {kk}.")
                converged = True
                final_kk = kk
                break
            
            if np.isnan(descent[kk]):
                print(f"\nNaN detected at iter {kk}. Stopping.")
                final_kk = kk
                break

            ######################################
            # 5. Trajectory Update
            ######################################

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

            ######################################
            # Debug Printing
            ######################################

            if debug_print:
                log_str = (f"Iter: {kk} | "
                    f"Cost: {JJ[kk]:.6e} | "
                    f"Stepsize: {stepsize:.3e} | "
                    f"Descent: {descent[kk]:.6e}")
                print(f"\r{log_str}".ljust(120), end="", flush=True)
            else:
                dots = "." * (kk % 4)
                print(f"\rComputing optimal trajectory{dots:3}", end="", flush=True)

        ######################################
        # [END INNER LOOP]
        ######################################
        
        current_total_iters = phase_iters[-1] 
        phase_iters.append(current_total_iters + final_kk + 1)
        
        # Extract Best Trajectory of this Phase
        idx_best = final_kk if converged else max_iters-1
        xx_phase_opt = xx[:, :, idx_best]
        uu_phase_opt = uu[:, :, idx_best]

        # Store Phase History
        phase_history.append({
            'time': time.copy(),
            'xx': xx[:, :, :idx_best+1].copy(),
            'uu': uu[:, :, :idx_best+1].copy(),
            'final_idx': idx_best,
            'phase_idx': jj
        })

        ######################################
        # Check Extension Criteria
        ######################################

        # Check Terminal Error
        term_error = np.linalg.norm(xx_phase_opt[:, -1] - x_eq2)
        print(f"\nPhase {jj} finished. Terminal Error: {term_error:.5f}")

        # Decide on Extension
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

            ######################################
            # Extend Trajectories
            ######################################
           
           # Extend State and Input Trajectories
            xx_init, uu_init, xx_ref, uu_ref = extend_trajectory(
                model, xx_phase_opt, uu_phase_opt, xx_ref, uu_ref, x_eq2, u_eq2, dt, extension_duration
            )
            
            # Extend Time Vector
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
    plots.trajectory_comparison_phases(phase_history, eq_pair1=eq_pair1, eq_pair2=eq_pair2, title="Evolution of Trajectories over Extension Phases", plot_flag=plot_flag, folder=folder)
    
    plots.gradient_error(gradient_error, total_iterations, plot_flag=plot_flag, folder=folder)

    anim.animate_comparison(model, xx_star, xx_ref, (x_eq1, u_eq1), (x_eq2, u_eq2), dt=dt, folder=folder)

    plt.close('all')

    return xx_star, uu_star, time


######################################
# Helper: Array Extension
######################################

def extend_trajectory(model: DoublePendulumFlexible, xx_curr: np.ndarray, uu_curr: np.ndarray, xx_ref:np.ndarray, uu_ref:np.ndarray, x_eq: np.ndarray, u_eq:np.ndarray, dt:float, extension_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extends the current and reference trajectories by a specified duration using the system dynamics and equilibrium points.
    Parameters:
        - model: Instance of the DoublePendulumFlexible model.
        - xx_curr: Current state trajectory (ns x T).
        - uu_curr: Current input trajectory (ni x T).
        - xx_ref: Reference state trajectory (ns x T).
        - uu_ref: Reference input trajectory (ni x T).
        - x_eq: Equilibrium state to extend towards (ns,).
        - u_eq: Equilibrium input to extend towards (ni,).
        - dt: Discretization time step.
        - extension_time: Duration to extend the trajectories.
    Returns:
        - xx_ext: Extended state trajectory.
        - uu_ext: Extended input trajectory.
        - xx_ref_ext: Extended reference state trajectory.
        - uu_ref_ext: Extended reference input trajectory.
    """

    tt_extra = int(np.ceil(extension_time / dt))
    
    if tt_extra <= 0:
        return xx_curr, uu_curr, xx_ref, uu_ref

    # 1. Extend Reference (Pad with final equilibrium)
    xx_ref_ext = np.pad(xx_ref, ((0, 0), (0, tt_extra)), mode='edge')
    xx_ref_ext[:, -tt_extra:] = x_eq[:, None]

    uu_ref_ext = np.pad(uu_ref, ((0, 0), (0, tt_extra)), mode='edge')
    uu_ref_ext[:, -tt_extra:] = u_eq[:, None]

    # 2. Extend Current Solution (Warm Start for next iteration)
    uu_pad = np.tile(u_eq[:, None], (1, tt_extra))
    uu_ext = np.hstack((uu_curr, uu_pad)) # This introduces a step at the junction

    T_old = xx_curr.shape[1]
    uu_ext[:, T_old - 1] = u_eq

    # For State: Simulate dynamics forward from the last known state
    xx_ext = np.pad(xx_curr, ((0, 0), (0, tt_extra)), mode='edge') 
    
    curr_x = xx_curr[:, -1]
    for k in range(tt_extra):
        curr_x = model.discrete_dynamics(curr_x, u_eq)
        xx_ext[:, -tt_extra + k] = curr_x

    return xx_ext, uu_ext, xx_ref_ext, uu_ref_ext
