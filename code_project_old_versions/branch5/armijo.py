#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         armijo.py
# Description:  Implementation of Armijo's rule for stepsize selection.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         December 2025
#########################################################################################

#######################################
# Imports
#######################################

# Python libraries
import numpy as np

# Model
from model import DoublePendulumFlexible

# Costs
import costs as cst

# Plots
import plots as plots

#######################################
# Functions
#######################################

def select_stepsize_newton(model: DoublePendulumFlexible, stepsize_0: float, armijo_maxiters: int, c: float, beta: float, 
                           ss_k: np.ndarray, KK_k: np.ndarray, xx_ref: np.ndarray, uu_ref: np.ndarray, x0: np.ndarray, 
                           xx: np.ndarray, uu: np.ndarray, JJ: float, descent_arm: float, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray,
                           kk: int, plot: bool = False, folder: str = None, save_armijo: bool = False, save_armijo_iteration: int = 10) -> tuple[float, float]:
    """
    Select the stepsize using the Armijo rule for the Newton method.

    Parameters
    ----------
    model : DoublePendulumFlexible
        System model.
    stepsize_0 : float
        Initial stepsize to start the Armijo search.
    armijo_maxiters : int
        Maximum number of iterations for the Armijo search.
    c : float
        Armijo condition constant.
    beta : float
        Stepsize reduction factor.
    ss_k : np.ndarray
        Sigma gain from affine LQR.
    KK_k : np.ndarray
        Feedback gain matrix.
    xx_ref : np.ndarray
        Reference state trajectory.
    uu_ref : np.ndarray
        Reference control trajectory.
    x0 : np.ndarray
        Initial state.
    xx : np.ndarray
        Current state trajectory.
    uu : np.ndarray
        Current control trajectory.
    JJ : float
        Current cost.
    descent_arm : float
        Descent condition value.
    QQt : np.ndarray
        State cost matrix.
    RRt : np.ndarray
        Control cost matrix.
    QQT : np.ndarray
        Terminal state cost matrix.
    kk : int
        Current iteration.
    plot : bool, optional
        Whether to plot the Armijo descent, by default False.

    Returns
    -------
    stepsize : float
        Selected stepsize.
    gradient_error : float
        Error between numeric and analytic gradient.
    """

    

 
    TT = uu.shape[1]
    ns = xx_ref.shape[0]
    ni = uu.shape[0]

    stepsize = stepsize_0
    stepsizes = []
    costs_armijo = []

    if descent_arm > 0:
        print("WARNING: Descent direction is positive! Regularization needed.")

    for ii in range(armijo_maxiters):
        xx_temp = np.zeros((ns, TT  ))
        uu_temp = np.zeros((ni, TT))
        xx_temp[:, 0] = x0
        JJ_temp = 0

        # Forward integration (Closed-loop)
        for tt in range(TT - 1):
            deltax = xx_temp[:, tt] - xx[:, tt]
            uu_temp[:, tt] = uu[:, tt] + stepsize * ss_k[:, tt] +  KK_k[:, :, tt] @ deltax
            xx_temp[:, tt+1] = model.discrete_dynamics(xx_temp[:, tt], uu_temp[:, tt])
        
        # Compute total cost
        JJ_temp = cst.total_cost(xx_temp, uu_temp, xx_ref, uu_ref, QQt, RRt, QQT)
        
        uu_temp[:, -1] = uu_temp[:, -2]

        stepsizes.append(stepsize)      # Save the stepsize
        costs_armijo.append(JJ_temp)    # Save the cost associated to the stepsize

        # Armijo condition
        if JJ_temp > JJ + c*stepsize*descent_arm:
            stepsize = beta*stepsize
        else:
            break

        if ii == armijo_maxiters -1:
            stepsize = 0
            print("WARNING: no stepsize was found with armijo rule!")

    
    ############################
    # Gradient Check
    ############################
    
    gradient_error = 0.0
    step = 1e-6

    xx_step = np.zeros((ns,TT))
    uu_step = np.zeros((ni,TT))
    xx_step[:,0] = x0

    for tt in range(TT - 1):
        delta_x = xx_step[:, tt] - xx[:, tt]
        uu_step[:, tt] = uu[:, tt] + KK_k[:, :, tt] @ delta_x + step * ss_k[:, tt] 
        xx_step[:, tt+1] = model.discrete_dynamics(xx_step[:, tt], uu_step[:, tt])
    
    uu_temp[:, -1] = uu_temp[:, -2]
    cost_step = cst.total_cost(xx_step, uu_step, xx_ref, uu_ref, QQt, RRt, QQT) 
    slope_numeric = (cost_step - JJ)/step
    slope_analytic = descent_arm

    gradient_error = slope_numeric - slope_analytic
    
    ############################
    # Descent Plots
    ############################

    if plot or (folder is not None and save_armijo):
        steps = np.linspace(0,stepsize_0,int(20))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):
            step = steps[ii]

            # Temp solution update
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))
            xx_temp[:,0] = x0
            JJ_temp = 0

            for tt in range(TT - 1):
                delta_x = xx_temp[:, tt] - xx[:, tt]
                uu_temp[:, tt] = uu[:, tt] + KK_k[:, :, tt] @ delta_x + step * ss_k[:, tt] 
                xx_temp[:, tt+1] = model.discrete_dynamics(xx_temp[:, tt], uu_temp[:, tt])
            uu_step[:, -1] = uu_step[:, -2]
            JJ_temp = cst.total_cost(xx_temp, uu_temp, xx_ref, uu_ref, QQt, RRt, QQT) 
            costs[ii] = JJ_temp

        if folder is not None and save_armijo:
            save = kk % save_armijo_iteration == 0
        else:
            save = False

        plots.armijo_descent(steps, costs, JJ, descent_arm, c, stepsizes, costs_armijo, kk, plot=plot, save=save, folder=folder)

    return stepsize, gradient_error
