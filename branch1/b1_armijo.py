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
import matplotlib.pyplot as plt

# Model
from b1_model import DoublePendulumFlexible, PARAM_SET_2
model = DoublePendulumFlexible(**PARAM_SET_2)

# Costs
import b1_costs as cst

# Solver
import b1_solver as solver

# Plots
import b1_plots as plots

#######################################
# Functions
#######################################

def select_stepsize(stepsize_0: float, armijo_maxiters: int, c: float, beta: float, 
                    deltau: np.ndarray, xx_ref: np.ndarray, uu_ref: np.ndarray, x0: np.ndarray,
                    uu: np.ndarray,
                    JJ: float, descent_arm: float, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray, plot: bool = False):
    """
    Computes the stepsize using Armijo's rule.

    Parameters
    ----------
    stepsize_0 : float
        Initial stepsize.
    armijo_maxiters : int
        Maximum number of iterations for Armijo's rule.
    c : float
        Armijo threshold c.
    beta : float
        Armijo stepsize beta.
    deltau : np.ndarray
        Descent direction for the control inputs.
    xx_ref : np.ndarray
        Reference state trajectory.
    uu_ref : np.ndarray
        Reference control trajectory.
    x0 : np.ndarray
        Initial state.
    uu : np.ndarray
        Current control trajectory.
    JJ : float
        Current cost.
    descent_arm : float
        Descent value for Armijo's condition.
    QQt : np.ndarray
        State cost weight matrix.
    RRt : np.ndarray
        Control cost weight matrix.
    QQT : np.ndarray
        Terminal cost weight matrix.
    plot : bool, optional
        Whether to plot the Armijo descent (default is False).

    Returns
    -------
    stepsize : float
        Selected stepsize according to Armijo's rule.
    """

    TT = uu.shape[1]        # Time horizon
    stepsizes = []          # List of stepsizes
    costs_armijo = []       # List of costs associated to the stepsizes
    stepsize = stepsize_0   # Initial stepsize

    ns = xx_ref.shape[0]    # Number of states
    ni = uu_ref.shape[0]    # Number of inputs

    #Armijo loop
    for ii in range(armijo_maxiters):
        xx_temp = np.zeros((ns, TT))    # Temporary state trajectory
        uu_temp = np.zeros((ni, TT))    # Temporary control trajectory
        xx_temp[:, 0] = x0              # Set initial state
        JJ_temp = 0                     # Temporary cost

        # Cost computation
        for tt in range(TT-1):
            # Update Gradient method control input
            uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]

            # Simulate system dynamics
            xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])

            # Compute stage cost
            temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
            JJ_temp += temp_cost

        # Compute terminal cost
        temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QQT)[0]
        JJ_temp += temp_cost

        stepsizes.append(stepsize)      # Save the stepsize
        costs_armijo.append(JJ_temp)    # Save the cost associated to the stepsize

        # Armijo condition check
        if JJ_temp > JJ + c*stepsize*descent_arm:
            stepsize = beta*stepsize
        else:
            if plot:
                print('Armijo stepsize = {:.3e}'.format(stepsize))
            break

        if ii == armijo_maxiters -1 and plot:
            print("WARNING: no stepsize was found with armijo rule!")


    ############################
    # Descent Plot
    ############################

    if plot:
        steps = np.linspace(0,stepsize_0,int(2e1))  # Space for function g(gamma)
        costs = np.zeros(len(steps))                # Costs associated to g(gamma)

        for ii in range(len(steps)):
            step = steps[ii]

            # Temp solution update
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))
            xx_temp[:,0] = x0
            JJ_temp = 0

            for tt in range(TT-1):
                uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
                xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])
                temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
                JJ_temp += temp_cost
            temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QQT)[0]
            JJ_temp += temp_cost

            costs[ii] = JJ_temp

        plots.armijo_descent(steps, costs, JJ, descent_arm, c, stepsizes, costs_armijo)

    return stepsize

def select_stepsize_newton(stepsize_0: float, armijo_maxiters: int, c: float, beta: float, 
                           ss_k: np.ndarray, KK_k: np.ndarray, xx_ref: np.ndarray, uu_ref: np.ndarray, x0: np.ndarray, 
                           xx: np.ndarray, uu: np.ndarray, JJ: float, descent_arm: float, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray, plot: bool = False):
    """
    Select the stepsize using the Armijo rule for the Newton method.

    Parameters
    ----------
    stepsize_0 : float
        Initial stepsize to start the Armijo search.
    armijo_maxiters : int
        Maximum number of iterations for the Armijo search.
    c : float
        Armijo condition constant.
    beta : float
        Stepsize reduction factor.
    ss_k : np.ndarray
        Descent direction for the control inputs.
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
    plot : bool, optional
        Whether to plot the Armijo descent, by default False.

    Returns
    -------
    stepsize : float
        Selected stepsize.
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
        xx_temp = np.zeros((ns, TT))
        uu_temp = np.zeros((ni, TT))
        xx_temp[:, 0] = x0
        JJ_temp = 0

        # Forward integration (Closed-loop)
        for tt in range(TT - 1):
            delta_x = xx_temp[:, tt] - xx[:, tt]
            uu_temp[:, tt] = uu[:, tt] + KK_k[:, :, tt] @ delta_x + stepsize * ss_k[:, tt]
            xx_temp[:, tt+1] = model.discrete_dynamics(xx_temp[:, tt], uu_temp[:, tt])
            cost = cst.stagecost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt], QQt, RRt)[0]
            JJ_temp += cost
        cost_term = cst.termcost(xx_temp[:, -1], xx_ref[:, -1], QQT)[0]
        JJ_temp += cost_term

        uu_temp[:, -1] = uu_temp[:, -2]

        stepsizes.append(stepsize)      # save the stepsize
        costs_armijo.append(JJ_temp)    # save the cost associated to the stepsize

        if JJ_temp > JJ + c*stepsize*descent_arm:
            stepsize = beta*stepsize
        else:
            if plot:
                print('Armijo stepsize = {:.3e}'.format(stepsize))
            break

        if ii == armijo_maxiters -1 and plot:
            print("WARNING: no stepsize was found with armijo rule!")

    ############################
    # Descent Plot
    ############################

    if plot:
        steps = np.linspace(0,stepsize_0,int(2e1))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):
            step = steps[ii]

            # temp solution update
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))
            xx_temp[:,0] = x0
            JJ_temp = 0

            for tt in range(TT - 1):
                delta_x = xx_temp[:, tt] - xx[:, tt]
                uu_temp[:, tt] = uu[:, tt] + KK_k[:, :, tt] @ delta_x + step * ss_k[:, tt]
                xx_temp[:, tt+1] = model.discrete_dynamics(xx_temp[:, tt], uu_temp[:, tt])
                cost = cst.stagecost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt], QQt, RRt)[0]
                JJ_temp += cost
            cost_term = cst.termcost(xx_temp[:, -1], xx_ref[:, -1], QQT)[0]
            JJ_temp += cost_term

            costs[ii] = JJ_temp

        plots.armijo_descent(steps, costs, JJ, descent_arm, c, stepsizes, costs_armijo)        

    return stepsize
