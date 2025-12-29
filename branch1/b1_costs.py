#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         costs.py
# Description:  Cost functions.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         December 2025
#########################################################################################

#######################################
# Imports
#######################################
import numpy as np

# Default cost matrices (for reference)
# QQt = np.diag([100.0, 100.0, 10.0, 10.0]) * 1e-1
# RRt = 0.01 * np.eye(1)

#######################################
# Functions
#######################################

def get_cost_matrices(name: str) -> tuple[np.ndarray, np.ndarray]:      #COST MATRICES CHOICE
    """
    Returns the state and input cost matrices for the given maneuver name.

    Parameters
    ----------
    name : str
        The name of the maneuver
    
    Returns
    -------
    QQt : np.ndarray
        The state cost matrix
    RRt : np.ndarray
        The input cost matrix
    """
    match name:
        case 'elbow_up':
            # ti = 0.1, tf = 0.4, Tend = 4.0, poly7
            QQt = np.diag([100.0, 100.0, 1.0, 1.0]) * 1e-1
            RRt = 0.01 * np.eye(1)
            return QQt, RRt
        case 'swing_up':
            # ti = 0.1, tf = 0.4, Tend = 4.0, poly7
            QQt = np.diag([100.0, 100.0, 10.0, 10.0]) * 1e-3
            RRt = 0.01 * np.eye(1)
            return QQt, RRt
        case 'left_to_right_elbow_up':
            # ti = 0.3, tf = 0.7, Tend = 4.0, poly7
            QQt = np.diag([100.0, 100.0, 1.0, 1.0]) * 1e-1
            RRt = 0.01 * np.eye(1)
            return QQt, RRt
        case 'top_to_elbow_up':
            # ti = 0.2, tf = 0.5, Tend = 4.0, poly7
            QQt = np.diag([100.0, 100.0, 10.0, 10.0]) * 1e-1
            RRt = 0.01 * np.eye(1)
            return QQt, RRt
        case 'top_to_elbow_down':
            # ti = 0.3, tf = 0.6, Tend = 4.0 / 8.0, poly7 / stepwise
            QQt = np.diag([100.0, 100.0, 10.0, 10.0]) * 1e-1
            RRt = 0.01 * np.eye(1)
            return QQt, RRt
        case 'custom':
            QQt = np.diag([10.0, 10.0, 0.1, 0.1]) * 1
            RRt = 0.1 * np.eye(1)
            return QQt, RRt

SSt = np.zeros((1, 4))

def stagecost(x: np.ndarray, u:np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray, QQt: np.ndarray, RRt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the stage cost and its derivatives.
    
    Parameters
    ----------
    x : np.ndarray
        Current state (4,)
    u : np.ndarray
        Current input (1,)
    x_ref : np.ndarray
        Reference state (4,)
    u_ref : np.ndarray
        Reference input (1,)
    QQt : np.ndarray
        State cost matrix (4,4)
    RRt : np.ndarray
        Input cost matrix (1,1)

    Returns
    -------
    ll : float
        Stage cost
    lx : np.ndarray
        Gradient of stage cost w.r.t. state (4,1)
    lu : np.ndarray
        Gradient of stage cost w.r.t. input (1,1)
    lxx : np.ndarray
        Hessian of stage cost w.r.t. state (4,4)
    luu : np.ndarray
        Hessian of stage cost w.r.t. input (1,1)
    lux : np.ndarray
        Cross derivative of stage cost w.r.t. input and state (1,4)
    """
    x = x[:, None] # make column vector (4,1)
    u = u[:, None] # make column vector (1,1)

    x_ref = x_ref[:, None] # make column vector (4,1)
    u_ref = u_ref[:, None] # make column vector (1,1)

    # Stage cost
    ll = 0.5 * (x - x_ref).T @ QQt @ (x - x_ref) + 0.5 * (u - u_ref).T @ RRt @ (u - u_ref)

    # First Derivatives
    lx = QQt @ (x - x_ref)
    lu = RRt @ (u - u_ref)

    # Second Derivatives
    lxx = QQt
    luu = RRt
    lux = SSt

    return ll.squeeze(), lx, lu, lxx, luu, lux

def termcost(xT: np.ndarray, xT_ref: np.ndarray, QT: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the terminal cost and its derivatives.

    Parameters
    ----------
    xT : np.ndarray
        Terminal state (4,)
    xT_ref : np.ndarray
        Reference terminal state (4,)
    QT : np.ndarray
        Terminal state cost matrix (4,4)

    Returns
    -------
    llT : float
        Terminal cost
    lTx : np.ndarray
        Gradient of terminal cost w.r.t. terminal state (4,1)
    lTxx : np.ndarray
        Hessian of terminal cost w.r.t. terminal state (4,4)
    """
    # Terminal cost
    llT = 0.5 * (xT - xT_ref).T @ QT @ (xT - xT_ref)

    # First Derivative
    lTx = QT @ (xT - xT_ref)

    # Second Derivative
    lTxx = QT

    return llT.squeeze(), lTx, lTxx

def total_cost(xx: np.ndarray, uu: np.ndarray, xx_ref: np.ndarray, uu_ref: np.ndarray,
               QQt: np.ndarray, RRt: np.ndarray, QT: np.ndarray) -> float:
    
    """
    Computes the total cost over the trajectory.
    
    Parameters
    ----------
    xx : np.ndarray
        State trajectory (n, T)
    uu : np.ndarray
        Input trajectory (m, T)
    xx_ref : np.ndarray
        Reference state trajectory (n, T)
    uu_ref : np.ndarray
        Reference input trajectory (m, T)
    QQt : np.ndarray
        State cost matrix (n, n)
    RRt : np.ndarray
        Input cost matrix (m, m)
    QT : np.ndarray
        Terminal state cost matrix (n, n)
    tf : float
        Final relative time for the trajectory segment.

    Returns
    -------
    total_ll : float
        Total cost over the trajectory
    """
    
    total_ll = 0.0
    TT = xx.shape[1]
    
    
    for t in range(TT-1):
        ll, _, _, _, _, _ = stagecost(xx[:, t], uu[:, t], xx_ref[:, t], uu_ref[:, t], QQt, RRt)
        total_ll += ll

   
    llT, _, _ = termcost(xx[:, -1], xx_ref[:, -1], QT)
    total_ll += llT

    return total_ll