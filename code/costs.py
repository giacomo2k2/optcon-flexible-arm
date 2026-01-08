#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         costs.py
# Description:  This module contains functions to compute stage and terminal costs and their derivatives.
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

#######################################
# Functions
#######################################

def stagecost(x: np.ndarray, u:np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray, QQt: np.ndarray, RRt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the stage cost and its derivatives.
    
    Parameters
    ----------
    x : np.ndarray
        Current state (ns,).
    u : np.ndarray
        Current input (ni,).
    x_ref : np.ndarray
        Reference state (ns,).
    u_ref : np.ndarray
        Reference input (ni,).
    QQt : np.ndarray
        State cost matrix (ns,ns).
    RRt : np.ndarray
        Input cost matrix (ni,ni).

    Returns
    -------
    ll : float
        Stage cost.
    lx : np.ndarray
        Gradient of stage cost w.r.t. state (ns,1).
    lu : np.ndarray
        Gradient of stage cost w.r.t. input (ni,1).
    lxx : np.ndarray
        Hessian of stage cost w.r.t. state (ns,ns).
    luu : np.ndarray
        Hessian of stage cost w.r.t. input (ni,ni).
    lux : np.ndarray
        Cross derivative of stage cost w.r.t. input and state (ni,ns).
    """
    x = x[:, None] # Make column vector (ns,1)
    u = u[:, None] # Make column vector (ni,1)

    x_ref = x_ref[:, None] # Make column vector (ns,1)
    u_ref = u_ref[:, None] # Make column vector (ni,1)

    # Stage cost
    ll = 0.5 * (x - x_ref).T @ QQt @ (x - x_ref) + 0.5 * (u - u_ref).T @ RRt @ (u - u_ref)

    # First Derivatives
    lx = QQt @ (x - x_ref)
    lu = RRt @ (u - u_ref)

    # Second Derivatives
    lxx = QQt
    luu = RRt
    lux = np.zeros((u.shape[0], x.shape[0]))  # Cross term is zero

    return ll.squeeze(), lx, lu, lxx, luu, lux

def termcost(xT: np.ndarray, xT_ref: np.ndarray, QT: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the terminal cost and its derivatives.

    Parameters
    ----------
    xT : np.ndarray
        Terminal state (ns,).
    xT_ref : np.ndarray
        Reference terminal state (ns,).
    QT : np.ndarray
        Terminal state cost matrix (ns,ns).
        
    Returns
    -------
    llT : float
        Terminal cost.
    lTx : np.ndarray
        Gradient of terminal cost w.r.t. terminal state (4,1).
    lTxx : np.ndarray
        Hessian of terminal cost w.r.t. terminal state (4,4).
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
        State trajectory (ns, T).
    uu : np.ndarray
        Input trajectory (ni, T).
    xx_ref : np.ndarray
        Reference state trajectory (ns, T).
    uu_ref : np.ndarray
        Reference input trajectory (ni, T).
    QQt : np.ndarray
        State cost matrix (ns, ns).
    RRt : np.ndarray
        Input cost matrix (ni, ni).
    QT : np.ndarray
        Terminal state cost matrix (ns, ns).
    tf : float
        Final relative time for the trajectory segment.

    Returns
    -------
    total_ll : float
        Total cost over the trajectory.
    """
    total_ll = 0.0
    TT = xx.shape[1]

    for t in range(TT-1):
        ll, _, _, _, _, _ = stagecost(xx[:, t], uu[:, t], xx_ref[:, t], uu_ref[:, t], QQt, RRt)
        total_ll += ll
    llT, _, _ = termcost(xx[:, -1], xx_ref[:, -1], QT)
    
    total_ll += llT

    return total_ll