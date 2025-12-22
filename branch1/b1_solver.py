#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         solver.py
# Description:  Solver functions for system dynamics.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         December 2025
#########################################################################################

#######################################
# Imports
#######################################

import numpy as np
import sympy as sp

#######################################
# Chaces
#######################################

_JACOBIAN_CACHE = {}
_HESS_PROD_CACHE = {}

#######################################
# Functions
#######################################

def solver(model: object, x0:np.ndarray, steps:int, u: float = 0.0) -> np.ndarray:
    """
    Solves the system dynamics for a given number of steps.

    Parameters
    ----------
    model : object
        Dynamic model with a 'dynamics' method.
    x0 : np.ndarray
        Initial state vector.
    dt : float
        Time step for the simulation.
    steps : int
        Number of time steps to simulate.
    u : float, optional
        Control input (default is 0.0).

    Returns
    -------
    x_traj : np.ndarray
        States trajectory over the simulation time.
    """
    # Initialize trajectory array
    x_traj = np.zeros((x0.shape[0], steps))

    # Initial states
    x_traj[:,0] = x0

    # Solve the system dynamics
    for i in range(1, steps):
        if isinstance(u, np.ndarray):
            x_traj[:,i] = solver_step(model, x_traj[:,i-1], u[:, i-1].item())
        else:
            x_traj[:,i] = solver_step(model, x_traj[:,i-1], u)

    return x_traj

def solver_step(model: object, x:np.ndarray, u: float = 0.0) -> np.ndarray:
    """
    Solves one time step of the system dynamics.

    Parameters
    ----------
    model : object
        Dynamic model with a 'dynamics' method.
    x : np.ndarray
        State vector.
    u : float, optional
        Control input (default is 0.0).

    Returns
    -------
    x_new : np.ndarray
        New state vector after one time step.
    """
    if model.discr_method == 'rk4':
        k1 = model.dynamics(x, u)
        k2 = model.dynamics(x + model.dt * k1 / 2, u)
        k3 = model.dynamics(x + model.dt * k2 / 2, u)
        k4 = model.dynamics(x + model.dt * k3, u)

        new_states = x + model.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    elif model.discr_method == 'fe':
        new_states = x + model.dt * model.dynamics(x, u)
    else:
        raise ValueError("Choose a valid method: 'fe' or 'rk4'.")
    return new_states

def jacobian_solver(A_sym, B_sym, x_val, u_val, dt: float = 0.01):
    """
    Compute the discrete-time Jacobian matrices A, B.

    Parameters
    ----------
    A_sym : sympy.Matrix
        Symbolic continuous-time state Jacobian matrix.
    B_sym : sympy.Matrix
        Symbolic continuous-time input Jacobian matrix.
    x_val : np.ndarray
        State vector at which to evaluate the Jacobians.
    u_val : float or np.ndarray
        Input value at which to evaluate the Jacobians.
    dt : float, optional
        Discretization time step (default is 0.01).

    Returns
    -------
    Ad : np.ndarray
        Discrete-time state Jacobian matrix.
    Bd : np.ndarray
        Discrete-time input Jacobian matrix.
    """
    global _JACOBIAN_CACHE

    cache_key = (id(A_sym), id(B_sym))
    
    if cache_key not in _JACOBIAN_CACHE:
        q1, q2, q1_dot, q2_dot = sp.symbols('q1 q2 q1_dot q2_dot')
        u = sp.symbols('u')

        func_A = sp.lambdify((q1, q2, q1_dot, q2_dot, u), A_sym, 'numpy')
        func_B = sp.lambdify((q1, q2, q1_dot, q2_dot, u), B_sym, 'numpy')
        
        _JACOBIAN_CACHE[cache_key] = (func_A, func_B)
    
    f_A, f_B = _JACOBIAN_CACHE[cache_key]

    u_scalar = u_val[0] if hasattr(u_val, "__len__") else u_val
    args = (*x_val, u_scalar)
    Ac = f_A(*args)
    Bc = f_B(*args)

    ns = Ac.shape[0]
    Ad = np.eye(ns) + dt * Ac
    Bd = dt * Bc
    
    return Ad, Bd

def hessian_numerical_solver(Hxx_sym, Huu_sym, Hxu_sym, x_val, u_val, l_next_val, dt = 0.01):
    """
    Valuta numericamente i termini grad^2 f * lambda.
    
    Ritorna:
    - Hxx: (ns x ns) matrice numerica
    - Huu: (ni x ni) matrice numerica
    - Hxu: (ns x ni) matrice numerica
    """
    global _HESS_PROD_CACHE
    
    # Identificatore per la cache
    cache_key = (id(Hxx_sym), id(Huu_sym), id(Hxu_sym))
    
    if cache_key not in _HESS_PROD_CACHE:
        # Definiamo i simboli per la lambdify
        q1, q2, q1_dot, q2_dot = sp.symbols('q1 q2 q1_dot q2_dot')
        u = sp.symbols('u')
        l1, l2, l3, l4 = sp.symbols('l1 l2 l3 l4')
        
        args = (q1, q2, q1_dot, q2_dot, u, l1, l2, l3, l4)
        
        # Trasformiamo in funzioni numeriche NumPy
        f_Hxx = sp.lambdify(args, Hxx_sym, 'numpy')
        f_Huu = sp.lambdify(args, Huu_sym, 'numpy')
        f_Hxu = sp.lambdify(args, Hxu_sym, 'numpy')
        
        _HESS_PROD_CACHE[cache_key] = (f_Hxx, f_Huu, f_Hxu)
    
    # Estraiamo le funzioni dalla cache
    f_Hxx, f_Huu, f_Hxu = _HESS_PROD_CACHE[cache_key]
    
    # Prepariamo i valori numerici (appiattiamo l_next se necessario)
    u_scalar = u_val[0] if hasattr(u_val, "__len__") else u_val
    l_vals = l_next_val.flatten()
    
    call_args = (*x_val, u_scalar, *l_vals)
    
    # Valutazione e discretizzazione
    # Poiché la slide parla di f_t (dinamica discreta), se f è continua:
    # f_discreta = x + dt * f_continua -> grad^2 f_discreta = dt * grad^2 f_continua
    Hxx_num = f_Hxx(*call_args) * dt
    Huu_num = f_Huu(*call_args) * dt
    Hxu_num = f_Hxu(*call_args) * dt
    
    return Hxx_num, Huu_num, Hxu_num

def ltv_LQR_step(At, Bt, QQt, RRt, SSt, PPtp, pptp, qt, rt):
    """
    Single time-step LQR backward pass computation.
    
    Computes the Riccati recursion for a single time step, given the next time-step values.
    
    Parameters
    ----------
    At : np.ndarray, (ns x ns)
        State dynamics matrix at time t.
    Bt : np.ndarray, (ns x ni)
        Input dynamics matrix at time t.
    QQt : np.ndarray, (ns x ns)
        State cost matrix at time t.
    RRt : np.ndarray, (ni x ni)
        Input cost matrix at time t.
    SSt : np.ndarray, (ni x ns)
        Cross term cost matrix at time t.
    PPtp : np.ndarray, (ns x ns)
        Riccati matrix at time t+1.
    pptp : np.ndarray, (ns,)
        Affine term of value function at time t+1.
    qt : np.ndarray, (ns, 1)
        State affine cost term at time t.
    rt : np.ndarray, (ni, 1)
        Input affine cost term at time t.
    
    Returns
    -------
    Kt : np.ndarray, (ni x ns)
        State feedback gain matrix at time t.
    st : np.ndarray, (ni,)
        Affine control term at time t.
    PPt : np.ndarray, (ns x ns)
        Riccati matrix at time t.
    ppt : np.ndarray, (ns,)
        Affine term of value function at time t.
    """
    # Ensure pptp is column vector
    if pptp.ndim == 1:
        pptp = pptp[:, None]
    
    # Compute intermediate terms
    MMt_inv = np.linalg.inv(RRt + Bt.T @ PPtp @ Bt)
    mmt = rt + Bt.T @ pptp

    # Compute feedback gain and affine term
    Kt = -MMt_inv @ (Bt.T @ PPtp @ At + SSt)
    st = -MMt_inv @ mmt
    
    # Riccati recursion for P and p
    PPt = At.T @ PPtp @ At - (Bt.T @ PPtp @ At + SSt).T @ MMt_inv @ (Bt.T @ PPtp @ At + SSt) + QQt
    ppt = At.T @ pptp - (Bt.T @ PPtp @ At + SSt).T @ MMt_inv @ mmt + qt
        
    return Kt, st.squeeze(), PPt, ppt.squeeze()