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
import casadi as ca

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

def jacobian_solver(A_sym: sp.Matrix, B_sym: sp.Matrix, x_val: np.ndarray, u_val, dt: float = 0.01):
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

def hessian_numerical_solver(Hxx_sym: sp.Matrix, Huu_sym: sp.Matrix, Hxu_sym: sp.Matrix, x_val: np.ndarray, u_val, l_next_val: np.ndarray, dt: float = 0.01):
    """
    Compute the discrete-time Hessian matrices Hxx, Huu, Hxu.

    Parameters
    ----------
    Hxx_sym : sympy.Matrix
        Symbolic continuous-time Hessian matrix w.r.t. states.
    Huu_sym : sympy.Matrix
        Symbolic continuous-time Hessian matrix w.r.t. inputs.
    Hxu_sym : sympy.Matrix
        Symbolic continuous-time Hessian matrix w.r.t. states and inputs.
    x_val : np.ndarray
        State vector at which to evaluate the Hessians.
    u_val : float or np.ndarray
        Input value at which to evaluate the Hessians.
    l_next_val : np.ndarray
        Costate vector at the next time step.
    dt : float, optional
        Discretization time step (default is 0.01).
    
    Returns
    -------
    Hxx_num : np.ndarray
        Discrete-time Hessian matrix w.r.t. states.
    Huu_num : np.ndarray
        Discrete-time Hessian matrix w.r.t. inputs.
    Hxu_num : np.ndarray
        Discrete-time Hessian matrix w.r.t. states and inputs.
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

def ltv_affine_LQR_step(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray,
                        SSt: np.ndarray, PPtp: np.ndarray, pptp: np.ndarray, qt: np.ndarray, rt: np.ndarray):
    """
    Single time-step affine LQR backward pass computation.
    
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

def ltv_LQR_step(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray, PPtp: np.ndarray):
    """
    Computes the Riccati recursion for a single time step, given the next time-step value.

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
    PPtp : np.ndarray, (ns x ns)
        Riccati matrix at time t+1.

    Returns
    -------
    Kt : np.ndarray, (ni x ns)
        State feedback gain matrix at time t.
    PPt : np.ndarray, (ns x ns)
        Riccati matrix at time t.
    """
    MMt_inv = np.linalg.inv(RRt + Bt.T @ PPtp @ Bt)
    Kt = -MMt_inv @ (Bt.T @ PPtp @ At)
    PPt = At.T @ PPtp @ At - (At.T @ PPtp @ Bt) @ MMt_inv @ (Bt.T @ PPtp @ At) + QQt
    return Kt, PPt

def unconstrained_ltv_lqr(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray,
                          x0: np.ndarray, TT: int, ipopt_opts=None):
    """
    CasAdi solver for unconstrained LTV LQR problem.

    Parameters
    ----------
    At : np.ndarray, (ns x ns x TT)
        Sequence of state dynamics matrices.
    Bt : np.ndarray, (ns x ni x TT)
        Sequence of input dynamics matrices.
    QQt : np.ndarray, (ns x ns)
        State cost matrix.
    RRt : np.ndarray, (ni x ni)
        Input cost matrix.
    QQT : np.ndarray, (ns x ns)
        Terminal state cost matrix.
    x0 : np.ndarray, (ns,)
        Initial state vector.
    xx_star : np.ndarray, (ns x TT)
        Optimal state trajectory to track.
    uu_star : np.ndarray, (ni x TT)
        Optimal input trajectory to track.
    TT : int
        Time horizon length.
    ipopt_opts : dict, optional
        Options for the IPOPT solver.

    Returns
    -------
    X : np.ndarray, (ns x TT)
        Optimal state trajectory.
    U : np.ndarray, (ni x TT)
        Optimal input trajectory.
    """
    # Unpack dimensions
    ns = At.shape[0]
    ni = Bt.shape[1]

    # Initialize CasAdi optimization
    opti = ca.Opti()

    # Decision variables
    dX = opti.variable(ns, TT)
    dU = opti.variable(ni, TT)

    # Objective function
    cost = 0
    for tt in range(TT - 1):
        # Stage cost
        cost += ca.mtimes([dX[:, tt].T, QQt, dX[:, tt]]) + ca.mtimes([dU[:, tt].T, RRt, dU[:, tt]])

        # Dynamic constraint
        opti.subject_to(dX[:, tt + 1] == ca.DM(At[:,:,tt]) @ dX[:, tt] + ca.DM(Bt[:,:,tt]) @ dU[:, tt])

    # Terminal cost
    cost += ca.mtimes([dX[:, TT - 1].T, QQT, dX[:, TT - 1]])
    
    # Initial condition
    opti.subject_to(dX[:, 0] == ca.DM(x0))

    # Cost to minimize
    opti.minimize(cost)

    # Solver options
    if ipopt_opts is None:
        ipopt_opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 1000}
    opti.solver("ipopt", ipopt_opts)

    # Solve the optimization problem
    sol = opti.solve()
    
    return np.asarray(sol.value(dX)), np.asarray(sol.value(dU)).reshape(ni, TT)

def ltv_mpc(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray,
            xxt: np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray, umax: float = 1000, umin: float = -1000,
            x1_max: float = 20, x1_min: float = -20, x2_max: float = 20, x2_min: float = -20,
            x3_max: float = 20, x3_min: float = -20, x4_max: float = 20, x4_min: float = -20,
            T_pred: int = 50, ipopt_opts=None):
    """
    ...
    """
    # Unpack dimensions
    ns = Bt.shape[0] 
    ni = Bt.shape[1]

    # Initialize CasAdi optimization
    opti = ca.Opti()

    # Decision variables for the upcoming time window
    dX = opti.variable(ns, T_pred)
    dU = opti.variable(ni, T_pred)

    # Objective function
    cost = 0
    for t_p in range(T_pred - 1):
        # Stage cost
        cost += ca.mtimes([dX[:, t_p].T, QQt, dX[:, t_p]]) + ca.mtimes([dU[:, t_p].T, RRt, dU[:, t_p]])

        # Dynamic constraint
        opti.subject_to(dX[:, t_p + 1] == ca.DM(At[:,:,t_p]) @ dX[:, t_p] + ca.DM(Bt[:,:,t_p]) @ dU[:, t_p])

        # Input constraints
        u_real = ca.DM(u_ref[:, t_p]) + dU[:, t_p]
        # opti.subject_to(opti.bounded(umin, u_real, umax))
        opti.subject_to(u_real <= umax)
        opti.subject_to(u_real >= umin)
        
        # States constraints
        x1_real = ca.DM(x_ref[0, t_p]) + dX[0, t_p]
        x2_real = ca.DM(x_ref[1, t_p]) + dX[1, t_p]
        x3_real = ca.DM(x_ref[2, t_p]) + dX[2, t_p]
        x4_real = ca.DM(x_ref[3, t_p]) + dX[3, t_p]
        # opti.subject_to(opti.bounded(x1_min, x1_real, x1_max))
        # opti.subject_to(opti.bounded(x2_min, x2_real, x2_max))
        # opti.subject_to(opti.bounded(x3_min, x3_real, x3_max))
        # opti.subject_to(opti.bounded(x4_min, x4_real, x4_max))
        opti.subject_to(x1_real <= x1_max)
        opti.subject_to(x1_real >= x1_min)
        opti.subject_to(x2_real <= x2_max)
        opti.subject_to(x2_real >= x2_min)
        opti.subject_to(x3_real <= x3_max)
        opti.subject_to(x3_real >= x3_min)
        opti.subject_to(x4_real <= x4_max)
        opti.subject_to(x4_real >= x4_min)

    # Terminal cost
    cost += ca.mtimes([dX[:, T_pred - 1].T, QQT, dX[:, T_pred - 1]])
    
    # Initial condition
    opti.subject_to(dX[:, 0] == ca.DM(xxt - x_ref[:, 0]))

    # Cost to minimize
    opti.minimize(cost)

    # Solver options
    if ipopt_opts is None:
        ipopt_opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 1000}
    opti.solver("ipopt", ipopt_opts)

    try:
        # Solve the optimization problem
        sol = opti.solve()
    except RuntimeError:
        # Try with zero initial guess then fail gracefully
        opti.set_initial(dX, np.zeros((ns, T_pred)))
        opti.set_initial(dU, np.zeros((ni, T_pred)))
        try:
            sol = opti.solve()
        except Exception as e:
            print("solver_ltv_mpc: solver failed:", e)
            return None, None, None

    return np.asarray(sol.value(dU[:, 0])), np.asarray(sol.value(dX)), np.asarray(sol.value(dU))