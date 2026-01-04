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
from scipy.optimize import fsolve

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
            x_traj[:,i] = solver_step(model, x_traj[:,i-1], u[:, i-1].item()) # Assuming u is 2D array with shape (ni, steps)
        else:
            x_traj[:,i] = solver_step(model, x_traj[:,i-1], u) # u is scalar

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



def jacobian_analytic(x_val: np.ndarray, u_val, model: object) -> tuple[np.ndarray, np.ndarray]:
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
    model : object
        System model.

    Returns
    -------
    Ad : np.ndarray
        Discrete-time state Jacobian matrix.
    Bd : np.ndarray
        Discrete-time input Jacobian matrix.
    """
   

    dt = model.dt
    method = model.discr_method

   
    Ac, Bc = model.jacobian(x_val, u_val)  
    u_scalar = u_val[0] if hasattr(u_val, "__len__") else u_val         
    
    ns = Ac.shape[0]

    
    if method == 'fe':
        Ad = np.eye(ns) + model.dt * Ac           # Discrete-time A
        Bd = model.dt * Bc                        # Discrete-time B
    elif method == 'rk4':
        k1 = model.dynamics(x_val, u_scalar)
        k2 = model.dynamics(x_val + 0.5 * model.dt * k1, u_scalar)
        k3 = model.dynamics(x_val + 0.5 * model.dt * k2, u_scalar)
        k4 = model.dynamics(x_val + model.dt * k3, u_scalar)
        
        A1, B1 = model.jacobian(x_val, u_scalar)

        x2 = x_val + 0.5 * model.dt * k1
        A2, B2 = model.jacobian(x2, u_scalar)

        x3 = x_val + 0.5 * model.dt * k2
        A3, B3 = model.jacobian(x3, u_scalar)

        x4 = x_val + model.dt * k3
        A4, B4 = model.jacobian(x4, u_scalar)

        dk1_dx = A1
        dk2_dx = A2 @ (np.eye(ns) + 0.5 * model.dt * dk1_dx)    
        dk3_dx = A3 @ (np.eye(ns) + 0.5 * model.dt * dk2_dx)
        dk4_dx = A4 @ (np.eye(ns) + model.dt * dk3_dx)

        Ad = np.eye(ns) + (model.dt / 6) * (dk1_dx + 2 * dk2_dx + 2 * dk3_dx + dk4_dx)

        dk1_du = B1
        dk2_du = A2 @ (0.5 * model.dt * dk1_du) + B2
        dk3_du = A3 @ (0.5 * model.dt * dk2_du) + B3
        dk4_du = A4 @ (model.dt * dk3_du) + B4

        Bd = (model.dt / 6) * (dk1_du + 2 * dk2_du + 2 * dk3_du + dk4_du)
    else:
        raise ValueError(f"Method {method} not supported.")
    
    return Ad, Bd



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
    rho = 0.1              # Regularization parameter

    # Ensure pptp is column vector
    if pptp.ndim == 1:
        pptp = pptp[:, None]

    # Compute intermediate terms
    MMt = RRt + Bt.T @ PPtp @ Bt# + rho * np.eye(Bt.shape[1]) #TODO Here's a regularization
    MMt_reg = MMt.copy()
    MMt_inv = np.linalg.inv(MMt_reg)

    mmt = rt + Bt.T @ pptp

    # Compute feedback gain and affine term
    Kt = -MMt_inv @ (Bt.T @ PPtp @ At + SSt)
    st = -MMt_inv @ mmt
    
    # Riccati recursion for P and p
    PPt = At.T @ PPtp @ At - (Bt.T @ PPtp @ At + SSt).T @ MMt_inv @ (Bt.T @ PPtp @ At + SSt) + QQt
    ppt = At.T @ pptp - (Bt.T @ PPtp @ At + SSt).T @ MMt_inv @ mmt + qt
        
    return Kt, st.squeeze(), PPt, ppt.squeeze()

def ltv_LQR_step(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray, PPtp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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



def ltv_LQR(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray,
                        x0: np.ndarray, xx_star: np.ndarray, uu_star: np.ndarray, model: object) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear time-varying LQR solver.
    
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
    model : object
        System model.

    Returns
    -------
    xx_lqr : np.ndarray, (ns x TT)
        LQR state trajectory.
    uu_lqr : np.ndarray, (ni x TT)
        LQR input trajectory.
    """
    # Unpack dimensions
    ns = model.ns
    ni = model.ni
    TT = xx_star.shape[1]

    xx_lqr = np.zeros((ns, TT))      # State seq.
    uu_lqr = np.zeros((ni, TT))
    PP = np.zeros((ns, ns, TT))      # P matrices for LQR
    KK = np.zeros((ni, ns, TT))      # Feedback gain matrices

    xx_lqr[:,0] = x0

    # Solve Riccati difference equation backward in time
    PP[:,:,TT-1] = QQT

    
    for tt in reversed(range(TT-1)):
        KK[:,:,tt], PP[:,:,tt] = ltv_LQR_step(At[:,:,tt], Bt[:,:,tt], QQt, RRt, PP[:,:,tt+1])

    for tt in range(TT-1):
        uu_lqr[:,tt] = uu_star[:,tt] + KK[:,:,tt] @ (xx_lqr[:,tt] - xx_star[:,tt])
        xx_lqr[:,tt+1] = model.discrete_dynamics(xx_lqr[:,tt], uu_lqr[:,tt])

    # For plotting purposes
    uu_lqr[:,-1] = uu_lqr[:,-2]
    
    return xx_lqr, uu_lqr

def ltv_mpc(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray,
            xxt: np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray, umax: float = 1000, umin: float = -1000,
            T_pred: int = 50, ipopt_opts=None):
    """
    CasAdi solver for constrained LTV MPC problem.

    Parameters
    ----------
    ipopt_opts : dict, optional
        Options for the IPOPT solver.
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





