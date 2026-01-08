#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         solver.py
# Description:  This module contains solver functions for system dynamics.
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
import casadi as ca

# Custom Libraries
from model import DoublePendulumFlexible

#######################################
# Functions
#######################################


def solver(model: DoublePendulumFlexible, x:np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Computes the next state using the specified discretization method.
    Parameters:
        - model: DoublePendulumFlexible 
        - x: state vector at current time step
        - u: input vector at current time step
    Returns:
        - new_states: state vector at next time step
    """
    
    # Compute the next state based on the chosen discretization method
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

def jacobian_discrete_analytic(model: DoublePendulumFlexible, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the discrete-time Jacobians of the system dynamics using the specified discretization method.
    Parameters:
        - model : DoublePendulumFlexible
        - x: state vector at current time step
        - u: input vector at current time step
    Returns:
        - Ad: Discrete-time state Jacobian.
        - Bd: Discrete-time input Jacobian.
    """
   
    # Unpack model discretization parameters
    dt = model.dt
    method = model.discr_method

    # Continuous-time Jacobians
    Ac, Bc = model.jacobian(x, u)  

    # State dimension
    ns = Ac.shape[0]
    
    # Discrete-time Jacobians
    if method == 'fe':
        Ad = np.eye(ns) + dt * Ac                 
        Bd = model.dt * Bc                        
    
    elif method == 'rk4':
        k1 = model.dynamics(x, u)
        k2 = model.dynamics(x + 0.5 * dt * k1, u)
        k3 = model.dynamics(x + 0.5 * dt * k2, u)
        k4 = model.dynamics(x + model.dt * k3, u)
        
        A1, B1 = model.jacobian(x, u)

        x2 = x + 0.5 * model.dt * k1
        A2, B2 = model.jacobian(x2, u)

        x3 = x + 0.5 * model.dt * k2
        A3, B3 = model.jacobian(x3, u)

        x4 = x + model.dt * k3
        A4, B4 = model.jacobian(x4, u)

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
                        SSt: np.ndarray, PPtp: np.ndarray, pptp: np.ndarray, qt: np.ndarray, rt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Computes the Riccati recursion for a single time step of an affine LTV system, given the next time-step value.
    Parameters:
        - At : State dynamics matrix at time t.
        - Bt : Input dynamics matrix at time t.
        - QQt : State cost matrix at time t.
        - RRt : Input cost matrix at time t.
        - SSt : State-input cross term cost matrix at time t.
        - PPtp : Riccati matrix at time t+1.
        - pptp : Affine term at time t+1.
        - qt : State cost vector at time t.
        - rt : Input cost vector at time t.
    Returns:
        - Kt : State feedback gain matrix at time t.
        - st : Affine term at time t.
        - PPt : Riccati matrix at time t.
        - ppt : Affine term at time t.  
    """

    # Ensure pptp is column vector
    if pptp.ndim == 1:
        pptp = pptp[:, None]

    # Compute intermediate terms
    MMt = RRt + Bt.T @ PPtp @ Bt
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
    Computes the Riccati recursion for a single time step of a linear time-varying system, given the next time-step value.
    Parameters:
        - At : State dynamics matrix at time t.
        - Bt : Input dynamics matrix at time t.
        - QQt : State cost matrix at time t.
        - RRt : Input cost matrix at time t.
        - PPtp : Riccati matrix at time t+1.
    Returns:
        - Kt : State feedback gain matrix at time t.
        - PPt : Riccati matrix at time t.
    """


    # Inverse term
    MMt_inv = np.linalg.inv(RRt + Bt.T @ PPtp @ Bt)  

    # Feedback gain
    Kt = -MMt_inv @ (Bt.T @ PPtp @ At)

    # Riccati recursion
    PPt = At.T @ PPtp @ At - (At.T @ PPtp @ Bt) @ MMt_inv @ (Bt.T @ PPtp @ At) + QQt

    return Kt, PPt

def ltv_LQR(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray,
                        x0: np.ndarray, xx_star: np.ndarray, uu_star: np.ndarray, model: DoublePendulumFlexible) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Solves the finite-horizon LTV LQR problem and computes the optimal state and input trajectories.
    Parameters:
        - At : State dynamics matrices over the horizon (ns x ns x T).
        - Bt : Input dynamics matrices over the horizon (ns x ni x T).
        - QQt : State cost matrix.  
        - RRt : Input cost matrix.
        - QQT : Terminal state cost matrix.
        - x0 : Initial state vector.
        - xx_star : Reference state trajectory over the horizon (ns x T).
        - uu_star : Reference input trajectory over the horizon (ni x T).
    Returns:
        - xx_lqr : Optimal state trajectory (ns x T).
        - uu_lqr : Optimal input trajectory (ni x T).
    """

    # Unpack dimensions
    ns = model.ns
    ni = model.ni
    TT = xx_star.shape[1]

    # Initialize arrays
    xx_lqr = np.zeros((ns, TT))     # State seq.
    uu_lqr = np.zeros((ni, TT))     # Input seq.
    PP = np.zeros((ns, ns, TT))     # P matrices for LQR
    KK = np.zeros((ni, ns, TT))     # Feedback gain matrices

    # Initial state
    xx_lqr[:,0] = x0

    # Solve Riccati difference equation backward in time
    PP[:,:,TT-1] = QQT
    for tt in reversed(range(TT-1)):
        KK[:,:,tt], PP[:,:,tt] = ltv_LQR_step(At[:,:,tt], Bt[:,:,tt], QQt, RRt, PP[:,:,tt+1])

    # Compute LQR control and state trajectories forward in time
    for tt in range(TT-1):
        uu_lqr[:,tt] = uu_star[:,tt] + KK[:,:,tt] @ (xx_lqr[:,tt] - xx_star[:,tt])
        xx_lqr[:,tt+1] = model.discrete_dynamics(xx_lqr[:,tt], uu_lqr[:,tt])

    # For plotting purposes
    uu_lqr[:,-1] = uu_lqr[:,-2]
    
    return xx_lqr, uu_lqr

def ltv_mpc(At: np.ndarray, Bt: np.ndarray, QQt: np.ndarray, RRt: np.ndarray, QQT: np.ndarray,
            xxt: np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray, umax: float = 1000, umin: float = -1000,
            T_pred: int = 50, ipopt_opts: dict = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the LTV MPC optimization problem using CasAdi.
    Parameters:
        - At : State dynamics matrices over the prediction horizon (ns x ns x T_pred).
        - Bt : Input dynamics matrices over the prediction horizon (ns x ni x T_pred).
        - QQt : State cost matrix.  
        - RRt : Input cost matrix.
        - QQT : Terminal state cost matrix.
        - xxt : Current state vector.
        - x_ref : Reference state trajectory over the prediction horizon (ns x T_pred).
        - u_ref : Reference input trajectory over the prediction horizon (ni x T_pred).
        - umax : Maximum allowable input.
        - umin : Minimum allowable input.
        - T_pred : Prediction horizon length.
        - ipopt_opts : Options for the IPOPT solver.
    Returns:
        - u_opt : Optimal input at the current time step.
        - x_opt : Optimal state trajectory over the prediction horizon.
        - u_traj : Optimal input trajectory over the prediction horizon.    
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