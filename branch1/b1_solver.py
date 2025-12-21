import numpy as np
import sympy as sp
_JACOBIAN_CACHE = {}

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

def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None):
    """
    LQR for LTV system with (time-varying) affine cost.

    Parameters
    ----------
    AAin : np.ndarray, (nn x nn x TT)
        State dynamics matrices over time.
    BBin : np.ndarray, (nn x mm x TT)
        Input dynamics matrices over time.
    QQin : np.ndarray, (nn x nn x TT)
        State cost matrices over time.
    RRin : np.ndarray, (mm x mm x TT)
        Input cost matrices over time.
    SSin : np.ndarray, (mm x nn x TT)
        Cross term cost matrices over time.
    QQfin : np.ndarray, (nn x nn)
        Terminal state cost matrix.
    x0 : np.ndarray, (nn,)
        Initial state.
    qqin : np.ndarray, (nn x TT), optional
        State affine cost terms over time (default is None).
    rrin : np.ndarray, (mm x TT), optional
        Input affine cost terms over time (default is None).
    qqfin : np.ndarray, (nn,), optional
        Terminal state affine cost term (default is None).

    Returns
    -------
    KK : np.ndarray, (mm x nn x TT-1)
        State feedback gain matrices over time.
    sigma : np.ndarray, (mm x TT-1)
        Affine control terms over time.
    PP : np.ndarray, (nn x nn x TT)
        Riccati matrices over time.
    xxout : np.ndarray, (nn x TT)
        State trajectory.
    uuout : np.ndarray, (mm x TT)
        Input trajectory.
    """

    # check if matrix is (.. x .. x TT) - 3 dimensional array - if not, make it
    try:  
        ns, lA = AAin.shape[1:]
    except:
        AAin = AAin[:,:,None]
        ns, lA = AAin.shape[1:]
    try:  
        ni, lB = BBin.shape[1:]
    except:
        BBin = BBin[:,:,None]
        ni, lB = BBin.shape[1:]
    try:
        nQ, lQ = QQin.shape[1:]
    except:
        QQin = QQin[:,:,None]
        nQ, lQ = QQin.shape[1:]
    try:
        nR, lR = RRin.shape[1:]
    except:
        RRin = RRin[:,:,None]
        nR, lR = RRin.shape[1:]
    try:
        nSi, nSs, lS = SSin.shape
    except:
        SSin = SSin[:,:,None]
        nSi, nSs, lS = SSin.shape

    # Check dimensions consistency -- safety
    if nQ != ns:
        print("Matrix Q does not match number of states")
        exit()
    if nR != ni:
        print("Matrix R does not match number of inputs")
        exit()
    if nSs != ns:
        print("Matrix S does not match number of states")
        exit()
    if nSi != ni:
        print("Matrix S does not match number of inputs")
        exit()

    # Repeat matrices if needed
    if lA < TT:
        AAin = AAin.repeat(TT, axis=2)
    if lB < TT:
        BBin = BBin.repeat(TT, axis=2)
    if lQ < TT:
        QQin = QQin.repeat(TT, axis=2)
    if lR < TT:
        RRin = RRin.repeat(TT, axis=2)
    if lS < TT:
        SSin = SSin.repeat(TT, axis=2)

    # Check for affine terms
    augmented = False
    if qqin is not None or rrin is not None or qqfin is not None:
        augmented = True
        print("Augmented term!")

    # Initialize matrices
    KK = np.zeros((ni, ns, TT))
    sigma = np.zeros((ni, TT))
    PP = np.zeros((ns, ns, TT))
    pp = np.zeros((ns, TT))

    QQ = QQin
    RR = RRin
    SS = SSin
    QQf = QQfin

    qq = qqin
    rr = rrin

    qqf = qqfin

    AA = AAin
    BB = BBin

    xx = np.zeros((ns, TT))
    uu = np.zeros((ni, TT))

    xx[:,0] = x0

    PP[:,:,-1] = QQf
    pp[:,-1] = qqf

    # Solve Riccati equation
    for tt in reversed(range(TT-1)):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]
        PPtp = PP[:,:,tt+1]
        pptp = pp[:, tt+1][:,None]

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        mmt = rrt + BBt.T @ pptp

        PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
        ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

        PP[:,:,tt] = PPt
        pp[:,tt] = ppt.squeeze()


    # Evaluate KK and sigma
    for tt in range(TT-1):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]

        PPtp = PP[:,:,tt+1]
        pptp = pp[:,tt+1][:,None]

        # Check positive definiteness
        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        mmt = rrt + BBt.T @ pptp

        # for other purposes we could add a regularization step here...

        KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
        sigma_t = -MMt_inv@mmt

        sigma[:,tt] = sigma_t.squeeze()

    # Trajectory
    for tt in range(TT - 1):
        uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
        xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]

        xx[:,tt+1] = xx_p

        xxout = xx
        uuout = uu

    return KK, sigma, PP, xxout, uuout