import numpy as np

def solver(model: object, x0:np.ndarray, dt:float, steps:int, u:float=0.0, method: str = 'fe') -> np.ndarray:
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
    method : str, optional
        Integration method: ('fe', 'rk4'), default is 'fe'.

    Returns
    -------
    x_traj : np.ndarray
        States trajectory over the simulation time.
    """
    # Initialize trajectory array
    x_traj = np.zeros((steps, len(x0)))

    # Initial states
    x_traj[0] = x0

    # Solve the system dynamics
    for i in range(1, steps):
        x_traj[i] = solver_step(model, x_traj[i-1], dt, u[i-1], method)

    return x_traj

def solver_step(model: object, x:np.ndarray, dt: float, u: float=0.0, method: str = 'fe') -> np.ndarray:
    """
    Solves one time step of the system dynamics.

    Parameters
    ----------
    model : object
        Dynamic model with a 'dynamics' method.
    x : np.ndarray
        State vector.
    dt : float
        Time step for the simulation.
    u : float, optional
        Control input (default is 0.0).
    method : str, optional
        Integration method: ('fe', 'rk4'), default is 'fe'.

    Returns
    -------
    x_new : np.ndarray
        New state vector after one time step.
    """
    if method == 'rk4':
        k1 = model.dynamics(x, u)
        k2= model.dynamics(x + dt * k1 / 2, u)
        k3= model.dynamics(x + dt * k2 / 2, u)
        k4= model.dynamics(x + dt * k3, u)

        new_states = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    elif method == 'fe':
        new_states = x + dt * model.dynamics(x, u)
    else:
        raise ValueError("Choose a valid method: 'fe' or 'rk'")
    return new_states