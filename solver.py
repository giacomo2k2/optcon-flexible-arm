import numpy as np

def solver(model, x0:np.ndarray, dt:float, Tend:float, u:float=0.0, method: str = 'fe') -> np.ndarray:
    # Time vector
    time = np.arange(0, Tend, dt)
    
    # Initialize trajectory array
    x_traj = np.zeros((len(time), len(x0)))

    # Initial states
    x_traj[0] = x0

    # Solve the system dynamics
    for i in range(1, len(time)):
        x_traj[i] = solver_step(model, x_traj[i-1], dt, u, method)

    return x_traj

def solver_step(model, x:np.ndarray, dt: float, u: float=0.0, method: str = 'fe') -> np.ndarray:
    # Solve one time step using the specified method
    if method == 'rk':
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