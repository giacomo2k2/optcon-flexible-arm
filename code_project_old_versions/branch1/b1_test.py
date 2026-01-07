from b1_model import DoublePendulumFlexible, PARAM_SET_2
from b1_model_symbolic import DoublePendulumFlexibleSymbolic
from b1_solver import solver, jacobian_solver
from b1_animation import animate
from b1_trajectories import trajectory_generator
from b1_plots import plot_reference_traj, plot_equilibrium_manifold
import numpy as np

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


########## VECTOR NOTATION ###########
# axis 0: different states/inputs
# axis 1: time steps


########## VECTOR NOTATION DURING ITERATIONS ###########
# axis 0: different states/inputs
# axis 1: time steps
# axis 2: different iterations

def check_animation(model, x0, dt, steps):
    # Solve the system dynamics
    x_traj = solver(model, x0, dt, steps, method='rk4')
    # Animate the results
    animate(model, x_traj, dt, target_fps=30)

def check_jacobian(model_sym, x0, u0):
    # Compute Jacobians at the initial state
    A_sym, B_sym = model_sym.symbolic_jacobian()
    A_num, B_num = jacobian_solver(A_sym, B_sym, x0, u0)

    print(f"State Jacobian (sym): {A_sym.shape}", A_sym)
    print(f"Input Jacobian (sym): {B_sym.shape}", B_sym)

    print(f"State Jacobian (num): {A_num.shape}", A_num)
    print(f"Input Jacobian (num): {B_num.shape}", B_num)

def check_trajectory_generation(model, T0, Tend, dt):
    x_eq1, u_eq1 = model.find_equilibrium(fix_type='q2', value=np.pi/2)
    model.plot_equilibrium_point(x_eq1, u_eq1, title="1")
    x_eq2, u_eq2 = model.find_equilibrium(fix_type='q2', value=-3*np.pi/4)
    model.plot_equilibrium_point(x_eq2, u_eq2, title="2")
    print(f"Shape x_eq1: {x_eq1.shape}")
    # Select some equilibria as setpoints
    x_setpoints = np.array([
                            x_eq1,
                            x_eq2
                        ])

    u_setpoints = np.array([
                            u_eq1,
                            u_eq2
                        ]).reshape(-1,1)
    
    print(f"Shape x_setpoints: {x_setpoints.shape}, Shape u_setpoints: {u_setpoints.shape}")

    method = 'poly7'  # 'poly3', 'poly5', 'poly7', 'stepwise', 'cubic_spline'
    time, x_ref = trajectory_generator(T0, Tend, dt=dt, setpoints=x_setpoints, method=method)
    time, u_ref = trajectory_generator(T0, Tend, dt=dt, setpoints=u_setpoints, method=method)

    print(f"Shape time: {time.shape}, Shape x_ref: {x_ref.shape}, Shape u_ref: {u_ref.shape}")

    plot_reference_traj(time, x_ref, u_ref, method)

    return time, x_ref, u_ref

def check_equilibrium_points(model):
    # Compute equilibrium manifold
     data_dict = model.compute_equilibrium_manifold()
     # Plot equilibrium manifold
     plot_equilibrium_manifold(data_dict)

     #########################################
     # Ex. 1: q1 = 45 deg
     x_1, u_1 = model.find_equilibrium(fix_type='q1', value=np.pi - 0.01)
     model.plot_equilibrium_point(x_1, u_1)

     # Ex. 2: u = 5.0 Nm
     x_2, u_2 = model.find_equilibrium(fix_type='u', value=5.0)
     model.plot_equilibrium_point(x_2, u_2)

     # Ex. 3: q2 = 45 deg
     x_3, u_3 = model.find_equilibrium(fix_type='q2', value=np.pi/4)
     model.plot_equilibrium_point(x_3, u_3)

def main():
    #################################
    # Setup
    #################################

    # Create model instance
    model = DoublePendulumFlexible(**PARAM_SET_2)
    model_sym = DoublePendulumFlexibleSymbolic(**PARAM_SET_2)

    # Initial state
    x0 = model.random_init()
    u0 = 0.0

    # Simulation parameters
    T0 = 0.0
    dt = model.dt
    Tend = 10.0
    steps = np.arange(T0, Tend, dt).shape[0]

    #################################
    # Tests
    #################################

    # Animation check
    #check_animation(model, x0, dt, steps)

    # Jacobian check
    #check_jacobian(model_sym, x0, u0)

    # Trajectory generation check
    #time, x_ref, u_ref = check_trajectory_generation(model, T0, Tend, dt)

    # Check equilibrium points
    check_equilibrium_points(model) 
    

if __name__ == "__main__":
    main()

################
