from model import DoublePendulumFlexible, PARAM_SET_1, PARAM_SET_2, PARAM_SET_3
from model_symbolic import DoublePendulumFlexible as DoublePendulumSymbolic
from solver import solver
from animation import animate
import numpy as np
import sympy as sp

# Q1: is only the second arm flexible?
# Q2: first row in G is not correct, we correct it in model.py
# Q3: we manually computed the equilibrium


def main():
    # Define model parameters
    params = PARAM_SET_2

    # Create model instance
    model = DoublePendulumFlexible(**params)
    model_sym = DoublePendulumSymbolic(**params)

    # Initial state
    x0 = np.array([np.pi/4, np.pi/4, 0.0, 0.0])  # [q1, q2, q1_dot, q2_dot]
    x0 = model.random_init()

    # Simulation parameters
    dt = 0.01
    Tend = 10.0

    # Solve the system dynamics
    x_traj = solver(model, x0, dt, Tend, method='rk')

    # Animate the results
    # animate(model, x_traj, dt, target_fps=30)

    # Compare symbolic and numeric Jacobians at given state and input pair
    A_sym, B_sym = model_sym.symbolic_jacobian(x0, u_val=0.0)
    A_num, B_num = model.jacobian(x0, u=0.0)

    # Check if they are the same
    print("State jacobians are the same!" if np.allclose(A_sym, A_num) else "State jacobians differ:\n" + str(A_sym - A_num))
    print("Input jacobians are the same!" if np.allclose(B_sym, B_num) else "Input jacobians differ:\n" + str(B_sym - B_num))

if __name__ == "__main__":
    main()