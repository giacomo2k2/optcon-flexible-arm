#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         model.py
# Description:  This module contains the model class for the flexible double pendulum system.
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
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numba import njit
from matplotlib.lines import Line2D

# Set font family
plt.rcParams['font.family'] = 'serif'

# Solver
import solver as solver

# Plots
import plots as plots

# Color palette
from plots import colors

# Flag for fast method usage (Numba-optimized)
USE_FAST_METHOD = False

# Parameters settings
PARAM_SET_1 = {
    'm1': 1.0,      # Mass of link 1 [kg]
    'm2': 1.0,      # Mass of link 2 [kg]
    'l1': 1.0,      # Length of link 1 [m]
    'l2': 1.0,      # Length of link 2 [m]
    'lc1': 0.5,     # Center of mass of link 1 [m]
    'lc2': 0.5,     # Center of mass of link 2 [m]
    'I1': 0.33,     # Inertia of link 1 [kg*m^2]
    'I2': 0.33,     # Inertia of link 2 [kg*m^2]
    'g': 9.81,      # Gravity acceleration [m/s^2]
    'f1': 1.0,      # Friction coefficient link 1 [N*m*s/rad]
    'f2': 1.0,      # Friction coefficient link 2 [N*m*s/rad]
    'k': 0.5        # Flexibility coefficient [N*m/rad]
}

# This is the one used in this project
PARAM_SET_2 = {
    'm1': 2.0,     
    'm2': 2.0,    
    'l1': 1.5,    
    'l2': 1.5,      
    'lc1': 0.75,    
    'lc2': 0.75,   
    'I1': 1.5,   
    'I2': 1.5,      
    'g': 9.81,   
    'f1': 1.0,      
    'f2': 1.0,       
    'k': 0.6        
}

PARAM_SET_3 = {
    'm1': 1.5,    
    'm2': 1.5,    
    'l1': 2.0,      
    'l2': 2.0,    
    'lc1': 1.0,    
    'lc2': 1.0,     
    'I1': 2.0,     
    'I2': 2.0,      
    'g': 9.81,      
    'f1': 1.0,        
    'f2': 1.0,        
    'k': 0.7        
}

#######################################
# Model Class
#######################################

class DoublePendulumFlexible:
    """
    Class representing a flexible double pendulum system.
    """
    def __init__(self, m1:float, m2:float, l1:float, l2:float, lc1:float, lc2:float,
                  I1:float, I2:float, g:float, f1:float, f2:float, k:float) -> None:
        """
        Create a Double Pendulum system.

        Parameters
        ----------
        m1 : float
            Mass of link 1 [kg]
        m2 : float
            Mass of link 2 [kg]
        l1 : float
            Length of link 1 [m]
        l2 : float
            Length of link 2 [m]
        lc1 : float
            Center of mass of link 1 [m]
        lc2 : float
            Center of mass of link 2 [m]
        I1 : float
            Inertia of link 1 [kg*m^2]
        I2 : float
            Inertia of link 2 [kg*m^2]
        g : float
            Gravity acceleration [m/s^2]
        f1 : float
            Friction coefficient link 1 [N*m*s/rad]
        f2 : float
            Friction coefficient link 2 [N*m*s/rad]
        k : float
            Flexibility coefficient [N*m/rad]
        """
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.I1 = I1
        self.I2 = I2
        self.g = g
        self.f1 = f1
        self.f2 = f2
        self.k = k

        # Parameters as tuple and dict for easy access if needed
        self.param_tuple = (self.m1, self.m2, self.l1, self.l2, self.lc1, self.lc2, self.I1, self.I2, self.g, self.f1, self.f2, self.k)
        self.param_dict = {
            'm1': self.m1,
            'm2': self.m2,
            'l1': self.l1,
            'l2': self.l2,
            'lc1': self.lc1,
            'lc2': self.lc2,
            'I1': self.I1,
            'I2': self.I2,
            'g': self.g,
            'f1': self.f1,
            'f2': self.f2,
            'k': self.k
        }
        self.param_names = ['m1', 'm2', 'l1', 'l2', 'lc1', 'lc2', 'I1', 'I2', 'g', 'f1', 'f2', 'k']

        self.ns = 4                 # State dimension
        self.ni = 1                 # Input dimension
        self.dt = 0.01              # Discretization time step
        self.discr_method = 'rk4'   # Discretization method

    def get_params_as_tuple(self) -> tuple:
        """
        Return model parameters as a tuple.

        Returns
        -------
        params : tuple
            Model parameters (m1, m2, l1, l2, lc1, lc2, I1, I2, g, f1, f2, k).
        """
        return self.param_tuple

    def get_param(self, param: str) -> float:
        """
        Get a single model parameter.

        Parameters
        ----------
        param : str
            Name of the parameter to get.

        Returns
        -------
        value : float
            Value of the requested parameter.
        """
        try:
            if param not in self.param_names:
                raise ValueError(f"Parameter '{param}' is not valid.")
        except ValueError as e:
            print(f"\n{e}\nValid paramters name are: {', '.join(self.param_names)}.\n")
            exit(1)
        
        return getattr(self, param)

    def set_param(self, param: str, value: float) -> None:
        """
        Set a single model parameter.

        Parameters
        ----------
        param : str
            Name of the parameter to set.
        value : float
            New value for the parameter.
        """
        try:
            if param not in self.param_names:
                raise ValueError(f"Parameter '{param}' is not valid.")
            if not isinstance(value, (int, float)):
                raise ValueError("Parameter value must be numeric (int or float).")
        except ValueError as e:
            print(f"\n{e}\nValid paramters name are: {', '.join(self.param_names)}.\n")
            exit(1)
        
        setattr(self, param, value)
        self.param_dict[param] = value
        self.param_tuple = tuple(self.param_dict[name] for name in self.param_names)

    def set_params(self, params: dict) -> None:
        """
        Set multiple model parameters from a dictionary.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters to set.
        """
        try:
            for param, value in params.items():
                if param not in self.param_names:
                    raise ValueError(f"Parameter '{param}' is not valid.")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Value for parameter '{param}' must be numeric (int or float).")
        except ValueError as e:
            print(f"\n{e}\nValid paramters name are: {', '.join(self.param_names)}.\n")
            exit(1)
        
        for param, value in params.items():
            setattr(self, param, value)
            self.param_dict[param] = value
        self.param_tuple = tuple(self.param_dict[name] for name in self.param_names)
 
    def dynamic_model(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the dynamic model of the Double Pendulum.
        
        Parameters
        ----------
        x : np.ndarray
            State vector [q1, q2, q1_dot, q2_dot].

        Returns
        -------
        M : np.ndarray
            Mass matrix.
        M_inv : np.ndarray
            Inverse of the mass matrix.
        C : np.ndarray
            Coriolis and centrifugal matrix.
        G : np.ndarray
            Gravity vector.
        F : np.ndarray
            Friction matrix.
        """
        # State variables
        q1 = x[0]
        q2 = x[1]
        q1_dot = x[2]
        q2_dot = x[3]
        
        # Precompute sines and cosines
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        
        # Combined angles sines and cosines
        c12 = np.cos(q1+q2)
        s12 = np.sin(q1+q2)

        # Inertia matrix
        M = np.array([
            [self.I1 + self.I2 + self.lc1**2 * self.m1 + self.m2 * (self.l1**2 + 2 * self.l1 * self.lc2 * c2 + self.lc2**2), self.I2 + self.lc2 * self.m2 * (self.l1 * c2 + self.lc2)],
            [self.I2 + self.lc2 * self.m2 * (self.l1 * c2 + self.lc2), self.I2 + self.m2 * self.lc2 ** 2]
        ])
    
        # Coriolis and centrifugal matrix
        C = np.array([
            [- self.l1 * self.lc2 * self.m2 * q2_dot * s2, - self.l1 * self.lc2 * self.m2 * (q1_dot + q2_dot) * s2],
            [ self.l1 * self.lc2 * self.m2 * q1_dot * s2, 0.0]
        ])

        # Gravity vector        
        G = np.array([
            [self.g * self.lc1 * self.m1 * s1 + self.g * self.m2 * self.l1 * s1 + self.m2 * self.g * self.lc2 * s12],
            [self.g * self.m2 * self.lc2 * s12 + 3 * self.k * s2 * c2**2]
        ])
        
        # Friction matrix
        F = np.array([
            [self.f1, 0.0],
            [0.0, self.f2]
        ])
        
        # Intertia Matrix inverse
        M_inv = np.linalg.inv(M)

        return M, M_inv, C, G, F

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the continuous-time dynamics of the Double Pendulum.

        Parameters
        ----------
        x : np.ndarray
            State vector [q1, q2, q1_dot, q2_dot].
        u : np.ndarray
            Input vector.

        Returns
        -------
        x_dot : np.ndarray
            Time derivative of the state vector.
        """
        # Angular velocities, scalars
        q1_dot = x[2]
        q2_dot = x[3]

        # Compute dynamic model matrices with possible fast method        
        if USE_FAST_METHOD:
            params = self.get_params_as_tuple()
            M, M_inv, C, G, F = fast_dynamic_model(x, params)
        else:
            M, M_inv, C, G, F = self.dynamic_model(x)

        if isinstance(u, np.ndarray):
            u = u.item()  # Convert to scalar if u is an array

        # Angular velocity vector
        q_dot = np.array([
                            [q1_dot],
                            [q2_dot]
                         ])

        # Input Torque vector
        T = np.array([
                        [u],
                        [0.0]
                    ])
        
        # Right-hand side of the dynamics equation
        rhs = - F @ q_dot - C @ q_dot - G + T
        
        # Angular accelerations
        q_ddot = M_inv @ rhs

        # State at next time step
        x_dot = np.concatenate([q_dot.flatten(), q_ddot.flatten()]).flatten()

        return x_dot  

    def discrete_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the discrete-time dynamics of the Double Pendulum.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.ndarray
            Current input vector.

        Returns
        -------
        x_next : np.ndarray
            State vector at the next time step.
        """
        return solver.solver_step(self, x, u)
    
    def jacobian(self, x: np.ndarray, u:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the continuous-time Jacobian matrices A, B analytically.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.ndarray
            Current input vector.
        
        Returns
        -------
        A : np.ndarray
            Continuous-time state Jacobian matrix.
        B : np.ndarray
            Continuous-time input Jacobian matrix.
        """

        # Compute Jacobian with possible fast method
        if USE_FAST_METHOD:
            params = self.get_params_as_tuple()
            A, B = fast_jacobian(x, u.item(), params)
            return A, B
        else:
            # Zeros entries
            a11 = np.zeros((2, 2))
            a12 = np.eye(2)

            # Extract state variables
            q1 = x[0]
            q2 = x[1]
            q1_dot = x[2]
            q2_dot = x[3]

            # Angular velocity vector
            q_dot = np.array([
                                [q1_dot],
                                [q2_dot]
                            ])

            # Precompute sines and cosines
            s1 = np.sin(q1)
            c1 = np.cos(q1)
            s2 = np.sin(q2)
            c2 = np.cos(q2)

            # Combined angles sines and cosines
            c12 = np.cos(q1 + q2)
            s12 = np.sin(q1 + q2)

            # Compute dynamic model matrices
            M, M_inv, C, G, F = self.dynamic_model(x)

            if isinstance(u, np.ndarray):
                u = u.item() # Convert to scalar if u is an array

            # Input Torque vector
            T = np.array([
                            np.array([u]),
                            np.array([0.0])
                        ]).reshape((2,1))

            # Derivative of Coriolis matrix w.r.t q_dot
            dc_dq_dot = 2*self.l1 * self.lc2 * self.m2 * s2 * np.array([
                                                                        [-q2_dot, -q1_dot - q2_dot],
                                                                        [q1_dot, 0.0]
                                                                    ])

            # Lower-right block
            a22 = M_inv @ (-F - dc_dq_dot)

            # Derivative of M w.r.t q1
            dM_dq1 = np.array([
                [0, 0],
                [0, 0]
            ])

            # Derivative of M w.r.t q2
            dM_dq2 = np.array([
                                [- 2 * self.l1 * self.lc2 * self.m2 * s2, - self.l1 * self.lc2 *self.m2 * s2],
                                [- self.l1 * self.lc2 *self.m2 * s2, 0.0]
                            ])
            
            # Compute tau'
            tau = - F @ q_dot - C @ q_dot - G + T

            # Derivative of M_inv * tau w.r.t q1 and q2
            dM_inv_dq_tau_1 = - M_inv @ dM_dq1 @ M_inv @ tau
            dM_inv_dq_tau_2 = - M_inv @ dM_dq2 @ M_inv @ tau

            # Combine derivatives
            dM_inv_dq_tau = np.array([
                [dM_inv_dq_tau_1[0, 0], dM_inv_dq_tau_2[0, 0]],
                [dM_inv_dq_tau_1[1, 0], dM_inv_dq_tau_2[1, 0]]
            ])

            # Derivative of Coriolis matrix w.r.t q1 and q2
            dc_dq = self.l1 * self.lc2 * self.m2 * c2 * np.array([
                                                                    [0.0, -2*q1_dot*q2_dot - q2_dot**2],
                                                                    [0.0, q1_dot**2]
                                                                ])
            
            # Derivative of Gravity vector w.r.t q1 and q2
            dG_dq = np.array([
                            [self.g * self.lc1 * self.m1 * c1 + self.g * self.m2 * self.l1 * c1 + self.m2 * self.g * self.lc2 * c12, self.m2 * self.g * self.lc2 * c12],
                            [self.m2 * self.g * self.lc2 * c12, self.m2 * self.g * self.lc2 * c12 + 3 * self.k * (c2**2 - 2*s2**2) * c2]
                            ])
            
            # Lower-left block
            a21 = dM_inv_dq_tau + M_inv @ (- dc_dq - dG_dq)

            # Analytic State Jacobian matrix
            A = np.block([
                [a11, a12],
                [a21, a22]
            ])

            # Zero entry
            b1 = np.zeros((2, 1))

            # Lower block
            b2 = M_inv @ np.array([
                                    [1.0],
                                    [0.0]
                                ])

            # Analytic Input Jacobian matrix
            B = np.vstack((b1, b2))

            return A, B
    
    def _get_forward_kinematics(self, x: np.ndarray, com: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the forward kinematics of the double pendulum.

        Parameters
        ----------
        x : np.ndarray
            State vector [q1, q2, q1_dot, q2_dot].
        com : bool, optional
            If True, compute positions of centers of mass; otherwise, joint positions (default is False).

        Returns
        -------
        p1 : np.ndarray
            Position of the first joint [x1, y1].
        p2 : np.ndarray
            Position of the end-effector [x2, y2].
        pc1 : np.ndarray, optional
            Position of the center of mass of link 1 [xc1, yc1] (returned if com=True).
        pc2 : np.ndarray, optional
            Position of the center of mass of link 2 [xc2, yc2] (returned if com=True).
        """
        # Extract joint angles
        q1 = x[0]
        q2 = x[1]

        # Compute cartesian positions
        p1 = np.array([self.l1 * np.sin(q1), -self.l1 * np.cos(q1)])
        p2 = p1 + np.array([self.l2 * np.sin(q1 + q2), -self.l2 * np.cos(q1 + q2)])

        # Compute centers of mass positions if requested
        if com:
            pc1 = np.array([self.lc1 * np.sin(q1), -self.lc1 * np.cos(q1)])
            pc2 = np.array([self.l1 * np.sin(q1) + self.lc2 * np.sin(q1 + q2), -self.l1 * np.cos(q1) - self.lc2 * np.cos(q1 + q2)])
            return p1, p2, pc1, pc2
        else:
            return p1, p2
        
    def find_equilibrium(self, fix_type: str, value_fixed: float = 0.0, right: bool = True, up: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Find an equilibrium point by fixing either q1 or q2.

        Parameters
        ----------
        fix_type : str
            Type of variable to fix ('q1' or 'q2').
        value_fixed : float, optional
            Value to fix the variable at (default is 0.0).
        right : bool, optional
            If fixing q2, choose the solution with q1 > 0 (default is True).
        up: bool, optional
            If fixing q1, choose the solution with q2 > 0 (default is True).
        """
        # Define the error function for fsolve
        def error_func(guess):
            if fix_type == 'q1':
                x_test = np.array([value_fixed, float(guess), 0.0, 0.0])
            else:
                x_test = np.array([float(guess), value_fixed, 0.0, 0.0])
            
            _, _, _, G, _ = self.dynamic_model(x_test)
            return G[1, 0] 

        # Search full circle
        guesses = [0.0, np.pi, -np.pi, np.pi/2, -np.pi/2, 0.5, -0.5]
        valid_solutions = []
        seen_vals = []

        # Solve for each guess
        for guess in guesses:
            sol, info, ier, msg = fsolve(error_func, guess, full_output=True)
            
            current_error = np.abs(info['fvec'])
            if ier == 1 or current_error < 1e-6:
                # Normalize to [-pi, pi]
                val = np.arctan2(np.sin(sol[0]), np.cos(sol[0]))
                
                # Check uniqueness
                is_new = True
                for existing in seen_vals:
                    if np.abs(val - existing) < 1e-3 or np.abs(val + np.pi) < 1e-3: 
                        is_new = False
                        break
                
                # Double check residual
                if is_new and abs(error_func(val)) < 1e-4:
                    # Calculate required u
                    if fix_type == 'q1':
                        x_final = np.array([value_fixed, val, 0.0, 0.0])
                    else:
                        x_final = np.array([val, value_fixed, 0.0, 0.0])

                    _, _, _, G_final, _ = self.dynamic_model(x_final)
                    u_calc = G_final[0, 0]
                    if fix_type == 'q1':
                        valid_solutions.append({'q1': value_fixed, 'q2': val, 'u': u_calc})
                    else:
                        valid_solutions.append({'q1': val, 'q2': value_fixed, 'u': u_calc})
                    seen_vals.append(val)

        index = 0    
        if fix_type == 'q1':
            for i, sol in enumerate(valid_solutions):
                q2 = sol['q2']
                if up:
                    if value_fixed >= 0 and q2 >= 0 or value_fixed < 0 and q2 <= 0:
                        index = i
                if not up:
                    if value_fixed >= 0 and q2 <= 0 or value_fixed < 0 and q2 >= 0:
                        index = i

        else:            
            for i, sol in enumerate(valid_solutions):
                
                q1 = sol['q1']
                if right:
                    if q1 >= 0: 
                        index = i
                if not right:
                    if q1 < 0:
                        index = i

        return (np.array([valid_solutions[index]['q1'], valid_solutions[index]['q2'], 0.0, 0.0]), np.array([valid_solutions[index]['u']]))
        
    def compute_equilibrium_manifold(self, n_points: int = 50, q_max: float = np.pi) -> dict:
        """
        Compute the equilibrium manifold by fixing q1 and q2 over specified ranges.
        
        Note: This function sweeps positions (q1 and q2) and calculates the required 
        torque (u). It does not sweep 'u' directly as the solver is position-based.

        Parameters
        ----------
        n_points : int, optional
            Number of points to sample in each range (default is 50).
        q_max : float, optional
            Maximum absolute value for joint angles (default is pi).

        Returns
        -------
        manifold_data : dict
            Dictionary containing equilibrium data for 'q1_fix' and 'q2_fix'.
            Format: {'q1_fix': np.array([[q1, q2, u], ...]), ...}
        """
        q_range = np.linspace(-q_max, q_max, n_points)
        
        # We only initialize lists for q1 and q2 sweeps
        res = {'q1_fix': [], 'q2_fix': []}

        # 1. Sweep q1 -> Find q2 and required u
        for val in q_range:
            try:
                # The new signature returns (q1, q2, u) directly
                x_eq1, u_eq1 = self.find_equilibrium('q1', value_fixed=val, up=False)
                res['q1_fix'].append([x_eq1[0], x_eq1[1], u_eq1[0]])
                x_eq2, u_eq2 = self.find_equilibrium('q1', value_fixed=val, up=True)
                res['q1_fix'].append([x_eq2[0], x_eq2[1], u_eq2[0]])
            except Exception as e:
                # Useful to catch empty valid_solutions list or solver errors
                pass

        # 2. Sweep q2 -> Find q1 and required u
        for val in q_range:
            try:
                # The new signature returns (q1, q2, u) directly
                x_eq1, u_eq1 = self.find_equilibrium('q2', value_fixed=val, right=False)
                res['q2_fix'].append([x_eq1[0], x_eq1[1], u_eq1[0]])
                x_eq2, u_eq2 = self.find_equilibrium('q2', value_fixed=val, right=True)
                res['q2_fix'].append([x_eq2[0], x_eq2[1], u_eq2[0]])
            except Exception as e:
                pass

        # Convert lists to numpy arrays for easier plotting
        for key in res:
            res[key] = np.array(res[key])
                
        return res
    
    def draw_pendulums(self, eq_pair1: tuple, eq_pair2: tuple, title: str = "", plot_flag: bool = True, folder: str = None) -> None:
        """
        Visualize the double pendulum at the equilibrium point.
        
        Parameters
        ----------
        x_eq : np.ndarray
            Equilibrium state [q1, q2, 0, 0].
        u_eq : float
            Equilibrium input torque.
        title : str, optional
            Title for the plot (default is empty string).
        plot_flag : bool, optional
            If True, display the plot (default is True).
        folder : str, optional
            If provided, save the plot to the specified folder (default is None).
        """
        # Unpack equilibrium pairs
        x_eq1, u_eq1 = eq_pair1
        x_eq2, u_eq2 = eq_pair2

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        self.draw_frame(x_eq1, ax[0])
        ax[0].set_title(f"u = {u_eq1[0]:.2f} Nm, q1 = {x_eq1[0]:.2f} rad, q2 = {x_eq1[1]:.2f} rad")
        self.draw_frame(x_eq2, ax[1])
        ax[1].set_title(f"u = {u_eq2[0]:.2f} Nm, q1 = {x_eq2[0]:.2f} rad, q2 = {x_eq2[1]:.2f} rad")
        
        # Title for the two subplots
        fig.suptitle(f"{title.replace('_', ' ').title()}", fontsize=16, fontweight='bold')

        # Show plot if requested
        if plot_flag:
            plt.show()

        # Save plot if folder is provided
        if folder is not None:
            fig.savefig(f"{folder}/double_pendulum_equilibria.png", dpi=300)

    def draw_pendulum_envelope(self, eq_pair1: tuple, eq_pair2: tuple, xx_ref: np.ndarray, title: str = "", plot_flag: bool = True, folder: str = None) -> None:
        """
        Visualize the envelope of the double pendulum trajectory.
        
        Parameters
        ----------
        eq_pair1 : tuple
            First equilibrium pair (x_eq, u_eq).
        eq_pair2 : tuple
            Second equilibrium pair (x_eq, u_eq).
        xx_ref : np.ndarray
            Reference trajectory states over time (shape: [ns, N]).
        title : str, optional
            Title for the plot (default is empty string).
        plot_flag : bool, optional
            If True, display the plot (default is True).
        folder : str, optional
            If provided, save the plot to the specified folder (default is None).
        """
        n_mid_pendulum = 15  # Increased slightly for smoother motion look
        indices = np.linspace(0, xx_ref.shape[1]-1, n_mid_pendulum+2, dtype=int)

        # Estrazione x_eq e u_eq dai pair
        x_eq1, u_eq1 = eq_pair1
        x_eq2, u_eq2 = eq_pair2

        # Figure and Axes
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Draw initial equilibrium frame
        self.draw_frame(xx_ref[:, indices[0]], ax=ax, alpha=1.0)
        p1_s, p2_s, _, _ = self._get_forward_kinematics(xx_ref[:, indices[0]], com=True)
        ax.text(p2_s[0], p2_s[1], "  Start", fontsize=12, fontweight='bold', color='green', verticalalignment='center')

        # Draw ghost frames for the envelope
        for idx in indices[1:-1]:
            self.draw_frame(xx_ref[:, idx], ax=ax, alpha=0.15)

        # Draw final equilibrium frame
        self.draw_frame(xx_ref[:, indices[-1]], ax=ax, alpha=1.0)
        p1_e, p2_e, _, _ = self._get_forward_kinematics(xx_ref[:, indices[-1]], com=True)
        ax.text(p2_e[0], p2_e[1], "  End", fontsize=12, fontweight='bold', color='red', verticalalignment='center')
        
        # Title for the plot
        plt.title(f"{title.replace('_', ' ').title()}", fontsize=16, fontweight='bold')

        # Helper
        def fmt_v(v):
            if isinstance(v, (np.ndarray, list)):
                return "[" + ", ".join([f"{x:.2f}" for x in v]) + "]"
            return f"{v:.2f}"

        # Labels for legend
        label1 = rf'$x_{{eq,1}} = {fmt_v(x_eq1)}^\top$' + '\n' + rf'$u_{{eq,1}} = {fmt_v(u_eq1)}$'
        label2 = rf'$x_{{eq,2}} = {fmt_v(x_eq2)}^\top$' + '\n' + rf'$u_{{eq,2}} = {fmt_v(u_eq2)}$'

        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label=label1),
            Line2D([0], [0], color='red', lw=2, label=label2),
            Line2D([0], [0], color='gray', alpha=0.3, lw=1.5, label='Trajectory Envelope')
        ]
        
        # Posizioniamo la legenda fuori dal grafico o in un punto meno invasivo
        ax.legend(handles=legend_elements, loc='upper left',
              fontsize=9, frameon=True, shadow=True, title="Equilibrium Points")

        # Show plot if requested
        if plot_flag:
            plt.show()

        # Save plot if folder is provided
        if folder is not None:
            fig.savefig(f"{folder}/double_pendulum_envelope.png", dpi=300)
        
    def draw_frame(self, x: np.ndarray, ax: plt.Axes, alpha: float = 1.0) -> None:
        """
        Create an animation frame for the double pendulum at state x.
        
        Parameters
        ----------
        x : np.ndarray
            State vector [q1, q2, q1_dot, q2_dot].
        ax : plt.Axes
            Matplotlib Axes to draw on.
        alpha : float, optional
            Transparency level for the drawing (default is 1.0).
        """
        # Get coordinates
        p1, p2, pc1, pc2 = self._get_forward_kinematics(x, com=True)

        # Limit for total pendolum length
        lim_size = (self.l1 + self.l2) * 1.5

        # Plot settings
        ax.set_xlim([-lim_size, lim_size])
        ax.set_ylim([-lim_size, lim_size])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Double Pendulum')

        # === Draw Links and Joints ===
        # Link 1
        ax.plot([0.0, p1[0]], [0.0, p1[1]], color=colors['grey'], linewidth=2.5, label='Link 1', alpha=alpha) # Link 1
        ax.scatter([0.0], [0.0], color=colors['black'], s=20, zorder=5, alpha=alpha) # Joint 1
        ax.scatter([pc1[0]], [pc1[1]], color=colors['red'], s=40, zorder=5, alpha=alpha)   # CM1

        # Link 2
        ax.scatter(p1[0], p1[1], color=colors['black'], s=20, zorder=5, alpha=alpha) # Joint 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors['grey'], linewidth=2.5, label='Link 2', alpha=alpha) # Link 2
        ax.scatter(pc2[0], pc2[1], color=colors['red'], s=40, zorder=5, alpha=alpha) # CM2

    def draw_comparison_frame(self, ax: plt.Axes, x_curr: np.ndarray, x_ref: np.ndarray, x_eq1: np.ndarray, x_eq2: np.ndarray) -> None:
        """
        Draw a comparison frame between current state, reference state, and equilibrium.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib Axes to draw on.
        x_curr : np.ndarray
            Current state vector [q1, q2, q1_dot, q2_dot].
        x_ref : np.ndarray
            Reference state vector [q1, q2, q1_dot, q2_dot].
        x_eq1 : np.ndarray
            Equilibrium state vector for current state [q1, q2, 0, 0].
        x_eq2 : np.ndarray
            Equilibrium state vector for reference state [q1, q2, 0, 0].
        """
        # Get coordinates
        p1_eq1, p2_eq1, pc1_eq1, pc2_eq1 = self._get_forward_kinematics(x_eq1, com=True)
        p1_eq2, p2_eq2, pc1_eq2, pc2_eq2 = self._get_forward_kinematics(x_eq2, com=True) 
        p1_ref, p2_ref, pc1_ref, pc2_ref = self._get_forward_kinematics(x_ref, com=True)
        p1_curr, p2_curr, pc1_curr, pc2_curr = self._get_forward_kinematics(x_curr, com=True)

        # Limit for total plot size
        lim_size = (self.l1 + self.l2) * 1.5

        plots.set_animation_figure(ax, lim_size)

        # === Draw Equilibrium 1 ===
        plots.draw_pendulum_configuration(ax, p1_eq1, p2_eq1, pc1_eq1, pc2_eq1, 'Equilibrium 1')

        # === Draw Equilibrium 2 ===
        plots.draw_pendulum_configuration(ax, p1_eq2, p2_eq2, pc1_eq2, pc2_eq2, 'Equilibrium 2')

        # === Draw Reference State ===
        plots.draw_pendulum_configuration(ax, p1_ref, p2_ref, pc1_ref, pc2_ref, 'Reference')

        # === Draw Current State ===
        plots.draw_pendulum_configuration(ax, p1_curr, p2_curr, pc1_curr, pc2_curr, 'Current')


@njit(cache=True, fastmath=True)
def fast_dynamic_model(x, params):
    m1, m2, l1, l2, lc1, lc2, I1, I2, g, f1, f2, k = params
    
    q1, q2, q1_dot, q2_dot = x[0], x[1], x[2], x[3]
    
    c1 = np.cos(q1)
    s1 = np.sin(q1)
    c2 = np.cos(q2)
    s2 = np.sin(q2)
    s12 = np.sin(q1 + q2)

    # Mass Matrix (M)
    M = np.zeros((2, 2))
    M[0, 0] = I1 + I2 + lc1**2 * m1 + m2 * (l1**2 + 2 * l1 * lc2 * c2 + lc2**2)
    M[0, 1] = I2 + lc2 * m2 * (l1 * c2 + lc2)
    M[1, 0] = M[0, 1]
    M[1, 1] = I2 + m2 * lc2**2
    
    # Coriolis (C)
    C = np.zeros((2, 2))
    C[0, 0] = -l1 * lc2 * m2 * q2_dot * s2
    C[0, 1] = -l1 * lc2 * m2 * (q1_dot + q2_dot) * s2
    C[1, 0] = l1 * lc2 * m2 * q1_dot * s2
    
    # Gravity (G)
    G = np.zeros((2, 1))
    G[0, 0] = g * lc1 * m1 * s1 + g * m2 * l1 * s1 + m2 * g * lc2 * s12
    G[1, 0] = g * m2 * lc2 * s12 + 3 * k * s2 * c2**2
    
    # Friction (F)
    F = np.zeros((2, 2))
    F[0, 0] = f1
    F[1, 1] = f2

    # Invert M manually for 2x2 to avoid overhead
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    M_inv = np.zeros((2, 2))
    M_inv[0, 0] = M[1, 1] / det_M
    M_inv[0, 1] = -M[0, 1] / det_M
    M_inv[1, 0] = -M[1, 0] / det_M
    M_inv[1, 1] = M[0, 0] / det_M
    
    return M, M_inv, C, G, F


#######################################
# Numba-accelerated Jacobian computation
#######################################

@njit(cache=True, fastmath=True)
def fast_jacobian(x: np.ndarray, u_scalar: float, params: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the continuous-time Jacobian matrices A, B for the DoublePendulumFlexible model using Numba acceleration.

    Parameters
    ----------
    x : np.ndarray
        State vector [q1, q2, q1_dot, q2_dot].
    u_scalar : float
        Input torque (scalar).
    params : tuple
        Model parameters tuple.
    
    Returns
    -------
    A : np.ndarray
        Continuous-time state Jacobian matrix.
    B : np.ndarray
        Continuous-time input Jacobian matrix.
    """
    # Unpack parameters
    m1, m2, l1, l2, lc1, lc2, I1, I2, g, f1, f2, k = params
    
    # Extract joint angles and velocities
    q1, q2 = x[0], x[1]
    q1_dot, q2_dot = x[2], x[3]

    # Precompute trig
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)

    # Get Dynamics Matrices
    _, M_inv, C, G, F = fast_dynamic_model(x, params)

    # Helper vectors
    q_dot = np.array([[q1_dot], [q2_dot]])
    T = np.array([[u_scalar], [0.0]])
    
    # --- Compute a22 = M_inv @ (-F - dc_dq_dot) ---
    # dc_dq_dot construction
    term_common = 2 * l1 * lc2 * m2 * s2
    dc_dq_dot = np.zeros((2, 2))
    dc_dq_dot[0, 0] = -q2_dot * term_common
    dc_dq_dot[0, 1] = -(q1_dot + q2_dot) * term_common
    dc_dq_dot[1, 0] = q1_dot * term_common
    # dc_dq_dot[1, 1] is 0.0

    # Calculate inner term for a22
    neg_F_minus_dc = -F - dc_dq_dot
    a22 = M_inv @ neg_F_minus_dc

    # --- Compute a21 term ---
    # Calculate tau (torque balance)
    tau = - (F @ q_dot) - (C @ q_dot) - G + T
    
    # dM_dq1 is zero matrix in original code, so dM_inv_dq_tau_1 is zero.
    # dM_dq2 construction
    dM_dq2 = np.zeros((2, 2))
    term_M2 = -l1 * lc2 * m2 * s2
    dM_dq2[0, 0] = 2 * term_M2
    dM_dq2[0, 1] = term_M2
    dM_dq2[1, 0] = term_M2
    
    # Calculate dM_inv/dq * tau contribution
    # Formula: d(M_inv)/dq * tau = - M_inv * (dM/dq) * M_inv * tau
    temp_vec = M_inv @ tau
    vec_dq2 = dM_dq2 @ temp_vec
    dM_inv_dq_tau_2 = - M_inv @ vec_dq2

    # dM_inv_dq_tau matrix (col 1 is 0 because dM_dq1 is 0)
    dM_inv_dq_tau = np.zeros((2, 2))
    dM_inv_dq_tau[0, 1] = dM_inv_dq_tau_2[0, 0]
    dM_inv_dq_tau[1, 1] = dM_inv_dq_tau_2[1, 0]

    # dc_dq construction
    dc_dq = np.zeros((2, 2))
    term_c2 = l1 * lc2 * m2 * c2
    dc_dq[0, 1] = (-2 * q1_dot * q2_dot - q2_dot**2) * term_c2
    dc_dq[1, 1] = (q1_dot**2) * term_c2

    # dG_dq construction
    dG_dq = np.zeros((2, 2))
    term_g_c1 = g * (lc1 * m1 + m2 * l1) * c1
    term_g_c12 = m2 * g * lc2 * c12
    
    dG_dq[0, 0] = term_g_c1 + term_g_c12
    dG_dq[0, 1] = term_g_c12
    dG_dq[1, 0] = term_g_c12
    dG_dq[1, 1] = term_g_c12 + 3 * k * (c2**2 - 2 * s2**2) * c2

    # Final a21
    inner_a21 = -dc_dq - dG_dq
    a21 = dM_inv_dq_tau + (M_inv @ inner_a21)

    # --- Assemble A (4x4) ---
    A = np.zeros((4, 4))
    # a11 is zeros (already done)
    # a12 is identity
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    # a21
    A[2, 0], A[2, 1] = a21[0, 0], a21[0, 1]
    A[3, 0], A[3, 1] = a21[1, 0], a21[1, 1]
    # a22
    A[2, 2], A[2, 3] = a22[0, 0], a22[0, 1]
    A[3, 2], A[3, 3] = a22[1, 0], a22[1, 1]

    # --- Assemble B (4x1) ---
    B = np.zeros((4, 1))
    # b1 is zeros
    # b2 is M_inv @ [1, 0]. This is effectively the first column of M_inv
    B[2, 0] = M_inv[0, 0]
    B[3, 0] = M_inv[1, 0]

    return A, B