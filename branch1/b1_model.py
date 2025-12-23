#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         model.py
# Description:  Model class for the flexible double pendulum system.
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

plt.rcParams['font.family'] = 'serif'

# Solver
import b1_solver as solver

# Color palette
from b1_plots import colors

# Plots
import b1_plots as plots

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

        self.ns = 4 # State dimension
        self.ni = 1 # Input dimension
        self.dt = 0.01 # Default time step
        self.discr_method = 'fe'  # 'fe' or 'rk4'

    def random_init(self) -> np.ndarray:
        """
        Generate a random initial state for the double pendulum.

        Returns
        -------
            q_init : np.ndarray
                     Random initial state [q1, q2, q1_dot, q2_dot]
        """
        q1 = np.random.uniform(-np.pi, np.pi)
        q2 = np.random.uniform(-np.pi, np.pi)
        q1_dot = np.random.uniform(-1.0, 1.0)
        q2_dot = np.random.uniform(-1.0, 1.0)
        return np.array([q1, q2, q1_dot, q2_dot])

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

        # Angular velocity vector
        q_dot = np.array([
                            [q1_dot],
                            [q2_dot]
                         ])

        # Extract dynamic model matrices
        M, M_inv, C, G, F = self.dynamic_model(x)

        if isinstance(u, np.ndarray):
            u = u.item()  # Convert to scalar if u is an array

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
    
    # TODO: remove this in next versions. Symbolic jacobian is used
    def discrete_jacobian(self, x: np.ndarray, u: np.ndarray, method: str = 'fe') -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the discrete-time Jacobian matrices A and B.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.ndarray
            Current input vector.
        method : str, optional
            Discretization method ('fe' for Forward Euler, 'rk4' for Runge-Kutta 4), by default 'fe'.
        
        Returns
        -------
        A_discr : np.ndarray
            Discrete-time state Jacobian matrix.
        B_discr : np.ndarray
            Discrete-time input Jacobian matrix.
        """
        if method == 'fe':            
            return self._discrete_jacobian_fe(x, u)
        elif method == 'rk4':            
            return self._discrete_jacobian_rk4(x, u)
        else:
            raise ValueError(f"Discretization method '{method}' not supported. Use 'fe' or 'rk4'.")
        
    def _discrete_jacobian_fe(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the discrete-time Jacobian matrices A and B using Forward Euler integration.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.ndarray
            Current input vector.
        
        Returns
        -------
        A_discr : np.ndarray
            Discrete-time state Jacobian matrix.
        B_discr : np.ndarray
            Discrete-time input Jacobian matrix.
        """
        A_cont, B_cont = self.jacobian(x, u)

        A_discr = np.eye(self.ns) + A_cont * self.dt
        B_discr = B_cont * self.dt

        return A_discr, B_discr
    
    def _discrete_jacobian_rk4(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the discrete-time Jacobian matrices A and B using RK4 integration.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.ndarray
            Current input vector.
        
        Returns
        -------
        A_discr : np.ndarray
            Discrete-time state Jacobian matrix.
        B_discr : np.ndarray
            Discrete-time input Jacobian matrix.
        """
        epsilon = 1e-6 

        # Compute next state
        x_next_base = self.discrete_dynamics(x, u)

        # Initialize A_discr and B_discr
        A_discr = np.zeros((self.ns, self.ns))
        B_discr = np.zeros((self.ns, self.ni))

        # --- Compute A_discr (Derivative w.r.t. x) ---
        for j in range(self.ns):
            x_perturbed = x.copy()
            x_perturbed[j] += epsilon
            
            x_next_perturbed = self.discrete_dynamics(x_perturbed, u)
            
            # TODO: check this discretization method (with epsilon sufficently small should be ok)
            A_discr[:, j] = (x_next_perturbed - x_next_base) / epsilon

        # --- Compute B_discr (Derivative w.r.t. u) ---
        for j in range(self.ni): 
            u_perturbed = u.copy()
            u_perturbed[j] += epsilon
            
            x_next_perturbed = self.discrete_dynamics(x, u_perturbed)
            
            B_discr[:, j] = (x_next_perturbed - x_next_base) / epsilon
        
        return A_discr, B_discr
        
    def find_equilibrium(self, fix_type: str, value: float) -> tuple[np.ndarray, float]:
        """
        Find the equilibrium state and input fixing one of (q1, q2, u).
        
        Parameters
        ----------
        fix_type : str
            One of 'q1', 'q2', or 'u'.
        value : float
            The fixed value for the chosen variable.
            
        Returns
        -------
        x_eq : np.ndarray
            Equilibrium state [q1, q2, 0, 0].
        u_eq : float
            Equilibrium input torque.
        """
        def equations(vars, fix_type, value):
            if fix_type == 'q1':
                q1 = value
                q2, u = vars
            elif fix_type == 'q2':
                q2 = value
                q1, u = vars
            elif fix_type == 'u':
                u = value
                q1, q2 = vars
            else:
                raise ValueError("fix_type must be one of 'q1', 'q2', or 'u'.")
            
            x_temp = np.array([q1, q2, 0.0, 0.0])
            G = self.dynamic_model(x_temp)[3].flatten()

            eq1 = G[0] - u
            eq2 = G[1]

            return [eq1, eq2]
        
        initial_guess = [0.0, 0.0]
        sol, info, ier, msg = fsolve(equations, initial_guess, args=(fix_type, value), full_output=True)
        
        if ier != 1:
            raise ValueError(f"fsolve did not converge: {msg}")

        if fix_type == 'q1':
            return np.array([value, sol[0], 0.0, 0.0]), float(sol[1])
        elif fix_type == 'q2':
            return np.array([sol[0], value, 0.0, 0.0]), float(sol[1])
        else:
            return np.array([sol[0], sol[1], 0.0, 0.0]), value
        
    def plot_equilibrium_point(self, x_eq: np.ndarray, u_eq: float, title: str = "") -> None:
        """ Visualize the double pendulum at the equilibrium point.
        
        Parameters
        ----------
        x_eq : np.ndarray
            Equilibrium state [q1, q2, 0, 0].
        u_eq : float
            Equilibrium input torque.
        title : str, optional
            Title for the plot (default is empty string).
        """
        fig, ax = plt.subplots(figsize=(6,6))
        self.animation(x_eq, ax)
        ax.set_title(f"Equilibria {title}: u = {u_eq:.2f} Nm\nq1 = {x_eq[0]:.2f}, q2 = {x_eq[1]:.2f}")
        plt.show()
        
    def compute_equilibrium_manifold(self, n_points: int = 50, q_max: float = np.pi, u_max: float = 20.0) -> dict:
        """
        Compute the equilibrium manifold by fixing q1, q2, and u over specified ranges.

        Parameters
        ----------
        n_points : int, optional
            Number of points to sample in each range (default is 50).
        q_max : float, optional
            Maximum absolute value for joint angles q1 and q2 (default is pi).
        u_max : float, optional
            Maximum absolute value for input torque u (default is 20.0).

        Returns
        -------
        manifold_data : dict
            Dictionary containing equilibrium data for 'q1_fix', 'q2_fix', and 'u_fix'.
        """
        q_range = np.linspace(-q_max, q_max, n_points)
        u_range = np.linspace(-u_max, u_max, n_points)

        res = {'q1_fix': [], 'q2_fix': [], 'u_fix': []}

        print(f"Computing manifold for {self.__class__.__name__}...")

        for val in q_range:
            # 1. Fix q1 --> find q2, u
            try:
                x_eq, u_eq = self.find_equilibrium('q1', val)
                res['q1_fix'].append([val, x_eq[1], u_eq])
            except: pass

            # 2. Fix q2 --> find q1, u
            try:
                x_eq, u_eq = self.find_equilibrium('q2', val)
                res['q2_fix'].append([x_eq[0], val, u_eq])
            except: pass

        for val in u_range:
            # 3. Fix u --> find q1, q2
            try:
                x_eq, u_eq = self.find_equilibrium('u', val)
                q_wrapped = np.arctan2(np.sin(x_eq[:2]), np.cos(x_eq[:2]))
                res['u_fix'].append([q_wrapped[0], q_wrapped[1], val])
            except: pass

        # Convert lists to numpy arrays
        for key in res:
            res[key] = np.array(res[key])
                
        return res
    
    def _get_forward_kinematics(self, x:np.ndarray, com:bool = False) -> tuple[np.ndarray, np.ndarray]:
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
        q1 = x[0]
        q2 = x[1]

        p1 = np.array([self.l1 * np.sin(q1), -self.l1 * np.cos(q1)])
        p2 = p1 + np.array([self.l2 * np.sin(q1 + q2), -self.l2 * np.cos(q1 + q2)])

        if com:
            pc1 = np.array([self.lc1 * np.sin(q1), -self.lc1 * np.cos(q1)])
            pc2 = np.array([self.l1 * np.sin(q1) + self.lc2 * np.sin(q1 + q2), -self.l1 * np.cos(q1) - self.lc2 * np.cos(q1 + q2)])
            return p1, p2, pc1, pc2
        else:
            return p1, p2

    def draw_frame(self, x: np.ndarray, ax: plt.Axes) -> None:
        """
        Create an animation frame for the double pendulum at state x.
        
        Parameters
        ----------
        x : np.ndarray
            State vector [q1, q2, q1_dot, q2_dot].
        ax : plt.Axes
            Matplotlib Axes to draw on.
        """

        # TODO: consider flexible arm for animation
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
        ax.plot([0.0, p1[0]], [0.0, p1[1]], color=colors['grey'], linewidth=2.5, label='Link 1') # Link 1
        ax.scatter([0.0], [0.0], color=colors['black'], s=20, zorder=5) # Joint 1
        ax.scatter([pc1[0]], [pc1[1]], color=colors['red'], s=40, zorder=5)   # CM1

        # Link 2
        ax.scatter(p1[0], p1[1], color=colors['black'], s=20, zorder=5) # Joint 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors['grey'], linewidth=2.5, label='Link 2') # Link 2
        ax.scatter(pc2[0], pc2[1], color=colors['red'], s=40, zorder=5) # CM2

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