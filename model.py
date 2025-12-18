import numpy as np
import matplotlib.pyplot as plt

from solver import solver_step

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

# Q1: first row in G is not correct, we wrote m2*g*lc2*sin(q1+q2)


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
        G_paper = np.array([
                           [self.g * self.lc1 * self.m1 * s1 + self.g * self.m2 * self.l1 * s1 + self.lc2 * s12],
                           [self.g * self.m2 * self.lc2 * s12 + 3 * self.k * s2 * c2**2]
                        ])
        
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

        # Input Torque vector
        T = np.array([
                        u,
                        np.array([0.0])
                    ])
        
        # Right-hand side of the dynamics equation
        rhs = - F @ q_dot - C @ q_dot - G + T
        
        # Angular accelerations
        q_ddot = M_inv @ rhs

        # State at next time step
        x_dot = np.concatenate([q_dot.flatten(), q_ddot.flatten()])
        
        return x_dot    
    
    def _discrete_jacobian_rk4(self, x: np.ndarray, u: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the discrete-time Jacobian matrices A and B using RK4 integration.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.ndarray
            Current input vector.
        dt : float
            Time step for discretization.
        
        Returns
        -------
        A_discr : np.ndarray
            Discrete-time state Jacobian matrix.
        B_discr : np.ndarray
            Discrete-time input Jacobian matrix.
        """
        epsilon = 1e-6 

        # Compute next state
        x_next_base = solver_step(x, u, dt, method='rk4')

        # Initialize A_discr and B_discr
        A_discr = np.zeros((self.ns, self.ns))
        B_discr = np.zeros((self.ns, self.ni))

        # --- Compute A_discr (Derivative w.r.t. x) ---
        for j in range(self.ns):
            x_perturbed = x.copy()
            x_perturbed[j] += epsilon
            
            x_next_perturbed = solver_step(x_perturbed, u, dt, method='rk4')
            
            A_discr[:, j] = (x_next_perturbed - x_next_base) / epsilon

        # --- Compute B_discr (Derivative w.r.t. u) ---
        for j in range(self.ni): 
            u_perturbed = u.copy()
            u_perturbed[j] += epsilon
            
            x_next_perturbed = solver_step(x, u_perturbed, dt, method='rk4')
            
            B_discr[:, j] = (x_next_perturbed - x_next_base) / epsilon
        
        return A_discr, B_discr
    
    def discrete_jacobian(self, x: np.ndarray, u: np.ndarray, dt: float, method: str = 'fe') -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the discrete-time Jacobian matrices A and B.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.ndarray
            Current input vector.
        dt : float
            Time step for discretization.
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
            A, B = self.jacobian(x, u)
            
            A_discr = np.eye(self.ns) + A * dt
            B_discr = B * dt
            
            return A_discr, B_discr
        
        elif method == 'rk4':            
            return self._discrete_jacobian_rk4(x, u, dt)
            
        else:
            raise ValueError(f"Discretization method '{method}' not supported. Use 'fe' or 'rk4'.")

    def discrete_dynamics(self, x: np.ndarray, u: np.ndarray, dt: float, method: str = 'fe') -> np.ndarray:
        """
        Compute the discrete-time dynamics of the Double Pendulum.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.ndarray
            Current input vector.
        dt : float
            Time step for discretization.
        method : str, optional
            Discretization method ('fe', 'rk4'), by default 'fe'.

        Returns
        -------
        x_next : np.ndarray
            State vector at the next time step.
        """
        if method == 'fe':
            x_dot = self.dynamics(x, u)
            x_next = x + x_dot * dt
            return x_next
        elif method == 'rk4':
            return solver_step(self, x, u, dt, method='rk4')
        else:
            raise ValueError(f"Discretization method '{method}' not supported. Use 'fe' or 'rk4'.")

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
        a11 = np.zeros((2, 2))
        a12 = np.eye(2)

        q1 = x[0]
        q2 = x[1]
        q1_dot = x[2]
        q2_dot = x[3]

        q_dot = np.array([
                            [q1_dot],
                            [q2_dot]
                         ])

        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s2 = np.sin(q2)
        c2 = np.cos(q2)

        c12 = np.cos(q1 + q2)
        s12 = np.sin(q1 + q2)

        M, M_inv, C, G, F = self.dynamic_model(x)

        T = np.array([
                        u ,
                        np.array([0.0])
                    ])

        dc_dq_dot = 2*self.l1 * self.lc2 * self.m2 * s2 * np.array([
                                                                    [-q2_dot, -q1_dot - q2_dot],
                                                                    [q1_dot, 0.0]
                                                                ])

        
        a22 = M_inv @ (-F - dc_dq_dot)

        

        dM_dq1 = np.array([
            [0, 0],
            [0, 0]
        ])

        dM_dq2 = np.array([
                            [- 2 * self.l1 * self.lc2 * self.m2 * s2, - self.l1 * self.lc2 *self.m2 * s2],
                            [- self.l1 * self.lc2 *self.m2 * s2, 0.0]
                        ])
        

        tau = - F @ q_dot - C @ q_dot - G + T

        dM_inv_dq_tau_1 = - M_inv @ dM_dq1 @ M_inv @ tau
        dM_inv_dq_tau_2 = - M_inv @ dM_dq2 @ M_inv @ tau

        dM_inv_dq_tau = np.array([
            [dM_inv_dq_tau_1[0, 0], dM_inv_dq_tau_2[0, 0]],
            [dM_inv_dq_tau_1[1, 0], dM_inv_dq_tau_2[1, 0]]
        ])

        dc_dq = self.l1 * self.lc2 * self.m2 * c2 * np.array([
                                                                [0.0, -2*q1_dot*q2_dot - q2_dot**2],
                                                                [0.0, q1_dot**2]
                                                            ])
            

        dG_dq = np.array([
                           [self.g * self.lc1 * self.m1 * c1 + self.g * self.m2 * self.l1 * c1 + self.m2 * self.g * self.lc2 * c12, self.m2 * self.g * self.lc2 * c12],
                           [self.m2 * self.g * self.lc2 * c12, self.m2 * self.g * self.lc2 * c12 + 3 * self.k * (c2**2 - 2*s2**2) * c2]
                        ])
        
        a21 = dM_inv_dq_tau + M_inv @ (- dc_dq - dG_dq)


        A = np.block([
            [a11, a12],
            [a21, a22]
        ])

        b1 = np.zeros((2, 1))
        b2 = M_inv @ np.array([
                                [1.0],
                                [0.0]
                             ])
        
        B = np.vstack((b1, b2))


        return A, B
    
    def equilibrium_input(self, x_eq: np.ndarray) -> float:
        """
        Compute the equilibrium input torque for a given equilibrium state.

        Parameters
        ----------
        x_eq : np.ndarray
            Equilibrium state vector.
        
        Returns
        -------
        u_eq : float
            Equilibrium input torque.
        """
        G = self.dynamic_model(x_eq)[3]

        # The equilibrium input only needs to balance gravity
        u_eq = G[0]

        return u_eq


    def animation(self, x: np.ndarray, ax: plt.Axes) -> None:
        """
        Create an animation frame for the double pendulum at state x.
        
        Parameters
        ----------
        x : np.ndarray
            State vector [q1, q2, q1_dot, q2_dot].
        ax : plt.Axes
            Matplotlib Axes to draw on.
        """
        # Extract joints position
        q1 = x[0]
        q2 = x[1]
        
        # Compute cartesians position (Forward Kinematics)
        p1 = np.array([self.l1 * np.sin(q1), -self.l1 * np.cos(q1)])
        p2 = np.array([self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2), -self.l1 * np.cos(q1) - self.l2 * np.cos(q1 + q2)])

        pc1 = np.array([self.lc1 * np.sin(q1), -self.lc1 * np.cos(q1)])
        pc2 = np.array([self.l1 * np.sin(q1) + self.lc2 * np.sin(q1 + q2), -self.l1 * np.cos(q1) - self.lc2 * np.cos(q1 + q2)])

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
        ax.plot([0.0, p1[0]], [0.0, p1[1]], 'grey', linewidth=2.5, label='Link 1')
        ax.scatter([0.0], [0.0], color='black', s=20, zorder=5) # Base pivot
        ax.scatter([pc1[0]], [pc1[1]], color='red', s=40, zorder=5)   # CM1

        # Link 2
        ax.scatter(p1[0], p1[1], color='black', s=20, zorder=5)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', linewidth=2.5, label='Link 2')
        ax.scatter(pc2[0], pc2[1], color='red', s=40, zorder=5)

