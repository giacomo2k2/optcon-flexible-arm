import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, m1, m2, l1, l2, lc1, lc2, I1, I2, g, f1, f2, k):
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
        q1 = np.random.uniform(-np.pi, np.pi)
        q2 = np.random.uniform(-np.pi, np.pi)
        q1_dot = np.random.uniform(-1.0, 1.0)
        q2_dot = np.random.uniform(-1.0, 1.0)
        return np.array([q1, q2, q1_dot, q2_dot])

    def dynamic_model(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        # Mass matrix
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
        
        M_inv = np.linalg.inv(M)

        return M, M_inv, C, G, F

    def dynamics(self, x: np.ndarray, u: float = 0.0) -> np.ndarray:
        q1_dot = x[2]
        q2_dot = x[3]
        q_dot = np.array([
                            [q1_dot],
                            [q2_dot]
                         ])
        
        M, M_inv, C, G, F = self.dynamic_model(x)

        T = np.array([
                        [u] ,
                        [0.0]
                    ])
        
        rhs = - F @ q_dot - C @ q_dot - G + T
        
        q_ddot = M_inv @ rhs

        # State at next time step
        x_dot = np.concatenate([q_dot.flatten(), q_ddot.flatten()])
        
        return x_dot    

    def jacobian(self, x: np.ndarray, u:float=0.0) -> np.ndarray:

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
                        [u] ,
                        [0.0]
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
        G = self.dynamic_model(x_eq)[3]

        u_eq = G[0]

        return u_eq


    def animation(self, x: np.ndarray   , ax: plt.Axes) -> None:
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

