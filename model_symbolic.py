import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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
    'm1': 2.0, 'm2': 2.0, 'l1': 1.5, 'l2': 1.5, 'lc1': 0.75, 'lc2': 0.75,   
    'I1': 1.5, 'I2': 1.5, 'g': 9.81, 'f1': 1.0, 'f2': 1.0, 'k': 0.6        
}

PARAM_SET_3 = {
    'm1': 1.5, 'm2': 1.5, 'l1': 2.0, 'l2': 2.0, 'lc1': 1.0, 'lc2': 1.0, 
    'I1': 2.0, 'I2': 2.0, 'g': 9.81, 'f1': 1.0, 'f2': 1.0, 'k': 0.7        
}


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

    def dynamic_model(self, x, backend=np):
        """
        Computes M, C, G, F matrices. 
        'backend' can be numpy (for simulation) or sympy (for Jacobians).
        """
        # Unpack state
        q1, q2, q1_dot, q2_dot = x[0], x[1], x[2], x[3]
        
        # Select math functions based on backend
        cos = backend.cos
        sin = backend.sin
        
        # Precompute sines and cosines
        c1 = cos(q1)
        s1 = sin(q1)
        c2 = cos(q2)
        s2 = sin(q2)
        c12 = cos(q1+q2)
        s12 = sin(q1+q2)

        # Helper to create matrices consistent with the backend
        if backend == sp:
            mat_constructor = sp.Matrix
        else:
            mat_constructor = np.array


        # TODO: check if all terms are correct
        # Mass matrix
        M = mat_constructor([
            [self.I1 + self.I2 + self.lc1**2 * self.m1 + self.m2 * (self.l1**2 + 2 * self.l1 * self.lc2 * c2 + self.lc2**2), self.I2 + self.lc2 * self.m2 * (self.l1 * c2 + self.lc2)],
            [self.I2 + self.lc2 * self.m2 * (self.l1 * c2 + self.lc2), self.I2 + self.m2 * self.lc2 ** 2]
        ])
    
        # Coriolis and centrifugal matrix
        C = mat_constructor([
            [- self.l1 * self.lc2 * self.m2 * q2_dot * s2, - self.l1 * self.lc2 * self.m2 * (q1_dot + q2_dot) * s2],
            [  self.l1 * self.lc2 * self.m2 * q1_dot * s2, 0.0]
        ])

        # Gravity vector
        G_paper = mat_constructor([
            [self.g * self.lc1 * self.m1 * s1 + self.g * self.m2 * self.l1 * s1 + self.lc2 * s12],
            [self.g * self.m2 * self.lc2 * s12 + 3 * self.k * s2 * c2**2] 
        ])

        G  = mat_constructor([
            [self.g * self.lc1 * self.m1 * s1 + self.g * self.m2 * self.l1 * s1 + self.m2 * self.g * self.lc2 * s12],
            [self.g * self.m2 * self.lc2 * s12 + 3 * self.k * s2 * c2**2]
        ])

        # Friction matrix
        F = mat_constructor([
            [self.f1, 0.0],
            [0.0, self.f2]
        ])

        M_inv = np.linalg.inv(M) if backend == np else M.inv()
        
        return M, M_inv, C, G, F

    def dynamics(self, x, u=0.0, backend=np):
        q1_dot = x[2]
        q2_dot = x[3]
        
        # Handle vector creation based on backend
        if backend == sp:
            q_dot = sp.Matrix([q1_dot, q2_dot])
            T = sp.Matrix([u, 0.0])
        else:
            q_dot = np.array([[q1_dot], [q2_dot]])
            T = np.array([[u], [0.0]])
        
        # Get matrices using the specific backend
        M, M_inv, C, G, F = self.dynamic_model(x, backend=backend)

        # Equation of Motion: M * q_ddot + C * q_dot + F * q_dot + G = T
        rhs = T - F @ q_dot - C @ q_dot - G
        
        # Solve for q_ddot
        if backend == sp:
            q_ddot = M_inv @ rhs 
            
            # Construct state derivative x_dot = [q_dot; q_ddot]
            x_dot = sp.Matrix([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])
        else:
            # Numerical solution
            q_ddot = M_inv @ rhs
            x_dot = np.concatenate([q_dot.flatten(), q_ddot.flatten()])
        
        return x_dot
    

    def symbolic_jacobian(self, x_val=None, u_val=0.0):
        # Symbolic variables
        q1, q2, q1_dot, q2_dot = sp.symbols('q1 q2 q1_dot q2_dot')
        u = sp.symbols('u')
        
        # State vector and Input vector (SymPy Matrices)
        state = sp.Matrix([q1, q2, q1_dot, q2_dot])
        input_var = sp.Matrix([u])
        
        # Symbolic dynamics f(x, u)
        # We pass the symbols as a list/Matrix, and specify backend=sp
        f_expr = self.dynamics(state, u, backend=sp)
        
        # Compute Jacobians
        # A = df/dx, B = df/du
        A_sym = f_expr.jacobian(state)
        B_sym = f_expr.jacobian(input_var)

        # Substitute numerical values if provided
        if x_val is not None:
            subs_map = {'q1': x_val[0], 'q2': x_val[1], 'q1_dot': x_val[2], 'q2_dot': x_val[3], 'u': u_val}
            A_sym = A_sym.evalf(subs=subs_map)
            B_sym = B_sym.evalf(subs=subs_map)
        
        # Convert to numpy arrays
        A_sym = np.array(A_sym).astype(np.float64)
        B_sym = np.array(B_sym).astype(np.float64)
        
        return A_sym, B_sym
    
    def animation(self, x: np.ndarray, ax: plt.Axes) -> None:
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

