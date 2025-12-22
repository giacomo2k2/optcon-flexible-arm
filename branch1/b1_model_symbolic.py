#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         model_symbolic.py
# Description:  Symbolic model of the flexible double pendulum.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         December 2025
#########################################################################################

#######################################
# Imports
#######################################
import sympy as sp


#######################################
# Symbolic Model Class
#######################################
class DoublePendulumFlexibleSymbolic:
    """Symbolic model of the flexible double pendulum."""
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

    def dynamic_model(self, x):
        # Unpack state
        q1, q2, q1_dot, q2_dot = x[0], x[1], x[2], x[3]
        
        # Precompute sines and cosines
        c1 = sp.cos(q1)
        s1 = sp.sin(q1)
        c2 = sp.cos(q2)
        s2 = sp.sin(q2)
        c12 = sp.cos(q1+q2)
        s12 = sp.sin(q1+q2)

        # Inertia matrix
        M = sp.Matrix([
            [self.I1 + self.I2 + self.lc1**2 * self.m1 + self.m2 * (self.l1**2 + 2 * self.l1 * self.lc2 * c2 + self.lc2**2), self.I2 + self.lc2 * self.m2 * (self.l1 * c2 + self.lc2)],
            [self.I2 + self.lc2 * self.m2 * (self.l1 * c2 + self.lc2), self.I2 + self.m2 * self.lc2 ** 2]
        ])
    
        # Coriolis and centrifugal matrix
        C = sp.Matrix([
            [- self.l1 * self.lc2 * self.m2 * q2_dot * s2, - self.l1 * self.lc2 * self.m2 * (q1_dot + q2_dot) * s2],
            [  self.l1 * self.lc2 * self.m2 * q1_dot * s2, 0.0]
        ])

        # Gravity vector
        G  = sp.Matrix([
            [self.g * self.lc1 * self.m1 * s1 + self.g * self.m2 * self.l1 * s1 + self.m2 * self.g * self.lc2 * s12],
            [self.g * self.m2 * self.lc2 * s12 + 3 * self.k * s2 * c2**2]
        ])

        # Friction matrix
        F = sp.Matrix([
            [self.f1, 0.0],
            [0.0, self.f2]
        ])
        
        # Intertia Matrix inverse
        M_inv = sp.Matrix.inv(M)
        
        return M, M_inv, C, G, F

    def dynamics(self, x, u=0.0):
        q1_dot = x[2]
        q2_dot = x[3]
        
        q_dot = sp.Matrix([
                            [q1_dot], 
                            [q2_dot]
                           ])
        
        M, M_inv, C, G, F = self.dynamic_model(x)

        T = sp.Matrix([
                        [u],
                        [0.0]
                    ])

        # Equation of Motion: M * q_ddot + C * q_dot + F * q_dot + G = T
        rhs = T - F @ q_dot - C @ q_dot - G
        
        q_ddot = M_inv @ rhs 
        x_dot = sp.Matrix([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])
        
        return x_dot
    
    def symbolic_jacobian(self):
        """
        Computes the symbolic Jacobian matrices A and B.

        Returns
        -------
        A_sym : sp.Matrix
            Symbolic state Jacobian matrix.
        B_sym : sp.Matrix
            Symbolic input Jacobian matrix.
        """
        # Symbolic variables
        q1, q2, q1_dot, q2_dot = sp.symbols('q1 q2 q1_dot q2_dot')
        u = sp.symbols('u')
        
        # State and input vectors
        state = sp.Matrix([q1, q2, q1_dot, q2_dot])
        input_var = sp.Matrix([u])
        
        # Symbolic dynamics f(x, u)
        f_expr = self.dynamics(state, u)
        
        # Compute Jacobians
        A_sym = f_expr.jacobian(state)
        B_sym = f_expr.jacobian(input_var)
        
        return A_sym, B_sym
    
    def symbolic_hessians_lambda_product(self):
        """
        Computes the symbolic Hessian matrices of the Hamiltonian with respect to state and input.

        Returns
        -------
        H_xx : sp.Matrix
            Symbolic Hessian of the Hamiltonian with respect to state.
        H_uu : sp.Matrix
            Symbolic Hessian of the Hamiltonian with respect to input.
        H_xu : sp.Matrix
            Symbolic mixed Hessian of the Hamiltonian with respect to state and input.
        """
        # Symbolic variables
        q1, q2, q1_dot, q2_dot = sp.symbols('q1 q2 q1_dot q2_dot')
        u = sp.symbols('u')
        l1, l2, l3, l4 = sp.symbols('l1 l2 l3 l4') 
        
        # State, lambda and input vectors
        state = sp.Matrix([q1, q2, q1_dot, q2_dot])
        lmbd = sp.Matrix([l1, l2, l3, l4])
        input_var = sp.Matrix([u])
        
        # Continous dynamics
        f_expr = self.dynamics(state, u)
        
        # Scalar product
        H_dyn = f_expr.dot(lmbd)
        
        # Grad xx
        H_xx = sp.hessian(H_dyn, state)
        
        # Grad uu
        H_uu = sp.hessian(H_dyn, input_var)
        
        # Grad xu
        grad_x = sp.Matrix([sp.diff(H_dyn, s) for s in state])
        H_xu = grad_x.jacobian(input_var)
        
        return H_xx, H_uu, H_xu