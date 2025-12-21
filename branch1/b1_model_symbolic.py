import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class DoublePendulumFlexibleSymbolic:
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

        self.ns = 4
        self.ni = 1

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
        G_paper = sp.Matrix([
            [self.g * self.lc1 * self.m1 * s1 + self.g * self.m2 * self.l1 * s1 + self.lc2 * s12],
            [self.g * self.m2 * self.lc2 * s12 + 3 * self.k * s2 * c2**2] 
        ])

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