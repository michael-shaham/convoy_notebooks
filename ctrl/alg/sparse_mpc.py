from alg.controller_base import ControllerBase
from dyn.linear_long import LinearLong

import copy
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag

class SparseMPC(ControllerBase):
    """
    Predecessor follower linear MPC controller.

    Parameters:
        Q: move suppression cost term
        Q_f: terminal cost  
        R: input cost
        dyn: dynamics base class based on DynamicsBase, linear
        u_min/max: input bounds (assume box constraints)
        x_min/max: state bounds (assume box constraints)
        H: horizon
    """
    def __init__(self,
                 Q: np.ndarray, Q_f: np.ndarray, R: np.ndarray,
                 dyn: LinearLong,
                 u_min: np.ndarray, u_max: np.ndarray,
                 x_min: np.ndarray, x_max: np.ndarray,
                 H: int):
        n = dyn.n
        m = dyn.m
        y_dim = H*(m+n)
        self.n = n
        self.m = m
        self.H = H

        self.y_dim = y_dim 

        # cost function: assume Q, Q_f, R all diagonal
        Q_bar = np.zeros((y_dim, y_dim))
        Q_tilde = block_diag(R, Q)

        Q_bar[:, :] = np.kron(np.eye(H), Q_tilde)
        Q_bar[-n:, -n:] = Q_f 
        self.Q_bar = csr_matrix(Q_bar)

        # equality constraint: Ay = b 
        A_i = dyn.Ad
        self.A_i = A_i
        B_i = dyn.Bd
        self.B_i = B_i
        I = np.eye(n)

        AB_tilde = np.zeros((n, 2*n + m))
        AB_tilde[:, :n] = -A_i
        AB_tilde[:, n:n+m] = -B_i
        AB_tilde[:, -n:] = I
        self.A = np.zeros((n*(H+1), y_dim))
        self.A[:n, :m] = -B_i 
        self.A[:n, m:m+n] = I
        self.A[-n:, -n:] = I

        j = m
        for i in range(n, n*H, n):
            self.A[i:i+n, j:j+2*n+m] = AB_tilde 
            j += n+m

        # inequality constraint: Cy <= d
        C = np.zeros((y_dim*2, y_dim))
        I_tilde = np.zeros((2*(m+n), m+n))
        I_tilde[:m, :m] = np.eye(m)
        I_tilde[m:2*m, :m] = -1 * np.eye(m)
        I_tilde[-2*n:-n, -n:] = np.eye(n)
        I_tilde[-n:, -n:] = -1 * np.eye(n)
        C = np.kron(np.eye(H), I_tilde)
        self.C = csr_matrix(C)
 
        self.d = np.kron(np.ones(H), np.r_[u_max, -u_min, x_max, -x_min])
    
    def control(self, x_0, z, x_f=None):
        prob, y_opt = self.mpc_problem(x_0, z, x_f) 
        prob.solve()
        return y_opt.value, prob

    def mpc_problem(self, x_0, z, x_f=None):
        n, H = self.n, self.H
        y = cp.Variable(self.y_dim)
        cost = cp.quad_form(y, self.Q_bar)

        assert z.shape == (n, H+1), \
            f"z dim should be {(n, H+1)}, was {z.shape}"

        b = np.zeros((self.A.shape[0]))
        b[:n] = self.A_i @ x_0 - z[:, 1]
        for k in range(1, H):
            b[k*n:k*n+n] = self.A_i @ z[:, k] - z[:, k+1]
        if type(x_f) == np.ndarray:
            b[-n:] = x_f - z[:, H]
            self.A[-n:,-n:] = np.eye(n)
        else:
            b[-n:] = np.zeros(n)
            self.A[-n:,-n:] = np.zeros((n, n))

        A = csr_matrix(self.A)
        constraints = [
            A @ y == b,
            self.C @ y <= self.d
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, y
