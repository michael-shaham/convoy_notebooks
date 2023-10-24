import copy

from alg.controller_base import ControllerBase
from dyn.linear_long import LinearLong

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix

class Sparse1NormPFMPC(ControllerBase):
    """
    PF linear MPC controller based on sum of l1-norms cost function.

    Parameters:
        W (np.ndarray): W = diag(w1, w2, optionally w3)
            w1 (float): weight on position deviation
            w2 (float): weight on velocity deviation
            w3 (float): weight on acceleration deviation
        q_s: move suppresion term (0 if pinned to leader or no trail, 1 o.w.)
        q_p: predecessor relative error weight (vs. move suppression term)
        r: weighting term for input cost
        dyn: dynamics base class based on DynamicsBase, must be linear
        u_min/max: min/max input acceleration
        H: horizon
    """
    def __init__(self, W: np.ndarray, q_s: float, q_p: float, r: float, 
                 dyn: LinearLong, u_min: np.ndarray, u_max: np.ndarray, H: int):
        self.dyn = dyn
        self.W = W
        self.q_s, self.q_p, self.r = q_s, q_p, r
        n, m, p = dyn.n, dyn.m, dyn.p
        self.n, self.m, self.p, self.H = n, m, p, H
        z_dim = (H-1)*n + 2*H*m + (3*H-2)*p
        self.z_dim = z_dim
        self.u_min, self.u_max = u_min[0], u_max[0]
        A, B, C = dyn.Ad, dyn.Bd, dyn.C

        # cost function is linear: c^T z
        self.c = np.zeros(z_dim)
        c_tilde = np.zeros(n+2*m+3*p)
        c_tilde[n+m+p:] = np.ones(2*p+m)
        self.c[m+p:2*m+p] = np.ones(m)
        self.c[2*m+p:] = np.kron(np.ones(H-1), c_tilde)

        # equalty constraint matrix (A in Az=b)
        A1 = np.block([[-B, np.zeros((n, p)), np.zeros((n, m))],
                       [np.zeros((p, m)), np.eye(p), np.zeros((p, m))]])
        A2 = np.block([[-A, -B, np.zeros((n, p)), np.zeros((n, m+2*p))],
                       [-C, np.zeros((p, m)), np.eye(p), np.zeros((p, m+2*p))]])
        A3 = copy.deepcopy(A2)
        A3[:n, :] *= -1
        I_tilde = np.zeros_like(A2)
        I_tilde[:n, :n] = np.eye(n)

        self.A = np.zeros((H*(n+p), z_dim))
        self.A[:n+p, :2*m+p] = A1
        self.A[n+p:-(n+p), 2*m+p:-(n+2*m+3*p)] += np.kron(np.eye(H-2), A2)
        self.A[:-(n+p), 2*m+p:] += np.kron(np.eye(H-1), I_tilde)
        self.A[-(n+p):, -(n+2*m+3*p):] = A3

        # inequality constraint matrix (C in Cz<=d)
        C1 = np.array([
            [-1, 0, 0, 0],
            [1, 0, 0, 0],
            [r, 0, 0, -1],
            [-r, 0, 0, -1]
        ])
        C2 = np.block([
            [np.zeros((m, n)), -1, np.zeros((m, 3*p+m))],
            [np.zeros((m, n)), 1, np.zeros((m, 3*p+m))],
            [np.zeros((m, n)), r, np.zeros((m, 3*p)), -1],
            [np.zeros((m, n)), -r, np.zeros((m, 3*p)), -1],
            [np.zeros((p, n+m)), q_s*W, -np.eye(p), np.zeros((p, p+m))],
            [np.zeros((p, n+m)), -q_s*W, -np.eye(p), np.zeros((p, p+m))],
            [np.zeros((p, n+m)), q_p*W, np.zeros((p, p)), -np.eye(p), np.zeros((p, m))],
            [np.zeros((p, n+m)), -q_p*W, np.zeros((p, p)), -np.eye(p), np.zeros((p, m))],
        ])
        self.C = block_diag(C1, np.kron(np.eye(H-1), C2))

        self.A, self.C = csr_matrix(self.A), csr_matrix(self.C)
    
    def control(self, x_0, y_a, y_p_a, d_des):
        prob, z = self.mpc_problem(x_0, y_a, y_p_a, d_des)
        prob.solve()
        return z.value, prob

    def mpc_problem(self, x_0, y_a, y_p_a, d_des):
        d_tilde = np.array([d_des, 0])
        n, m, p, H = self.n, self.m, self.p, self.H
        z = cp.Variable(self.z_dim)

        # equality constraint RHS (b in Az=b)
        b = np.zeros(self.A.shape[0])
        b[:n] = self.dyn.Ad @ x_0
        b[n:n+p] = self.dyn.C @ x_0
        b[-(p+n):-p] = np.block([y_p_a[:, -1] - d_tilde, 0])

        # inequality constraint RHS (d in Cz<=d)
        d = np.zeros(self.C.shape[0])
        d[:m] = -self.u_min
        d[m:2*m] = self.u_max
        for k, i in enumerate(range(4*m, self.C.shape[1], 4*p + 4*m)):
            d[i:i+m] = -self.u_min
            d[i+m:i+2*m] = self.u_max
            d[i+4*m:i+4*m+p] = self.q_s * self.W @ y_a[:, k+1]
            d[i+4*m+p:i+4*m+2*p] = -self.q_s * self.W @ y_a[:, k+1]
            d[i+4*m+2*p:i+4*m+3*p] = self.q_p * self.W @ (y_p_a[:, k+1] - d_tilde)
            d[i+4*m+3*p:i+4*m+4*p] = -self.q_p * self.W @ (y_p_a[:, k+1] - d_tilde)
        
        cost = self.c @ z
        constraints = [
            self.A @ z == b, 
            self.C @ z <= d,
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, z