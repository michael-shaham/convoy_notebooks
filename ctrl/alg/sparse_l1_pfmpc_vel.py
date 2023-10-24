import copy

from alg.controller_base import ControllerBase
from dyn.linear_long_vel import LinearLongVel

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix

class Sparse1NormPFMPCVel(ControllerBase):
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
        v_min/max: min/max velocity
        a_max: max allowed accel (mgnitude)
        H: horizon
    """
    def __init__(self, W: np.ndarray, q_s: float, q_p: float, r: float, 
                 dyn: LinearLongVel, v_min: float, v_max: float, a_max: float, 
                 H: int):
        self.dyn = dyn
        self.W = W
        self.q_s, self.q_p, self.r = q_s, q_p, r
        n, m = dyn.n, dyn.m
        self.n, self.m, self.H = n, m, H
        z_dim = 3*(H-1)*n + 2*H*m
        self.z_dim = z_dim
        self.v_min, self.v_max = v_min, v_max
        self.a_max = a_max
        Ad, Bd = dyn.Ad, dyn.Bd
        self.dt = dyn.dt

        # cost function is linear: c^T z
        self.c = np.zeros(z_dim)
        c_tilde = np.zeros(3*n+2*m)
        c_tilde[n+m:] = np.ones(2*n+m)
        self.c[m:2*m] = np.ones(m)
        self.c[2*m:] = np.kron(np.ones(H-1), c_tilde)

        # equalty constraint matrix (A in Az=b)
        A1 = np.block([-Bd, np.zeros((n, m))])
        A2 = np.block([-Ad, -Bd, np.zeros((n, m+2*n))])
        A3 = copy.deepcopy(A2)
        A3[:n, :] *= -1
        I_tilde = np.zeros_like(A2)
        I_tilde[:n, :n] = np.eye(n)

        self.A = np.zeros((H*n, z_dim))
        self.A[:n, :2*m] = A1
        self.A[n:-n, 2*m:-(3*n+2*m)] += np.kron(np.eye(H-2), A2)
        self.A[:-n, 2*m:] += np.kron(np.eye(H-1), I_tilde)
        self.A[-n:, -(3*n+2*m):] = A3

        # inequality constraint matrix (C in Cz<=d)
        self.C = np.zeros((4*(H-1)*n + 6*H*m, 3*(H-1)*n + 2*H*m))
        C1 = np.array([
            [-1, 0], # input constraint
            [1, 0],  # input constraint
            [r, -1], # min control
            [-r, -1], # min control
        ])
        C2 = np.block([
            [q_s*W, np.zeros((n, m)), -np.eye(n), np.zeros((n, n+m))],  # MS term
            [-q_s*W, np.zeros((n, m)), -np.eye(n), np.zeros((n, n+m))], # MS term
            [q_p*W, np.zeros((n, m)), np.zeros((n, n)), -np.eye(n), np.zeros((n, m))],  # PRE term
            [-q_p*W, np.zeros((n, m)), np.zeros((n, n)), -np.eye(n), np.zeros((n, m))], # PRE term
            [np.zeros((m, n)), -1, np.zeros((m, 2*n+m))],  # input constraint
            [np.zeros((m, n)), 1, np.zeros((m, 2*n+m))],   # input constraint
            [np.zeros((m, n)), r, np.zeros((m, 2*n)), -1],  # min control
            [np.zeros((m, n)), -r, np.zeros((m, 2*n)), -1], # min control
            [np.array([[0, -1]]), np.zeros(2*n+2*m)],  # accel constraint
            [np.array([[0, 1]]), np.zeros(2*n+2*m)],   # accel constraint
        ])
        I1 = np.zeros((2, 3*n+2*m))
        I1[-2, 1] = 1
        I1[-1, 1] = -1
        I2 = np.zeros_like(C2)
        I2[-2, 1] = 1
        I2[-1, 1] = -1
        self.C[:6*m, :3*n+4*m] = block_diag(C1, I1)
        self.C[6*m:, 2*m:] += np.kron(np.eye(H-1), C2)
        self.C[6*m:-(4*n+6*m), 4*m+3*n:] += np.kron(np.eye(H-2), I2)

        self.A, self.C = csr_matrix(self.A), csr_matrix(self.C)
    
    def control(self, x_0, x_a, x_p_a, d_des):
        prob, z = self.mpc_problem(x_0, x_a, x_p_a, d_des)
        prob.solve(solver='ECOS')

        if prob.status != 'optimal':
            return None, None, prob

        z_opt = z.value
        x_opt = np.zeros((self.n, self.H+1))
        u_opt = np.zeros((self.m, self.H))

        x_opt[:, 0] = x_0
        u_opt[:, 0] = z_opt[:self.m]
        for t, i in enumerate(range(2*self.m, self.z_dim, 3*self.n+2*self.m)):
            x_opt[:, t+1] = z_opt[i:i+self.n]
            u_opt[:, t+1] = z_opt[i+self.n:i+self.n+self.m]
        x_opt[:, self.H] = self.dyn.forward(x_opt[:, self.H-1], 
                                            u_opt[:, self.H-1])

        return u_opt, x_opt, prob

    def mpc_problem(self, x_0, x_a, x_p_a, d_des):
        n, m, H = self.n, self.m, self.H
        z = cp.Variable(self.z_dim)

        d_tilde = np.array([d_des, 0])
        u_des = x_0[1]

        # equality constraint RHS (b in Az=b)
        b = np.zeros(self.A.shape[0])
        b[:n] = self.dyn.Ad @ x_0
        b[-n:] = x_p_a[:, -1] - d_tilde

        # inequality constraint RHS (d in Cz<=d)
        d = np.zeros(self.C.shape[0])
        d[:m] = -self.v_min
        d[m:2*m] = self.v_max
        d[2*m:3*m] = self.r * u_des
        d[3*m:4*m] = -self.r * u_des
        d[4*m:5*m] = self.dt*self.a_max + x_0[1]
        d[5*m:6*m] = self.dt*self.a_max - x_0[1]
        for k, i in enumerate(range(6*m, self.C.shape[0], 4*n + 6*m)):
            d[i:i+n] = self.q_s * self.W @ x_a[:, k+1]
            d[i+n:i+2*n] = -self.q_s * self.W @ x_a[:, k+1]
            d[i+2*n:i+3*n] = self.q_p * self.W @ (x_p_a[:, k+1] - d_tilde)
            d[i+3*n:i+4*n] = -self.q_p * self.W @ (x_p_a[:, k+1] - d_tilde)
            d[i+4*n:i+4*n+m] = -self.v_min
            d[i+4*n+m:i+4*n+2*m] = self.v_max
            d[i+4*n+2*m:i+4*n+3*m] = self.r * u_des
            d[i+4*n+3*m:i+4*n+4*m] = -self.r * u_des
            d[i+4*n+4*m:i+4*n+5*m] = self.dt*self.a_max
            d[i+4*n+5*m:i+4*n+6*m] = self.dt*self.a_max
        d[-2] = self.dt*self.a_max - x_p_a[1, -1]
        d[-1] = self.dt*self.a_max + x_p_a[1, -1]
        
        cost = self.c @ z
        constraints = [
            self.A @ z == b, 
            self.C @ z <= d,
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, z