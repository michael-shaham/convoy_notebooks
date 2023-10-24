from alg.controller_base import ControllerBase
from dyn.linear_long import LinearLong

import copy
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag

class SparseQuadPFMPC(ControllerBase):
    """
    Predecessor follower linear MPC controller.

    Parameters:
        Q: move suppression cost term
        Q_p: predecessor relative error cost term
        R: input cost
        dyn: dynamics base class based on DynamicsBase, linear
        u_min/max: input bounds (assume box constraints)
        H: horizon
        da_max: max allowed jerk [m/s^3]
    """
    def __init__(self,
                 Q: np.ndarray, Q_p: np.ndarray, R: np.ndarray,
                 dyn: LinearLong,
                 u_min: np.ndarray, u_max: np.ndarray,
                 H: int,
                 da_max: float = 0.):
        n = dyn.n
        m = dyn.m
        p = dyn.p
        z_dim = (H-1)*n + H*m + (3*H - 2)*p
        self.n = n
        self.m = m
        self.p = p
        self.H = H
        u_min, u_max = u_min[0], u_max[0]

        self.z_dim = z_dim

        # cost function: assume Q, Q_p, R all diagonal
        self.Q_p = Q_p
        Q_bar = np.zeros((z_dim, z_dim))
        Q_tilde = block_diag(np.zeros((n, n)), R, np.zeros((p, p)), Q, Q_p)
        Q_bar[:m, :m] = R
        Q_bar[m+p:, m+p:] = np.kron(np.eye(H-1), Q_tilde)
        self.Q_bar = csr_matrix(Q_bar)

        # equality constraint: Az = b 
        A = np.zeros((H*n + (3*H - 2)*p, z_dim))
        A_i = dyn.Ad
        self.A_i = A_i
        B_i = dyn.Bd
        C_i = dyn.C
        self.C_i = C_i

        A_tilde_1 = np.zeros((n + p, m + p))
        A_tilde_1[:n, :m] = -B_i
        A_tilde_1[-p:,-p:] = np.eye(p)

        A_tilde_2 = np.zeros((n + 3*p, n + m + 3*p))
        A_tilde_2[:n, :n] = -A_i
        A_tilde_2[:n, n:n+m] = -B_i
        A_tilde_2[n:n+p, :n] = -C_i
        A_tilde_2[n:n+p, n+m:n+m+p] = np.eye(2)
        A_tilde_2[n+p:n+2*p, n+m:n+m+p] = np.eye(2)
        A_tilde_2[n+p:n+2*p, n+m+p:n+m+2*p] = -np.eye(2)
        A_tilde_2[n+2*p:n+3*p, n+m:n+m+p] = np.eye(2)
        A_tilde_2[n+2*p:n+3*p, n+m+2*p:n+m+3*p] = -np.eye(2)

        A_tilde_3 = copy.deepcopy(A_tilde_2)
        if Q_p[0, 0] == 0:
            A_tilde_3[:n, :n] = np.array([[0, 0, 0], [0, 1, dyn.dt], [0, 0, 1-1/dyn.tau]])
            A_tilde_3[:n, n:n+m] *= -1
        else:
            A_tilde_3[:n, :n+m] *= -1

        I_tilde_1 = np.zeros((n + p, n + m + 3*p))
        I_tilde_1[:n, :n] = np.eye(n)

        I_tilde_2 = np.zeros((n + 3*p, n + m + 3*p))
        I_tilde_2[:n, :n] = np.eye(n)

        A = np.zeros((H*n + (3*H-2)*p, z_dim))
        A[:n+p, :m+p] += A_tilde_1
        A[:n+p, m+p:n+2*m+4*p] += I_tilde_1
        A[n+p:-(n+3*p), m+p:-(n+m+3*p)] += np.kron(np.eye(H-2), A_tilde_2)
        A[n+p:-(n+3*p), n+2*m+4*p:] += np.kron(np.eye(H-2), I_tilde_2)
        A[-(n+3*p):, -(n+m+3*p):] += A_tilde_3
        self.A = csr_matrix(A)

        # inequality constraint: Cz <= d
        da_max_dt = da_max * dyn.dt
        self.da_max_dt = da_max_dt
        if da_max > 0.0:
            # C_tilde_1 = np.zeros((2*m + 2, n + m + 3*p))
            # C_tilde_1[0, n-1] = 1
            # C_tilde_1[1, n-1] = -1
            # C_tilde_1[2:2+m, n:n+m] = np.eye(m)
            # C_tilde_1[2+m:2+2*m, n:n+m] = -np.eye(m)

            # C_tilde_2 = np.zeros((2*m + 2, n + m + 3*p))
            # C_tilde_2[0, n-1] = -1
            # C_tilde_2[1, n-1] = 1

            # C = np.zeros((2*H*m + 2*H, z_dim))
            # C[:m, :m] += np.eye(m)
            # C[m:2*m, :m] += -np.eye(m)
            # C[2*m:-2, m+p:] += np.kron(np.eye(H-1), C_tilde_1)
            # C[4*m+2:-2, m+p:-(n+m+3*p)] += np.kron(np.eye(H-2), C_tilde_2)
            # C[-2, -(n+m+3*p)+n-1] = 1
            # C[-1, -(n+m+3*p)+n-1] = -1
            # self.d = np.zeros(C.shape[0])
            # self.d[:2*m] = np.array([u_max, -u_min])
            # self.d[2*m:-2] = np.tile(np.array([da_max_dt, da_max_dt, u_max, -u_min]), H-1)
            # self.d[-2:] = np.array([da_max_dt, da_max_dt])

            # only constraining next acceleration vs current
            C = np.zeros((2*H*m + 2, z_dim))
            C_tilde = np.zeros((2*m, n + m + 3*p))
            C_tilde[:m, n:n+m] += np.eye(m)
            C_tilde[m:2*m, n:n+m] += -np.eye(m)
            C[:m, :m] += np.eye(m)
            C[m:2*m, :m] += -np.eye(m)
            C[2*m, m+p+n-1] = 1
            C[2*m+1, m+p+n-1] = -1
            C[2*m+2:, m+p:] += np.kron(np.eye(H-1), C_tilde)
            self.d = np.zeros(C.shape[0])
            self.d[:2*m] = np.array([u_max, -u_min])
            self.d[2*m+2:] = np.repeat(np.array([u_max, -u_min]), H-1)
        else:
            C_tilde = np.zeros((2*m, n + m + 3*p))
            C_tilde[:m, n:n+m] += np.eye(m)
            C_tilde[m:2*m, n:n+m] += -np.eye(m)

            C = np.zeros((2*H*m, z_dim))
            C[:m, :m] += np.eye(m)
            C[m:2*m, :m] += -np.eye(m)
            C[2*m:, m+p:] += np.kron(np.eye(H-1), C_tilde)
            self.d = np.repeat(np.array([u_max, -u_min]), H)

        self.C = csr_matrix(C)


    def control(self, x_0, y_a, y_p_a, d_des):
        prob, z = self.mpc_problem(x_0, y_a, y_p_a, d_des)
        prob.solve(solver='OSQP')
        return z.value, prob

    def mpc_problem(self, x_0, y_a, y_p_a, d_des):
        n, m, p, H = self.n, self.m, self.p, self.H
        z = cp.Variable(self.z_dim)
        cost = cp.quad_form(z, self.Q_bar)

        assert y_a.shape == (p, H + 1), \
            f"y_a dim should be {(p, H + 1)}, was {y_a.shape}"
        assert y_p_a.shape == (p, H + 1), \
            f"y_p_a dim should be {(p, H + 1)}, was {y_p_a.shape}"
        d_tilde = np.array([d_des, 0])

        b = np.zeros(self.A.shape[0])
        b[:n] += self.A_i @ x_0
        b[n:n+p] += self.C_i @ x_0
        for k, i in enumerate(range(n + p, len(b), n + 3*p)):
            b[i+n+p:i+n+2*p] += y_a[:, k+1]
            b[i+n+2*p:i+n+3*p] += y_p_a[:, k+1] - d_tilde
        b[-(3*p+n):-(2*p+n)] += y_p_a[:, H] - d_tilde
        if self.Q_p[0, 0] == 0:
            b[-(3*p+n)] = 0.0

        if self.da_max_dt > 0.0:
            self.d[2*m] = self.da_max_dt + x_0[n-1]
            self.d[2*m+1] = self.da_max_dt - x_0[n-1]
        constraints = [
            self.A @ z == b,
            self.C @ z <= self.d
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, z
