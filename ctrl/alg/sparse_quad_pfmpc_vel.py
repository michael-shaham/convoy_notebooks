from alg.controller_base import ControllerBase
from dyn.linear_long_vel import LinearLongVel

import copy
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix

class SparseQuadPFMPCVel(ControllerBase):
    """
    Predecessor follower linear MPC controller based on velocity dynamics model.

    Parameters:
        Q: move suppression cost term
        Q_p: predecessor relative error cost term
        R: input cost
        dyn: dynamics base class based on DynamicsBase, linear
        v_min/v_max: minimum and maximum velocity allowed (in state)
        a_max: maximum (magnitude) accel allowed
        H: horizon
    """
    def __init__(self,
                 Q: np.ndarray, Q_p: np.ndarray, R: np.ndarray,
                 dyn: LinearLongVel,
                 v_min: float, v_max: float, a_max: float,
                 H: int):
        self.dyn = dyn
        self.dt = dyn.dt
        n = dyn.n
        m = dyn.m
        z_dim = (3*n+m)*(H-1) + m
        self.n = n
        self.m = m
        self.H = H
        self.a_max = a_max

        self.z_dim = z_dim
        # z = (u_0, x_1, u_1, x_1_tilde_1, x_1_tilde_2, ..., 
        #      x_{H-1}, u_{H-1}, x_{H-1}_tilde_1, x_{H-1}_tilde_2)
        # x_k_tilde_1 = x_k - x_a_{k}
        # x_k_tilde_2 = x_k - x_p_a_{k} + d

        # cost function: assume Q, Q_p, R all diagonal
        Q_bar = np.zeros((z_dim, z_dim))
        Q_tilde = np.diag(np.concatenate((np.zeros(n), 
                                          np.diag(R), 
                                          np.diag(Q), 
                                          np.diag(Q_p)))
        )
        Q_bar[:m, :m] = R
        Q_bar[m:, m:] = np.kron(np.eye(H-1), Q_tilde)
        self.Q_bar = csr_matrix(Q_bar)

        # equality constraint: Az = b 
        A = np.zeros((3*H*n - 2*n, z_dim))
        A_i = dyn.Ad
        self.A_i = A_i
        B_i = dyn.Bd
        self.B_i = B_i

        A_tilde = np.zeros((3*n, 3*n+m))
        A_tilde[:n, :n] = -A_i
        A_tilde[:n, n:n+m] = -B_i
        A_tilde[n:2*n, :n] = np.eye(n)
        A_tilde[n:2*n, n+m:2*n+m] = -np.eye(n)
        A_tilde[2*n:3*n, :n] = np.eye(n)
        A_tilde[2*n:3*n, 2*n+m:] = -np.eye(n)

        I_tilde = np.zeros((3*n, 3*n+m))
        I_tilde[:n, :n] = np.eye(n)

        A = np.zeros((3*H*n - 2*n, z_dim))
        A[:n, :m] = -B_i
        A[:n, m:m+n] = np.eye(n)

        A[n:, m:] += np.kron(np.eye(H-1), A_tilde)
        A[n:-3*n, 3*n+2*m:] += np.kron(np.eye(H-2), I_tilde)
        A[-3*n:-2*n, -(3*n+m):-2*n] *= -1
        self.A = csr_matrix(A)

        # inequality constraint: Cz <= d
        C_tilde = np.zeros((2, 3*n+m))
        C_tilde[0, 1] = 1
        C_tilde[1, 1] = -1
        C_bar = np.block([[C_tilde, -C_tilde],
                          [np.zeros_like(C_tilde), C_tilde]])
        C = np.zeros((4*H-2, z_dim))
        C[:2, m:2*m+3*n] = C_tilde
        C[2:4, m:2*m+3*n] = C_tilde
        for j, i in enumerate(range(4, 4*H-4, 4)):
            C[i:i+4, 1+j*(3*n+m):1+(j+2)*(3*n+m)] = C_bar
        C[-2:, -(3*n+m):] = C_tilde
        self.C = csr_matrix(C)

        self.d = np.zeros(4*H - 2)
        self.d[2:4] = np.array([v_max, -v_min])
        self.d[4:-2] = np.tile(
            np.array([self.dt*a_max, self.dt*a_max, v_max, -v_min]), H-2)

    def control(self, x_0, x_a, x_p_a, d_des):
        prob, z = self.mpc_problem(x_0, x_a, x_p_a, d_des)
        prob.solve()
        if prob.status != 'optimal':
            return None, None, prob

        z = z.value
        x_opt = np.zeros((self.n, self.H+1))
        u_tilde_opt = np.zeros((self.m, self.H))

        x_opt[:, 0] = x_0
        u_tilde_opt[:, 0] = z[:self.m]
        for t, i in enumerate(range(self.m, len(z), 3*self.n+self.m)):
            x_opt[:, t+1] = z[i:i+self.n]
            u_tilde_opt[:, t+1] = z[i+self.n:i+self.n+self.m]

        u_opt = u_tilde_opt + x_p_a[1, self.H]
        x_opt[:, self.H] = self.dyn.forward(x_opt[:, self.H-1], 
                                            u_opt[:, self.H-1])

        return u_opt, x_opt, prob
    
    def mpc_problem(self, x_0, x_a, x_p_a, d_des):
        n, m, H = self.n, self.m, self.H
        z = cp.Variable(self.z_dim)
        cost = cp.quad_form(z, self.Q_bar)

        assert x_a.shape == (n, H + 1), \
            f"x_a dim should be {(n, H + 1)}, was {x_a.shape}"
        assert x_p_a.shape == (n, H + 1), \
            f"x_p_a dim should be {(n, H + 1)}, was {x_p_a.shape}"

        d_tilde = np.array([d_des, 0])
        u_des = np.array([x_p_a[1, H]])

        b = np.zeros(self.A.shape[0])
        b[:n] += self.A_i @ x_0 + self.B_i @ u_des
        for k, i in enumerate(range(n, len(b), 3*n)):
            b[i:i+n] += self.B_i @ u_des 
            b[i+n:i+2*n] += x_a[:, k+1]
            b[i+2*n:i+3*n] += x_p_a[:, k+1] - d_tilde
        b[-(3*n):-(2*n)] = x_p_a[:, H] - d_tilde - self.B_i @ u_des

        self.d[:2] = np.array([self.dt * self.a_max + x_0[1], 
                               self.dt * self.a_max - x_0[1]])
        self.d[-2:] = np.array([self.dt * self.a_max + x_p_a[1, H], 
                                self.dt * self.a_max - x_p_a[1, H]])

        constraints = [
            self.A @ z == b,
            self.C @ z <= self.d
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, z