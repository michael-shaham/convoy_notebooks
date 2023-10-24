import control as ct
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import typing


class CTPlatoon:

    def __init__(self, tau: float, 
                 M: np.ndarray, P: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, 
                 use_integrator: bool = False):
        """
        params:
            N: number of vehicles in platoon
            taus: list of tau values (inertial delay to longitudinal dynamics)
            Q, R, W, V: matrices for controller/observer
            M: adjacency matrix for platoon followers
            P: pinning matrix (which followers access leader)
            use_integrator: whether or not to augment sys with position integrator
        """
        self.N = M.shape[0]
        N = self.N
        self.use_integrator = use_integrator
        self.M, self.P = M, P
        self.D = np.diag([sum([M[i, j] for j in range(N)]) for i in range(N)])
        self.L = self.D - self.M
        self.tau = tau
        self.A0 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        self.B0 = np.array([[0], [0], [1]])
        self.A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1/tau]])
        self.B = np.array([[0], [0], [1/tau]])
        self.n = self.A0.shape[0]
        self.m = self.B0.shape[1]

        self.A_aug = np.block([[0, np.zeros((1, self.n))],
                               [np.zeros((self.n, 1)), self.A]])
        self.B_aug = np.block([[0], [self.B]])
        if use_integrator:
            self.A_aug[0, 1] = 1

        print(self.A_aug.shape, Q.shape)
        self.K = self.select_control_gains(Q, R)
    
    def select_control_gains(self, Q: np.ndarray, R: np.ndarray, c: float = 1.):
        assert c > 1 / (2 * min(np.linalg.eigvals(self.L + self.P)))
        P = sp.linalg.solve_continuous_are(self.A_aug, self.B_aug, Q, R)
        return np.linalg.inv(R) @ self.B_aug.T @ P
    
    def sim_sys(self, dt: float, t0: float, tf: float, a0: float, 
                td1: float, td2: float, v0: float, d_des: float):
        """
        params:
            dt: timestep
            t0: start time
            tf: end time
            a0: leader acceleration (impulse disturbance)
            td1: leader acceleration start time
            td2: leader acceleration end time
            v0: platoon initial velocity
            d_des: desired distance between vehicles
        """
        tau = self.tau
        n = self.n
        N = self.N
        d = np.array([self.P[i,i]*d_des*i + 
                      sum([self.M[i,j]*d_des*(i-j) for j in range(i)]) 
                      for i in range(N)])
        k0, k1, k2, k3 = self.K[0, :]
        if np.isclose(k0, 0): k0 = 0

        A_diag = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [-k1/tau, -k0/tau, -k1/tau, -k2/tau, -(1+k3)/tau]
        ])
        A_offdiag = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, k1/self.tau, k2/self.tau, k3/self.tau],
        ])
        if self.use_integrator:
            A_diag[1, :] = np.array([1, 0, 1, 0, 0])
            A_offdiag[1, :] = np.array([0, 0, -1, 0, 0])
        
        A_bar = np.block([
            [self.A0, 
             np.zeros((n, N*(n+2)))],
            [np.zeros((N*(n+2), n)), 
             np.kron(np.eye(N), A_diag) + np.kron(self.M, A_offdiag)]
        ])
        for k, i in enumerate(range(n, n + N*(n+2), n+2)):
            A_bar[i+1, 0] = -self.P[k, k]
            A_bar[i+n+1, 0] = self.P[k,k]*k1/tau
            A_bar[i+n+1, 1] = self.P[k,k]*k2/tau
            A_bar[i+n+1, 2] = self.P[k,k]*k3/tau
        B_bar = np.zeros((n+N*(n+2), 1))
        B_bar[n-1, 0] = a0
        C_bar = np.eye(n+N*(n+2))
        D_bar = np.zeros((n+N*(n+2), 1))

        sys = ct.ss(A_bar, B_bar, C_bar, D_bar)
        sys_neg = ct.ss(A_bar, -B_bar, C_bar, D_bar)

        x0 = np.zeros(n + N*(n+2))
        x0[1] = v0
        for k, i in enumerate(range(n, n+N*(n+2), n+2)):
            x0[i] = d[k]
            x0[i+2] = -d_des*(k)
            x0[i+3] = v0

        T1 = np.arange(0, td1, dt)
        T2 = np.arange(td1, td2, dt)
        T3 = np.arange(td2, tf + dt, dt)

        res1 = ct.initial_response(sys, T1, x0)
        res2 = ct.impulse_response(sys, T2, res1.outputs[:, -1])
        res3 = ct.impulse_response(sys_neg, T3, res2.outputs[:, 0, -1])
        T = np.concatenate((res1.time, res2.time, res3.time))
        x_traj = np.concatenate((res1.outputs, res2.outputs[:, 0, :], res3.outputs[:, 0, :]), axis=1)

        traj = [np.zeros((n, len(T))) for _ in range(N+1)]
        inputs = [np.zeros(len(T)) for _ in range(N+1)]
        pred_errors = [np.zeros((n, len(T))) for _ in range(N)]
        lead_errors = [np.zeros((n, len(T))) for _ in range(N)]

        traj[0] = x_traj[:n, :]
        inputs[0] = x_traj[n-1, :]

        for k, i in enumerate(range(n, n+N*(n+2), n+2)):
            traj[k+1][0, :] = x_traj[i+2, :]
            traj[k+1][1, :] = x_traj[i+3, :]
            traj[k+1][2, :] = x_traj[i+4, :]

            pred_errors[k][0, :] = x_traj[i+2, :] - x_traj[i-3, :] + d[k]
            pred_errors[k][1, :] = x_traj[i+3, :] - x_traj[i-2, :]
            pred_errors[k][2, :] = x_traj[i+4, :] - x_traj[i-1, :]

            lead_errors[k][0, :] = x_traj[i+2, :] - x_traj[0, :] + k*d_des
            lead_errors[k][1, :] = x_traj[i+3, :] - x_traj[1, :]
            lead_errors[k][2, :] = x_traj[i+4, :] - x_traj[2, :]

            inputs[k+1] = -(k0*x_traj[i+1,:] + k1*pred_errors[k][0,:] + 
                k2*pred_errors[k][1,:] + k3*pred_errors[k][2,:])

        return T, traj, inputs, pred_errors, lead_errors