from alg.controller_base import ControllerBase
from dyn.dynamics_base import DynamicsBase

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm


class NormDMPC(ControllerBase):
    """
    Platoon MPC controller with multi-objective tradeoff between norms of 
    errors.

    Parameters:
        Q: penalizes error from leader trajectory (zeros if no access to leader)
        R: input cost
        F: move suppression term
        G: neighbor relative error term (zero if no neighbors)
        dyn: dynamics base class based on DynamicsBase, linear
        x_min/max: state bounds (assume box constraints)
        u_min/max: input bounds (assume box constraints)
        H: horizon
    """
    def __init__(self,
                 Q: np.ndarray, R: np.ndarray, F: np.ndarray, G: np.ndarray,
                 dyn: DynamicsBase,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray,
                 H: int):
        self.dyn = dyn
        self.x_min, self.x_max = x_min, x_max
        self.u_min, self.u_max = u_min, u_max
        self.H = H
        self.n = self.dyn.n
        self.m = self.dyn.m
        self.p = self.dyn.p

        self.Q = sqrtm(Q)
        self.R = sqrtm(R)
        self.F = sqrtm(F)
        self.G = sqrtm(G)
    
    def control(self, x_0, y_a, info_set, y_i_a, d_i):
        """
        x_0: np.ndarray, shape (3,), initial state 
            (longitudinal position, velocity, acceleration)
        y_a: np.ndarray, shape (2, H+1), assumed longitudinal trajectory
        y_l_a: np.ndarray, shape (2, H+1), leader assumed longitudinal 
            trajectory (None if no access to leader)
        info_set: list of indices of the vehicles whose information this 
            controller has access to
        y_i_a: list of np.ndarray (one for each vehicle in info_set), 
            shape (2, H+1), assumed longitudinal trajectory for corresponding 
            vehicle in info_set. 
        d_i: list of floats (one for each vehicle in info_set
            desired distances from corresponding vehicles in info_set
        """
        prob, x, y, u = self.mpc_problem(x_0, y_a, info_set, y_i_a, d_i)
        prob.solve(solver='GUROBI')
        return u.value, x.value, y.value, prob.value, prob.status
    
    def mpc_problem(self, x_0, y_a, info_set, y_i_a, d_i):
        x = cp.Variable((self.n, self.H + 1))
        y = cp.Variable((self.p, self.H + 1))
        u = cp.Variable((self.m, self.H))
        cost = 0.
        constraints = []

        d_i_tilde = [np.array([d_i[i], 0.]) for i in range(len(info_set))]
        # construct cost
        for k in range(self.H):
            cost += cp.norm(self.R @ u[:, k])
            cost += cp.norm(self.F @ (y[:, k] - y_a[:, k]))
            for j, i in enumerate(info_set):
                if i == 0:
                    cost += cp.norm(
                        self.Q @ (y[:, k] - y_i_a[j][:, k] + d_i_tilde[j])
                    )
                else:
                    cost += cp.norm(
                        self.G @ (y[:, k] - y_i_a[j][:, k] + d_i_tilde[j])
                    )

        # construct constraints
        # initial and terminal constraints
        constraints += [
            x[:, 0] == x_0,                  # initial condition
            y[:, 0] == self.dyn.sense(x_0),  # initial condition
            x[2, self.H] == 0.,              # terminal constraint: 0 accel
        ]
        # terminal constraint: output is average of neighbors in info set
        constraints += [
            y[:, self.H] == 1 / len(y_i_a) * \
                cp.sum([y_i_a[i][:, self.H] - d_i_tilde[i] 
                        for i in range(len(y_i_a))])
        ] 

        # dynamics and state/input constraints
        for k in range(self.H):
            constraints += [
                x[:, k+1] == self.dyn.forward(x[:, k], u[:, k]),
                y[:, k+1] == self.dyn.sense(x[:, k+1]),
                u[:, k] >= self.u_min, 
                u[:, k] <= self.u_max
            ]
        
        # construct problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, x, y, u