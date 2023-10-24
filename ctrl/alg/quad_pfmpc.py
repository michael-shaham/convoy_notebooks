from alg.controller_base import ControllerBase
from dyn.dynamics_base import DynamicsBase

import cvxpy as cp
import numpy as np


class QuadPFMPC(ControllerBase):
    """
    Predecessor follower linear MPC controller.

    Parameters:
        Q: move suppression term
        Q_p: predecessor relative error term
        R: input cost
        dyn: dynamics base class based on DynamicsBase, linear
        x_min/max: state bounds (assume box constraints)
        u_min/max: input bounds (assume box constraints)
        H: horizon
    """
    def __init__(self,
                 Q: np.ndarray, Q_p: np.ndarray, R: np.ndarray,
                 dyn: DynamicsBase,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray,
                 H: int):
        self.Q, self.Q_p, self.R = Q, Q_p, R
        self.dyn = dyn
        self.x_min, self.x_max = x_min, x_max
        self.u_min, self.u_max = u_min, u_max
        self.H = H
        self.n = self.dyn.n
        self.m = self.dyn.m
        self.p = self.dyn.p
    
    def control(self, x_0: np.ndarray, y_a: np.ndarray, 
                y_pred_a: np.ndarray, d: float):
        prob, x, y, u = self.mpc_problem(x_0, y_a, y_pred_a, d)
        prob.solve()
        return u.value, x.value, y.value, prob.value
    
    def mpc_problem(self, x_0: np.ndarray, y_a: np.ndarray, 
                    y_pred_a: np.ndarray, d: float) -> cp.Problem:
        d_tilde = np.array([d, 0.])
        x = cp.Variable((self.n, self.H + 1))
        y = cp.Variable((self.p, self.H + 1))
        u = cp.Variable((self.m, self.H))
        cost = 0.
        constraints = []

        # construct cost
        for k in range(self.H):
            cost += cp.quad_form(u[:, k], self.R)
            cost += cp.quad_form(y[:, k] - y_a[:, k], self.Q)
            cost += cp.quad_form(y[:, k] - y_pred_a[:, k] + d_tilde, self.Q_p)

        # construct constraints
        constraints += [
            x[:, 0] == x_0,                  # initial condition
            y[:, 0] == self.dyn.sense(x_0),  # initial condition
            x[2, self.H] == 0.,              # end with 0 accel
            y[:, self.H] == y_pred_a[:, self.H] - d_tilde  # end at assumed
        ]
        for k in range(self.H):
            constraints += [
                x[:, k+1] == self.dyn.forward(x[:, k], u[:, k]),
                y[:, k+1] == self.dyn.sense(x[:, k+1]),
                x[:, k] >= self.x_min, 
                x[:, k] <= self.x_max, 
                u[:, k] >= self.u_min, 
                u[:, k] <= self.u_max
            ]
        constraints += [x[:, self.H] >= self.x_min,
                        x[:, self.H] <= self.x_max]
        
        # construct problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, x, y, u