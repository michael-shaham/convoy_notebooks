from alg.controller_base import ControllerBase
from dyn.dynamics_base import DynamicsBase

import cvxpy as cp
import numpy as np


class QuadPFMPCVel(ControllerBase):
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
                 u_min: np.ndarray,
                 H: int):
        self.Q, self.Q_p, self.R = Q, Q_p, R
        self.dyn = dyn
        self.u_min = u_min
        self.H = H
        self.n = self.dyn.n
        self.m = self.dyn.m
    
    def control(self, x_0: np.ndarray, x_a: np.ndarray, 
                x_pred_a: np.ndarray, d: float):
        prob, x, u = self.mpc_problem(x_0, x_a, x_pred_a, d)
        prob.solve()
        return u.value, x.value, prob
    
    def mpc_problem(self, x_0: np.ndarray, x_a: np.ndarray, 
                    x_pred_a: np.ndarray, d: float) -> cp.Problem:
        d_tilde = np.array([d, 0.])
        x = cp.Variable((self.n, self.H + 1))
        u = cp.Variable((self.m, self.H))
        cost = 0.
        constraints = []

        # construct cost
        for k in range(self.H):
            cost += cp.quad_form(u[:, k], self.R)
            cost += cp.quad_form(x[:, k] - x_a[:, k], self.Q)
            cost += cp.quad_form(x[:, k] - x_pred_a[:, k] + d_tilde, self.Q_p)

        # construct constraints
        constraints += [
            x[:, 0] == x_0,                  # initial condition
            x[:, self.H] == x_pred_a[:, self.H] - d_tilde  # end at assumed
        ]
        for k in range(self.H):
            constraints += [
                x[:, k+1] == self.dyn.forward(x[:, k], u[:, k]),
                u[:, k] >= self.u_min, 
            ]
        
        # construct problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, x, u