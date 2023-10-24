from alg.controller_base import ControllerBase
from dyn.dynamics_base import DynamicsBase

import cvxpy as cp
import numpy as np


class QuadBDMPC(ControllerBase):
    """
    Bidirectional linear MPC controller.

    Parameters:
        Q: move suppression cost term
        Q_p: predecessor relative error cost term
        Q_s: successor relative error cost term
        R: input cost
        q_p: weight on terminal constraint for predecessor trajectory
        q_s: weight on terminal constraint for successor trajectory
        dyn: dynamics base class based on DynamicsBase, linear
        x_min/max: state bounds (assume box constraints)
        u_min/max: input bounds (assume box constraints)
        H: horizon
    """
    def __init__(self,
                 Q: np.ndarray, Q_p: np.ndarray, Q_s: np.ndarray, R: np.ndarray,
                 q_p: float, q_s: float,
                 dyn: DynamicsBase,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray,
                 H: int):
        self.Q, self.Q_p, self.Q_s, self.R = Q, Q_p, Q_s, R
        self.q_p, self.q_s = q_p, q_s
        self.dyn = dyn
        self.x_min, self.x_max = x_min, x_max
        self.u_min, self.u_max = u_min, u_max
        self.H = H
        self.n = self.dyn.n
        self.m = self.dyn.m
        self.p = self.dyn.p
    
    def control(self, x_0: np.ndarray, y_a: np.ndarray,
                y_pred_a: np.ndarray, y_succ_a: np.ndarray,
                d_pred: float, d_succ: float):
        """
        arguments:
            x_0: current state
            y_a: assumed output trajectory
            y_pred_a: assumed output trajectory of predecessor
            y_succ_a: assumed output trajectory of successor (can be None)
            d_pred: desired distance to predecessor
            d_succ: desired distance to successor (can be None)
        """
        prob, x, y, u = self.mpc_problem(x_0, y_a, y_pred_a, y_succ_a, 
                                          d_pred, d_succ)
        prob.solve()
        return u.value, x.value, y.value, prob.value
    
    def mpc_problem(self, x_0: np.ndarray, y_a: np.ndarray,
                     y_pred_a: np.ndarray, y_succ_a: np.ndarray,
                     d_pred: float, d_succ: float):
        d_p_tilde = np.array([d_pred, 0.])
        if type(y_succ_a) == np.ndarray:
            d_s_tilde = np.array([-d_succ, 0.])
        x = cp.Variable((self.n, self.H + 1))
        y = cp.Variable((self.p, self.H + 1))
        u = cp.Variable((self.m, self.H))
        cost = 0.
        constraints = []

        # construct cost
        for k in range(self.H):
            cost += cp.quad_form(u[:, k], self.R)
            cost += cp.quad_form(y[:, k] - y_a[:, k], self.Q)
            cost += cp.quad_form(y[:, k] - y_pred_a[:, k] + d_p_tilde, self.Q_p)
            if type(y_succ_a) == np.ndarray:
                cost += cp.quad_form(y[:, k] - y_succ_a[:, k] + d_s_tilde, self.Q_s)
        
        # construct constraints
        constraints += [
            x[:, 0] == x_0,       # initial condition
            x[2, self.H] == 0.,  # end with 0 accel
        ]
        # TODO: try ending at pred assumed and average of pred and succ assumed
        if type(y_succ_a) == np.ndarray:
            constraints += [
                y[:, self.H] == y_pred_a[:, self.H] - d_p_tilde,  # end at pred 
                # y[:, self.H] == 0.5 * (
                #     self.q_p * (y_pred_a[:, self.H] - d_p_tilde) + 
                #     self.q_s * (y_succ_a[:, self.H] - d_s_tilde)
                # )  # end at average of pred and succ
            ]
        else:
            constraints += [
                y[:, self.H] == y_pred_a[:, self.H] - d_p_tilde  # end at pred 
            ]
        for k in range(self.H):
            constraints += [
                x[:, k+1] == self.dyn.forward(x[:, k], u[:, k]),
                y[:, k] == self.dyn.sense(x[:, k]),
                x[:, k] >= self.x_min, 
                x[:, k] <= self.x_max, 
                u[:, k] >= self.u_min, 
                u[:, k] <= self.u_max
            ]
        
        # construct problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        return prob, x, y, u