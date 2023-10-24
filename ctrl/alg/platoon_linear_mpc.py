from .controller_base import ControllerBase
from dyn.dynamics_base import DynamicsBase

import cvxpy as cp
import numpy as np


class PlatoonLinearMPC(ControllerBase):
    """
    Platoon distributed MPC controller.

    Parameters:
        Q: desired trajectory stage cost
        R: input stage cost
        F: assumed trajectory deviation stage cost
        G: neighbor assumed trajectory deviation stage cost
        dyn: dynamics base class based on DynamicsBase, linear
        x_min/max: state bounds (assume box constraints)
        u_min/max: input bounds (assume box constraints)
        T: horizon
    """
    def __init__(self, 
                 Q: np.ndarray, R: np.ndarray, F: np.ndarray, G: np.ndarray,
                 dyn: DynamicsBase,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray,
                 T: int,
                 solver='cvx'):
        self.Q, self.R, self.F, self.G = Q, R, F, G
        self.dyn = dyn
        self.x_min, self.x_max = x_min, x_max
        self.u_min, self.u_max = u_min, u_max
        self.T = T
        self.n = self.dyn.n
        self.m = self.dyn.m
        self.p = self.dyn.p
        self.solver = solver

    def control(self, x_0: np.ndarray, y_a: np.ndarray, 
                y_neighbor_a: list, d_neighbor: list):
        if self.solver == 'cvx':
            prob, x, y, u = self.mpc_problem(x_0, y_a, y_neighbor_a, 
                                             d_neighbor)
            prob.solve()
            return u.value, x.value, y.value, prob.value

    def mpc_problem(self, x_0: np.ndarray, y_a: np.ndarray, 
                    y_neighbor_a: list, d_neighbor: list) -> cp.Problem:
        d_tilde = [np.array([d_neighbor[i], 0.]) 
                   for i in range(len(d_neighbor))]

        if self.solver == 'cvx':
            x = cp.Variable((self.n, self.T + 1))
            y = cp.Variable((self.p, self.T + 1))
            u = cp.Variable((self.m, self.T))
            cost = 0.
            constraints = []

            # construct cost
            for t in range(self.T):
                cost += cp.quad_form(u[:, t], self.R)
                cost += cp.quad_form(y[:, t] - y_a[:, t], self.F)
                for i in range(len(y_neighbor_a)):
                    cost += cp.quad_form(
                        y[:, t] - y_neighbor_a[i][:, t] + d_tilde[i], self.G
                    )
            
            # construct constraints
            constraints += [
                x[:, 0] == x_0,                       # initial condition
                y[:, 0] == self.dyn.sense(x[:, 0]),   # initial condition
                x[2, self.T] == 0.,                   # end with 0 acceleration
                y[:, self.T] == 1 / len(y_neighbor_a) * \
                    cp.sum([y_neighbor_a[i][:, self.T] - d_tilde[i] 
                            for i in range(len(y_neighbor_a))])
            ]
            for t in range(self.T):
                constraints += [
                    x[:, t+1] == self.dyn.forward(x[:, t], u[:, t]),
                    y[:, t+1] == self.dyn.sense(x[:, t+1]),
                    x[:, t] >= self.x_min, 
                    x[:, t] <= self.x_max, 
                    u[:, t] >= self.u_min, 
                    u[:, t] <= self.u_max
                ]
            constraints += [
                y[:, self.T] == self.dyn.sense(x[:, self.T]),
                x[:, self.T] >= self.x_min,
                x[:, self.T] <= self.x_max
            ]

            # construct problem
            prob = cp.Problem(cp.Minimize(cost), constraints)

            return prob, x, y, u
