from alg.controller_base import ControllerBase
from dyn.dynamics_base import DynamicsBase

import cvxpy as cp
import numpy as np


class LinearMPC(ControllerBase):
    """
    Vanilla linear MPC controller.

    Parameters:
        Q: state stage cost
        Q_f: final state stage cost (can be None)
        R: input stage cost
        dyn: dynamics class based on DynamicsBase, linear
        x_min/max: state bounds (assume box constraints)
        u_min/max: input bounds (assume box constraints)
        T: horizon
        solver: string, options are 'cvx', 'gurobi'
    """

    def __init__(self, 
                 Q: np.ndarray, Q_f: np.ndarray, R: np.ndarray, 
                 dyn: DynamicsBase,
                 x_min: np.ndarray, x_max: np.ndarray, 
                 u_min: np.ndarray, u_max: np.ndarray,
                 T: int,
                 solver='cvx'):
        self.Q, self.Q_f, self.R = Q, Q_f, R
        self.dyn = dyn
        self.x_min, self.x_max = x_min, x_max
        self.u_min, self.u_max = u_min, u_max
        self.T = T
        self.n = self.dyn.n
        self.m = self.dyn.m
        self.solver = solver

    def control(self, x_0: np.ndarray, x_T: np.ndarray, z: np.ndarray):
        """
        Returns the optimal control sequence u_0, ..., u_{T-1}, the 
        corresponding states x_0, ..., x_{T}, and the optimal cost.

        Parameters:
            x_0: starting state/initial condition
            x_T: desired end state, can be None
            z: reference trajectory to track
        """
        if self.solver == 'cvx':
            prob, x, u = self.mpc_problem(x_0, x_T, z)
            prob.solve()
            return u.value, x.value, prob.value
    
    def mpc_problem(self, x_0: np.ndarray, x_T: np.ndarray, z: np.ndarray):
        """
        Constructs the MPC problem using the solver specified in the 
        constructor.
        """
        if self.solver == 'cvx':
            x = cp.Variable((self.n, self.T + 1))
            u = cp.Variable((self.m, self.T))
            cost = 0.
            constraints = []

            # construct cost
            for t in range(self.T):
                cost += cp.quad_form(x[:, t] - z[:, t], self.Q)
                cost += cp.quad_form(u[:, t], self.R)
            if type(self.Q_f) == np.ndarray:
                cost += cp.quad_form(x[:, self.T] - z[:, self.T], self.Q_f)

            # construct constraints
            constraints += [x[:, 0] == x_0]
            if type(x_T) == np.ndarray:
                constraints += [x[:, self.T] == x_T]
            for t in range(self.T):
                constraints += [
                    x[:, t] >= self.x_min, 
                    x[:, t] <= self.x_max, 
                    u[:, t] >= self.u_min, 
                    u[:, t] <= self.u_max, 
                    x[:, t+1] == self.dyn.forward(x[:, t], u[:, t])
                ]
            constraints += [x[:, self.T] >= self.x_min,
                            x[:, self.T] <= self.x_max]

            # construct problem
            prob = cp.Problem(cp.Minimize(cost), constraints)

            return prob, x, u