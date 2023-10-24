import numpy as np

from .dynamics_base import DynamicsBase


class NonlinearLong(DynamicsBase):
    """Nonlinear longitudinal dynamics, using the model described in Zhang et 
    al., 'Distributed model predictive control for heterogeneous vehicle 
    platoons under unidirectional topologies.'

    Parameters:
        dt: discrete timestep
        x_min/max: state bounds
        u_min/max: control bounds
        m: vehicle mass
        C_A: aerodynamic drag coefficient
        g: gravitational constant
        f_r: coefficient of rolling resistance
        tau: inertial lag of longitudinal dynamics
        R: tire radius
        eta_T: mechanical efficiency of driveline
    """

    def __init__(self,
                 dt: float,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray,
                 m: float, c_a: float, g: float, f_r: float, tau: float, 
                 r: float, eta_T: float):
        super().__init__(dt, x_min, x_max, u_min, u_max)
        self.m = m
        self.c_a = c_a
        self.g = g
        self.f_r = f_r
        self.tau = tau
        self.r = r
        self.eta_T = eta_T

        self.B = np.array([[0], [0], [dt / tau]])
        self.C = np.array([[1, 0, 0],
                            [0, 1, 0]])
        self.n, self.m = self.B.shape
        self.p = self.C.shape[0]
    
    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.f(x) + self.B @ u
    
    def sense(self, x: np.ndarray) -> np.ndarray:
        return self.C @ x

    def _F_veh(self, v: float) -> float:
        return self.c_a * v**2 + self.m * self.g * self.f_r
    
    def _f(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            x[0] + self.dt * x[1],
            x[1] + self.dt/self.m*(self.eta_T/self.r*x[2] - self.F_veh(x[1])),
            x[2] - self.dt / self.tau * x[2]
        ])
    
    def h(self, v: float) -> float:
        return self.r / self.eta_T * self.F_veh(v)