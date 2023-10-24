import numpy as np

from .dynamics_base import DynamicsBase


class LinearLong(DynamicsBase):
    """
    Linear longitudinal dynamics, using the model described in Zhang et al., 
    'Stability and scalability of homogeneous vehicular platoon: Study on the 
    influence of information flow topologies.'

    Parameters:
        dt: discrete timestep
        x_min/max: state bounds
        u_min/max: control bounds
        tau: inertial lag of longitudinal dynamics
    """

    def __init__(self, 
                 dt: float,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray,
                 tau: float,
                 full_obs: bool = False):
        super().__init__(dt, x_min, x_max, u_min, u_max)
        self.tau = tau

        # continuous-time dynamics
        self.Ac = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1/tau]])
        self.Bc = np.array([[0], [0], [1/tau]])

        self.n = self.Ac.shape[0]  # x dimension
        self.m = self.Bc.shape[1]  # u dimension

        # discrete-time dynamics
        self.Ad = np.eye(self.n) + self.dt * self.Ac
        self.Bd = self.dt * self.Bc

        if full_obs:
            self.C = np.eye(self.n)
        else:
            self.C = np.array([[1, 0, 0], [0, 1, 0]])

        self.p = self.C.shape[0]   # y dimension

    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.Ad @ x + self.Bd @ u
    
    def sense(self, x: np.ndarray) -> np.ndarray:
        return self.C @ x