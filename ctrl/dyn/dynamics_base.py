from abc import ABC, abstractmethod
import numpy as np


class DynamicsBase(ABC):
    """Base dynamics class. Takes as input state and control limits."""

    def __init__(self, dt: float,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray):
        self.dt = dt
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max

    @abstractmethod
    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sense(self, x: np.ndarray) -> np.ndarray:
        pass