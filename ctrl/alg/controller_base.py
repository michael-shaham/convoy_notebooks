from abc import ABC, abstractmethod
import numpy as np


class ControllerBase(ABC):
    """Base controller class. Takes some state(s) or state representation as 
       input and outputs control action(s)."""

    def __init__(self):
        pass

    @abstractmethod
    def control(self) -> np.ndarray:
        pass