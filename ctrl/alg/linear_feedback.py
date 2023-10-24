from .controller_base import ControllerBase


class LinearFeedback(ControllerBase):
    """
    Linear feedback controller. Uses the position error, velocity error, and 
    (sometimes) acceleration error to calculate a control command.

    Paramters:
        k_p: position gain
        k_v: velocity gain
        k_a: acceleration gain
    """

    def __init__(self, k_p: float, k_v: float, k_a: float):
        self.k_p = k_p
        self.k_v = k_v
        self.k_a = k_a
    
    def control(self, pos_err: float, vel_err: float, accel_err: float) -> float:
        return -(self.k_p * pos_err + self.k_v * vel_err + self.k_a * accel_err)