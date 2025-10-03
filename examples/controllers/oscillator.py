"""
"""

__all__ = ["Oscillator"]

import numpy as np

from ssl_simulator import Controller

class Oscillator(Controller):
    def __init__(self, context, A, omega, speed):
        super().__init__(context)
        # Controller settings
        self.A = A
        self.omega = omega
        self.speed = speed

        # Controller variables
        self.gamma = None

        # ---------------------------        
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "A": self.A,
            "omega": self.omega,
            "speed": self.speed,
            "gamma": self.gamma,
        }

        # Controller interface for other controller to interact with it
        self.register_interface(self._set_osc_omega)

    def _set_osc_omega(self, omega):
        self.omega = omega

    def compute_control(self, time, dt):
        """
        Follow y = gamma(t) = A * sin(w t) at constant speed ||v|| = s
        """
        state = self.context.get_robot_state()
        N = state["p"].shape[0]
  
        if (self.A * self.omega > self.speed).any():
            raise ValueError("A * omega should be <= speed!")
    
        gamma = self.A * np.sin(self.omega * time)
        gamma_dot = self.A*self.omega * np.cos(self.omega * time) 
        x_dot = np.sqrt(self.speed**2 - gamma_dot**2)

        self.gamma = gamma
        self.tracked_vars["gamma"] = self.gamma

        self.control_vars["u"] = np.zeros((N,2))
        self.control_vars["u"][:,0] = x_dot
        self.control_vars["u"][:,1] = gamma_dot