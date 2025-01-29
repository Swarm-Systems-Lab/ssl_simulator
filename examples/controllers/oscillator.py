"""
"""

__all__ = ["Oscillator"]

import numpy as np

from ssl_simulator.controllers import Controller

class Oscillator(Controller):
    def __init__(self, A, omega, speed):

        # Controller settings
        self.A = A
        self.omega = omega
        self.speed = speed

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "A_gamma": self.A,
            "omega_gamma": self.omega,
            "speed": self.speed,
            "gamma": None,
        }

        # Controller data
        self.init_data()

    def compute_control(self, time, state):
        """
        Follow y = gamma(t) = A * sin(w t) at constant speed ||v|| = s
        """
        N = state["p"].shape[0]
  
        if (self.A * self.omega > self.speed).any():
            raise ValueError("A * omega should be <= speed!")
    
        gamma = self.A * np.sin(self.omega * time)
        gamma_dot = self.A*self.omega * np.cos(self.omega * time) 
        x_dot = np.sqrt(self.speed**2 - gamma_dot**2)

        self.tracked_vars["gamma"] = gamma

        self.control_vars["u"] = np.zeros((N,2))
        self.control_vars["u"][:,0] = x_dot * np.ones(N)
        self.control_vars["u"][:,1] = gamma_dot * np.ones(N)

        return self.control_vars