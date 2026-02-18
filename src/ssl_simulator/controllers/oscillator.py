""" """

__all__ = ["Oscillator"]

import numpy as np

from ssl_simulator.core._controller import Controller


class Oscillator(Controller):
    def __init__(self, context, A, omega, speed):
        super().__init__(context)

        robot_state = self.context.get_robot_state()
        if not robot_state:
            raise ValueError("Robot state is empty. Oscillator requires a non-empty robot state.")
        first_state = next(iter(robot_state.values()))
        self.n_agents = first_state.shape[0]

        # Controller variables
        self.A = A
        self.omega = omega
        self.speed = speed
        self.gamma = None
        self.ctrl_u = None

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": lambda: self.ctrl_u,
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

    # ---------------------------------------------------------------------------------

    def _set_osc_omega(self, omega):
        self.omega = omega

    def compute_control(self, time, dt):
        """Follow y = gamma(t) = A * sin(w t) at constant speed ||v|| = s."""
        if np.any(self.A * self.omega > self.speed):
            raise ValueError("A * omega should be <= speed!")

        self.gamma = self.A * np.sin(self.omega * time)
        gamma_dot = self.A * self.omega * np.cos(self.omega * time)
        x_dot = np.sqrt(self.speed**2 - gamma_dot**2)

        self.ctrl_u = np.zeros((self.n_agents, 2))
        self.ctrl_u[:, 0] = x_dot
        self.ctrl_u[:, 1] = gamma_dot
