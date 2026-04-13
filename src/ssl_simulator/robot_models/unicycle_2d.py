""" """

import numpy as np

from ssl_simulator.core._robot_model import RobotModel
from ssl_simulator.math import check_and_parse_dimensions


class Unicycle2D(RobotModel):
    def __init__(self, context, initial_state, omega_lims=None):
        super().__init__(context)

        # Robot model state
        self.state = {
            "p": check_and_parse_dimensions(initial_state[0], (None, 2)),
            "speed": check_and_parse_dimensions(initial_state[1], (None,)),
            "theta": check_and_parse_dimensions(initial_state[2], (None,)),
        }

        self.omega_lims = omega_lims

        # Robot model state time variation
        self.state_dot = {}
        for key, value in self.state.items():
            self.state_dot.update({key + "_dot": value * 0})

        # Robot model control inputs
        self.control_inputs = {"omega": np.zeros_like(self.state["theta"])}

    # ---------------------------------------------------------------------------------

    def dynamics(self, time):
        state = self.state
        control_vars = self.control_inputs

        speed = state["speed"]
        theta = state["theta"]
        omega = control_vars["omega"] + np.zeros_like(theta)  # broadcasts to (N,)

        self.state_dot["p_dot"] = (speed * np.array([np.cos(theta), np.sin(theta)])).T

        if self.omega_lims is not None:
            self.state_dot["theta_dot"] = np.clip(omega, self.omega_lims[0], self.omega_lims[1])
        else:
            self.state_dot["theta_dot"] = omega

        return self.state_dot
