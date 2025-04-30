"""
"""

__all__ = ["Unicycle2D"]

import numpy as np

from ._robot_models import RobotModel

#######################################################################################

class Unicycle2D(RobotModel):
    def __init__(self, initial_state, omega_lims = None):

        # Robot model state
        self.state = {
            "p": initial_state[0],
            "speed": initial_state[1],
            "theta": initial_state[2],
        }

        self.omega_lims = omega_lims

        # Robot model state time variation
        self.state_dot = {}
        for key,value in self.state.items():
            self.state_dot.update({key+"_dot": value*0})

        # Robot model data
        self.init_data()

    # ---------------------------------------------------------------------------------

    def dynamics(self, state, control_vars):
        speed = state["speed"]
        theta = state["theta"]
        omega = next(iter(control_vars.values())) * np.ones(theta.shape)
        
        self.state_dot["p_dot"] = (speed * np.array([np.cos(theta), np.sin(theta)])).T

        if self.omega_lims is not None:
            self.state_dot["theta_dot"] = np.clip(omega, self.omega_lims[0], self.omega_lims[1])
        else:
            self.state_dot["theta_dot"] = omega

        #print(self.state["p"].shape, self.state_dot["p_dot"].shape)

        return self.state_dot

#######################################################################################