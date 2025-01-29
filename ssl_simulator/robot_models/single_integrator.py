"""
"""

__all__ = ["SingleIntegrator"]

import numpy as np

from ._robot_model import RobotModel

#######################################################################################

class SingleIntegrator(RobotModel):
    def __init__(self, initial_state):

        # Robot model state
        self.state = {
            "p": initial_state[0],
        }

        # Robot model state time variation
        self.state_dot = {}
        for key,value in self.state.items():
            self.state_dot.update({key+"_dot": value*0})

        # Robot model data
        self.init_data()

    # ---------------------------------------------------------------------------------
    def dynamics(self, state, control_vars):
        self.state_dot["p_dot"] = next(iter(control_vars.values())) * np.ones(state["p"].shape)
        return self.state_dot

#######################################################################################