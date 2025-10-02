import numpy as np
from ssl_simulator import RobotModel

#######################################################################################

class SingleIntegrator(RobotModel):
    def __init__(self, context, initial_state):
        super().__init__(context)

        # Robot model state
        self.state = {
            "p": initial_state[0],
        }

        # Robot model state time variation
        self.state_dot = {}
        for key,value in self.state.items():
            self.state_dot.update({key+"_dot": value*0})

        # Robot model control inputs
        self.control_inputs = {
            "u": np.zeros_like(self.state["p"])
        }

    # ---------------------------------------------------------------------------------
    def dynamics(self, time):
        state = self.state
        ctrl_vars = self.control_inputs
        
        u = np.atleast_2d(ctrl_vars["u"]) # shape becomes (1, m) if 1D
        self.state_dot["p_dot"] = u + np.zeros_like(state["p"])  # broadcasts to (N, m)
        return self.state_dot

#######################################################################################