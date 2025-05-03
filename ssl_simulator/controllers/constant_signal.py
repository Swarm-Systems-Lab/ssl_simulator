"""
"""

import numpy as np
from ssl_simulator import Controller

#######################################################################################

class ConstantSignal(Controller):
    def __init__(self, constant):

        # Controller settings
        self.constant = np.array([constant])

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "k": self.constant,
        }

        # Controller data
        self.init_data()
    
    # ---------------------------------------------------------------------------------
    def compute_control(self, time, state):
        self.control_vars["u"] = self.constant
        return self.control_vars

#######################################################################################