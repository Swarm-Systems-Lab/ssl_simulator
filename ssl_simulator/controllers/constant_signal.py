"""
"""

import numpy as np
from ssl_simulator import Controller

#######################################################################################

class ConstantSignal(Controller):
    def __init__(self, context, signal):
        super().__init__(context)

        # Controller settings
        self.signal = np.array([signal])

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": None,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "k": self.signal,
        }

        # Controller interface for other controller to interact with it
        self.register_interface(self._set_signal)
    
    # ---------------------------------------------------------------------------------
    def _set_signal(self, signal):
        self.signal = signal

    def compute_control(self, time):
        self.control_vars["u"] = self.signal

#######################################################################################