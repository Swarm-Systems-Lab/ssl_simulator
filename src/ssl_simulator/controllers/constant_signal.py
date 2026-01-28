""" """

import numpy as np

from ssl_simulator.core._controller import Controller

#######################################################################################


class ConstantSignal(Controller):
    def __init__(self, context, signal):
        super().__init__(context)

        # Controller variables
        self.signal = np.asarray(signal)
        self.ctrl_u = None

        # ---------------------------
        # Controller output variables
        self.control_vars = {
            "u": lambda: self.ctrl_u,
        }

        # Controller variables to be tracked by logger
        self.tracked_vars = {
            "k": lambda: self.signal,
        }

        # Controller interface for other controller to interact with it
        self.register_interface(self._set_signal)

    # ---------------------------------------------------------------------------------

    def _set_signal(self, signal):
        self.signal = signal

    def compute_control(self, time, dt):
        self.ctrl_u = self.signal


#######################################################################################
