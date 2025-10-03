__all__ = ["DummyOscillatorHL"]

import numpy as np
from ssl_simulator import Controller

class DummyOscillatorHL(Controller):
    def __init__(self, context, oscillator_key):
        super().__init__(context)

        self.oscillator_key = oscillator_key

    def compute_control(self, time, dt):
        self.context.call_interface(self.oscillator_key, "_set_osc_omega", time*np.pi/10)