import numpy as np

from ssl_simulator.core._robot_model import RobotModel
from ssl_simulator.math import check_and_parse_dimensions


class SingleIntegrator(RobotModel):
    def __init__(self, context, initial_state):
        super().__init__(context)

        # Robot model state
        self.state = {"p": check_and_parse_dimensions(initial_state[0], (None, [1, 2, 3]))}

        # Robot model state time variation
        self.state_dot = {}
        for key, value in self.state.items():
            self.state_dot.update({key + "_dot": value * 0})

        # Robot model control inputs
        self.control_inputs = {"u": np.zeros_like(self.state["p"])}

        # Cache shape info validated at init to skip re-validation on every dynamics call
        self._n_dims: int = self.state["p"].shape[1]
        self._n_agents: int = self.state["p"].shape[0]

    # ---------------------------------------------------------------------------------
    def dynamics(self, time):
        state = self.state
        ctrl_vars = self.control_inputs

        # np.atleast_2d: (m,) → (1, m); (N, m) stays (N, m). Avoids the full
        # check_and_parse_dimensions validation on every step (shape was verified at init).
        u = np.atleast_2d(ctrl_vars["u"])

        self.state_dot["p_dot"] = np.broadcast_to(u, state["p"].shape).copy()
        return self.state_dot
