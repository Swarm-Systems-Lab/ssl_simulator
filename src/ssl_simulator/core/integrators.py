import logging

import lieplusplus as lpp
import numpy as np

from ssl_simulator.math import check_and_parse_dimensions

SO3 = getattr(lpp, "SO3", None)

logger = logging.getLogger(__name__)


class EulerIntegrator:
    def integrate(self, context, dt, test=False):
        """
        Perform one step of Euler integration.

        Parameters
        ----------
            dynamics (callable): Function that computes state derivatives.
            state (dict): Current state of the system.
            dynamics_input (dict): Input to the dynamics function.
            dt (float): Time step for integration.
            test (bool): If True, perform dimension checks during integration.

        Returns
        -------
            dict: New state after integration.
        """
        state = context.get_robot_state()
        state_dot = context.get_robot_state_dot()

        # Perform checks on first call or whenever explicit test mode is requested.
        if test or not context.initialized:
            for key, state_value in state.items():
                dot_key = key + "_dot"
                if dot_key not in state_dot:
                    raise KeyError(f"Missing derivative '{dot_key}' for state '{key}'.")

                state_dot_value = state_dot[dot_key]
                if isinstance(state_value, np.ndarray):
                    check_and_parse_dimensions(
                        state_dot_value,
                        expected_shape=state_value.shape,
                        name=f"state_dot['{dot_key}']",
                    )
                elif SO3 is not None and isinstance(state_value, SO3):
                    state_dot_value = np.asarray(state_dot_value)
                    if state_dot_value.shape == (3, 3):
                        raise ValueError(
                            f"state_dot['{dot_key}'] is a 3x3 matrix, which is not allowed for integration."
                        )
                    if state_dot_value.shape not in [(3, 1), (1, 3), (3,)]:
                        raise ValueError(
                            f"state_dot['{dot_key}'] must have shape (3,) for SO3 integration, got {state_dot_value.shape}."
                        )
                    if state_dot_value.shape == (1, 3):
                        logger.warning(
                            f"state_dot['{dot_key}'] has shape (1, 3). Consider reshaping it to (3, 1) for consistency with SO3 integration."
                        )

        # Perform integration
        new_state = {}
        for key in state:
            if SO3 is not None and isinstance(state[key], SO3):
                state_dot_dt = dt * state_dot[key + "_dot"]
                if state_dot_dt.shape == (1, 3):
                    state_dot_dt = state_dot_dt.T
                new_state[key] = state[key] * SO3.exp(state_dot_dt)
            else:
                integration = state[key] + dt * state_dot[key + "_dot"]
                new_state.update({key: integration})
        return new_state


# class RK4Integrator: # TODO: implement properly
#     def integrate(self, dynamics, state, control_input, dt):
#         k1 = dynamics(state, control_input)
#         k2 = dynamics(state + 0.5 * dt * k1, control_input)
#         k3 = dynamics(state + 0.5 * dt * k2, control_input)
#         k4 = dynamics(state + dt * k3, control_input)
#         return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
