import numpy as np
import lieplusplus as lpp

from ssl_simulator.math.lie import so3_rotate_with_step
from ssl_simulator.math import check_and_parse_dimensions

from ssl_simulator.config import CONFIG

#######################################################################################

class EulerIntegrator:
    def integrate(self, context, dt, test=False):
        """
        Perform one step of Euler integration.

        Parameters:
            dynamics (callable): Function that computes state derivatives.
            state (dict): Current state of the system.
            dynamics_input (dict): Input to the dynamics function.
            dt (float): Time step for integration.
            test (bool): If True, perform dimension checks during integration.

        Returns:
            dict: New state after integration.
        """
        state = context.get_robot_state()
        state_dot = context.get_robot_state_dot()

        # Perform dimension checks if test mode is enabled
        if not context.initialized:
            for key in state.keys():
                if key + "_dot" in state_dot:
                    if isinstance(state[key], np.ndarray):
                        check_and_parse_dimensions(
                            state_dot[key + "_dot"],
                            expected_shape=state[key].shape,
                            name=f"state_dot[{key}]"
                        )

        # Perform integration
        new_state = {}
        for key in state.keys():
            if isinstance(state[key], lpp.SO3):
                if isinstance(state_dot[key + "_dot"], np.ndarray) and state_dot[key + "_dot"].shape == (3, 3):
                    raise ValueError(f"state_dot[{key + '_dot'}] is a 3x3 matrix, which is not allowed for integration.")
                new_state[key] = state[key] * lpp.SO3.exp(dt * state_dot[key + "_dot"])
                # so3_rotate_with_step(state["R"], dt * omega_hat, step=CONFIG["SO3_STEP"])
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

#######################################################################################

