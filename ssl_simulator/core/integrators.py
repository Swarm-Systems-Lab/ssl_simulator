"""
"""

# TODO: global settings for integrators (e.g., step size for exp map)
from numpy import pi
from ssl_simulator.math.lie import so3_rotate_with_step
from ssl_simulator.math import check_and_parse_dimensions

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
        if test:
            for key in state.keys():
                if key + "_dot" in state_dot:
                    check_and_parse_dimensions(
                        state_dot[key + "_dot"],
                        expected_shape=state[key].shape,
                        name=f"state_dot[{key}]"
                    )

        # Perform integration
        new_state = {}
        for key in state.keys():
            if key == "R":  # special case for rotation matrices
                omega_hat = state_dot["R_dot"]
                new_state["R"] = so3_rotate_with_step(state["R"], dt * omega_hat, step=pi/12)
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

