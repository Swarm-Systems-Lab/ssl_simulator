"""
"""

# TODO: global settings for integrators (e.g., step size for exp map)
from numpy import pi
from ssl_simulator.math.lie import so3_rotate_with_step

#######################################################################################

class EulerIntegrator:
    def integrate(self, dynamics, state, dynamics_input, dt):
        new_state = {}
        state_dot = dynamics(state, dynamics_input)
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

