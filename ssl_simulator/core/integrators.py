"""
"""

#######################################################################################

class EulerIntegrator:
    def integrate(self, dynamics, state, dynamics_input, dt):
        new_state = {}
        state_dot = dynamics(state, dynamics_input)
        for key in state.keys():
            integration = state[key] + dt * state_dot[key+"_dot"]
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

