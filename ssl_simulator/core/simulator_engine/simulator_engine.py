"""
"""
from tqdm import tqdm

from . integrators import *
from ..data_manager.logger import DataLogger

from .integrators import INTEGRATORS

#######################################################################################

class SimulationEngine:
    def __init__(self, robot_model, controller, time_step=0.01, integrator="euler"):
        
        self.robot_model = robot_model
        self.controller = controller
        self.time_step = time_step
        
        # Check if the integrator exists in INTEGRATORS
        if integrator in INTEGRATORS:
            self.integrator = INTEGRATORS[integrator]()
        else:
            supported_integrators = ", ".join(INTEGRATORS.keys())
            raise ValueError(
                f"Unsupported integrator '{integrator}'. "
                f"Supported integrators are: {supported_integrators}."
            )
        
        self.time = 0.0

        # Initialize the logger with the labels of the variables to be tracked
        labels = [*self.robot_model.get_labels(), *self.controller.get_labels()]

        self.logger = DataLogger(labels)

        # Log initial state of the system
        state = self.robot_model.get_state()
        self.controller.compute_control(self.time, state)
        self.log_data()

    def log_data(self):
        data = self.robot_model.get_data()
        data.update(self.controller.get_data())
        self.logger.log(self.time, data)

    def step(self):
        # Get the actual robots' state
        state = self.robot_model.get_state()

        # Calculate the control action
        control_input = self.controller.compute_control(self.time, state)

        # Integrate the robots' dynamics
        new_state = self.integrator.integrate(self.robot_model.dynamics, state, control_input, self.time_step)
        self.time += self.time_step

        # Update robots' state
        self.robot_model.set_state(new_state)

        # Log data
        self.log_data()

    def run(self, duration, eta=True):
        steps = int(duration / self.time_step)
        for _ in tqdm(range(steps), desc="Running simulation", disable=not eta):
            self.step()

#######################################################################################