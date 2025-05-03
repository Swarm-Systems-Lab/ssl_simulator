"""
"""
from tqdm import tqdm

from .integrators import EulerIntegrator
from .loggers import DataLogger, RealTimeLogger

INTEGRATORS = {
    "euler": EulerIntegrator, 
    #"rk4": RK4Integrator
}

#######################################################################################

class SimulationEngine:
    def __init__(self, robot_model, controller, time_step=0.01, integrator="euler",
                 log_filename=None, log_time_step=None):
        
        self.robot_model = robot_model
        self.controller = controller
        self.log_time_step = log_time_step
        
        self.set_time_step(time_step)
        self.time = 0.0
        self.current_step = 1

        # Initialize integrator
        if integrator in INTEGRATORS: # check if the integrator exists in INTEGRATORS
            self.integrator = INTEGRATORS[integrator]()
        else:
            supported_integrators = ", ".join(INTEGRATORS.keys())
            raise ValueError(   
                f"Unsupported integrator '{integrator}'. "
                f"Supported integrators are: {supported_integrators}."
            )
        
        # Set labels of the variables to be tracked
        labels = [*self.robot_model.get_labels(), *self.controller.get_labels()]

        # Initialize logger
        if log_filename is not None:
            self.logger = DataLogger(labels, log_filename)
        else:
            self.logger = RealTimeLogger(labels)
        
        # Log settings and initial state 
        state = self.robot_model.get_state()
        self.controller.compute_control(self.time, state)

        self._log_settings()
        self._log_data()

    def run(self, duration, eta=True):
        steps = int(duration / self.time_step)
        for _ in tqdm(range(steps), desc="Running simulation", disable=not eta):
            self.step()

    def set_time_step(self, time_step):
        self.time_step = time_step
        
        # Set logger log interval_steps
        if self.log_time_step is not None:
            if self.log_time_step < time_step:
                raise ValueError(
                    f"log_time_step ({self.log_time_step}) must be greater than or equal to time_step ({time_step})."
                )
            self.log_interval_steps = int(round(self.log_time_step / time_step))
        else:
            self.log_interval_steps = None

    def _log_settings(self):
        pass

    def _log_data(self):
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
        if self.log_interval_steps is None:
            self._log_data()
        elif self.current_step % self.log_interval_steps == 0:
            self._log_data()

        self.current_step += 1

#######################################################################################