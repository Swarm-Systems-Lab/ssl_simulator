"""
"""
from tqdm import tqdm

from ._controller_manager import ControllerManager
from ._robot_model import RobotModel
from .integrators import EulerIntegrator
from .loggers import DataLogger

INTEGRATORS = {
    "euler": EulerIntegrator, 
    #"rk4": RK4Integrator
}

#######################################################################################

class SimulationEngine:
    def __init__(
        self,
        robot_model: RobotModel,
        controller_manager: ControllerManager,
        time_step: float = 0.01,
        integrator: str = "euler",
        log_filename: str = None,
        log_time_step: float = None,
        log_size: int = 10
    ):
        
        self.robot_model = robot_model
        self.controller_manager = controller_manager
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
        
        # Set labels of the variables to be tracked and settings
        labels = [*self.robot_model.get_labels(), *self.controller_manager.get_labels()]
        settings = self.robot_model.get_settings()
        settings.update(self.controller_manager.get_settings())

        # Initialize logger
        self.logger = DataLogger(labels, log_filename, log_size, settings)
        self.data = self.logger.data # shortcut to avoid refering to logger
        self.settings = self.logger.settings # shortcut to avoid refering to logger
        
        # Log settings and initial state 
        state = self.robot_model.get_state()
        self.controller_manager.compute_controls(self.time, state)

        self._step_test()
        self._log_data()

    def _log_data(self):
        data = self.robot_model.get_data()
        data.update(self.controller_manager.get_data())
        self.logger.log(self.time, data)

    def _step_test(self):
        state = self.robot_model.get_state()
        control_input = self.controller_manager.compute_controls(self.time, state)
        self.integrator.integrate(self.robot_model.dynamics, state, control_input, self.time_step, debug=True)

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

    def step(self):
        # Get the actual robots' state
        state = self.robot_model.get_state()

        # Calculate the control action using the ControllerManager
        control_input = self.controller_manager.compute_controls(self.time, state)

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