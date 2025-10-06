from tqdm import tqdm

from ._robot_model import RobotModel
from .integrators import EulerIntegrator
from .loggers import DataLogger
from .simulation_context import SimulationContext

INTEGRATORS = {
    "euler": EulerIntegrator, 
    #"rk4": RK4Integrator
}

#######################################################################################

class SimulationEngine:
    def __init__(
        self,
        time_step: float = 0.01,
        integrator: str = "euler",
        log_filename: str = None,
        log_time_step: float = None,
        log_size: int = 10
    ):
        self.context = SimulationContext()

        self.log_filename = log_filename
        self.log_time_step = log_time_step
        self.log_size = log_size
        
        self._set_time_step(time_step)
        self.time = 0.0
        self.current_step = 1

        # Initialize the integrator
        if integrator in INTEGRATORS: # check if the integrator exists in INTEGRATORS
            self.integrator = INTEGRATORS[integrator]()
        else:
            supported_integrators = ", ".join(INTEGRATORS.keys())
            raise ValueError(   
                f"Unsupported integrator '{integrator}'. "
                f"Supported integrators are: {supported_integrators}."
            )

    def __getattr__(self, name):
        """
        Delegate attribute access to self.context if the attribute is not found in self.
        """
        if hasattr(self.context, name):
            return getattr(self.context, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _set_time_step(self, time_step):
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

    def _step_test(self):
        self.context.compute_controls(self.time, self.time_step)
        self.context.compute_robot_dynamics(self.time)
        self.integrator.integrate(self.context, self.time_step, test=True)
        
    def _log_data(self):
        data = self.context.get_data()
        self.logger.log(self.time, data)

    def run(self, duration, eta=True):
        if not self.context.initialized:
            # Initialize logger
            labels = self.context.get_labels()     # labels of the variables to be tracked
            settings = self.context.get_settings() # settings
            self.logger = DataLogger(labels, self.log_filename, self.log_size, settings)
            self.data = self.logger.data           # shortcut to avoid refering to logger
            self.settings = self.logger.settings   # shortcut to avoid refering to logger
            
            # Log settings and initial state
            self._step_test()
            self._log_data()
            self.context.initialized = True

        steps = int(duration / self.time_step)
        for _ in tqdm(range(steps), desc="Running simulation", disable=not eta):
            self.step()

    def step(self):
        # Integrate the robots' dynamics
        self.context.compute_controls(self.time, self.time_step)
        self.context.compute_robot_dynamics(self.time)
        new_state = self.integrator.integrate(self.context, self.time_step)
        self.time += self.time_step

        # Update robots' state
        self.context.set_robot_state(new_state)

        # Log data
        if self.log_interval_steps is None:
            self._log_data()
        elif self.current_step % self.log_interval_steps == 0:
            self._log_data()

        self.current_step += 1

#######################################################################################