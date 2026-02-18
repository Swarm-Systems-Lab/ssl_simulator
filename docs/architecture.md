# Architecture

This page explains the core design of `ssl_simulator`: how robot models, controllers, integrators, and the simulation engine are structured and how they connect during a simulation run.

---

## Layered overview

The simulator is organised in three layers. User code (experiments, notebooks) sits at the top and only talks to `SimulationEngine`. The engine owns a `SimulationContext` that wires robot model, controllers, and the integrator together. The math/utility packages support all layers laterally.

```
┌──────────────────────────────────────────────────────────┐
│  User code  │  Examples / notebooks / custom scripts      │
├──────────────────────────────────────────────────────────┤
│  SimulationEngine  (time-loop + DataLogger)               │
│    └─ SimulationContext                                   │
│         ├─ RobotModel   (state, dynamics)                 │
│         ├─ Controller…  (control_vars, interface)         │
│         └─ EulerIntegrator  (state ← state + dt·ẋ)       │
├──────────────────────────────────────────────────────────┤
│  Support  │  math · components · visualization · utils   │
└──────────────────────────────────────────────────────────┘
```

---

## Core concepts

### SimulationEngine

`SimulationEngine` is the single entry point for running a simulation. It owns:

- a `SimulationContext` (the wiring layer)
- an `Integrator` (numerical integration strategy)
- a `DataLogger` (in-memory ring-buffer + optional CSV sink)
- the global simulation clock (`time`, `time_step`, `current_step`)

Calling `engine.run(duration)` executes a fixed-step loop. Each iteration calls:

1. `context.compute_controls(t, dt)` — ask every controller to update its output variables
2. `context.compute_robot_dynamics(t)` — ask the robot model to compute `state_dot`
3. `integrator.integrate(context, dt)` — advance the state
4. `context.set_robot_state(new_state)` — write back
5. `DataLogger.log(t, data)` — record the snapshot (at the configured log frequency)

### SimulationContext

`SimulationContext` is the shared blackboard that binds everything together. It holds:

| Attribute | Type | Purpose |
|---|---|---|
| `robot_model` | `RobotModel \| None` | The single kinematic model for all agents |
| `controllers` | `dict[str, Controller]` | Named controllers (multiple allowed) |
| `connections` | `list[tuple]` | Named mappings from controller outputs to robot inputs |
| `control_vars` | `dict` | Aggregated last control output snapshot |
| `ctrl_interfaces` | `dict` | Callable interfaces controllers expose to each other |

Controllers are stacked: `compute_controls` iterates them in **reverse insertion order** so a high-level controller can feed into a lower-level one without needing explicit references.

Connections are declared once with `connect_controller_to_robot()`, specifying which `control_vars` key maps to which `control_inputs` key in the robot model.

### RobotModel

A `RobotModel` describes the kinematics for a **fleet of N agents** (arrays are always shape `(N, …)`). Subclasses must define:

| Attribute | Description |
|---|---|
| `state` | `dict` of named state arrays (e.g. `"p"`, `"theta"`) |
| `state_dot` | Matching `dict` for time derivatives (`"p_dot"`, `"theta_dot"`) |
| `control_inputs` | `dict` of inputs the model reads from connected controllers |
| `dynamics(time)` | Fills `state_dot` from `state` + `control_inputs`, returns `state_dot` |

**Built-in models**

| Class | State | Control input |
|---|---|---|
| `SingleIntegrator` | `p` — position in ℝ^m | `u` — velocity |
| `Unicycle2D` | `p`, `speed`, `theta` | `omega` — angular rate |

### Controller

A `Controller` computes one or more **control variables** that are fed to the robot model (or to other controllers). Subclasses define:

| Attribute | Description |
|---|---|
| `control_vars` | `dict` of output variables (may be lazy callables) |
| `tracked_vars` | Extra variables recorded by the logger |
| `control_interface` | Named callables other controllers can invoke |
| `compute_control(time, dt)` | Runs the control law, updates `control_vars` |

Controllers can **register an interface** (`register_interface(*methods)`) so that a higher-level controller can change their internal parameters (e.g. an outer loop adjusting the frequency of an `Oscillator`).

**Built-in controllers**

| Class | What it does |
|---|---|
| `Oscillator` | Generates a sinusoidal trajectory `γ(t) = A sin(ωt)` at a given speed |
| `ConstantSignal` | Emits a fixed control vector |

### Integrator

Integrators implement a single method `integrate(context, dt) → new_state`.

The only shipped integrator is `EulerIntegrator`:

$$x(t+\Delta t) = x(t) + \Delta t \cdot \dot{x}(t)$$

For `SO3` states (rotation matrices) it uses the Lie-group exponential map provided by `lieplusplus`:

$$R(t+\Delta t) = R(t) \cdot \exp\bigl(\Delta t\, \hat{\omega}(t)\bigr)$$

A `RK4Integrator` skeleton exists in the file but is not yet implemented.

### DataLogger

`DataLogger` records every tracked quantity at a configurable interval. Data is:

- kept in a NumPy ring-buffer in memory (`log_size` controls buffer depth; `None` keeps all history)
- optionally streamed to a CSV file with a JSON settings header for reproducibility

After `engine.run()`, results are accessible at `engine.data` (dict of arrays) and `engine.settings`.

---

## Step-by-step simulation loop

```
engine.run(T)
│
├─ [once] DataLogger initialised with labels from context
│  └─ _step_test() + _log_data()  ← logs t=0
│
└─ for each step 1 … T/dt
   ├─ context.compute_controls(t, dt)
   │    ├─ controller_N.compute_control(t, dt)   ← high-level first (reversed)
   │    ├─ …
   │    └─ controller_0.compute_control(t, dt)   ← lowest-level last
   │         └─ propagates control_vars → robot_model.control_inputs via connections
   ├─ context.compute_robot_dynamics(t)
   │    └─ robot_model.dynamics(t)  → fills state_dot
   ├─ integrator.integrate(context, dt)  → new_state
   ├─ context.set_robot_state(new_state)
   └─ DataLogger.log(t, data)   ← at log_interval_steps cadence
```

---

## Extending the simulator

### Add a new robot model

```python
from ssl_simulator.core._robot_model import RobotModel

class MyRobot(RobotModel):
    def __init__(self, context, initial_state):
        super().__init__(context)
        self.state = {"p": initial_state}
        self.state_dot = {"p_dot": initial_state * 0}
        self.control_inputs = {"u": np.zeros_like(initial_state)}

    def dynamics(self, time):
        self.state_dot["p_dot"] = self.control_inputs["u"]
        return self.state_dot
```

Place it in `src/ssl_simulator/robot_models/` and export it from `__init__.py`.

### Add a new controller

```python
from ssl_simulator.core._controller import Controller

class MyController(Controller):
    def __init__(self, context, gain):
        super().__init__(context)
        self.gain = gain
        self.ctrl_u = None
        self.control_vars = {"u": lambda: self.ctrl_u}

    def compute_control(self, time, dt):
        state = self.context.get_robot_state()
        self.ctrl_u = -self.gain * state["p"]
```

Place it in `src/ssl_simulator/controllers/` and export it from `__init__.py`.

---

## Package map

| Package | Key contents |
|---|---|
| `core` | `SimulationEngine`, `SimulationContext`, `RobotModel`, `Controller`, `EulerIntegrator`, `DataLogger`, `types` |
| `robot_models` | `SingleIntegrator`, `Unicycle2D` |
| `controllers` | `Oscillator`, `ConstantSignal` |
| `components` | `network`, `scalar_fields`, `gvf` — reusable building blocks for controllers |
| `math` | Algebra, Lie groups, distributions, graph utilities, source-seeking helpers |
| `visualization` | Plotting helpers for trajectories, scalar fields, and SO(3) data |
| `utils` | File I/O, path helpers, dict operations, debugging |
| `apps` | Ready-made experiment wrappers |
| `examples` | Jupyter notebooks illustrating typical use cases |
