"""
Basic Simulation Example
=========================

This example demonstrates the core workflow of the ssl_simulator package:
1. Configure logging (optional)
2. Set up initial conditions
3. Create and configure the simulation engine
4. Add robot model and controllers
5. Connect controllers to robot inputs
6. Run simulation
7. Load and visualize results

This is the simplest possible example using a constant velocity controller.
"""

import os

import numpy as np
from visualization import PlotBasic

from ssl_simulator import SimulationEngine, load_sim, set_log_format, set_log_level
from ssl_simulator.controllers import ConstantSignal
from ssl_simulator.robot_models import SingleIntegrator
from ssl_simulator.visualization import set_paper_parameters

# ======================================================================================
# CONFIGURATION
# ======================================================================================

# Optional: Configure logging fotmat
set_log_format("simple")  # (default)
# set_log_format("compact")

# Optional: Configure logging verbosity
# set_log_level("DEBUG")     # Show detailed execution info
set_log_level("INFO")  # Show general progress (default)
# set_log_level("WARNING")   # Show only warnings and errors

# Set plotting parameters for publication-quality figures
set_paper_parameters(fontsize=12)

# Define output paths
OUTPUT_DIR = os.path.join("..", "output")
SIMDATA_FILE = os.path.join(OUTPUT_DIR, "data_example.csv")

# ======================================================================================
# INITIAL CONDITIONS
# ======================================================================================

# Create a swarm of N robots with random initial positions
N = 5  # number of robots
positions = np.random.random((N, 2))  # Random positions in [0, 1] x [0, 1]

# Initial state for single integrator: [positions]
x0 = [positions]

# ======================================================================================
# SIMULATION SETUP
# ======================================================================================

# Create simulation engine
dt = 0.01  # Integration time step (seconds)
simulator = SimulationEngine(
    time_step=dt,
    log_filename=SIMDATA_FILE,
    log_time_step=0.1,  # Log every 0.1 seconds (reduces file size)
)

# Add robot model
simulator.set_robot_model(SingleIntegrator, x0)

# Add constant velocity controller
velocity = np.array([1.0, 1.0])  # Move at 1 m/s in x and y directions
simulator.add_controller("signal", ConstantSignal, velocity)

# Connect controller output to robot input
simulator.connect_controller_to_robot(
    controller_key="signal",
    mapping={"u": "u"},  # Map controller's "u" output to robot's "u" input
)

# ======================================================================================
# RUN SIMULATION
# ======================================================================================

tf = 1.0  # Simulation duration (seconds)
simulator.run(tf)

# ======================================================================================
# VISUALIZATION
# ======================================================================================

# Load simulation results
simulation_data, simulation_settings = load_sim(SIMDATA_FILE)

# Create and display plot
plotter = PlotBasic(simulation_data)
plotter.plot()
