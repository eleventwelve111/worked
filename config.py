import numpy as np
import os
import json

# Unit conversions
CM_TO_FT = 30.48  # 1 foot = 30.48 cm
INCH_TO_CM = 2.54  # 1 inch = 2.54 cm

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
STATE_POINT_DIR = os.path.join(BASE_DIR, "statepoints")

# Create directories if they don't exist
for directory in [RESULTS_DIR, PLOTS_DIR, STATE_POINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Simulation parameters
BATCHES = 150
PARTICLES_PER_BATCH = 10000
INACTIVE_BATCHES = 10
SEED = 42

# Physical dimensions (all in cm)
WALL_THICKNESS = 2 * CM_TO_FT  # 2 ft in cm
SOURCE_TO_SHIELD_DISTANCE = 6 * CM_TO_FT  # 6 ft in cm
DETECTOR_INITIAL_DISTANCE = 1 * CM_TO_FT  # 1 ft in cm
PHANTOM_DIAMETER = 30.0  # 30 cm ICRU phantom

# Channel diameters in cm
CHANNEL_DIAMETERS = [0.05, 0.1, 0.25, 0.5, 1.0]  # 0.5 mm to 1 cm

# Source energy range (keV)
SOURCE_ENERGIES = [100, 500, 1000, 2000, 5000]  # 100 keV to 5 MeV

# Detector distances behind wall (cm)
DETECTOR_DISTANCES = [30, 40, 60, 80, 100, 150]

# Detector angles (degrees)
DETECTOR_ANGLES = [0, 5, 10, 15, 30, 45]  # Angles in degrees

# Mesh parameters
MESH_DIMENSION = 100  # Number of mesh elements in each direction
MESH_LENGTH = 300.0  # cm - Length of mesh in each direction

# Weight window parameters
USE_WEIGHT_WINDOWS = True
WEIGHT_LOW = 0.1
WEIGHT_HIGH = 1.0

def save_results(data, filename="simulation_results.json"):
    """Save simulation results to a JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_results(filename="simulation_results.json"):
    """Load simulation results from a JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None