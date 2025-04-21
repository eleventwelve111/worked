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

import os
import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any

# Add the missing constants required by geometry.py
WALL_THICKNESS = 60.96  # cm
SOURCE_TO_SHIELD_DISTANCE = 100.0  # cm
DETECTOR_INITIAL_DISTANCE = 30.0  # cm
PHANTOM_DIAMETER = 30.0  # cm

# Add the missing constants required by tally.py
DETECTOR_DISTANCES = [30, 50, 100, 150]  # cm
DETECTOR_ANGLES = [0, 15, 30, 45]  # degrees

@dataclass
class SimulationConfig:
    """Configuration for radiation streaming simulations."""
    # Physical parameters
    wall_thickness: float  # cm
    source_to_wall_distance: float  # cm
    world_width: float  # cm
    world_height: float  # cm
    world_depth: float  # cm
    phantom_diameter: float  # cm

    # Conversion factors
    foot_to_cm: float = 30.48

    # Default simulation parameters
    default_particles: int = 1000000
    default_batches: int = 10

    # Output directories
    output_dir: str = "output"
    plot_dir: str = "plots"
    results_dir: str = "results"
    tallies_dir: str = "tallies"

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate physical dimensions
        for param_name in ['wall_thickness', 'source_to_wall_distance',
                           'world_width', 'world_height', 'world_depth',
                           'phantom_diameter', 'foot_to_cm']:
            value = getattr(self, param_name)
            if value <= 0:
                raise ValueError(f"Parameter {param_name} must be positive, got {value}")

        # Validate simulation parameters
        if self.default_particles <= 0:
            raise ValueError(f"default_particles must be positive, got {self.default_particles}")
        if self.default_batches <= 0:
            raise ValueError(f"default_batches must be positive, got {self.default_batches}")

    @classmethod
    def from_file(cls, filepath: str) -> 'SimulationConfig':
        """Load configuration from a file (YAML or JSON)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            # Determine file type and load
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                with open(filepath, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError("Configuration file must be YAML or JSON")

            # Filter out unknown parameters
            valid_fields = cls.__annotations__.keys()
            filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}

            # Check if required fields are present
            required_fields = {'wall_thickness', 'source_to_wall_distance',
                               'world_width', 'world_height', 'world_depth',
                               'phantom_diameter'}
            missing = required_fields - set(filtered_data.keys())
            if missing:
                raise ValueError(f"Missing required configuration parameters: {missing}")

            return cls(**filtered_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON config: {str(e)}")
        except TypeError as e:
            raise ValueError(f"Configuration error: {str(e)}")

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a file (YAML or JSON)."""
        config_dict = asdict(self)

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        try:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                with open(filepath, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif filepath.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError("Output file must have .yaml, .yml, or .json extension")
        except Exception as e:
            raise IOError(f"Failed to save configuration to {filepath}: {str(e)}")

    def create_directories(self) -> None:
        """Create necessary output directories."""
        for directory in [self.output_dir, self.plot_dir, self.results_dir, self.tallies_dir]:
            os.makedirs(directory, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
   
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
