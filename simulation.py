ort
numpy as np
import time
import logging
import os
import json
from tqdm import tqdm

# Import local modules
from geometry import create_geometry
from materials import create_materials
from source import create_source
from tally import create_tallies
from dose import calculate_dose, analyze_dose_components
from uncertainty import propagate_uncertainties, analyze_simulation_convergence
from weight_windows import generate_weight_windows, analyze_weight_window_effectiveness
from visualization import plot_geometry, plot_results
from spectrum_analysis import analyze_dose_energy_dependence, estimate_effective_dose
from ml_analysis import DosePredictionModel, plot_model_performance
from config import load_configuration


class RadiationShieldingSimulation:
    """
    Main class for radiation shielding simulation and analysis.

    This class integrates all components of the gamma-ray shielding simulation framework
    and provides methods for setup, execution, analysis, and visualization of results.
    """

    def __init__(self, config_file=None, config_dict=None):
        """
        Initialize the simulation with configuration options.

        Parameters:
            config_file: Path to configuration file (JSON or YAML)
            config_dict: Configuration dictionary (overrides config_file)
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config_dict is not None:
            self.config = config_dict
        elif config_file is not None:
            self.config = load_configuration(config_file)
        else:
            self.config = {}  # Default configuration

        # Initialize components
        self.geometry = None
        self.materials = None
        self.source = None
        self.tallies = None
        self.model = None

        # Initialize results storage
        self.results = {}
        self.analyses = {}

        # Setup output directory
        self.output_dir = self.config.get('output', {}).get('directory', 'results')
        os.makedirs(self.output_dir, exist_ok=True)

        # Logging setup
        log_file = os.path.join(self.output_dir, 'simulation.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        # Initialize ML model if enabled
        self.ml_enabled = self.config.get('machine_learning', {}).get('enabled', False)
        self.ml_model = None
        if self.ml_enabled:
            model_type = self.config.get('machine_learning', {}).get('model_type', 'random_forest')
            self.ml_model = DosePredictionModel(model_type=model_type)

    def setup(self):
        """
        Set up the simulation components based on configuration.
        """
        self.logger.info("Setting up simulation components")

        # Create materials
        material_config = self.config.get('materials', {})
        self.materials = create_materials(material_config)
        self.logger.info(f"Created {len(self.materials)} materials")

        # Create geometry
        geometry_config = self.config.get('geometry', {})
        self.geometry = create_geometry(geometry_config, self.materials)
        self.logger.info("Created geometry")

        # Create source
        source_config = self.config.get('source', {})
        self.source = create_source(source_config)
        self.logger.info("Created source")

        # Create tallies
        tally_config = self.config.get('tallies', {})
        self.tallies = create_tallies(tally_config)
        self.logger.info(f"Created {len(self.tallies)} tallies")

        # Create model
        from openmc import Model
        self.model = Model(self.geometry, self.materials, self.source, self.tallies)
        self.logger.info("Model created successfully")

        # Setup weight windows if enabled
        if self.config.get('variance_reduction', {}).get('weight_windows', {}).get('enabled', False):
            ww_config = self.config.get('variance_reduction', {}).get('weight_windows', {})
            target_regions = ww_config.get('target_regions', [])
            energy_groups = ww_config.get('energy_groups', None)

            self.weight_windows = generate_weight_windows(
                self.geometry, self.source, target_regions, energy_groups
            )
            self.logger.info("Weight windows generated")

    def run(self, parameter_sets=None):
        """
        Run the simulation for a set of parameters.

        Parameters:
            parameter_sets: List of parameter dictionaries for parametric studies
                            If None, uses the default parameters from config

        Returns:
            dict: Simulation results
        """
        # If no parameter sets provided, use default from config
        if parameter_sets is None:
            parameter_sets = [self.config.get('parameters', {})]

        self.logger.info(f"Running {len(parameter_sets)} parameter sets")

        # Run each parameter set
        for i, params in enumerate(parameter_sets):
            param_id = params.get('id', f"param_set_{i}")
            self.logger.info(f"Running parameter set {param_id}")

            # Update model parameters
            self._update_model_parameters(params)

            # Run with weight windows if enabled
            use_ww = self.config.get('variance_reduction', {}).get('weight_windows', {}).get('enabled', False)

            if use_ww:
                # First run without weight windows to establish baseline
                self.logger.info("Running baseline simulation without weight windows")
                start_time = time.time()
                baseline_results = self._run_single_simulation(params, use_weight_windows=False)
                baseline_runtime = time.time() - start_time
                baseline_results['runtime'] = baseline_runtime

                # Run with weight windows
                self.logger.info("Running simulation with weight windows")
                start_time = time.time()
                ww_results = self._run_single_simulation(params, use_weight_windows=True)
                ww_runtime = time.time() - start_time
                ww_results['runtime'] = ww_runtime

                # Analyze weight window effectiveness
                ww_analysis = analyze_weight_window_effectiveness(ww_results, baseline_results)
                self.logger.info(f"Weight window efficiency gain: {ww_analysis['overall']['avg_fom_improvement']:.2f}x")

                # Store both results and analysis
                self.results[f"{param_id}_baseline"] = baseline_results
                self.results[param_id] = ww_results
                self.analyses[f"{param_id}_weight_windows"] = ww_analysis
            else:
                # Regular run without variance reduction
                self.logger.info("Running simulation")
                start_time = time.time()
                results = self._run_single_simulation(params)
                runtime = time.time() - start_time
                results['runtime'] = runtime

                self.results[param_id] = results

        # Save results to disk
        self._save_results()

        # Train ML model if enabled
        if self.ml_enabled and len(self.results) >= 5:  # Need sufficient data for training
            self.logger.info("Training machine learning model")
            self._train_ml_model()

        return self.results

    def _run_single_simulation(self, params, use_weight_windows=False):
        """
        Run a single simulation with the given parameters.

        Parameters:
            params: Dictionary of simulation parameters
            use_weight_windows: Whether to use weight windows

        Returns:
            dict: Simulation results
        """
        # Extract simulation parameters
        num_particles = params.get('particles', 1000000)
        energy_kev = params.get('energy_kev', 662)
        channel_diameter = params.get('channel_diameter', 5.0)

        # Set runtime parameters
        settings = {
            'particles': num_particles,
            'batches': params.get('batches', 10),
            'inactive': params.get('inactive', 5),
            'track': params.get('track', False)
        }

        # Apply weight windows if requested
        if use_weight_windows and hasattr(self, 'weight_windows'):
            settings['weight_windows'] = self.weight_windows.get('generator')

        # Execute simulation
        try:
            from openmc import run
            result_path = os.path.join(self.output_dir, f"results_{int(energy_kev)}_{channel_diameter}")
            run(self.model, path_output=result_path, **settings)

            # Process results
            simulation_results = self._process_results(result_path)

            # Add metadata
            simulation_results['parameters'] = {
                'energy_kev': energy_kev,
                'channel_diameter': channel_diameter,
                'num_particles': num_particles
            }

            # Calculate dose
            simulation_results['dose_data'] = calculate_dose(simulation_results, energy_kev)

            return simulation_results

        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            return {'error': str(e)}


def _update_model_parameters(self, params):
    """
    Update model parameters for the current simulation run.

    Parameters:
        params: Dictionary of simulation parameters
    """
    # Update source parameters
    energy_kev = params.get('energy_kev', 662)
    source_strength = params.get('source_strength', 1.0)

    if hasattr(self.source, 'energy') and hasattr(self.source.energy, 'set_parameters'):
        self.source.energy.set_parameters(energy=energy_kev / 1000.0)  # Convert to MeV

    if hasattr(self.source, 'strength'):
        self.source.strength = source_strength

    # Update geometry parameters
    channel_diameter = params.get('channel_diameter', 5.0)
    shield_thickness = params.get('shield_thickness', 10.0)

    # Update geometry dimensions if method available
    if hasattr(self.geometry, 'update_dimensions'):
        self.geometry.update_dimensions(
            channel_diameter=channel_diameter,
            shield_thickness=shield_thickness
        )

    # Update material compositions if needed
    material_updates = params.get('material_updates', {})
    for mat_name, mat_params in material_updates.items():
        if mat_name in self.materials:
            material = self.materials[mat_name]

            # Update density if provided
            if 'density' in mat_params:
                material.set_density(mat_params['density'][0], mat_params['density'][1])

            # Update composition if provided
            if 'composition' in mat_params:
                material.composition = mat_params['composition']

    self.logger.info(f"Updated model parameters: energy={energy_kev} keV, diameter={channel_diameter} cm")


def _process_results(self, result_path):
    """
    Process the simulation results from OpenMC output.

    Parameters:
        result_path: Path to simulation results

    Returns:
        dict: Processed results
    """
    # Import OpenMC statepoint module
    import openmc

    # Find the statepoint file
    statepoint_files = [f for f in os.listdir(result_path) if f.startswith('statepoint')]
    if not statepoint_files:
        self.logger.error(f"No statepoint file found in {result_path}")
        return {'error': 'No statepoint file found'}

    statepoint_file = os.path.join(result_path, sorted(statepoint_files)[-1])

    # Load statepoint file
    sp = openmc.StatePoint(statepoint_file)

    # Extract tally results
    tally_results = {}
    for tally_id, tally in self.tallies.items():
        if tally_id in sp.tallies:
            tally_data = sp.tallies[tally_id]

            # Extract mean and standard deviation
            mean = tally_data.mean.flatten()
            std_dev = tally_data.std_dev.flatten()

            # Calculate relative error
            rel_error = std_dev / mean
            where
            mean != 0
            rel_error = np.where(mean != 0, std_dev / mean, 0)

            # Extract energy grid if available
            energy_grid = None
            for filter_type in tally_data.filters:
                if isinstance(filter_type, openmc.EnergyFilter):
                    energy_grid = filter_type.values

            tally_results[tally_id] = {
                'mean': mean.tolist(),
                'std_dev': std_dev.tolist(),
                'rel_error': rel_error.tolist(),
                'energy_grid': energy_grid.tolist() if energy_grid is not None else None
            }

    # Extract keff if present
    k_combined = None
    if hasattr(sp, 'k_combined'):
        k_combined = {
            'nominal_value': float(sp.k_combined[0]),
            'std_dev': float(sp.k_combined[1])
        }

    # Process global tallies if present
    global_tallies = None
    if hasattr(sp, 'global_tallies'):
        global_tallies = {}
        for i, gt in enumerate(sp.global_tallies):
            global_tallies[f"global_tally_{i}"] = {
                'mean': float(gt[0]),
                'std_dev': float(gt[1])
            }

    # Create results dictionary
    results = {
        'tallies': tally_results,
        'k_combined': k_combined,
        'global_tallies': global_tallies,
        'num_particles': sp.n_particles
    }

    return results


def analyze(self):
    """
    Analyze the simulation results.

    Returns:
        dict: Analysis results
    """
    self.logger.info("Analyzing simulation results")

    # Perform different types of analysis for each parameter set
    for param_id, results in self.results.items():
        if 'error' in results:
            self.logger.warning(f"Skipping analysis for {param_id} due to error: {results['error']}")
            continue

        # Extract parameters
        params = results.get('parameters', {})
        energy_kev = params.get('energy_kev', 662)
        channel_diameter = params.get('channel_diameter', 5.0)

        # Dose component analysis
        self.logger.info(f"Performing dose component analysis for {param_id}")
        dose_analysis = analyze_dose_components(results, energy_kev, channel_diameter)
        self.analyses[f"{param_id}_dose_components"] = dose_analysis

        # Uncertainty analysis
        self.logger.info(f"Performing uncertainty analysis for {param_id}")
        uncertainty_analysis = propagate_uncertainties(results, energy_kev, channel_diameter)
        self.analyses[f"{param_id}_uncertainty"] = uncertainty_analysis

        # Energy dependence analysis
        self.logger.info(f"Analyzing energy dependence for {param_id}")
        energy_analysis = analyze_dose_energy_dependence(results)
        self.analyses[f"{param_id}_energy_dependence"] = energy_analysis

        # Effective dose estimation
        self.logger.info(f"Estimating effective dose for {param_id}")
        effective_dose = estimate_effective_dose(results)
        self.analyses[f"{param_id}_effective_dose"] = effective_dose

    # Save analyses to disk
    self._save_analyses()

    return self.analyses


def visualize(self):
    """
    Generate visualizations of geometry and results.

    Returns:
        dict: Paths to generated visualization files
    """
    self.logger.info("Generating visualizations")

    visualization_paths = {}

    # Generate geometry visualization
    geometry_plot_path = os.path.join(self.output_dir, 'geometry.png')
    plot_geometry(self.geometry, filename=geometry_plot_path)
    visualization_paths['geometry'] = geometry_plot_path

    # Generate results plots for each parameter set
    for param_id, results in self.results.items():
        if 'error' in results:
            continue

        # Extract parameters
        params = results.get('parameters', {})
        energy_kev = params.get('energy_kev', 662)
        channel_diameter = params.get('channel_diameter', 5.0)

        # Plot dose results
        dose_plot_path = os.path.join(self.output_dir, f"dose_results_{param_id}.png")
        plot_results(results, filename=dose_plot_path)
        visualization_paths[f"dose_results_{param_id}"] = dose_plot_path

        # Plot uncertainty analysis
        if f"{param_id}_uncertainty" in self.analyses:
            uncertainty_plot_path = os.path.join(self.output_dir, f"uncertainty_{param_id}.png")
            from uncertainty import plot_uncertainty_analysis
            plot_uncertainty_analysis(self.analyses[f"{param_id}_uncertainty"], filename=uncertainty_plot_path)
            visualization_paths[f"uncertainty_{param_id}"] = uncertainty_plot_path

        # Plot energy dependence
        if f"{param_id}_energy_dependence" in self.analyses:
            energy_plot_path = os.path.join(self.output_dir, f"energy_dependence_{param_id}.png")
            from spectrum_analysis import plot_energy_dependence
            plot_energy_dependence(self.analyses[f"{param_id}_energy_dependence"], filename=energy_plot_path)
            visualization_paths[f"energy_dependence_{param_id}"] = energy_plot_path

        # Plot weight window analysis if available
        if f"{param_id}_weight_windows" in self.analyses:
            ww_plot_path = os.path.join(self.output_dir, f"weight_windows_{param_id}.png")
            from weight_windows import plot_weight_window_importance_map
            if hasattr(self, 'weight_windows'):
                plot_weight_window_importance_map(self.weight_windows, filename=ww_plot_path)
                visualization_paths[f"weight_windows_{param_id}"] = ww_plot_path

    # Plot ML model performance if available
    if self.ml_enabled and hasattr(self, 'ml_training_results'):
        ml_plot_path = os.path.join(self.output_dir, "ml_performance.png")
        plot_model_performance(self.ml_training_results, filename=ml_plot_path)
        visualization_paths["ml_performance"] = ml_plot_path

    return visualization_paths


def _train_ml_model(self):
    """
    Train the machine learning model on simulation results.

    Returns:
        dict: Training results
    """
    # Skip if ML is not enabled or not enough data
    if not self.ml_enabled or len(self.results) < 5:
        return None

    # Prepare data
    features_df, targets_series = self.ml_model.prepare_data(self.results)

    # Train model with hyperparameter tuning
    hyperparameter_tuning = self.config.get('machine_learning', {}).get('hyperparameter_tuning', False)
    training_results = self.ml_model.train(features_df, targets_series, hyperparameter_tuning)

    # Save model
    model_path = os.path.join(self.output_dir, "dose_prediction_model.pkl")
    self.ml_model.save_model(model_path)

    # Store training results
    self.ml_training_results = training_results

    # Log performance
    self.logger.info(f"ML model trained - Test R²: {training_results['metrics']['test']['r2']:.3f}")

    return training_results


def predict(self, new_parameters):
    """
    Make predictions using the trained ML model.

    Parameters:
        new_parameters: List of parameter dictionaries

    Returns:
        dict: Prediction results
    """
    if not self.ml_enabled or not hasattr(self, 'ml_training_results'):
        self.logger.error("ML model not available for prediction")
        return {'error': 'ML model not available'}

    # Convert parameters to features format
    features = []
    for params in new_parameters:
        energy_kev = params.get('energy_kev', 662)
        channel_diameter = params.get('channel_diameter', 5.0)
        distance = params.get('distance', 30.0)
        angle = params.get('angle', 0.0)

        feature = {
            'energy_kev': energy_kev,
            'channel_diameter': channel_diameter,
            'distance': distance,
            'angle': angle
        }
        features.append(feature)

    # Make predictions
    predictions = self.ml_model.predict(pd.DataFrame(features))

    # Format results
    prediction_results = {
        'parameters': new_parameters,
        'predicted_dose': predictions.tolist(),
        'model_info': {
            'type': self.ml_model.model_type,
            'test_r2': self.ml_training_results['metrics']['test']['r2'],
            'test_mape': self.ml_training_results['metrics']['test']['mape']
        }
    }

    return prediction_results


def _save_results(self):
    """
    Save simulation results to disk.
    """
    results_file = os.path.join(self.output_dir, "simulation_results.json")

    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in self.results.items():
        if isinstance(value, dict):
            serializable_results[key] = self._make_serializable(value)
        else:
            serializable_results[key] = value

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    self.logger.info(f"Results saved to {results_file}")


def _save_analyses(self):
    """
    Save analysis results to disk.
    """
    analyses_file = os.path.join(self.output_dir, "analysis_results.json")

    # Convert numpy arrays to lists for JSON serialization
    serializable_analyses = {}
    for key, value in self.analyses.items():
        if isinstance(value, dict):
            serializable_analyses[key] = self._make_serializable(value)
        else:
            serializable_analyses[key] = value

    with open(analyses_file, 'w') as f:
        json.dump(serializable_analyses, f, indent=2)

    self.logger.info(f"Analyses saved to {analyses_file}")


def _make_serializable(self, obj):
    """
    Convert an object to a JSON serializable format.
    """
    if isinstance(obj, dict):
        return {k: self._make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [self._make_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is None:
        return None
    else:
        return str(obj)


"""
Simulation module for gamma-ray streaming simulation.
Contains functions to run the Monte Carlo particle transport simulation.
"""

import numpy as np
import time
import os
from tqdm import tqdm
import pickle
import h5py
import multiprocessing
import threading
import uuid

from logging import log_info, log_error, log_warning, log_debug


class Particle:
    """Class representing a particle in the simulation."""

    def __init__(self, position=(0, 0, 0), direction=(0, 0, 1), energy=1.0, weight=1.0, particle_id=None):
        """
        Initialize a particle.

        Parameters:
            position: Tuple (x, y, z) of particle position (cm)
            direction: Tuple (dx, dy, dz) of particle direction (normalized)
            energy: Particle energy (MeV)
            weight: Statistical weight of the particle
            particle_id: Unique identifier for the particle
        """
        self.position = position

        # Normalize direction vector
        direction_norm = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        if direction_norm > 0:
            self.direction = (
                direction[0] / direction_norm,
                direction[1] / direction_norm,
                direction[2] / direction_norm
            )
        else:
            self.direction = (0, 0, 1)  # Default to forward direction

        self.energy = energy
        self.weight = weight
        self.alive = True

        # Track history
        self.path_length = 0.0
        self.interactions = 0
        self.energy_deposition = 0.0
        self.birth_energy = energy
        self.history = []

        # Assign ID if not provided
        if particle_id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = particle_id

    def move(self, distance):
        """
        Move the particle along its direction.

        Parameters:
            distance: Distance to move (cm)

        Returns:
            tuple: New position
        """
        # Update position
        new_position = (
            self.position[0] + distance * self.direction[0],
            self.position[1] + distance * self.direction[1],
            self.position[2] + distance * self.direction[2]
        )

        # Record history if tracking
        self.history.append({
            'old_position': self.position,
            'new_position': new_position,
            'energy': self.energy,
            'direction': self.direction,
            'distance': distance
        })

        # Update position and path length
        self.position = new_position
        self.path_length += distance

        return self.position

    def scatter(self, new_direction, energy_loss=0):
        """
        Change the particle direction due to scattering.

        Parameters:
            new_direction: New direction tuple (dx, dy, dz)
            energy_loss: Energy lost in the scattering (MeV)

        Returns:
            tuple: New direction
        """
        # Normalize new direction
        direction_norm = np.sqrt(new_direction[0] ** 2 + new_direction[1] ** 2 + new_direction[2] ** 2)
        if direction_norm > 0:
            self.direction = (
                new_direction[0] / direction_norm,
                new_direction[1] / direction_norm,
                new_direction[2] / direction_norm
            )

        # Update energy
        old_energy = self.energy
        self.energy -= energy_loss
        self.energy_deposition += energy_loss

        # Increment interaction counter
        self.interactions += 1

        # Record history if tracking
        self.history.append({
            'interaction': 'scatter',
            'old_energy': old_energy,
            'new_energy': self.energy,
            'energy_deposition': energy_loss,
            'new_direction': self.direction
        })

        return self.direction

    def absorb(self):
        """
        Absorb the particle, depositing all its energy.

        Returns:
            float: Deposited energy
        """
        # Deposit all energy
        deposited_energy = self.energy
        self.energy_deposition += deposited_energy
        self.energy = 0

        # Kill the particle
        self.alive = False

        # Increment interaction counter
        self.interactions += 1

        # Record history if tracking
        self.history.append({
            'interaction': 'absorption',
            'energy_deposition': deposited_energy
        })

        return deposited_energy

    def kill(self, reason="unknown"):
        """
        Kill the particle without energy deposition.

        Parameters:
            reason: Reason for killing the particle

        Returns:
            bool: True if particle was killed
        """
        # Kill the particle
        self.alive = False

        # Record history if tracking
        self.history.append({
            'interaction': 'killed',
            'reason': reason
        })

        return True

    def split(self, n_particles):
        """
        Split the particle into multiple particles with reduced weight.

        Parameters:
            n_particles: Number of particles to create

        Returns:
            list: List of new particles
        """
        if n_particles <= 1:
            return [self]

        new_weight = self.weight / n_particles

        # Create new particles
        particles = []
        for i in range(n_particles):
            new_particle = Particle(
                position=self.position,
                direction=self.direction,
                energy=self.energy,
                weight=new_weight,
                particle_id=f"{self.id}_{i}"
            )

            # Copy history-related attributes
            new_particle.path_length = self.path_length
            new_particle.interactions = self.interactions
            new_particle.energy_deposition = self.energy_deposition
            new_particle.birth_energy = self.birth_energy

            particles.append(new_particle)

        # Record history if tracking
        self.history.append({
            'interaction': 'split',
            'n_particles': n_particles,
            'new_weight': new_weight
        })

        return particles

    def roulette(self, survival_probability):
        """
        Apply Russian roulette to the particle.

        Parameters:
            survival_probability: Probability of survival

        Returns:
            bool: True if particle survives, False otherwise
        """
        if np.random.random() < survival_probability:
            # Particle survives with increased weight
            self.weight /= survival_probability

            # Record history if tracking
            self.history.append({
                'interaction': 'roulette',
                'survived': True,
                'new_weight': self.weight
            })

            return True
        else:
            # Particle is killed
            self.alive = False

            # Record history if tracking
            self.history.append({
                'interaction': 'roulette',
                'survived': False
            })

            return False

        def transport_particle(particle, geometry, materials, tallies, settings):
            """
            Transport a single particle through the geometry until it is absorbed or escapes.

            Parameters:
                particle: Particle object to transport
                geometry: Geometry object defining the model
                materials: Dictionary of material objects
                tallies: Dictionary of tally objects
                settings: Dictionary of simulation settings

            Returns:
                Particle: The transported particle (may be dead)
            """
            max_steps = settings.get('max_steps', 1000)
            russian_roulette = settings.get('russian_roulette', True)
            rr_threshold = settings.get('rr_threshold', 0.1)
            rr_survival = settings.get('rr_survival', 0.5)
            cutoff_energy = settings.get('cutoff_energy', 0.001)  # MeV

            step = 0

            while particle.alive and step < max_steps:
                step += 1

                # Check energy cutoff
                if particle.energy < cutoff_energy:
                    if russian_roulette and particle.weight < rr_threshold:
                        if not particle.roulette(rr_survival):
                            break
                    else:
                        particle.absorb()
                        break

                # Find current region and material
                region = geometry.find_region(particle.position)

                if region is None:
                    # Particle is outside the geometry
                    particle.kill(reason="escaped geometry")
                    break

                material_name = region.material
                material = materials.get(material_name)

                if material is None:
                    # Region has no material (void)
                    # Calculate distance to next boundary
                    distance_to_boundary = geometry.distance_to_boundary(particle.position, particle.direction)

                    if distance_to_boundary <= 0:
                        # Numerical error or at boundary already, take a small step
                        particle.move(1e-6)
                    else:
                        # Move to the boundary
                        particle.move(distance_to_boundary)

                    continue

                # Calculate interaction probability (cross section)
                cross_section = material.get_cross_section(particle.energy)

                if cross_section <= 0:
                    # No interaction possible, move to boundary
                    distance_to_boundary = geometry.distance_to_boundary(particle.position, particle.direction)

                    if distance_to_boundary <= 0:
                        # Numerical error or at boundary already, take a small step
                        particle.move(1e-6)
                    else:
                        # Move to the boundary
                        particle.move(distance_to_boundary)

                    continue

                # Mean free path (cm)
                mfp = 1.0 / cross_section

                # Sample distance to next interaction
                distance_to_interaction = -mfp * np.log(np.random.random())

                # Calculate distance to next boundary
                distance_to_boundary = geometry.distance_to_boundary(particle.position, particle.direction)

                if distance_to_boundary < distance_to_interaction:
                    # Particle reaches boundary before interacting
                    particle.move(distance_to_boundary)
                else:
                    # Particle interacts before reaching boundary
                    particle.move(distance_to_interaction)

                    # Determine interaction type
                    interaction_type = material.sample_interaction(particle.energy)

                    if interaction_type == 'photoelectric':
                        # Photoelectric absorption
                        particle.absorb()
                    elif interaction_type == 'compton':
                        # Compton scattering
                        # Sample scattering angle and energy loss
                        cos_theta, energy_loss = material.sample_compton_scattering(particle.energy)

                        # Calculate new direction
                        new_direction = sample_direction_given_cosine(particle.direction, cos_theta)

                        # Update particle direction and energy
                        particle.scatter(new_direction, energy_loss)
                    elif interaction_type == 'pair':
                        # Pair production
                        # For gamma transport only, we ignore the positron-electron pair
                        # and just track any potential annihilation photons
                        particle.absorb()

                        # Two 0.511 MeV photons might be created here
                        # For simplicity, we ignore these secondary photons for now
                    else:
                        log_warning(f"Unknown interaction type: {interaction_type}")
                        particle.kill(reason=f"unknown interaction {interaction_type}")

                # Score the particle in all tallies
                for tally in tallies.values():
                    tally.score(particle)

            # If particle is still alive after max steps, kill it
            if particle.alive and step >= max_steps:
                particle.kill(reason="max steps exceeded")

            return particle

        def sample_direction_given_cosine(old_direction, cos_theta):
            """
            Sample a new direction given the old direction and cosine of the scattering angle.

            Parameters:
                old_direction: Original direction vector (dx, dy, dz)
                cos_theta: Cosine of the scattering angle

            Returns:
                tuple: New direction vector (dx, dy, dz)
            """
            # Sample azimuthal angle phi uniformly
            phi = 2.0 * np.pi * np.random.random()

            # Calculate sine of theta
            sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

            # Convert to Cartesian coordinates in the scattering frame
            x_scatter = sin_theta * np.cos(phi)
            y_scatter = sin_theta * np.sin(phi)
            z_scatter = cos_theta

            # Get the original direction
            u0, v0, w0 = old_direction

            # Construct an orthonormal basis
            if abs(w0) < 0.99999:
                # Use z-axis to construct the basis
                u1 = -v0
                v1 = u0
                w1 = 0
                norm = np.sqrt(u1 * u1 + v1 * v1)
                u1 /= norm
                v1 /= norm

                u2 = v1 * w0 - w1 * v0
                v2 = w1 * u0 - u1 * w0
                w2 = u1 * v0 - v1 * u0
            else:
                # Original direction is close to z-axis, use x-axis instead
                u1 = 0
                v1 = 1 if w0 > 0 else -1
                w1 = 0

                u2 = 1
                v2 = 0
                w2 = 0

            # Rotate the scattering vector to the global frame
            u = x_scatter * u1 + y_scatter * u2 + z_scatter * u0
            v = x_scatter * v1 + y_scatter * v2 + z_scatter * v0
            w = x_scatter * w1 + y_scatter * w2 + z_scatter * w0

            return (u, v, w)

        def run_simulation_batch(seed, n_particles, geometry, materials, tallies, settings, progress_queue=None):
            """
            Run a batch of particles in the simulation.

            Parameters:
                seed: Random seed for this batch
                n_particles: Number of particles to simulate
                geometry: Geometry object defining the model
                materials: Dictionary of material objects
                tallies: Dictionary of tally objects
                settings: Dictionary of simulation settings
                progress_queue: Queue to report progress

            Returns:
                dict: Dictionary of tally objects with results
            """
            # Set random seed for this batch
            np.random.seed(seed)

            # Get source configuration
            source = settings.get('source', {})
            source_type = source.get('type', 'point')
            source_position = source.get('position', (0, 0, 0))
            source_direction = source.get('direction', (0, 0, 1))
            source_energy = source.get('energy', 1.0)
            source_isotropic = source.get('isotropic', False)

            # Create copies of tallies for this batch
            batch_tallies = {}
            for tally_name, tally in tallies.items():
                # Create a new tally of the same type with same parameters
                batch_tallies[tally_name] = type(tally)(
                    name=tally.name,
                    region=tally.region
                )

                # Copy configuration parameters
                for key, value in tally.results.items():
                    if key not in ['flux', 'error', 'counts', 'deposition', 'squared_deposition',
                                   'pulse_height', 'distribution', 'path_length']:
                        setattr(batch_tallies[tally_name], key, value)

            # Track completed particles
            completed = 0

            # Create and transport particles
            for i in range(n_particles):
                # Sample source position if not a point source
                if source_type == 'point':
                    position = source_position
                elif source_type == 'volume':
                    # Sample uniformly within a sphere
                    radius = source.get('radius', 1.0)
                    r = radius * np.power(np.random.random(), 1 / 3)
                    theta = np.arccos(2 * np.random.random() - 1)
                    phi = 2 * np.pi * np.random.random()

                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)

                    position = (
                        source_position[0] + x,
                        source_position[1] + y,
                        source_position[2] + z
                    )
                elif source_type == 'plane':
                    # Sample uniformly within a disk
                    radius = source.get('radius', 1.0)
                    r = radius * np.sqrt(np.random.random())
                    phi = 2 * np.pi * np.random.random()

                    x = r * np.cos(phi)
                    y = r * np.sin(phi)
                    z = 0

                    position = (
                        source_position[0] + x,
                        source_position[1] + y,
                        source_position[2] + z
                    )
                else:
                    position = source_position

                # Sample source direction if isotropic
                if source_isotropic:
                    # Sample uniformly on the unit sphere
                    cos_theta = 2 * np.random.random() - 1
                    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
                    phi = 2 * np.pi * np.random.random()

                    dx = sin_theta * np.cos(phi)
                    dy = sin_theta * np.sin(phi)
                    dz = cos_theta

                    direction = (dx, dy, dz)
                else:
                    direction = source_direction

                # Sample source energy
                if isinstance(source_energy, (int, float)):
                    # Monoenergetic
                    energy = source_energy
                elif isinstance(source_energy, (list, tuple)) and len(source_energy) == 2:
                    # Uniform distribution
                    energy = source_energy[0] + (source_energy[1] - source_energy[0]) * np.random.random()
                elif isinstance(source_energy, dict) and 'spectrum' in source_energy:
                    # Energy spectrum
                    # This would require a more sophisticated sampling method
                    # For now, we'll use the mean energy
                    energy = source_energy.get('mean', 1.0)
                else:
                    energy = 1.0  # Default

                # Create the particle
                particle = Particle(
                    position=position,
                    direction=direction,
                    energy=energy,
                    weight=1.0,
                    particle_id=f"batch_{seed}_particle_{i}"
                )

                # Transport the particle
                transport_particle(particle, geometry, materials, batch_tallies, settings)

                # Update progress
                completed += 1
                if progress_queue is not None and completed % 100 == 0:
                    progress_queue.put(100)

            # Final progress update
            if progress_queue is not None:
                progress_queue.put(n_particles - completed)

            return batch_tallies

        def run_simulation(geometry, materials, source, tallies, params, output_dir=None):
            """
            Run the complete Monte Carlo simulation.

            Parameters:
                geometry: Geometry object defining the model
                materials: Dictionary of material objects
                source: Source object defining particle generation
                tallies: Dictionary of tally objects
                params: Dictionary of simulation parameters
                output_dir: Directory to save output files

            Returns:
                dict: Dictionary of simulation results
            """
            # Extract simulation parameters
            n_particles = params.get('n_particles', 10000)
            n_batches = params.get('n_batches', 1)
            use_multiprocessing = params.get('use_multiprocessing', True)
            n_processes = params.get('n_processes', None)

            # If n_processes is not specified, use available CPU cores
            if n_processes is None:
                n_processes = max(1, multiprocessing.cpu_count() - 1)

            # If n_batches is 1, use all particles in that batch
            if n_batches == 1:
                particles_per_batch = n_particles
            else:
                particles_per_batch = max(1, n_particles // n_batches)

            # Create output directory if needed
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Start timing
            start_time = time.time()

            # Progress tracking
            progress_bar = tqdm(total=n_particles, desc="Simulating particles")

            # Create a shared queue for progress updates
            progress_queue = multiprocessing.Queue()

            # Start a thread to update the progress bar
            def update_progress():
                while True:
                    try:
                        count = progress_queue.get(timeout=0.5)
                        if count is None:
                            break
                        progress_bar.update(count)
                    except:
                        pass

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()

            # Run simulation
            if use_multiprocessing and n_batches > 1:
                # Multiprocessing approach
                log_info(
                    f"Running simulation with {n_processes} processes, {n_batches} batches, {particles_per_batch} particles per batch")

                # Create a pool of workers
                with multiprocessing.Pool(processes=n_processes) as pool:
                    # Create tasks for each batch
                    tasks = []
                    for batch in range(n_batches):
                        seed = int(time.time()) + batch
                        tasks.append((seed, particles_per_batch, geometry, materials, tallies, params, progress_queue))

                    # Run tasks in parallel
                    batch_results = pool.starmap(run_simulation_batch, tasks)
            else:
                # Single process approach
                log_info(f"Running simulation with 1 process, {n_particles} particles")
                batch_results = [
                    run_simulation_batch(int(time.time()), n_particles, geometry, materials, tallies, params,
                                         progress_queue)]

            # Signal progress thread to exit
            progress_queue


progress_queue.put(None)
progress_thread.join()
progress_bar.close()

# End timing
end_time = time.time()
simulation_time = end_time - start_time

log_info(f"Simulation completed in {simulation_time:.2f} seconds")

# Merge results from all batches
for tally_name in tallies:
    tally_batches = [batch[tally_name] for batch in batch_results]

    # Combine results from all batches
    if hasattr(tallies[tally_name], 'merge'):
        # If tally has a merge method, use it
        tallies[tally_name].merge(tally_batches)
    else:
        # Otherwise merge manually based on tally type
        if hasattr(tallies[tally_name], 'flux'):
            # FluxTally
            for batch_tally in tally_batches:
                tallies[tally_name].flux += batch_tally.flux
                tallies[tally_name].counts += batch_tally.counts
        elif hasattr(tallies[tally_name], 'deposition'):
            # EnergyDepositionTally
            for batch_tally in tally_batches:
                tallies[tally_name].deposition += batch_tally.deposition
                tallies[tally_name].squared_deposition += batch_tally.squared_deposition
                tallies[tally_name].count += batch_tally.count
        elif hasattr(tallies[tally_name], 'pulse_height'):
            # PulseHeightTally
            for batch_tally in tally_batches:
                tallies[tally_name].pulse_height += batch_tally.pulse_height
        elif hasattr(tallies[tally_name], 'distribution'):
            # AngleTally
            for batch_tally in tally_batches:
                tallies[tally_name].distribution += batch_tally.distribution
        elif hasattr(tallies[tally_name], 'path_length'):
            # PathLengthTally
            for batch_tally in tally_batches:
                tallies[tally_name].path_length += batch_tally.path_length

# Normalize results
source_strength = params.get('source_strength', 1.0)
norm_factor = source_strength / n_particles

for tally in tallies.values():
    tally.normalize(norm_factor)

# Create results summary
results = {
    'simulation_time': simulation_time,
    'n_particles': n_particles,
    'n_batches': n_batches,
    'source_strength': source_strength,
    'tallies': {}
}

for tally_name, tally in tallies.items():
    results['tallies'][tally_name] = tally.get_results()

# Save results if output directory is provided
if output_dir:
    # Save simulation parameters
    with open(os.path.join(output_dir, 'simulation_params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # Save tally results
    save_tally_results(tallies, output_dir)

    # Save summary results
    with open(os.path.join(output_dir, 'simulation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

return results


def analyze_simulation_results(results, tallies, output_dir=None):
    """
    Analyze the simulation results and generate plots.

    Parameters:
        results: Dictionary of simulation results
        tallies: Dictionary of tally objects
        output_dir: Directory to save output files

    Returns:
        dict: Dictionary of analysis results
    """
    try:
        import matplotlib.pyplot as plt

        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Container for analysis results
        analysis = {
            'figures': {},
            'metrics': {}
        }

        # Loop through tallies
        for tally_name, tally in tallies.items():
            tally_results = tally.get_results()

            # Different plots for different tally types
            if isinstance(tally, FluxTally):
                # Energy spectrum plot
                plt.figure(figsize=(10, 6))

                # Get midpoints of energy bins
                energy_bins = tally_results['energy_bins']
                energy_mids = 0.5 * (energy_bins[1:] + energy_bins[:-1])

                # Get flux
                flux = tally_results['flux']

                # If flux has spatial dimension, sum over it
                if len(flux.shape) > 1:
                    flux = np.sum(flux, axis=1)

                # Plot spectrum
                plt.semilogx(energy_mids, flux, 'o-')
                plt.xlabel('Energy (MeV)')
                plt.ylabel('Flux')
                plt.title(f'Energy Spectrum - {tally_name}')
                plt.grid(True)

                # Save figure
                if output_dir:
                    fig_path = os.path.join(output_dir, f'{tally_name}_energy_spectrum.png')
                    plt.savefig(fig_path)
                    analysis['figures'][f'{tally_name}_energy_spectrum'] = fig_path

                plt.close()

                # Calculate metrics
                total_flux = tally.get_total_flux()
                mean_energy = tally.get_mean_energy()

                analysis['metrics'][f'{tally_name}_total_flux'] = total_flux
                analysis['metrics'][f'{tally_name}_mean_energy'] = mean_energy

                # If spatial bins exist, create spatial distribution plot
                if 'spatial_bins' in tally_results and tally_results['spatial_bins'] is not None:
                    plt.figure(figsize=(10, 6))

                    # Get spatial bin midpoints
                    spatial_bins = tally_results['spatial_bins']
                    spatial_mids = 0.5 * (spatial_bins[1:] + spatial_bins[:-1])

                    # Get spatial distribution (sum over energy)
                    spatial_flux = np.sum(flux, axis=0)

                    # Plot distribution
                    plt.plot(spatial_mids, spatial_flux, 'o-')
                    plt.xlabel('Position (cm)')
                    plt.ylabel('Flux')
                    plt.title(f'Spatial Distribution - {tally_name}')
                    plt.grid(True)

                    # Save figure
                    if output_dir:
                        fig_path = os.path.join(output_dir, f'{tally_name}_spatial_distribution.png')
                        plt.savefig(fig_path)
                        analysis['figures'][f'{tally_name}_spatial_distribution'] = fig_path

                    plt.close()

            elif isinstance(tally, EnergyDepositionTally):
                # Energy deposition plot
                plt.figure(figsize=(10, 6))

                deposition = tally_results['deposition']

                if 'spatial_bins' in tally_results and tally_results['spatial_bins'] is not None:
                    # Plot spatial distribution of energy deposition
                    spatial_bins = tally_results['spatial_bins']
                    spatial_mids = 0.5 * (spatial_bins[1:] + spatial_bins[:-1])

                    plt.plot(spatial_mids, deposition, 'o-')
                    plt.xlabel('Position (cm)')
                    plt.ylabel('Energy Deposition (MeV/cm³)')
                    plt.title(f'Energy Deposition - {tally_name}')
                    plt.grid(True)
                else:
                    # Single value, create a bar chart
                    plt.bar(['Total'], [deposition])
                    plt.ylabel('Energy Deposition (MeV)')
                    plt.title(f'Total Energy Deposition - {tally_name}')

                # Save figure
                if output_dir:
                    fig_path = os.path.join(output_dir, f'{tally_name}_energy_deposition.png')
                    plt.savefig(fig_path)
                    analysis['figures'][f'{tally_name}_energy_deposition'] = fig_path

                plt.close()

                # Calculate metrics
                total_deposition = tally.get_total_deposition()
                analysis['metrics'][f'{tally_name}_total_deposition'] = total_deposition

            elif isinstance(tally, PulseHeightTally):
                # Pulse height spectrum plot
                plt.figure(figsize=(10, 6))

                # Get energy bin midpoints
                energy_bins = tally_results['energy_bins']
                energy_mids = 0.5 * (energy_bins[1:] + energy_bins[:-1])

                # Get pulse height distribution
                pulse_height = tally_results['pulse_height']

                # Plot spectrum
                plt.semilogx(energy_mids, pulse_height, 'o-')
                plt.xlabel('Energy (MeV)')
                plt.ylabel('Counts')
                plt.title(f'Pulse Height Spectrum - {tally_name}')
                plt.grid(True)

                # Save figure
                if output_dir:
                    fig_path = os.path.join(output_dir, f'{tally_name}_pulse_height.png')
                    plt.savefig(fig_path)
                    analysis['figures'][f'{tally_name}_pulse_height'] = fig_path

                plt.close()

                # Calculate metrics
                total_counts = np.sum(pulse_height)
                analysis['metrics'][f'{tally_name}_total_counts'] = total_counts

            elif isinstance(tally, AngleTally):
                # Angular distribution plot
                plt.figure(figsize=(10, 6))

                # Get angle bin midpoints
                angle_bins = tally_results['angle_bins']
                angle_mids = 0.5 * (angle_bins[1:] + angle_bins[:-1])

                # Get angular distribution
                distribution = tally_results['distribution']

                # Plot distribution
                plt.plot(angle_mids, distribution, 'o-')
                plt.xlabel('Angle (degrees)')
                plt.ylabel('Counts')
                plt.title(f'Angular Distribution - {tally_name}')
                plt.grid(True)

                # Save figure
                if output_dir:
                    fig_path = os.path.join(output_dir, f'{tally_name}_angular_distribution.png')
                    plt.savefig(fig_path)
                    analysis['figures'][f'{tally_name}_angular_distribution'] = fig_path

                plt.close()

                # Calculate metrics
                total_counts = np.sum(distribution)
                analysis['metrics'][f'{tally_name}_total_counts'] = total_counts

            elif isinstance(tally, PathLengthTally):
                # Path length distribution plot
                plt.figure(figsize=(10, 6))

                # Get length bin midpoints
                length_bins = tally_results['length_bins']
                length_mids = 0.5 * (length_bins[1:] + length_bins[:-1])

                # Get path length distribution
                path_length = tally_results['path_length']

                # Plot distribution
                plt.semilogx(length_mids, path_length, 'o-')
                plt.xlabel('Path Length (cm)')
                plt.ylabel('Counts')
                plt.title(f'Path Length Distribution - {tally_name}')
                plt.grid(True)

                # Save figure
                if output_dir:
                    fig_path = os.path.join(output_dir, f'{tally_name}_path_length.png')
                    plt.savefig(fig_path)
                    analysis['figures'][f'{tally_name}_path_length'] = fig_path

                plt.close()

                # Calculate metrics
                total_counts = np.sum(path_length)
                analysis['metrics'][f'{tally_name}_total_counts'] = total_counts

        # Save analysis results
        if output_dir:
            with open(os.path.join(output_dir, 'analysis_results.pkl'), 'wb') as f:
                pickle.dump(analysis, f)

        return analysis

    except Exception as e:
        log_error(f"Error analyzing simulation results: {e}")
        return None


if __name__ == "__main__":
    # Example usage when module is run directly
    from geometry import create_geometry_from_config
    from materials import create_materials_from_config
    from tally import create_tallies

    # Create simple test configuration
    config = {
        'geometry': {
            'type': 'slab',
            'width': 10.0,
            'material': 'lead'
        },
        'materials': {
            'lead': {
                'density': 11.35,
                'composition': {
                    'Pb': 1.0
                }
            }
        },
        'source': {
            'type': 'point',
            'position': (0, 0, -1),
            'direction': (0, 0, 1),
            'energy': 1.0
        },
        'simulation': {
            'n_particles': 1000,
            'n_batches': 2,
            'use_multiprocessing': True
        },
        'tallies': {
            'flux_front': {
                'type': 'flux',
                'energy_bins': 30,
                'spatial_bins': None
            },
            'deposition': {
                'type': 'energy_deposition',
                'spatial_bins': 10
            }
        }
    }

    # Create simulation components
    geometry = create_geometry_from_config(config['geometry'])
    materials = create_materials_from_config(config['materials'])
    tallies = create_tallies(config['tallies'])

    # Run simulation
    results = run_simulation(
        geometry=geometry,
        materials=materials,
        source=config['source'],
        tallies=tallies,
        params=config['simulation'],
        output_dir='./output'
    )

    # Analyze results
    analysis = analyze_simulation_results(
        results=results,
        tallies=tallies,
        output_dir='./output'
    )

    print("Simulation completed successfully!")
    print(f"Results saved to: ./output")