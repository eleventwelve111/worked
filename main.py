#!/usr/bin/env python
"""
Main module for gamma-ray streaming simulation.
This is the entry point for running simulations and analysis.
"""

import os
import argparse
import json
import time
import sys
from datetime import datetime

# Local imports
from logging import setup_logging, log_info, log_error, log_warning
from config import load_config, save_config
from requirement import check_system_requirements, validate_config, validate_simulation_parameters, suggest_optimization
from geometry import create_geometry
from materials import create_materials
from source import create_source
from tally import create_tallies
from simulation import run_simulation
from dose import calculate_dose_rates
from spectrum_analysis import analyze_spectrum, compare_spectra
from visualization import visualize_results
from analysisreport import generate_report


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Gamma-ray streaming simulation through shielding channels."
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to store simulation results"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running simulation"
    )

    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Run with reduced particle count for quick testing"
    )

    parser.add_argument(
        "--report-formats",
        type=str,
        default="html,excel",
        help="Comma-separated list of report formats to generate (html,excel,pdf)"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks"
    )

    return parser.parse_args()


def setup_directories(output_dir):
    """
    Create necessary directories for simulation output.

    Parameters:
        output_dir: Base directory for outputs

    Returns:
        dict: Dictionary of output directories
    """
    dirs = {
        'base': output_dir,
        'data': os.path.join(output_dir, 'data'),
        'plots': os.path.join(output_dir, 'plots'),
        'reports': os.path.join(output_dir, 'reports'),
        'logs': os.path.join(output_dir, 'logs')
    }

    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        log_info(f"Created directory: {dir_path}")

    return dirs


def main():
    """
    Main function to run simulation and analysis.
    """
    start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Setup output directories
    dirs = setup_directories(args.output_dir)

    # Setup logging
    log_file = os.path.join(dirs['logs'], f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file, args.log_level)

    log_info("=" * 50)
    log_info("GAMMA-RAY STREAMING SIMULATION")
    log_info("=" * 50)
    log_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Load configuration
        log_info(f"Loading configuration from: {args.config_file}")
        config = load_config(args.config_file)

        # Update config with command line arguments
        config['output_dir'] = args.output_dir

        # If quick run, reduce particle count
        if args.quick_run:
            log_info("Quick run mode: reducing particle count")
            if 'simulation' in config:
                original_count = config['simulation'].get('num_particles', 0)
                config['simulation']['num_particles'] = min(original_count, 10000)
                log_info(f"Reduced particle count from {original_count} to {config['simulation']['num_particles']}")

        # Validate configuration
        if not args.skip_validation:
            log_info("Validating configuration...")
            config_valid, errors = validate_config(config)

            if not config_valid:
                log_error("Configuration validation failed:")
                for error in errors:
                    log_error(f"  - {error}")

                if args.validate_only:
                    return 1

                log_warning("Continuing despite validation errors...")

            # Validate simulation parameters
            sim_valid, sim_errors = validate_simulation_parameters(config)

            if not sim_valid:
                log_warning("Simulation parameter validation warnings:")
                for error in sim_errors:
                    log_warning(f"  - {error}")

            # Check system requirements
            if not check_system_requirements(config):
                log_error("System does not meet requirements for this simulation")
                if not args.skip_validation:
                    return 1

                log_warning("Continuing despite system requirement issues...")

                # If only validating, exit
                if args.validate_only:
                    log_info("Configuration validation completed. Exiting.")
                    return 0

                # Suggest optimizations
                suggestions = suggest_optimization(config)
                if suggestions:
                    log_info("Optimization suggestions:")
                    for suggestion in suggestions:
                        log_info(f"  - {suggestion}")

                # Save working configuration
                config_backup = os.path.join(dirs['data'], 'working_config.json')
                save_config(config, config_backup)
                log_info(f"Saved working configuration to: {config_backup}")

                # Create simulation components
                log_info("Creating simulation components...")

                # Create geometry
                geometry = create_geometry(config.get('geometry', {}))
                log_info("Geometry created successfully")

                # Create materials
                materials = create_materials(config.get('materials', {}))
                log_info("Materials created successfully")

                # Create source
                source = create_source(config.get('source', {}))
                log_info("Source created successfully")

                # Create tallies
                tallies = create_tallies(config.get('tally', {}))
                log_info("Tallies created successfully")

                # Run simulation
                log_info("Starting simulation...")
                simulation_results = run_simulation(
                    geometry=geometry,
                    materials=materials,
                    source=source,
                    tallies=tallies,
                    params=config.get('simulation', {}),
                    output_dir=dirs['data']
                )
                log_info("Simulation completed successfully")

                # Calculate dose rates
                log_info("Calculating dose rates...")
                dose_results = calculate_dose_rates(
                    simulation_results,
                    config.get('dose', {}),
                    output_dir=dirs['data']
                )
                log_info("Dose calculation completed")

                # Analyze spectrum
                log_info("Analyzing energy spectrum...")
                spectrum_results = analyze_spectrum(
                    simulation_results,
                    config.get('spectrum_analysis', {}),
                    output_dir=dirs['data']
                )
                log_info("Spectrum analysis completed")

                # Combine results
                all_results = {
                    'simulation': simulation_results,
                    'dose': dose_results,
                    'spectrum': spectrum_results
                }

                # Run parallel simulations if configured
                parallel_results = {}
                if 'parallel_configs' in config:
                    log_info("Running parallel simulations...")

                    for idx, parallel_config in enumerate(config['parallel_configs']):
                        log_info(f"Starting parallel simulation {idx + 1}...")

                        # Create components for this parallel sim
                        p_geometry = create_geometry(parallel_config.get('geometry', {}))
                        p_materials = create_materials(parallel_config.get('materials', {}))
                        p_source = create_source(parallel_config.get('source', {}))
                        p_tallies = create_tallies(parallel_config.get('tally', {}))

                        # Run simulation
                        p_results = run_simulation(
                            geometry=p_geometry,
                            materials=p_materials,
                            source=p_source,
                            tallies=p_tallies,
                            params=parallel_config.get('simulation', {}),
                            output_dir=os.path.join(dirs['data'], f'parallel_{idx + 1}')
                        )

                        # Calculate dose
                        p_dose = calculate_dose_rates(
                            p_results,
                            parallel_config.get('dose', {}),
                            output_dir=os.path.join(dirs['data'], f'parallel_{idx + 1}')
                        )

                        # Store results
                        parallel_results[f'config_{idx + 1}'] = {
                            'simulation': p_results,
                            'dose': p_dose,
                            'parameters': parallel_config
                        }

                    # Compare spectra if multiple simulations exist
                    if len(parallel_results) > 1:
                        log_info("Comparing spectra from parallel simulations...")
                        spectra = [v['simulation'].get('spectrum', None) for v in parallel_results.values()]
                        labels = [f"Config {i + 1}" for i in range(len(spectra))]

                        comparison_results = compare_spectra(
                            spectra,
                            labels,
                            output_dir=dirs['plots']
                        )
                        all_results['spectrum_comparison'] = comparison_results

                # Add parallel results to all results
                all_results['parallel'] = parallel_results

                # Visualization
                log_info("Generating visualizations...")
                viz_results = visualize_results(
                    all_results,
                    config,
                    output_dir=dirs['plots']
                )
                all_results['visualization'] = viz_results
                log_info("Visualization completed")

                # Generate reports
                log_info("Generating reports...")
                report_formats = args.report_formats.split(',')
                report_results = generate_report(
                    all_results,
                    output_dir=dirs['reports'],
                    formats=report_formats
                )

                # Log report file paths
                for fmt, path in report_results.items():
                    log_info(f"Generated {fmt.upper()} report: {path}")

                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time
                log_info(f"Total execution time: {execution_time:.2f} seconds")

                log_info("=" * 50)
                log_info("SIMULATION COMPLETED SUCCESSFULLY")
                log_info("=" * 50)

                return 0

            except Exception as e:
            log_error(f"Error in simulation: {e}", exc_info=True)
            return 1

        if __name__ == "__main__":
            sys.exit(main())
