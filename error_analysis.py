import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def propagate_uncertainties(results, energy_kev, channel_diameter):
    """
    Perform uncertainty propagation for dose calculations and quantify confidence intervals.

    Parameters:
        results: Simulation results dictionary
        energy_kev: Energy in keV
        channel_diameter: Channel diameter in cm

    Returns:
        dict: Uncertainty analysis results
    """
    # Extract relevant data
    dose_data = results['dose_data']

    # Initialize uncertainty analysis
    uncertainty_analysis = {
        'energy_kev': energy_kev,
        'channel_diameter': channel_diameter,
        'statistical_uncertainty': {},
        'systematic_uncertainty': {},
        'combined_uncertainty': {},
        'confidence_intervals': {}
    }

    # 1. Statistical uncertainty from Monte Carlo simulation
    for distance in dose_data:
        if distance == 'metadata':
            continue

        uncertainty_analysis['statistical_uncertainty'][distance] = {}

        for angle in dose_data[distance]:
            if angle == 'spectrum':
                continue

            dose_info = dose_data[distance][angle]

            # Extract statistical uncertainties if available
            if 'kerma_uncertainty' in dose_info:
                rel_uncertainty = dose_info['kerma_uncertainty'] / dose_info['kerma'] if dose_info['kerma'] > 0 else 0
                uncertainty_analysis['statistical_uncertainty'][distance][angle] = {
                    'relative': rel_uncertainty,
                    'absolute': dose_info['kerma_uncertainty']
                }
            elif 'dose_uncertainty' in dose_info:
                rel_uncertainty = dose_info['dose_uncertainty'] / dose_info['dose'] if dose_info['dose'] > 0 else 0
                uncertainty_analysis['statistical_uncertainty'][distance][angle] = {
                    'relative': rel_uncertainty,
                    'absolute': dose_info['dose_uncertainty']
                }
            else:
                # Estimate uncertainty based on number of particles if available
                if 'num_particles' in results and results['num_particles'] > 0:
                    # Uncertainty scales as 1/√N
                    est_rel_uncertainty = 1.0 / np.sqrt(results['num_particles'])

                    # Get dose value
                    dose_value = 0
                    if 'kerma' in dose_info:
                        dose_value = dose_info['kerma']
                    elif 'dose' in dose_info:
                        dose_value = dose_info['dose']

                    uncertainty_analysis['statistical_uncertainty'][distance][angle] = {
                        'relative': est_rel_uncertainty,
                        'absolute': est_rel_uncertainty * dose_value,
                        'estimated': True
                    }

    # 2. Systematic uncertainties
    # Sources: cross-section data, material compositions, geometry approximations

    # Cross-section uncertainty (typical range 1-10%)
    cross_section_uncertainty = 0.05  # 5% relative uncertainty

    # Material composition uncertainty (typical range 1-5%)
    material_uncertainty = 0.03  # 3% relative uncertainty

    # Geometry approximation uncertainty (typical range 1-3%)
    geometry_uncertainty = 0.02  # 2% relative uncertainty

    # Energy-dependent component (higher uncertainty at lower energies)
    energy_factor = max(0.01, 0.2 / np.sqrt(energy_kev / 100))  # Decreases with energy

    # Combine systematic uncertainties (in quadrature)
    systematic_rel_uncertainty = np.sqrt(
        cross_section_uncertainty ** 2 +
        material_uncertainty ** 2 +
        geometry_uncertainty ** 2 +
        energy_factor ** 2
    )

    # Apply to all positions
    for distance in dose_data:
        if distance == 'metadata':
            continue

        uncertainty_analysis['systematic_uncertainty'][distance] = {}

        for angle in dose_data[distance]:
            if angle == 'spectrum':
                continue

            dose_info = dose_data[distance][angle]

            # Get dose value
            dose_value = 0
            if 'kerma' in dose_info:
                dose_value = dose_info['kerma']
            elif 'dose' in dose_info:
                dose_value = dose_info['dose']

            uncertainty_analysis['systematic_uncertainty'][distance][angle] = {
                'relative': systematic_rel_uncertainty,
                'absolute': systematic_rel_uncertainty * dose_value
            }

    # 3. Combined uncertainty and confidence intervals
    for distance in dose_data:
        if distance == 'metadata':
            continue

        uncertainty_analysis['combined_uncertainty'][distance] = {}
        uncertainty_analysis['confidence_intervals'][distance] = {}

        for angle in dose_data[distance]:
            if angle == 'spectrum':
                continue

            if (distance in uncertainty_analysis['statistical_uncertainty'] and
                    angle in uncertainty_analysis['statistical_uncertainty'][distance] and
                    distance in uncertainty_analysis['systematic_uncertainty'] and
                    angle in uncertainty_analysis['systematic_uncertainty'][distance]):

                stat_unc = uncertainty_analysis['statistical_uncertainty'][distance][angle]['relative']
                sys_unc = uncertainty_analysis['systematic_uncertainty'][distance][angle]['relative']

                # Combined relative uncertainty (in quadrature)
                combined_rel_uncertainty = np.sqrt(stat_unc ** 2 + sys_unc ** 2)

                # Get dose value
                dose_info = dose_data[distance][angle]
                dose_value = 0
                if 'kerma' in dose_info:
                    dose_value = dose_info['kerma']
                elif 'dose' in dose_info:
                    dose_value = dose_info['dose']

                combined_abs_uncertainty = combined_rel_uncertainty * dose_value

                # Store combined uncertainty
                uncertainty_analysis['combined_uncertainty'][distance][angle] = {
                    'relative': combined_rel_uncertainty,
                    'absolute': combined_abs_uncertainty
                }

                # 95% confidence interval (assuming normal distribution)
                ci_95_lower = dose_value - 1.96 * combined_abs_uncertainty
                ci_95_upper = dose_value + 1.96 * combined_abs_uncertainty

                # 99% confidence interval
                ci_99_lower = dose_value - 2.576 * combined_abs_uncertainty
                ci_99_upper = dose_value + 2.576 * combined_abs_uncertainty

                uncertainty_analysis['confidence_intervals'][distance][angle] = {
                    'value': dose_value,
                    'ci_95': [max(0, ci_95_lower), ci_95_upper],
                    'ci_99': [max(0, ci_99_lower), ci_99_upper]
                }

    return uncertainty_analysis


def analyze_simulation_convergence(tally_results, num_batches):
    """
    Analyze convergence of simulation results with increasing number of particle batches.

    Parameters:
        tally_results: Dictionary of tally results at different batch numbers
        num_batches: Array of batch numbers

    Returns:
        dict: Convergence analysis results
    """
    # Extract tally values and relative errors
    tally_values = np.array([result['value'] for result in tally_results])
    rel_errors = np.array([result['rel_error'] for result in tally_results])

    # Calculate convergence rate
    # Expected: rel_error ~ 1/√N
    log_batches = np.log(num_batches)
    log_errors = np.log(rel_errors)

    # Linear fit to log-log data
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_batches, log_errors)

    # Theoretical convergence rate is -0.5
    theoretical_slope = -0.5
    deviation_from_theory = slope - theoretical_slope

    # Figure of Merit (FOM) = 1/(rel_error²·T)
    # FOM should be approximately constant if simulation is efficient
    # T ~ N for constant time per particle
    fom = 1.0 / (rel_errors ** 2 * num_batches)
    fom_variation = np.std(fom) / np.mean(fom)

    # Check for bias by comparing final result with average of last few results
    final_value = tally_values[-1]
    last_values = tally_values[int(0.8 * len(tally_values)):]  # Last 20% of results
    expected_value = np.mean(last_values)
    bias = (final_value - expected_value) / expected_value if expected_value != 0 else 0

    # Standard deviation of relative statistical error of the mean
    statistical_error_std = np.std([result.get('rel_error', 0) for result in tally_results])

    # Results
    convergence_results = {
        'num_batches': num_batches.tolist(),
        'tally_values': tally_values.tolist(),
        'rel_errors': rel_errors.tolist(),
        'convergence_rate': {
            'slope': float(slope),
            'theoretical_slope': theoretical_slope,
            'deviation': float(deviation_from_theory),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value)
        },
        'figure_of_merit': {
            'values': fom.tolist(),
            'mean': float(np.mean(fom)),
            'variation': float(fom_variation)
        },
        'bias_analysis': {
            'final_value': float(final_value),
            'expected_value': float(expected_value),
            'relative_bias': float(bias)
        },
        'statistical_quality': {
            'error_std': float(statistical_error_std),
            'converged': float(rel_errors[-1]) < 0.05  # Consider converged if relative error < 5%
        }
    }

    return convergence_results


def plot_uncertainty_analysis(uncertainty_analysis, filename=None):
    """
    Generate comprehensive plots for uncertainty analysis results.

    Parameters:
        uncertainty_analysis: Dictionary of uncertainty analysis results
        filename: Optional filename to save the plot

    Returns:
        matplotlib.figure.Figure: Figure with uncertainty plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    distances = []
    stat_uncs = []
    sys_uncs = []
    combined_uncs = []

    for distance in uncertainty_analysis['statistical_uncertainty']:
        for angle in uncertainty_analysis['statistical_uncertainty'][distance]:
            if angle == '0':  # Only consider angle = 0 for simplicity
                distances.append(int(distance))
                stat_uncs.append(uncertainty_analysis['statistical_uncertainty'][distance][angle]['relative'])
                sys_uncs.append(uncertainty_analysis['systematic_uncertainty'][distance][angle]['relative'])
                combined_uncs.append(uncertainty_analysis['combined_uncertainty'][distance][angle]['relative'])

    # Sort by distance
    sort_idx = np.argsort(distances)
    distances = np.array(distances)[sort_idx]
    stat_uncs = np.array(stat_uncs)[sort_idx]
    sys_uncs = np.array(sys_uncs)[sort_idx]
    combined_uncs = np.array(combined_uncs)[sort_idx]

    # Plot 1: Relative uncertainties by distance
    ax = axes[0, 0]
    ax.plot(distances, stat_uncs * 100, 'o-', label='Statistical')
    ax.plot(distances, sys_uncs * 100, 's-', label='Systematic')
    ax.plot(distances, combined_uncs * 100, '^-', label='Combined')
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Relative Uncertainty (%)')
    ax.set_title('Uncertainty Components vs. Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Uncertainty components breakdown
    ax = axes[0, 1]
    components = ['Statistical', 'Cross-section', 'Materials', 'Geometry', 'Energy-dependent']

    # These values should match those used in the propagate_uncertainties function
    values = [
        np.mean(stat_uncs) * 100,  # Average statistical uncertainty
        5.0,  # Cross-section uncertainty (5%)
        3.0,  # Material uncertainty (3%)
        2.0,  # Geometry uncertainty (2%)
        max(1.0, 20.0 / np.sqrt(uncertainty_analysis['energy_kev'] / 100))  # Energy factor
    ]

    ax.bar(components, values)
    ax.set_ylabel('Contribution (%)')
    ax.set_title('Uncertainty Components Breakdown')
    ax.grid(True, axis='y', alpha=0.3)

    # Plot 3: Confidence intervals for a specific position
    ax = axes[1, 0]

    # Select a reference distance (e.g., 30 cm)
    ref_distance = '30'
    if ref_distance in uncertainty_analysis['confidence_intervals'] and '0' in \
            uncertainty_analysis['confidence_intervals'][ref_distance]:
        ci_data = uncertainty_analysis['confidence_intervals'][ref_distance]['0']
        value = ci_data['value']
        ci_95 = ci_data['ci_95']
        ci_99 = ci_data['ci_99']

        ax.errorbar([1], [value], yerr=[[value - ci_95[0]], [ci_95[1] - value]],
                    fmt='o', capsize=5, label='95% CI')
        ax.errorbar([2], [value], yerr=[[value - ci_99[0]], [ci_99[1] - value]],
                    fmt='s', capsize=5, label='99% CI')

        ax.set_xlim(0, 3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['95% CI', '99% CI'])
        ax.set_ylabel('Dose (rem/hr)')
        ax.set_title(f'Confidence Intervals at {ref_distance} cm')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 4: Relative uncertainty vs. distance (log-log scale)
    ax = axes[1, 1]
    ax.loglog(distances, stat_uncs, 'o-', label='Statistical')

    # Theoretical 1/√r line
    ref_dist = distances[0]
    ref_unc = stat_uncs[0]
    theoretical_uncs = ref_unc * np.sqrt(ref_dist / np.array(distances))
    ax.loglog(distances, theoretical_uncs, 'k--', label='1/√r theory')

    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Relative Uncertainty')
    ax.set_title('Statistical Uncertainty Scaling')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig