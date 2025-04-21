import numpy as np
import matplotlib.pyplot as plt
import openmc


def generate_weight_windows(geometry, source, target_regions, energy_groups=None):
    """
    Generate weight windows for variance reduction in Monte Carlo simulations.

    Parameters:
        geometry: OpenMC geometry object
        source: Source definition
        target_regions: List of region IDs for which to optimize weight windows
        energy_groups: Energy group structure (defaults to 10 logarithmic groups)

    Returns:
        dict: Weight window parameters for simulation
    """
    # Default energy groups if none provided (10 logarithmic groups from 10 keV to 10 MeV)
    if energy_groups is None:
        energy_groups = np.logspace(-2, 1, 11)  # 10 groups, bounds in MeV

    # Create weight window generator mesh
    # Determine mesh boundaries from geometry
    xmin, xmax, ymin, ymax, zmin, zmax = -100, 100, -100, 100, -100, 100

    # Higher resolution near source and target regions
    nx, ny, nz = 30, 30, 30

    mesh = openmc.RegularMesh()
    mesh.dimension = [nx, ny, nz]
    mesh.lower_left = [xmin, ymin, zmin]
    mesh.upper_right = [xmax, ymax, zmax]

    # Create weight window generator
    ww_generator = openmc.WeightWindowGenerator(
        mesh,
        energy_bounds=energy_groups,
        particle_type='photon',
        target_weight=1.0
    )

    # Configure weight window parameters
    ww_generator.max_split = 5  # Maximum number of particle splits
    ww_generator.min_weight = 0.1  # Minimum particle weight before Russian roulette
    ww_generator.max_weight = 10.0  # Maximum particle weight before splitting
    ww_generator.survival_ratio = 0.5  # Survival probability in Russian roulette

    # Configure target regions for importance map
    for region_id in target_regions:
        ww_generator.add_target(region_id)

    # Return generator and parameters
    weight_windows = {
        'generator': ww_generator,
        'energy_groups': energy_groups.tolist(),
        'mesh': {
            'dimensions': [nx, ny, nz],
            'bounds': [xmin, xmax, ymin, ymax, zmin, zmax]
        },
        'parameters': {
            'max_split': 5,
            'min_weight': 0.1,
            'max_weight': 10.0,
            'survival_ratio': 0.5
        }
    }

    return weight_windows


def analyze_weight_window_effectiveness(results_with_ww, results_without_ww):
    """
    Analyze the effectiveness of weight windows by comparing simulations with and without them.

    Parameters:
        results_with_ww: Simulation results with weight windows
        results_without_ww: Simulation results without weight windows

    Returns:
        dict: Analysis of weight window effectiveness
    """
    # Initialize analysis
    analysis = {
        'figure_of_merit_improvement': {},
        'uncertainty_reduction': {},
        'efficiency_gain': {},
        'run_time_comparison': {}
    }

    # Extract performance metrics
    runtime_with_ww = results_with_ww.get('runtime', 1.0)
    runtime_without_ww = results_without_ww.get('runtime', 1.0)

    # Analyze tally results
    for tally_name in results_with_ww.get('tallies', {}):
        if tally_name in results_without_ww.get('tallies', {}):
            tally_with_ww = results_with_ww['tallies'][tally_name]
            tally_without_ww = results_without_ww['tallies'][tally_name]

            # Extract mean values and relative errors
            mean_with_ww = tally_with_ww.get('mean', 0.0)
            mean_without_ww = tally_without_ww.get('mean', 0.0)

            rel_err_with_ww = tally_with_ww.get('rel_err', 1.0)
            rel_err_without_ww = tally_without_ww.get('rel_err', 1.0)

            # Calculate Figure of Merit (FOM = 1/(σ²T))
            fom_with_ww = 1.0 / ((rel_err_with_ww ** 2) * runtime_with_ww) if rel_err_with_ww > 0 else 0
            fom_without_ww = 1.0 / ((rel_err_without_ww ** 2) * runtime_without_ww) if rel_err_without_ww > 0 else 0

            # Calculate improvement ratios
            fom_improvement = fom_with_ww / fom_without_ww if fom_without_ww > 0 else float('inf')
            uncertainty_reduction = rel_err_without_ww / rel_err_with_ww if rel_err_with_ww > 0 else float('inf')
            efficiency_gain = fom_improvement

            # Store in analysis
            analysis['figure_of_merit_improvement'][tally_name] = float(fom_improvement)
            analysis['uncertainty_reduction'][tally_name] = float(uncertainty_reduction)
            analysis['efficiency_gain'][tally_name] = float(efficiency_gain)

    # Overall runtime comparison
    analysis['run_time_comparison'] = {
        'with_weight_windows': float(runtime_with_ww),
        'without_weight_windows': float(runtime_without_ww),
        'ratio': float(runtime_with_ww / runtime_without_ww) if runtime_without_ww > 0 else float('inf')
    }

    # Overall assessment
    avg_fom_improvement = np.mean(list(analysis['figure_of_merit_improvement'].values()))

    analysis['overall'] = {
        'avg_fom_improvement': float(avg_fom_improvement),
        'runtime_ratio': float(runtime_with_ww / runtime_without_ww) if runtime_without_ww > 0 else float('inf'),
        'effective': avg_fom_improvement > 1.0
    }

    return analysis


def plot_weight_window_importance_map(weight_windows, filename=None):
    """
    Plot the importance map generated by weight windows.

    Parameters:
        weight_windows: Weight window parameters
        filename: Optional filename to save the plot

    Returns:
        matplotlib.figure.Figure: Figure with importance map plots
    """
    if 'importance_map' not in weight_windows:
        print("No importance map available in weight windows data")
        return None

    importance_map = weight_windows['importance_map']
    energy_groups = weight_windows['energy_groups']

    # Get mesh dimensions
    nx, ny, nz = weight_windows['mesh']['dimensions']

    # Select central slices
    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2

    # Select energy group for visualization (middle group)
    energy_idx = len(energy_groups) // 2 - 1

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Importance map - XY plane at middle Z
    ax = axes[0, 0]
    xy_slice = importance_map[mid_z, :, :, energy_idx]
    im = ax.imshow(xy_slice, cmap='viridis', origin='lower')
    ax.set_xlabel('X index')
    ax.set_ylabel('Y index')
    ax.set_title(f'XY Importance Map (Z={mid_z})')
    plt.colorbar(im, ax=ax)

    # Plot 2: Importance map - XZ plane at middle Y
    ax = axes[0, 1]
    xz_slice = importance_map[:, mid_y, :, energy_idx]
    im = ax.imshow(xz_slice, cmap='viridis', origin='lower')
    ax.set_xlabel('X index')
    ax.set_ylabel('Z index')
    ax.set_title(f'XZ Importance Map (Y={mid_y})')
    plt.colorbar(im, ax=ax)

    # Plot 3: Energy dependence at selected position
    ax = axes[1, 0]
    energy_dependence = importance_map[mid_z, mid_y, mid_x, :]
    energy_centers = 0.5 * (energy_groups[:-1] + energy_groups[1:])
    ax.semilogx(energy_centers, energy_dependence, 'o-')
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Importance')
    ax.set_title(f'Energy-Dependent Importance at Central Position')
    ax.grid(True, which='both', alpha=0.3)

    # Plot 4: Histogram of importance values
    ax = axes[1, 1]
    importance_values = importance_map[:, :, :, energy_idx].flatten()
    ax.hist(importance_values, bins=30, alpha=0.7)
    ax.set_xlabel('Importance Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Importance Values')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig


def analyze_weight_window_optimization(simulation_results, optimization_iterations):
    """
    Analyze the convergence and optimization of weight windows over multiple iterations.

    Parameters:
        simulation_results: List of simulation results for each iteration
        optimization_iterations: List of weight window optimization parameters for each iteration

    Returns:
        dict: Analysis of weight window optimization process
    """
    # Initialize analysis
    analysis = {
        'convergence': [],
        'figure_of_merit': [],
        'relative_error': [],
        'runtime': []
    }

    # Extract convergence metrics for each iteration
    for i, (results, optimization) in enumerate(zip(simulation_results, optimization_iterations)):
        # Extract key metrics
        fom = results.get('figure_of_merit', 0.0)
        rel_err = results.get('relative_error', 1.0)
        runtime = results.get('runtime', 0.0)

        # Store in analysis
        analysis['convergence'].append({
            'iteration': i + 1,
            'figure_of_merit': float(fom),
            'relative_error': float(rel_err),
            'runtime': float(runtime)
        })

        analysis['figure_of_merit'].append(float(fom))
        analysis['relative_error'].append(float(rel_err))
        analysis['runtime'].append(float(runtime))

    # Calculate convergence rates and improvements
    if len(analysis['figure_of_merit']) > 1:
        # FOM improvement ratios between consecutive iterations
        fom_ratios = [analysis['figure_of_merit'][i] / analysis['figure_of_merit'][i - 1]
                      if analysis['figure_of_merit'][i - 1] > 0 else float('inf')
                      for i in range(1, len(analysis['figure_of_merit']))]

        # Is optimization converging?
        is_converging = all(ratio >= 1.0 for ratio in fom_ratios)

        # Calculate convergence rate (power-law fit)
        iterations = np.arange(1, len(analysis['figure_of_merit']) + 1)
        log_iterations = np.log(iterations)
        log_rel_errors = np.log(analysis['relative_error'])

        # Linear fit on log scale
        if len(log_iterations) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_iterations, log_rel_errors)
            convergence_rate = slope
        else:
            convergence_rate = 0.0
            r_value = 0.0

        # Store convergence metrics
        analysis['convergence_metrics'] = {
            'fom_improvement_ratios': fom_ratios,
            'is_converging': is_converging,
            'convergence_rate': float(convergence_rate),
            'r_squared': float(r_value ** 2),
            'total_improvement': (analysis['figure_of_merit'][-1] / analysis['figure_of_merit'][0]
                                  if analysis['figure_of_merit'][0] > 0 else float('inf'))
        }

    return analysis