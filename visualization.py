import matplotlib.pyplot as plt
import numpy as np
import os
import base64
from io import BytesIO
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import datetime
import jinja2


def plot_geometry(geometry, filename=None):
    """
    Plot the geometry of the shielding configuration.

    Parameters:
        geometry: OpenMC geometry object
        filename: Optional filename to save the plot

    Returns:
        matplotlib.figure.Figure: Figure with geometry plots
    """
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: XY plane
    ax = axes[0, 0]
    geometry.plot(basis='xy', axes=ax, colors={
        'shield': 'gray',
        'collimator': 'blue',
        'air': 'white',
        'detector': 'red',
        'source': 'yellow'
    })
    ax.set_title('XY Plane View')

    # Plot 2: XZ plane
    ax = axes[0, 1]
    geometry.plot(basis='xz', axes=ax, colors={
        'shield': 'gray',
        'collimator': 'blue',
        'air': 'white',
        'detector': 'red',
        'source': 'yellow'
    })
    ax.set_title('XZ Plane View')

    # Plot 3: YZ plane
    ax = axes[1, 0]
    geometry.plot(basis='yz', axes=ax, colors={
        'shield': 'gray',
        'collimator': 'blue',
        'air': 'white',
        'detector': 'red',
        'source': 'yellow'
    })
    ax.set_title('YZ Plane View')

    # Plot 4: 3D view (if available)
    ax = axes[1, 1]
    try:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(2, 2, 4, projection='3d')

        # Simplified 3D representation
        # This is a placeholder - real 3D geometry visualization would be more complex
        ax.set_title('3D Representation (Simplified)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Add text description of 3D geometry
        ax.text2D(0.5, 0.5, 'Full 3D visualization\nrequires additional tools',
                  ha='center', va='center', transform=ax.transAxes)

    except Exception as e:
        # Fall back to text description if 3D plotting fails
        ax.text(0.5, 0.5, '3D visualization unavailable',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig


def plot_results(results, filename=None):
    """
    Plot simulation results for dose vs. distance.

    Parameters:
        results: Dictionary of simulation results
        filename: Optional filename to save the plot

    Returns:
        matplotlib.figure.Figure: Figure with results plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract dose data
    if 'dose_data' not in results:
        return fig

    dose_data = results['dose_data']
    parameters = results.get('parameters', {})
    energy_kev = parameters.get('energy_kev', 662)

    # Extract distance and dose values
    distances = []
    doses = []
    rel_errors = []

    for distance in dose_data:
        if distance == 'metadata':
            continue

        for angle in dose_data[distance]:
            if angle == '0':  # Only plot angle = 0 for clarity
                distances.append(float(distance))

                dose_info = dose_data[distance][angle]
                if 'kerma' in dose_info:
                    doses.append(dose_info['kerma'])
                    rel_errors.append(dose_info.get('kerma_rel_err', 0.0))
                elif 'dose' in dose_info:
                    doses.append(dose_info['dose'])
                    rel_errors.append(dose_info.get('dose_rel_err', 0.0))
                elif 'dose_equiv' in dose_info:
                    doses.append(dose_info['dose_equiv'])
                    rel_errors.append(dose_info.get('dose_equiv_rel_err', 0.0))

    # Sort by distance
    if distances:
        sort_indices = np.argsort(distances)
        distances = np.array(distances)[sort_indices]
        doses = np.array(doses)[sort_indices]
        rel_errors = np.array(rel_errors)[sort_indices]

        # Plot 1: Dose vs Distance (log-log)
        ax = axes[0, 0]
        ax.errorbar(distances, doses, yerr=doses * rel_errors, fmt='o-', capsize=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Dose (rem/hr)')
        ax.set_title(f'Dose vs Distance ({energy_kev} keV)')
        ax.grid(True, which='both', alpha=0.3)

        # Add 1/r² reference line
        if len(distances) > 1:
            ref_dist = distances[0]
            ref_dose = doses[0]
            x_range = np.logspace(np.log10(min(distances)), np.log10(max(distances)), 100)
            y_range = ref_dose * (ref_dist / x_range) ** 2
            ax.plot(x_range, y_range, 'k--', label='1/r² law')
            ax.legend()

        # Plot 2: Dose vs Distance (linear)
        ax = axes[0, 1]
        ax.errorbar(distances, doses, yerr=doses * rel_errors, fmt='s-', capsize=5)
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Dose (rem/hr)')
        ax.set_title(f'Dose vs Distance - Linear Scale ({energy_kev} keV)')
        ax.grid(True, alpha=0.3)

        # Plot 3: Relative Error vs Distance
        ax = axes[1, 0]
        ax.plot(distances, rel_errors * 100, 'o-')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Relative Error (%)')
        ax.set_title('Statistical Uncertainty vs Distance')
        ax.grid(True, alpha=0.3)

        # Add horizontal reference lines for error thresholds
        ax.axhline(y=5, color='g', linestyle='--', label='5% (Good)')
        ax.axhline(y=10, color='y', linestyle='--', label='10% (Acceptable)')
        ax.axhline(y=25, color='r', linestyle='--', label='25% (Poor)')
        ax.legend()

        # Plot 4: Dose * r² vs Distance (test of inverse square law)
        ax = axes[1, 1]
        dose_r2 = doses * distances ** 2
        dose_r2_err = dose_r2 * rel_errors

        # Normalize by the first point
        if len(dose_r2) > 0:
            norm_factor = dose_r2[0]
            dose_r2_norm = dose_r2 / norm_factor
            dose_r2_err_norm = dose_r2_err / norm_factor

            ax.errorbar(distances, dose_r2_norm, yerr=dose_r2_err_norm, fmt='o-', capsize=5)
            ax.set_xlabel('Distance (cm)')
            ax.set_ylabel('Normalized Dose × r²')
            ax.set_title('Inverse Square Law Test')
            ax.grid(True, alpha=0.3)

            # Add horizontal reference line at y=1
            ax.axhline(y=1.0, color='k', linestyle='--', label='Perfect 1/r²')
            ax.legend()

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig


def plot_parametric_results(results, parameter_name, filename=None):
    """
    Plot results from a parametric study focusing on a specific parameter.

    Parameters:
        results: Dictionary of simulation results
        parameter_name: Name of the parameter to plot (e.g., 'energy_kev')
        filename: Optional filename to save the plot

    Returns:
        matplotlib.figure.Figure: Figure with parametric study plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract parameter values and corresponding dose results
    param_values = []
    dose_values = {}  # Dictionary mapping distance to list of doses
    rel_error_values = {}  # Dictionary mapping distance to list of relative errors

    reference_distances = [10.0, 30.0, 50.0, 100.0]  # Reference distances to plot

    # Extract data from results
    for param_id, result in results.items():
        if 'parameters' not in result or 'dose_data' not in result:
            continue

        parameters = result['parameters']
        if parameter_name not in parameters:
            continue

        param_value = parameters[parameter_name]
        param_values.append(param_value)

        dose_data = result['dose_data']

        # Extract dose values at reference distances
        for ref_distance in reference_distances:
            ref_distance_str = str(int(ref_distance)) if ref_distance.is_integer() else str(ref_distance)

            if ref_distance_str in dose_data and '0' in dose_data[ref_distance_str]:
                dose_info = dose_data[ref_distance_str]['0']

                # Get dose value (prefer KERMA if available)
                dose_value = None
                rel_error = 0.0

                if 'kerma' in dose_info:
                    dose_value = dose_info['kerma']
                    rel_error = dose_info.get('kerma_rel_err', 0.0)
                elif 'dose' in dose_info:
                    dose_value = dose_info['dose']
                    rel_error = dose_info.get('dose_rel_err', 0.0)
                elif 'dose_equiv' in dose_info:
                    dose_value = dose_info['dose_equiv']
                    rel_error = dose_info.get('dose_equiv_rel_err', 0.0)

                if dose_value is not None:
                    if ref_distance not in dose_values:
                        dose_values[ref_distance] = []
                        rel_error_values[ref_distance] = []

                    dose_values[ref_distance].append(dose_value)
                    rel_error_values[ref_distance].append(rel_error)

    # Sort data by parameter value
    if param_values:
        # Get unique parameter values and sort them
        unique_param_values = list(set(param_values))
        unique_param_values.sort()

        # Create a mapping from parameter value to index
        param_to_index = {val: i for i, val in enumerate(unique_param_values)}

        # Create arrays for plotting
        param_array = np.array(unique_param_values)
        dose_arrays = {}
        error_arrays = {}

        for distance in reference_distances:
            if distance in dose_values:
                dose_arrays[distance] = np.zeros(len(unique_param_values))
                error_arrays[distance] = np.zeros(len(unique_param_values))

                # Fill arrays with data
                for i, param_val in enumerate(param_values):
                    if i < len(dose_values[distance]):
                        idx = param_to_index[param_val]
                        dose_arrays[distance][idx] = dose_values[distance][i]
                        error_arrays[distance][idx] = rel_error_values[distance][i]

        # Plot 1: Dose vs Parameter Value for different distances
        ax = axes[0, 0]
        for distance in reference_distances:
            if distance in dose_arrays:
                ax.errorbar(param_array, dose_arrays[distance],
                            yerr=dose_arrays[distance] * error_arrays[distance],
                            fmt='o-', capsize=5, label=f'{distance} cm')

        ax.set_xlabel(f'{parameter_name.replace("_", " ").title()}')
        ax.set_ylabel('Dose (rem/hr)')
        ax.set_title(f'Dose vs {parameter_name.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Normalized Dose vs Parameter Value
        ax = axes[0, 1]
        for distance in reference_distances:
            if distance in dose_arrays and dose_arrays[distance][0] > 0:
                normalized_dose = dose_arrays[distance] / dose_arrays[distance][0]
                normalized_error = error_arrays[distance]
                ax.errorbar(param_array, normalized_dose,
                            yerr=normalized_dose * normalized_error,
                            fmt='o-', capsize=5, label=f'{distance} cm')

        ax.set_xlabel(f'{parameter_name.replace("_", " ").title()}')
        ax.set_ylabel('Normalized Dose')
        ax.set_title(f'Normalized Dose vs {parameter_name.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Attenuation Factor vs Parameter Value
        ax = axes[1, 0]
        reference_distance = reference_distances[0]  # Use first distance as reference

        if reference_distance in dose_arrays:
            for distance in reference_distances[1:]:  # Skip the reference distance
                if distance in dose_arrays and dose_arrays[reference_distance][0] > 0:
                    # Calculate attenuation factor: dose_ref/dose * (dist/dist_ref)²
                    # Calculate attenuation factor: dose_ref/dose * (dist/dist_ref)²
                    attenuation = (dose_arrays[reference_distance] / dose_arrays[distance]) * (
                                (distance / reference_distance) ** 2)

                    # Error propagation for attenuation factor
                    attenuation_error = attenuation * np.sqrt(
                        error_arrays[reference_distance] ** 2 + error_arrays[distance] ** 2
                    )

                    ax.errorbar(param_array, attenuation, yerr=attenuation_error,
                                fmt='o-', capsize=5, label=f'{distance} cm')

                ax.set_xlabel(f'{parameter_name.replace("_", " ").title()}')
                ax.set_ylabel('Attenuation Factor')
                ax.set_title(f'Attenuation Factor vs {parameter_name.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Plot 4: Relative Error vs Parameter Value
                ax = axes[1, 1]
                for distance in reference_distances:
                    if distance in error_arrays:
                        ax.plot(param_array, error_arrays[distance] * 100, 'o-', label=f'{distance} cm')

                ax.set_xlabel(f'{parameter_name.replace("_", " ").title()}')
                ax.set_ylabel('Relative Error (%)')
                ax.set_title(f'Statistical Uncertainty vs {parameter_name.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)

                # Add horizontal reference lines for error thresholds
                ax.axhline(y=5, color='g', linestyle='--', label='5% (Good)')
                ax.axhline(y=10, color='y', linestyle='--', label='10% (Acceptable)')
                ax.axhline(y=25, color='r', linestyle='--', label='25% (Poor)')
                ax.legend()

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')

            return fig

        def plot_energy_spectrum(results, distance=None, angle=None, filename=None):
            """
            Plot energy spectrum of dose at a specific distance and angle.

            Parameters:
                results: Dictionary of simulation results
                distance: Distance at which to plot spectrum (if None, uses first available)
                angle: Angle at which to plot spectrum (if None, uses first available)
                filename: Optional filename to save the plot

            Returns:
                matplotlib.figure.Figure: Figure with energy spectrum plot
            """
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Extract dose data
            if 'dose_data' not in results:
                return fig

            dose_data = results['dose_data']
            parameters = results.get('parameters', {})
            energy_kev = parameters.get('energy_kev', 662)

            # Find first available distance and angle if not specified
            if distance is None or str(distance) not in dose_data:
                available_distances = [d for d in dose_data.keys() if d != 'metadata']
                if not available_distances:
                    return fig
                distance = available_distances[0]
            else:
                distance = str(distance)

            if angle is None or str(angle) not in dose_data[distance]:
                available_angles = [a for a in dose_data[distance].keys() if a != 'spectrum']
                if not available_angles:
                    return fig
                angle = available_angles[0]
            else:
                angle = str(angle)

            # Check if spectrum data is available
            if 'spectrum' not in dose_data[distance]:
                return fig

            spectrum_data = dose_data[distance]['spectrum']

            # Extract energy bins and spectrum values
            energy_bins = spectrum_data.get('energy_bins', [])
            spectrum_values = spectrum_data.get('values', [])
            spectrum_errors = spectrum_data.get('errors', [])

            if not energy_bins or not spectrum_values:
                return fig

            # Convert to arrays
            energy_bins = np.array(energy_bins)
            spectrum_values = np.array(spectrum_values)
            spectrum_errors = np.array(spectrum_errors) if spectrum_errors else np.zeros_like(spectrum_values)

            # Calculate bin centers and widths
            bin_centers = 0.5 * (energy_bins[1:] + energy_bins[:-1])
            bin_widths = energy_bins[1:] - energy_bins[:-1]

            # Plot 1: Energy Spectrum (linear scale)
            ax = axes[0, 0]
            ax.bar(bin_centers, spectrum_values, width=bin_widths, alpha=0.7,
                   yerr=spectrum_errors, capsize=3, color='skyblue', edgecolor='blue')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Dose Contribution')
            ax.set_title(f'Energy Spectrum at {distance} cm, {angle}°')
            ax.grid(True, alpha=0.3)

            # Plot 2: Energy Spectrum (log scale)
            ax = axes[0, 1]
            ax.bar(bin_centers, spectrum_values, width=bin_widths, alpha=0.7,
                   yerr=spectrum_errors, capsize=3, color='skyblue', edgecolor='blue')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Dose Contribution')
            ax.set_title(f'Energy Spectrum (Log Scale) at {distance} cm, {angle}°')
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.3)

            # Plot 3: Cumulative Spectrum
            ax = axes[1, 0]
            cum_spectrum = np.cumsum(spectrum_values)

            # Normalize to percentage
            if cum_spectrum[-1] > 0:
                cum_spectrum = 100 * cum_spectrum / cum_spectrum[-1]

            ax.plot(bin_centers, cum_spectrum, 'o-', color='green')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Cumulative Contribution (%)')
            ax.set_title('Cumulative Energy Spectrum')
            ax.grid(True, alpha=0.3)

            # Add horizontal lines at 25%, 50%, 75%
            ax.axhline(y=25, color='r', linestyle='--', label='25%')
            ax.axhline(y=50, color='g', linestyle='--', label='50%')
            ax.axhline(y=75, color='b', linestyle='--', label='75%')
            ax.legend()

            # Plot 4: Spectrum with Error Bars (Relative Error)
            ax = axes[1, 1]
            rel_errors = (spectrum_errors / spectrum_values) * 100
            rel_errors = np.where(np.isfinite(rel_errors), rel_errors, 0)

            ax.bar(bin_centers, rel_errors, width=bin_widths, alpha=0.7, color='salmon')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Relative Error (%)')
            ax.set_title('Spectrum Relative Errors')
            ax.grid(True, alpha=0.3)

            # Add horizontal reference lines for error thresholds
            ax.axhline(y=5, color='g', linestyle='--', label='5% (Good)')
            ax.axhline(y=10, color='y', linestyle='--', label='10% (Acceptable)')
            ax.axhline(y=25, color='r', linestyle='--', label='25% (Poor)')
            ax.legend()

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')

            return fig

        def fig_to_base64(fig):
            """
            Convert matplotlib figure to base64 string for embedding in HTML.

            Parameters:
                fig: matplotlib.figure.Figure object

            Returns:
                str: Base64 encoded string of the figure
            """
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            return image_base64


def create_radiation_outside_wall_heatmap(results, title=None):
    """
    Create an enhanced close-up Cartesian heatmap showing radiation distribution outside the wall
    with optimized visualization for this specific shielding problem
    """
    # Extract mesh data
    mesh_result = np.array(results['mesh_result'])

    # Create figure with higher resolution
    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

    # Define the extent of the plot focused specifically on the area outside the wall
    x_min = source_to_wall_distance + wall_thickness - 5  # Slightly before wall exit
    x_max = source_to_wall_distance + wall_thickness + 150  # 150 cm outside wall
    y_min = -75
    y_max = 75

    # Calculate indices in the mesh corresponding to these limits
    mesh_x_coords = np.linspace(-10, source_to_wall_distance + wall_thickness + 200, mesh_result.shape[0])
    mesh_y_coords = np.linspace(-50, 50, mesh_result.shape[1])

    x_indices = np.logical_and(mesh_x_coords >= x_min, mesh_x_coords <= x_max)
    y_indices = np.logical_and(mesh_y_coords >= y_min, mesh_y_coords <= y_max)

    # Extract the section of the mesh for the region of interest
    x_subset = mesh_x_coords[x_indices]
    y_subset = mesh_y_coords[y_indices]
    outside_wall_data = mesh_result[np.ix_(x_indices, y_indices)]

    # Create coordinate meshes for the plot
    X, Y = np.meshgrid(x_subset, y_subset)

    # Apply adaptive smoothing for better visualization
    from scipy.ndimage import gaussian_filter
    sigma = max(1, min(3, 5 / (results['channel_diameter'] + 0.1)))  # Smaller channels need more smoothing
    smoothed_data = gaussian_filter(outside_wall_data.T, sigma=sigma)

    # Set zero or very small values to NaN to make them transparent
    min_nonzero = np.max([np.min(smoothed_data[smoothed_data > 0]) / 10, 1e-12])
    smoothed_data[smoothed_data < min_nonzero] = np.nan

    # Create an enhanced custom colormap specifically for radiation visualization
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.0, 0.0, 0.3),  # Dark blue (background/low values)
        (0.0, 0.2, 0.6),  # Blue
        (0.0, 0.5, 0.8),  # Light blue
        (0.0, 0.8, 0.8),  # Cyan
        (0.0, 0.9, 0.3),  # Blue-green
        (0.5, 1.0, 0.0),  # Green
        (0.8, 1.0, 0.0),  # Yellow-green
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.8, 0.0),  # Yellow-orange
        (1.0, 0.6, 0.0),  # Orange
        (1.0, 0.0, 0.0)  # Red (highest intensity)
    ]

    cmap_name = 'EnhancedRadiation'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Use contourf for smoother visualization with more levels
    levels = np.logspace(np.log10(min_nonzero), np.log10(np.nanmax(smoothed_data)), 20)
    contour = ax.contourf(X, Y, smoothed_data,
                          levels=levels,
                          norm=LogNorm(),
                          cmap=custom_cmap,
                          alpha=0.95,
                          extend='both')

    # Add contour lines for a better indication of dose levels
    contour_lines = ax.contour(X, Y, smoothed_data,
                               levels=levels[::4],  # Fewer contour lines
                               colors='black',
                               alpha=0.3,
                               linewidths=0.5)

    # Add colorbar with scientific notation
    cbar = fig.colorbar(contour, ax=ax, format='%.1e', pad=0.01)
    cbar.set_label('Radiation Flux (particles/cm²/s)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add wall back position with improved styling
    wall_exit_x = source_to_wall_distance + wall_thickness
    ax.axvline(x=wall_exit_x, color='black', linestyle='-', linewidth=2.5, label='Wall Back')

    # Draw a small section of the wall for context
    wall_section = plt.Rectangle((x_min, y_min), wall_exit_x - x_min, y_max - y_min,
                                 color='gray', alpha=0.5, edgecolor='black')
    ax.add_patch(wall_section)

    # Add detector position with improved styling
    detector_x = results['detector_x']
    detector_y = results['detector_y']

    # Only show detector if it's in the displayed area
    if x_min <= detector_x <= x_max and y_min <= detector_y <= y_max:
        detector_circle = plt.Circle((detector_x, detector_y), detector_diameter / 2,
                                     fill=False, color='red', linewidth=2, label='Detector')
        ax.add_patch(detector_circle)

        # Add beam path from channel to detector with an arrow
        arrow_props = dict(arrowstyle='->', linewidth=2, color='yellow', alpha=0.9)
        beam_arrow = ax.annotate('', xy=(detector_x, detector_y), xytext=(wall_exit_x, 0),
                                 arrowprops=arrow_props)

    # Add channel exit with improved styling
    channel_radius = results['channel_diameter'] / 2
    channel_exit = plt.Circle((wall_exit_x, 0), channel_radius,
                              color='white', alpha=1.0, edgecolor='black', linewidth=1.5,
                              label='Channel Exit')
    ax.add_patch(channel_exit)

    # Add concentric circles to show distance from channel exit
    for radius in [25, 50, 75, 100]:
        # Draw dashed circle
        distance_circle = plt.Circle((wall_exit_x, 0), radius,
                                     fill=False, color='white', linestyle='--', linewidth=1, alpha=0.6)
        ax.add_patch(distance_circle)

        # Add distance label along 45° angle
        angle = 45
        label_x = wall_exit_x + radius * np.cos(np.radians(angle))
        label_y = radius * np.sin(np.radians(angle))
        ax.text(label_x, label_y, f"{radius} cm", color='white', fontsize=9,
                ha='center', va='center', rotation=angle,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    # Add detector angle indication if not at 0°
    angle = results['detector_angle']
    if angle > 0:
        # Draw angle arc
        angle_radius = 30
        arc = plt.matplotlib.patches.Arc((wall_exit_x, 0),
                                         angle_radius * 2, angle_radius * 2,
                                         theta1=0, theta2=angle,
                                         color='white', linewidth=2)
        ax.add_patch(arc)

        # Add angle text at arc midpoint
        angle_text_x = wall_exit_x + angle_radius * 0.7 * np.cos(np.radians(angle / 2))
        angle_text_y = angle_radius * 0.7 * np.sin(np.radians(angle / 2))
        ax.text(angle_text_x, angle_text_y, f"{angle}°", color='white',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

    # Set labels and title with improved styling
    ax.set_xlabel('Distance (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Lateral Distance (cm)', fontsize=14, fontweight='bold')

    if title is None:
        title = (f"Radiation Distribution Outside Wall\n"
                 f"{results['energy']} MeV Gamma, Channel Diameter: {results['channel_diameter']} cm")
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    # Add improved legend with better positioning
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(),
                       loc='upper right', framealpha=0.9, fontsize=11)
    legend.get_frame().set_edgecolor('black')

    # Add enhanced grid with better styling
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add detailed information box
    info_text = (f"Source: {results['energy']} MeV Gamma\n"
                 f"Wall: {wall_thickness / ft_to_cm:.1f} ft concrete\n"
                 f"Channel: {results['channel_diameter']} cm ∅\n"
                 f"Detector: {results['detector_distance']} cm from wall\n"
                 f"Angle: {results['detector_angle']}°\n"
                 f"Dose Rate: {results['dose_rem_per_hr']:.2e} rem/hr")

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Highlight the region of 10% or greater of the maximum dose
    if not np.isnan(np.max(smoothed_data)):
        high_dose_level = np.max(smoothed_data) * 0.1
        high_dose_contour = ax.contour(X, Y, smoothed_data,
                                       levels=[high_dose_level],
                                       colors=['red'],
                                       linewidths=2)

        # Add label for high dose region
        plt.clabel(high_dose_contour, inline=True, fontsize=9,
                   fmt=lambda x: "10% of Max Dose")

    # Ensure proper aspect ratio
    ax.set_aspect('equal')

    # Save high-resolution figure
    plt.savefig(f"results/outside_wall_E{results['energy']}_D{results['channel_diameter']}_" +
                f"dist{results['detector_distance']}_ang{results['detector_angle']}.png",
                dpi=300, bbox_inches='tight')

    return fig


def analyze_streaming_effect(results_by_diameter, output_file=None):
    """
    Analyze the streaming effect through channels of different diameters.

    Parameters:
        results_by_diameter: Dictionary mapping channel diameters to simulation results
        output_file: Optional filename to save the analysis figures

    Returns:
        dict: Analysis results of streaming effects
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract data for analysis
    diameters = []
    doses = []
    streaming_factors = []
    beam_widths = []
    scatter_fractions = []

    # Theoretical solid angle streaming model for comparison
    def theoretical_streaming(d, L, energy_mev):
        """Simple theoretical model for radiation streaming through a cylindrical channel"""
        # d: diameter, L: length (thickness)
        # Returns relative transmission compared to no shielding

        # Aspect ratio
        aspect = L / d

        # Simple collimator formula with energy-dependent parameters
        if energy_mev < 1.0:
            k1, k2 = 0.1, 0.8  # Lower energy: more scatter, less direct transmission
        else:
            k1, k2 = 0.2, 0.6  # Higher energy: less scatter, more direct transmission

        # Direct beam component through geometric cross-section
        direct = k1 * (d / L) ** 2

        # Scattered component - decreases with higher aspect ratio
        scatter = k2 * np.exp(-aspect / 2)

        return direct + scatter

    # Reference dose for no channel
    ref_dose = min([r['dose_rem_per_hr'] for r in results_by_diameter.values()]) * 0.01
    wall_thickness = next(iter(results_by_diameter.values()))['wall_thickness']
    energy = next(iter(results_by_diameter.values()))['energy']

    # Extract data from results
    for diameter, result in sorted(results_by_diameter.items()):
        if 'dose_rem_per_hr' not in result:
            continue

        dose = result['dose_rem_per_hr']
        diameters.append(diameter)
        doses.append(dose)

        # Calculate streaming factor (ratio of dose compared to no streaming case)
        streaming_factor = dose / ref_dose
        streaming_factors.append(streaming_factor)

        # Estimate beam width at 50 cm from wall exit (from mesh tally data if available)
        if 'mesh_result' in result:
            mesh = np.array(result['mesh_result'])

            # Find the exit position in the mesh
            wall_exit_x = result['source_to_wall_distance'] + wall_thickness

            # Define distance for beam width measurement
            measurement_distance = 50  # cm from wall exit
            measurement_x = wall_exit_x + measurement_distance

            # Find closest mesh position to measurement point
            mesh_x_coords = np.linspace(-10, wall_exit_x + 200, mesh.shape[0])
            x_idx = np.argmin(np.abs(mesh_x_coords - measurement_x))

            # Get lateral profile at this position
            mesh_y_coords = np.linspace(-50,
                                        # Get lateral profile at this position
                                        mesh_y_coords=np.linspace(-50, 50, mesh.shape[1])
            lateral_profile = mesh[x_idx, :]

            # Calculate FWHM (Full Width at Half Maximum) of the beam
            max_value = np.max(lateral_profile)
            half_max = max_value / 2.0

            # Find points where profile crosses half-maximum
            above_half_max = lateral_profile > half_max
            edges = np.where(np.diff(above_half_max.astype(int)))[0]

            if len(edges) >= 2:
                left_edge = mesh_y_coords[edges[0]]
            right_edge = mesh_y_coords[edges[-1]]
            beam_width = right_edge - left_edge
            else:
            # If can't determine properly, estimate based on channel diameter
            beam_width = diameter * (1 + measurement_distance / wall_thickness)

            beam_widths.append(beam_width)

            # Calculate scatter fraction (radiation outside primary beam)
            total_radiation = np.sum(lateral_profile)
            central_idx = len(lateral_profile) // 2
            channel_radius_idx = int(diameter / 2 * len(mesh_y_coords) / 100) + 1  # Convert cm to indices
            primary_beam = np.sum(
                lateral_profile[central_idx - channel_radius_idx:central_idx + channel_radius_idx + 1])
            scatter_fraction = 1.0 - (primary_beam / total_radiation)
            scatter_fractions.append(scatter_fraction)
            else:
            # Estimate if mesh data not available
            beam_widths.append(diameter * 1.5)
            scatter_fractions.append(0.3)

            # Plot 1: Dose vs Channel Diameter
            ax = axes[0, 0]
            ax.plot(diameters, doses, 'o-', color='blue', linewidth=2, markersize=8)
            ax.set_xlabel('Channel Diameter (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Dose (rem/hr)', fontsize=12, fontweight='bold')
            ax.set_title('Dose vs Channel Diameter', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Calculate threshold where streaming becomes significant
            threshold_diameter = None
            for i in range(1, len(diameters)):
                if
            doses[i] / doses[i - 1] > 2:  # If dose doubles with small diameter change
            threshold_diameter = diameters[i]
            break

            if threshold_diameter:
                ax.axvline(x=threshold_diameter, color='red', linestyle='--', linewidth=2)
                ax.text(threshold_diameter + 0.2, np.min(doses) * 2,
                        f'Streaming\nThreshold\n{threshold_diameter} cm',
                        color='red', fontsize=10, fontweight='bold')

            # Plot 2: Streaming Factor vs Aspect Ratio (L/D)
            ax = axes[0, 1]
            aspect_ratios = [wall_thickness / d for d in diameters]
            ax.plot(aspect_ratios, streaming_factors, 'o-', color='green', linewidth=2, markersize=8)

            # Add theoretical model
            theory_aspects = np.linspace(min(aspect_ratios), max(aspect_ratios), 100)
            theory_diameters = [wall_thickness / a for a in theory_aspects]
            theory_streaming = [theoretical_streaming(d, wall_thickness, energy) for d in theory_diameters]
            ax.plot(theory_aspects, theory_streaming, '--', color='red', linewidth=2, label='Theoretical Model')

            ax.set_xlabel('Aspect Ratio (Wall Thickness / Diameter)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Streaming Factor', fontsize=12, fontweight='bold')
            ax.set_title('Radiation Streaming vs Aspect Ratio', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.legend()

            # Plot 3: Beam Width vs Channel Diameter
            ax = axes[1, 0]
            ax.plot(diameters, beam_widths, 'o-', color='purple', linewidth=2, markersize=8)
            ax.set_xlabel('Channel Diameter (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Beam Width at 50 cm (cm)', fontsize=12, fontweight='bold')
            ax.set_title('Beam Spreading vs Channel Diameter', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add reference line for linear relationship
            x_ref = np.array([min(diameters), max(diameters)])
            y_ref = x_ref * 1.5  # Simple linear model
            ax.plot(x_ref, y_ref, '--', color='gray', linewidth=1.5, label='Linear Reference')
            ax.legend()

            # Plot 4: Scatter Fraction vs Channel Diameter
            ax = axes[1, 1]
            ax.plot(diameters, scatter_fractions, 'o-', color='orange', linewidth=2, markersize=8)
            ax.set_xlabel('Channel Diameter (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Scatter Fraction', fontsize=12, fontweight='bold')
            ax.set_title('Scattered Radiation Fraction vs Channel Diameter', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')

            # Compile analysis results
            analysis_results = {
                'diameters': diameters,
                'doses': doses,
                'streaming_factors': streaming_factors,
                'beam_widths': beam_widths,
                'scatter_fractions': scatter_fractions,
                'threshold_diameter': threshold_diameter,
                'wall_thickness': wall_thickness,
                'energy': energy
            }

            return analysis_results

        def analyze_radiation_damage(results, years_of_operation=10, output_file=None):
            """
            Analyze potential radiation damage to the shield over time due to streaming.

            Parameters:
                results: Dictionary of simulation results
                years_of_operation: Years of continuous operation to assess damage
                output_file: Optional filename to save the analysis figures

            Returns:
                dict: Analysis results of potential radiation damage
            """
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Extract relevant parameters
            channel_diameter = results.get('channel_diameter', 5)  # cm
            wall_thickness = results.get('wall_thickness', 61)  # cm (default ~2ft)
            energy_mev = results.get('energy', 0.662)  # MeV
            dose_rate = results.get('dose_rem_per_hr', 0.001)  # rem/hr

            # Estimate radiation field inside the channel
            # This is a simplification - in reality would come from the simulation
            source_to_wall = results.get('source_to_wall_distance', 100)  # cm
            source_strength = results.get('source_strength', 1e9)  # particles/sec

            # Create position arrays for plots
            # Axial positions: from source-facing side (0) to exit side (wall_thickness)
            axial_positions = np.linspace(0, wall_thickness, 100)

            # Radial positions: from center (0) to edge of channel (channel_diameter/2)
            radial_positions = np.linspace(0, channel_diameter / 2, 50)

            # 1. Calculate radiation field within the channel
            # Simple model: radiation decreases with inverse square from source
            # And exponential attenuation in material
            def dose_in_channel(z, r):
                """
                Calculate approximate dose at position (z, r) within channel
                z: axial position from entrance (cm)
                r: radial position from centerline (cm)
                """
                # Channel radius
                R = channel_diameter / 2

                # Distance from source (assuming source at z=-source_to_wall)
                distance = source_to_wall + z

                # Basic geometric attenuation with distance
                geometric_factor = 1.0 / (distance ** 2)

                # Include radial dependence (higher at edges due to scatter)
                # This is a phenomenological model, not exact physics
                if r == 0:
                    radial_factor = 1.0
                else:
                    # Enhanced radiation at wall interface due to scatter
                    radial_factor = 1.0 + 0.5 * (r / R) ** 2

                return geometric_factor * radial_factor * source_strength

            # 2. Calculate cumulative damage over time
            # Concrete damage parameters
            # These are simplified models based on literature
            rad_to_Gy = 0.01  # 1 rad = 0.01 Gy
            rem_to_rad = 1.0  # Approximate for gamma rays
            hours_per_year = 24 * 365.25  # Continuous operation

            # Damage thresholds for concrete
            threshold_minor = 1e6 * rad_to_Gy  # Gy for minor damage (discoloration)
            threshold_moderate = 1e7 * rad_to_Gy  # Gy for moderate damage (cracking)
            threshold_severe = 1e8 * rad_to_Gy  # Gy for severe damage (structural)

            # Calculate dose matrices for plotting
            Z, R = np.meshgrid(axial_positions, radial_positions)
            dose_matrix = np.zeros_like(Z)

            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    dose_matrix[i, j] = dose_in_channel(Z[i, j], R[i, j])

            # Convert to accumulated dose over time
            accumulated_dose = dose_matrix * rem_to_rad * hours_per_year * years_of_operation

            # 3. Calculate stress and damage to surrounding material
            # Create damage index matrix (0-1 scale)
            damage_index = np.clip(accumulated_dose / threshold_moderate, 0, 1)

            # Plot 1: Radiation Field in Channel (2D heatmap)
            ax = axes[0, 0]
            contour = ax.contourf(Z, R, dose_matrix, levels=20, cmap='hot')
            ax.set_xlabel('Axial Position (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Radial Position (cm)', fontsize=12, fontweight='bold')
            ax.set_title('Radiation Field Inside Channel', fontsize=14, fontweight='bold')
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label('Relative Radiation Intensity', fontsize=10, fontweight='bold')

            # Plot 2: Accumulated Dose Over Time
            ax = axes[0, 1]
            contour = ax.contourf(Z, R, accumulated_dose, levels=20, cmap='viridis')
            ax.set_xlabel('Axial Position (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Radial Position (cm)', fontsize=12, fontweight='bold')
            ax.set_title(f'Accumulated Dose After {years_of_operation} Years', fontsize=14, fontweight='bold')
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label('Accumulated Dose (Gy)', fontsize=10, fontweight='bold')

            # Add contour lines for damage thresholds
            if np.max(accumulated_dose) > threshold_minor:
                ax.contour(Z, R, accumulated_dose, levels=[threshold_minor],
                           colors=['yellow'], linestyles=['--'], linewidths=[2])
                ax.text(wall_thickness * 0.8, channel_diameter * 0.25, 'Minor Damage',
                        color='yellow', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

            if np.max(accumulated_dose) > threshold_moderate:
                ax.contour(Z, R, accumulated_dose, levels=[threshold_moderate],
                           colors=['orange'], linestyles=['--'], linewidths=[2])
                ax.text(wall_thickness * 0.6, channel_diameter * 0.25, 'Moderate Damage',
                        color='orange', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

            if np.max(accumulated_dose) > threshold_severe:
                ax.contour(Z, R, accumulated_dose, levels=[threshold_severe],
                           colors=['red'], linestyles=['--'], linewidths=[2])
                ax.text(wall_thickness * 0.4, channel_diameter * 0.25, 'Severe Damage',
                        color='red', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

            # Plot 3: Damage Index Profile Along Channel Wall
            ax = axes[1, 0]
            ax.plot(axial_positions, damage_index[-1, :], 'r-', linewidth=3)
            ax.set_xlabel('Axial Position (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Damage Index (0-1)', fontsize=12, fontweight='bold')
            ax.set_title('Radiation Damage Profile Along Channel Wall', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add reference lines for damage levels
            ax.axhline(y=0.3, color='yellow', linestyle='--', label='Minor Damage')
            ax.axhline(y=0.6, color='orange', linestyle='--', label='Moderate Damage')
            ax.axhline(y=0.9, color='red', linestyle='--', label='Severe Damage')
            ax.legend()

            # Plot 4: Material Degradation Timeline
            ax = axes[1, 1]

            # Define time points for degradation analysis
            years = np.linspace(0, 30, 31)  # 0 to 30 years

            # Calculate maximum damage index at different time points
            max_damage_indices = []
            for year in years:
                # Scale the damage index by ratio of years
                year_damage = damage_index * (year / years_of_operation) if years_of_operation > 0 else 0
                max_damage_indices.append(np.max(year_damage))

            # Plot degradation over time
            ax.plot(years, max_damage_indices, 'bo-', linewidth=2)
            ax.set_xlabel('Years of Operation', fontsize=12, fontweight='bold')
            ax.set_ylabel('Maximum Damage Index', fontsize=12, fontweight='bold')
            ax.set_title('Shield Degradation Timeline', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add reference lines for damage levels
            ax.axhline(y=0.3, color='yellow', linestyle='--', label='Minor Damage')
            ax.axhline(y=0.6, color='orange', linestyle='--', label='Moderate Damage')
            ax.axhline(y=0.9, color='red', linestyle='--', label='Severe Damage')

            # Calculate and mark time to reach critical damage levels
            damage_thresholds = [0.3, 0.6, 0.9]
            threshold_names = ["Minor", "Moderate", "Severe"]
            threshold_colors = ["yellow", "orange", "red"]

            for threshold, name, color in zip(damage_thresholds, threshold_names, threshold_colors):
                # Find when damage exceeds threshold
                if max(max_damage_indices) > threshold:
                    # Interpolate to find when threshold is crossed
                    years_to_threshold = np.interp(threshold, max_damage_indices, years)
                    ax.axvline(x=years_to_threshold, color=color, linestyle='--')
                    ax.text(years_to_threshold + 0.5, threshold - 0.05,
                            f"{name}: {years_to_threshold:.1f} years",
                            color=color, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7))

            ax.legend()

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')

            # Compile analysis results
            analysis_results = {
                'years_of_operation': years_of_operation,
                'channel_diameter': channel_diameter,
                'wall_thickness': wall_thickness,
                'energy_mev': energy_mev,
                'max_damage_index': max(max_damage_indices),
                'years_to_minor_damage': np.interp(0.3, max_damage_indices, years) if max(
                    max_damage_indices) > 0.3 else None,
                'years_to_moderate_damage': np.interp(0.6, max_damage_indices, years) if max(
                    max_damage_indices) > 0.6 else None,
                'years_to_severe_damage': np.interp(0.9, max_damage_indices, years) if max(
                    max_damage_indices) > 0.9 else None,
            }

            return analysis_results

        def analyze_channel_geometry_optimization(results_by_geometry, output_file=None):
            """
            Analyze how different channel geometries affect streaming and find optimal designs.

            Parameters:
                results_by_geometry: Dictionary mapping geometry parameters to simulation results
                output_file: Optional filename to save the analysis figures

            Returns:
                dict: Analysis results including optimal geometries
            """
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Extract geometry parameters and corresponding doses
            geometries = []
            doses = []
            streaming_factors = []
            effective_shielding = []

            # Get reference dose (no channel)
            wall_thickness = next(iter(results_by_geometry.values()))['wall_thickness']
            source_strength = next(iter(results_by_geometry.values())).get('source_strength', 1e9)
            energy = next(iter(results_by_geometry.values()))['energy']

            # Theoretical calculation for solid concrete (no channel)
            # Simple exponential attenuation model
            def attenuation_factor(thickness_cm, energy_mev):
                """Calculate radiation attenuation through concrete"""
                # Energy-dependent linear attenuation coefficient for concrete (cm^-1)
                if energy_mev < 0.1:
                    mu = 0.5  # Higher attenuation for low energy
                elif energy_mev < 1.0:
                    mu = 0.2
                else:
                    mu = 0.1  # Lower attenuation for high energy

                return np.exp(-mu * thickness_cm)

            # Reference dose with no channel
            ref_attenuation = attenuation_factor(wall_thickness, energy)
            ref_dose = source_strength * ref_attenuation / ((wall_thickness + 100) ** 2)

            # Process results
            for geometry, result in results_by_geometry.items():
                if isinstance(geometry, tuple) and len(geometry) >= 2:
                    # Assuming geometry is (diameter, shape_type, *other_params)
                    diameter = geometry[0]
                    shape_type = geometry[1]

                    dose = result.get('dose_rem_per_hr', 0)
                    doses.append(dose)
                    geometries.append(geometry)

                    # Calculate streaming factor
                    streaming_factor = dose / ref_dose if ref_dose > 0 else 0
                    streaming_factors.append(streaming_factor)

                    # Calculate effective shielding (dose relative to open beam)
                    open_beam_dose = source_strength / ((wall_thickness + 100) ** 2)
                    effective_shielding.append(1 - (dose / open_beam_dose))

            # Group results by shape type
            shape_types = set(g[1] for g in geometries)
            shape_colors = {'straight': 'blue', 'tapered': 'green', 'stepped': 'orange',
                            'offset': 'purple', 'curved': 'red', 'labyrinth': 'brown'}

            # Plot 1: Dose vs Diameter by Shape Type
            ax = axes[0, 0]

            for shape in shape_types:
                indices = [i for i, g in enumerate(geometries) if g[1] == shape]
                if not indices:
                    continue

                # Extract data for this shape
                shape_diameters = [geometries[i][0] for i in indices]
                shape_doses = [doses[i] for i in indices]

                # Sort by diameter
                sorted_indices = np.argsort(shape_diameters)
                shape_diameters = [shape_diameters[i] for i in sorted_indices]
                shape_doses = [shape_doses[i] for i in sorted_indices]

                # Plot
                color = shape_colors.get(shape, 'gray')
                ax.plot(shape_diameters, shape_doses, 'o-', color=color, linewidth=2, label=shape.capitalize())

            ax.set_xlabel('Channel Diameter (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Dose (rem/hr)', fontsize=12, fontweight='bold')
            ax.set_title('Dose vs Channel Diameter by Shape Type', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Plot 2: Streaming Factor vs L/D Ratio by Shape Type
            ax = axes[0, 1]

            for shape in shape_types:
                indices = [i for i, g in enumerate(geometries) if g[1] == shape]
                if not indices:
                    continue

                # Extract data for this shape
                shape_diameters = [geometries[i][0] for i in indices]
                shape_factors = [streaming_factors[i] for i in indices]
                shape_ratios = [wall_thickness / d for d in shape_diameters]

                # Sort by aspect ratio
                sorted_indices = np.argsort(shape_ratios)
                shape_ratios = [shape_ratios[i] for i in sorted_indices]
                shape_factors = [shape_factors[i] for i in sorted_indices]

                # Plot
                color = shape_colors.get(shape, 'gray')
                ax.plot(shape_ratios, shape_factors, 'o-', color=color, linewidth=2, label=shape.capitalize())

            ax.set_xlabel('Aspect Ratio (L/D)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Streaming Factor', fontsize=12, fontweight='bold')
            ax.set_title('Streaming vs Aspect Ratio by Shape Type', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Plot 3: Effective Shielding vs Channel Diameter
            ax = axes[1, 0]

            for shape in shape_types:
                indices = [i for i, g in enumerate(geometries) if g[1] == shape]
                if not indices:
                    continue

                # Extract data for this shape
                shape_diameters = [geometries[i][0] for i in indices]
                shape_shielding = [effective_shielding[i] for i in indices]

                # Sort by diameter
                sorted_indices = np.argsort(shape_diameters)
                shape_diameters = [shape_diameters[i] for i in sorted_indices]
                shape_shielding = [shape_shielding[i] for i in sorted_indices]

                # Plot
                color = shape_colors.get(shape, 'gray')
                ax.plot(shape_diameters, shape_shielding, 'o-', color=color, linewidth=2, label=shape.capitalize())

            ax.set_xlabel('Channel Diameter (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Effective Shielding (0-1)', fontsize=12, fontweight='bold')
            ax.set_title('Shielding Effectiveness vs Diameter by Shape Type', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Plot 4: Comparison of Best Design from Each Shape Type
            ax = axes[1, 1]

            # Find best design (lowest dose) for each shape type
            best_designs = {}
            for shape in shape_types:
                indices = [i for i, g in enumerate(geometries) if g[1] == shape]
                if not indices:
                    continue

                # Find minimum dose design for this shape
                min_dose_idx = indices[np.argmin([doses[i] for i in indices])]
                best_designs[shape] = (geometries[min_dose_idx], doses[min_dose_idx])

            # Create bar chart comparing best designs
            shapes = list(best_designs.keys())
            best_doses = [best_designs[shape][1] for shape in shapes]

            bar_colors = [shape_colors.get(shape, 'gray') for shape in shapes]
            bars = ax.bar(shapes, best_doses, color=bar_colors, alpha=0.8)

            ax.set_xlabel('Channel Shape', fontsize=12, fontweight='bold')
            ax.set_ylabel('Minimum Dose (rem/hr)', fontsize=12, fontweight='bold')
            ax.set_title('Comparison of Optimal Channel Designs', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')

            # Add design details to bars
            for bar, shape in zip(bars, shapes):
                design, _ = best_designs[shape]
                diameter = design[0]
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                        f'D={diameter}cm',
                        ha='center', va='bottom', rotation=0, fontsize=9)

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')

            # Find overall best design
            best_shape = min(best_designs.items(), key=lambda x: x[1][1])[0]
            best_geometry, best_dose = best_designs[best_shape]

            # Compile analysis results
            analysis_results = {
                'best_shape': best_shape,
                'best_geometry': best_geometry,
                'best_dose': best_dose,
                'best_designs_by_shape': best_designs,
                'ref_dose': ref_dose,
                'wall_thickness': wall_thickness,
                'energy': energy
            }

            return analysis_results

        def create_streaming_pathways_visualization(results, output_file=None):
            """
            Create a detailed visualization of radiation streaming pathways through the channel.

            Parameters:
                results: Dictionary of simulation results including particle tracks if available
                output_file: Optional filename to save the visualization

            Returns:
                fig: The matplotlib figure object
            """
            fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

            # Extract relevant parameters
            channel_diameter = results.get('channel_diameter', 5)  # cm
            wall_thickness = results.get('wall_thickness', 61)  # cm (default ~2ft)
            source_to_wall = results.get('source_to_wall_distance', 100)  # cm

            # Define wall boundaries
            wall_start_x = source_to_wall
            wall_end_x = source_to_wall + wall_thickness

            # Draw wall
            wall_rect = plt.Rectangle((wall_start_x, -50), wall_thickness, 100,
                                      color='gray', alpha=0.7, edgecolor='black')
            ax.add_patch(wall_rect)

            # Draw channel
            channel_radius = channel_diameter / 2
            channel_rect = plt.Rectangle((wall_start_x, -channel_radius), wall_thickness, channel_diameter,
                                         color='white', alpha=1.0, edgecolor='black')
            ax.add_patch(channel_rect)

            # Define number of simulated particles and their properties
            # This would ideally come from the actual simulation results
            num_particles = 1000

            # Generate random particle tracks
            # In a real implementation, these would come from the simulation
            np.random.seed(42)  # For reproducibility

            # Different types of particle paths:
            # 1. Direct transmission through channel
            # 2. Scattered from channel walls
            # 3. Penetration through wall material

            # Generate particle tracks
            tracks = []

            # Function to check if point is inside channel
            def is_in_channel(x, y):
                if x < wall_start_x or x > wall_end_x:
                    return False
                return abs(y) <= channel_radius

            # 1. Direct transmission particles (30%)
            for i in range(int(num_particles * 0.3)):
                # Start randomly before wall
                start_x = source_to_wall - np.random.uniform(10, 50)
                start_y = np.random.uniform(-channel_radius * 0.8, channel_radius * 0.8)

                # End randomly after wall
                end_x = wall_end_x + np.random.uniform(10, 100)

                # Small scatter angle
                angle = np.random.normal(0, 3)  # degrees, small standard deviation
                # Calculate end position based on scatter angle
                angle_rad = np.radians(angle)
                path_length = end_x - start_x
                end_y = start_y + path_length * np.tan(angle_rad)

                # Create track
                track = {
                    'type': 'direct',
                    'points': [(start_x, start_y), (end_x, end_y)],
                    'color': 'yellow',
                    'alpha': 0.8,
                    'linewidth': 1.5
                }
                tracks.append(track)

                # 2. Scattered particles from channel walls (50%)
            for i in range(int(num_particles * 0.5)):
                # Start randomly before wall
                start_x = source_to_wall - np.random.uniform(10, 50)
                start_y = np.random.uniform(-channel_radius * 1.5, channel_radius * 1.5)

                # Generate a scatter point on channel wall
                scatter_x = wall_start_x + np.random.uniform(0, wall_thickness)

                # Determine if top or bottom wall
                if np.random.rand() > 0.5:
                    scatter_y = channel_radius  # Top wall
                else:
                    scatter_y = -channel_radius  # Bottom wall

                # End randomly after wall
                end_x = wall_end_x + np.random.uniform(10, 100)

                # Scatter angle (larger than direct)
                if scatter_y > 0:  # Top wall, scatter downward
                    angle = np.random.uniform(-60, -10)
                else:  # Bottom wall, scatter upward
                    angle = np.random.uniform(10, 60)

                # Calculate end position based on scatter angle
                angle_rad = np.radians(angle)
                path_length = end_x - scatter_x
                end_y = scatter_y + path_length * np.tan(angle_rad)

                # Create track with a scatter point
                track = {
                    'type': 'scatter',
                    'points': [(start_x, start_y), (scatter_x, scatter_y), (end_x, end_y)],
                    'color': 'cyan',
                    'alpha': 0.6,
                    'linewidth': 1.0
                }
                tracks.append(track)

                # 3. Penetration through wall (20%)
            for i in range(int(num_particles * 0.2)):
                # Start randomly before wall
                start_x = source_to_wall - np.random.uniform(10, 50)
                start_y = np.random.uniform(-40, 40)

                # Generate a random entry point to wall (not in channel)
                entry_x = wall_start_x
                entry_y = np.random.uniform(-40, 40)
                while is_in_channel(entry_x, entry_y):
                    entry_y = np.random.uniform(-40, 40)

                # Generate random exit point from wall
                exit_x = wall_end_x

                # Calculate approximate attenuation for penetration
                penetration_depth = wall_thickness

                # Higher energy particles penetrate further
                energy = results.get('energy', 0.662)  # MeV
                attenuation_factor = np.exp(-0.1 * penetration_depth / energy)

                # Only show tracks with significant energy remaining
                if np.random.rand() < attenuation_factor * 0.1:  # Further reduce for visibility
                    exit_y = entry_y + np.random.normal(0, 5)  # Some scatter during penetration

                    # End randomly after wall
                    end_x = wall_end_x + np.random.uniform(10, 50)

                    # Small additional scatter angle
                    angle = np.random.normal(0, 10)  # degrees
                    angle_rad = np.radians(angle)
                    path_length = end_x - exit_x
                    end_y = exit_y + path_length * np.tan(angle_rad)

                    # Create track
                    track = {
                        'type': 'penetration',
                        'points': [(start_x, start_y), (entry_x, entry_y), (exit_x, exit_y), (end_x, end_y)],
                        'color': 'red',
                        'alpha': 0.3,
                        'linewidth': 0.5
                    }
                    tracks.append(track)

                # Plot particle tracks
            for track in tracks:
                points = track['points']
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                ax.plot(xs, ys, '-', color=track['color'], alpha=track['alpha'],
                        linewidth=track['linewidth'])

                # Draw source
            source_x = 0
            source_y = 0
            source_circle = plt.Circle((source_x, source_y), 5,
                                       color='orange', alpha=0.8, edgecolor='black')
            ax.add_patch(source_circle)
            ax.text(source_x, source_y, "Source", ha='center', va='center', fontweight='bold')

            # Highlight primary beam and scatter regions
            # Main beam path
            beam_patch = plt.Rectangle((source_x, -channel_radius), wall_end_x + 100, 2 * channel_radius,
                                       color='yellow', alpha=0.1, edgecolor='none')
            ax.add_patch(beam_patch)

            # First scatter region (conical shape outside the channel)
            scatter_points = [
                (wall_end_x, -channel_radius),
                (wall_end_x, channel_radius),
                (wall_end_x + 100, 3 * channel_radius),
                (wall_end_x + 100, -3 * channel_radius)
            ]
            scatter_polygon = plt.Polygon(scatter_points, color='cyan', alpha=0.1, edgecolor='none')
            ax.add_patch(scatter_polygon)

            # Highlight areas of potential wall damage
            # These are the regions where the channel meets the wall
            damage_top = plt.Rectangle((wall_start_x, channel_radius - 0.5), wall_thickness, 1,
                                       color='red', alpha=0.5, edgecolor='none')
            damage_bottom = plt.Rectangle((wall_start_x, -channel_radius - 0.5), wall_thickness, 1,
                                          color='red', alpha=0.5, edgecolor='none')
            ax.add_patch(damage_top)
            ax.add_patch(damage_bottom)

            # Add labels and annotations
            # Primary beam label
            ax.text(wall_start_x + wall_thickness / 2, 0, "Primary Beam",
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7))

            # Scatter region label
            ax.text(wall_end_x + 50, 2 * channel_radius, "Scatter Region",
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7))

            # Wall damage labels
            ax.text(wall_start_x + wall_thickness / 2, channel_radius + 2, "Potential Wall Damage",
                    ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
            ax.text(wall_start_x + wall_thickness / 2, -channel_radius - 2, "Potential Wall Damage",
                    ha='center', va='top', fontsize=9, color='red', fontweight='bold')

            # Legend items
            direct_line = plt.Line2D([0], [0], color='yellow', linewidth=2, label='Direct Transmission')
            scatter_line = plt.Line2D([0], [0], color='cyan', linewidth=2, label='Wall Scatter')
            penetration_line = plt.Line2D([0], [0], color='red', linewidth=1, label='Wall Penetration')
            damage_patch = plt.Patch(color='red', alpha=0.5, label='Damage-Prone Areas')

            ax.legend(handles=[direct_line, scatter_line, penetration_line, damage_patch],
                      loc='upper right', framealpha=0.9)

            # Set axis properties
            ax.set_xlim(source_x - 10, wall_end_x + 120)
            ax.set_ylim(-50, 50)
            ax.set_xlabel('Distance (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Lateral Distance (cm)', fontsize=12, fontweight='bold')

            # Set title and add information box
            title = f"Radiation Streaming Pathways Analysis\n{results.get('energy', 0.662)} MeV Gamma, Channel Diameter: {channel_diameter} cm"
            ax.set_title(title, fontsize=14, fontweight='bold')

            # Add information text box
            info_text = (f"Source Energy: {results.get('energy', 0.662)} MeV Gamma\n"
                         f"Wall Thickness: {wall_thickness / 30.48:.1f} ft concrete\n"
                         f"Channel Diameter: {channel_diameter} cm\n"
                         f"Key Streaming Mechanisms:\n"
                         f"• Direct transmission through channel\n"
                         f"• Scattering from channel walls\n"
                         f"• Limited wall penetration\n\n"
                         f"Potential wall damage increases with:\n"
                         f"• Higher energies\n"
                         f"• Smaller channel diameter\n"
                         f"• Longer operational time")

            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black')
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')

            return fig

        def create_streaming_effect_summary(results_dict, years=10, output_file=None):
            """
            Create a comprehensive summary of channel streaming effects and safety considerations.

            Parameters:
                results_dict: Dictionary with multiple simulation results
                years: Years of operation to consider for damage assessment
                output_file: Optional filename to save the summary

            Returns:
                fig: The matplotlib figure object
            """
            fig = plt.figure(figsize=(15, 20), dpi=150)
            gs = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.5])

            # Extract parameters from the results
            diameters = sorted([r.get('channel_diameter', 0) for r in results_dict.values() if 'channel_diameter' in r])
            energies = sorted(list(set([r.get('energy', 0) for r in results_dict.values() if 'energy' in r])))

            # Create a grid of diameter vs energy for streaming factors
            streaming_grid = np.zeros((len(diameters), len(energies)))
            damage_grid = np.zeros((len(diameters), len(energies)))
            safety_grid = np.zeros((len(diameters), len(energies)))

            for i, d in enumerate(diameters):
                for j, e in enumerate(energies):
                    # Find matching result
                    for result in results_dict.values():
                        if result.get('channel_diameter') == d and result.get('energy') == e:
                            # Calculate streaming factor (just example - would come from results)
                            streaming_grid[i, j] = result.get('dose_rem_per_hr', 0) * 1000

                            # Estimate damage potential
                            wall_thickness = result.get('wall_thickness', 61)
                            damage_grid[i, j] = streaming_grid[i, j] * years * 24 * 365.25 / 1e6 * (e / 0.5)

                            # Safety assessment (0-1 scale, 1 being safest)
                            aspect_ratio = wall_thickness / d if d > 0 else 9999
                            if aspect_ratio > 20:
                                # Very narrow channel
                                safety_grid[i, j] = 0.9
                            elif aspect_ratio > 10:
                                safety_grid[i, j] = 0.7
                            elif aspect_ratio > 5:
                                safety_grid[i, j] = 0.5
                            else:
                                safety_grid[i, j] = 0.3

                            # Adjust safety based on energy
                            safety_grid[i, j] *= (1.5 - e / 3) if e < 3 else 0.5
                            safety_grid[i, j] = max(0, min(1, safety_grid[i, j]))

                            break

            # 1. Top-left: Streaming intensity grid (diameter vs energy)
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(streaming_grid, cmap='hot', aspect='auto', origin='lower')
            ax1.set_xticks(range(len(energies)))
            ax1.set_yticks(range(len(diameters)))
            ax1.set_xticklabels([f"{e} MeV" for e in energies])
            ax1.set_yticklabels([f"{d} cm" for d in diameters])
            ax1.set_xlabel('Source Energy', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Channel Diameter', fontsize=11, fontweight='bold')
            ax1.set_title('Radiation Streaming Intensity (mrem/hr)', fontsize=13, fontweight='bold')
            plt.colorbar(im1, ax=ax1)

            # Add text annotations with values
            for i in range(len(diameters)):
                for j in range(len(energies)):
                    text = ax1.text(j, i, f"{streaming_grid[i, j]:.1e}",
                                    ha="center", va="center", color="black" if streaming_grid[i, j] < 100 else "white",
                                    fontsize=8, fontweight='bold')

            # 2. Top-right: Wall damage potential grid
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(damage_grid, cmap='Reds', aspect='auto', origin='lower')
            ax2.set_xticks(range(len(energies)))
            ax2.set_yticks(range(len(diameters)))
            ax2.set_xticklabels([f"{e} MeV" for e in energies])
            ax2.set_yticklabels([f"{d} cm" for d in diameters])
            ax2.set_xlabel('Source Energy', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Channel Diameter', fontsize=11, fontweight='bold')
            plt.colorbar(im2, ax=ax2)

            # Add text annotations with values
            for i in range(len(diameters)):
                for j in range(len(energies)):
                    text = ax2.text(j, i, f"{damage_grid[i, j]:.2f}",
                                    ha="center", va="center", color="black" if damage_grid[i, j] < 0.5 else "white",
                                    fontsize=8, fontweight='bold')

            # 3. Middle-left: Safety assessment grid
            ax3 = fig.add_subplot(gs[1, 0])
            im3 = ax3.imshow(safety_grid, cmap='RdYlGn', aspect='auto', origin='lower', vmin=0, vmax=1)
            ax3.set_xticks(range(len(energies)))
            ax3.set_yticks(range(len(diameters)))
            ax3.set_xticklabels([f"{e} MeV" for e in energies])
            ax3.set_yticklabels([f"{d} cm" for d in diameters])
            ax3.set_xlabel('Source Energy', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Channel Diameter', fontsize=11, fontweight='bold')
            ax3.set_title('Safety Assessment (0-1 scale)', fontsize=13, fontweight='bold')
            plt.colorbar(im3, ax=ax3)

            # Add text annotations with values
            for i in range(len(diameters)):
                for j in range(len(energies)):
                    text = ax3.text(j, i, f"{safety_grid[i, j]:.2f}",
                                    ha="center", va="center", color="white",
                                    fontsize=8, fontweight='bold')

            # 4. Middle-right: Optimal design criteria
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')

            # Find optimal configurations
            max_safety_idx = np.unravel_index(np.argmax(safety_grid), safety_grid.shape)
            opt_diameter = diameters[max_safety_idx[0]]
            opt_energy = energies[max_safety_idx[1]]
            min_streaming_idx = np.unravel_index(np.argmin(streaming_grid), streaming_grid.shape)
            low_damage_idx = np.unravel_index(np.argmin(damage_grid), damage_grid.shape)

            # Create optimal design criteria text
            title_text = "OPTIMAL DESIGN CRITERIA"

            design_text = [
                f"SAFETY RECOMMENDATIONS",
                f"--------------------------------",
                f"Optimal configuration: {opt_diameter} cm diameter, {opt_energy} MeV",
                f"Safety rating: {safety_grid[max_safety_idx]:.2f}",
                f"",
                f"STREAMING MINIMIZATION",
                f"--------------------------------",
                f"Diameter: {diameters[min_streaming_idx[0]]} cm",
                f"Energy: {energies[min_streaming_idx[1]]} MeV",
                f"Streaming: {streaming_grid[min_streaming_idx]:.1e} mrem/hr",
                f"",
                f"DAMAGE PREVENTION",
                f"--------------------------------",
                f"Diameter: {diameters[low_damage_idx[0]]} cm",
                f"Energy: {energies[low_damage_idx[1]]} MeV",
                f"Damage index: {damage_grid[low_damage_idx]:.2f}",
                f"",
                f"GENERAL GUIDANCE",
                f"--------------------------------",
                f"• Minimize channel diameter for thinner walls",
                f"• Use stepped or offset geometries for thicker walls",
                f"• For high energy sources, consider multiple small",
                f"  channels instead of one large channel",
                f"• Conduct periodic inspections every {max(1, int(10 / (np.max(damage_grid) + 0.1)))} years",
            ]

            props = dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.7, edgecolor='blue')
            ax4.text(0.5, 0.98, title_text, transform=ax4.transAxes, fontsize=14,
                     fontweight='bold', ha='center', va='top')

            ax4.text(0.05, 0.92, '\n'.join(design_text), transform=ax4.transAxes,
                     fontsize=10, va='top', bbox=props, family='monospace')

            # 5. Bottom: Streaming effect illustrations
            ax5 = fig.add_subplot(gs[2, :])

            # Example function for drawing diagrams
            def draw_diagram(ax, start_x, width, title, details):
                # Draw a mini diagram showing streaming effects
                # This is a placeholder - in a real implementation this would be more detailed
                ax.text(start_x + width / 2, 0.9, title, ha='center', va='top', fontsize=12, fontweight='bold')

                # Draw a simple wall with channel
                wall_rect = plt.Rectangle((start_x + 0.1, 0.3), width * 0.8, 0.4, color='gray', alpha=0.7)
                ax.add_patch(wall_rect)

                # Draw the channel
                channel_height = 0.1
                channel_rect = plt.Rectangle((start_x + 0.1, 0.45), width * 0.8, channel_height, color='white')
                ax.add_patch(channel_rect)

                # Add detail text
                ax.text(start_x + width / 2, 0.2, details, ha='center', va='top', fontsize=9,
                        wrap=True, bbox=dict(facecolor='white', alpha=0.8))

            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')

            # Draw three diagrams side by side
            draw_diagram(ax5, 0.0, 0.33, "Direct Transmission",
                         "Primary radiation travels straight through the channel with minimal interaction.")

            draw_diagram(ax5, 0.33, 0.33, "Wall Scatter",
                         "Radiation scatters from the channel walls, widening the beam profile.")

            draw_diagram(ax5, 0.66, 0.33, "Wall Damage",
                         "Over time, radiation damages the channel walls, potentially increasing streaming.")

            # 6. Bottom panel: Summary and recommendations
            ax6 = fig.add_subplot(gs[3, :])
            ax6.axis('off')

            # Calculate overall recommendations
            if np.max(streaming_grid) > 100:
                overall_risk = "HIGH"
                color = "red"
                recommendation = "Shield redesign recommended. Consider step-hole designs or offsets."
            elif np.max(streaming_grid) > 10:
                overall_risk = "MODERATE"
                color = "orange"
                recommendation = "Current design acceptable with periodic monitoring."
            else:
                overall_risk = "LOW"
                color = "green"
                recommendation = "Current design provides adequate protection."

            summary_text = (
                f"SUMMARY ASSESSMENT\n"
                f"--------------------------------\n"
                f"Overall Streaming Risk: {overall_risk}\n"
                f"Recommendation: {recommendation}\n"
                f"Annual inspections: {'Required' if overall_risk == 'HIGH' else ('Recommended' if overall_risk == 'MODERATE' else 'Optional')}"
            )

            props = dict(boxstyle='round,pad=1', facecolor=color, alpha=0.1, edgecolor=color)
            ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
                     fontweight='bold', ha='center', va='center', bbox=props, color=color)

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')

            # Compile analysis results
            analysis_results = {
                'diameters': diameters,
                'energies': energies,
                'streaming_grid': streaming_grid.tolist(),
                'damage_grid': damage_grid.tolist(),
                'safety_grid': safety_grid.tolist(),
                'optimal_diameter': opt_diameter,
                'optimal_energy': opt_energy,
                'overall_risk': overall_risk,
                'recommendation': recommendation
            }

            return fig, analysis_results

    def save_results_to_json(results, filename):
        """
        Save simulation and analysis results to a JSON file.

        Parameters:
            results: Dictionary of results to save
            filename: Output filename for the JSON file

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            processed_results = {}

            def process_value(value):
                if isinstance(value, (np.ndarray, list)):
                    if isinstance(value, np.ndarray):
                        return value.tolist()
                    else:
                        return [process_value(item) for item in value]
                elif isinstance(value, dict):
                    return {k: process_value(v) for k, v in value.items()}
                elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                    return int(value)
                elif isinstance(value, (np.float64, np.float32, np.float16)):
                    return float(value)
                else:
                    return value

            processed_results = process_value(results)

            # Add metadata
            processed_results['metadata'] = {
                'timestamp': datetime.datetime.now().isoformat(),
                'software_version': '1.0.0',
                'description': 'Gamma-ray streaming simulation results'
            }

            # Write to file
            with open(filename, 'w') as f:
                json.dump(processed_results, f, indent=2)

            print(f"Results successfully saved to {filename}")
            return True

        except Exception as e:
            print(f"Error saving results to JSON: {e}")
            return False

    def load_results_from_json(filename):
        """
        Load simulation and analysis results from a JSON file.

        Parameters:
            filename: Input JSON filename

        Returns:
            dict: The loaded results dictionary or None if loading failed
        """
        try:
            with open(filename, 'r') as f:
                results = json.load(f)

            print(f"Results successfully loaded from {filename}")
            return results

        except Exception as e:
            print(f"Error loading results from JSON: {e}")
            return None

