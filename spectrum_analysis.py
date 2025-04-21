import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
import json
from logging import log_info, log_error, log_warning


def load_spectrum_data(file_path):
    """
    Load energy spectrum data from a file.

    Parameters:
        file_path: Path to the spectrum data file

    Returns:
        tuple: (energies, counts) arrays or None if loading failed
    """
    try:
        # Determine file type based on extension
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            # CSV format
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            energies = data[:, 0]
            counts = data[:, 1]

        elif ext == '.txt':
            # Text format
            data = np.genfromtxt(file_path, skip_header=1)
            energies = data[:, 0]
            counts = data[:, 1]

        elif ext == '.json':
            # JSON format
            with open(file_path, 'r') as f:
                data = json.load(f)
            energies = np.array(data.get('energies', []))
            counts = np.array(data.get('counts', []))

        else:
            log_warning(f"Unsupported file extension: {ext}")
            return None

        return energies, counts

    except Exception as e:
        log_error(f"Error loading spectrum data from {file_path}: {e}")
        return None


def gaussian(x, amplitude, center, sigma):
    """
    Gaussian function for peak fitting.

    Parameters:
        x: x values
        amplitude: Peak amplitude
        center: Peak center
        sigma: Standard deviation

    Returns:
        array: Gaussian values
    """
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))


def analyze_spectrum(energies, counts, peak_threshold=0.3, output_file=None):
    """
    Analyze energy spectrum to identify peaks and calculate parameters.

    Parameters:
        energies: Array of energy values (MeV)
        counts: Array of count values
        peak_threshold: Relative threshold for peak detection
        output_file: Optional file path to save the plot

    Returns:
        dict: Analysis results including peaks and parameters
    """
    # Normalize counts
    counts_norm = counts / counts.max() if counts.max() > 0 else counts

    # Find peaks
    peak_indices, peak_properties = find_peaks(
        counts_norm,
        height=peak_threshold,
        distance=len(energies) // 20,  # Minimum distance between peaks
        prominence=0.1  # Minimum peak prominence
    )

    peak_energies = energies[peak_indices]
    peak_heights = counts[peak_indices]

    # Fit Gaussian to each peak
    peak_fits = []
    for i, peak_idx in enumerate(peak_indices):
        # Extract region around peak for fitting
        window = max(5, int(len(energies) * 0.05))  # 5% of spectrum or minimum 5 points
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(energies), peak_idx + window)

        x_data = energies[start_idx:end_idx]
        y_data = counts[start_idx:end_idx]

        try:
            # Initial guess for parameters
            p0 = [counts[peak_idx], energies[peak_idx], (energies[1] - energies[0]) * 5]

            # Fit Gaussian
            popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0)

            # Calculate FWHM (Full Width at Half Maximum)
            fwhm = 2.355 * popt[2]  # FWHM = 2.355 * sigma for Gaussian

            # Calculate resolution (FWHM/Center)
            resolution = fwhm / popt[1] if popt[1] > 0 else float('inf')

            # Calculate uncertainty (standard error) from covariance matrix
            perr = np.sqrt(np.diag(pcov))

            peak_fits.append({
                'peak_energy': popt[1],
                'peak_amplitude': popt[0],
                'sigma': popt[2],
                'fwhm': fwhm,
                'resolution': resolution,
                'uncertainty_energy': perr[1],
                'uncertainty_amplitude': perr[0],
                'uncertainty_sigma': perr[2]
            })

        except Exception as e:
            log_warning(f"Failed to fit peak at energy {energies[peak_idx]:.3f} MeV: {e}")

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot spectrum
    plt.step(energies, counts, where='mid', label='Spectrum', color='blue', linewidth=1.5)

    # Highlight peaks
    plt.plot(peak_energies, peak_heights, 'ro', label='Detected Peaks')

    # Plot Gaussian fits
    for i, fit in enumerate(peak_fits):
        x_fit = np.linspace(fit['peak_energy'] - 5 * fit['sigma'],
                            fit['peak_energy'] + 5 * fit['sigma'], 100)
        y_fit = gaussian(x_fit, fit['peak_amplitude'], fit['peak_energy'], fit['sigma'])
        plt.plot(x_fit, y_fit, '--', color='green', linewidth=1.5)

        # Add annotations
        plt.annotate(f"{fit['peak_energy']:.3f} MeV\nFWHM: {fit['fwhm']:.3f} MeV",
                     xy=(fit['peak_energy'], fit['peak_amplitude']),
                     xytext=(fit['peak_energy'] + fit['sigma'], fit['peak_amplitude'] * 1.1),
                     arrowprops=dict(arrowstyle="->", color='black'))

    plt.xlabel('Energy (MeV)', fontsize=12, fontweight='bold')
    plt.ylabel('Counts', fontsize=12, fontweight='bold')
    plt.title('Energy Spectrum Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add summary text
    summary_text = f"Total peaks detected: {len(peak_indices)}\n"
    if peak_fits:
        avg_resolution = np.mean([fit['resolution'] for fit in peak_fits])
        summary_text += f"Average resolution: {avg_resolution:.2%}\n"

    plt.figtext(0.02, 0.02, summary_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save plot if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        log_info(f"Spectrum analysis plot saved to {output_file}")

    # Calculate additional spectrum metrics
    total_counts = np.sum(counts)
    mean_energy = np.sum(energies * counts) / total_counts if total_counts > 0 else 0

    # Calculate energy calibration if possible (linear calibration)
    calibration = None
    if len(peak_fits) >= 2:
        # Sort by energy
        sorted_peaks = sorted(peak_fits, key=lambda x: x['peak_energy'])

        # Select at least 2 peaks
        calibration_peaks = sorted_peaks[:min(3, len(sorted_peaks))]

        # Extract channel numbers (indices) and energies
        indices = [np.abs(energies - peak['peak_energy']).argmin() for peak in calibration_peaks]
        peak_energies = [peak['peak_energy'] for peak in calibration_peaks]

        # Linear fit
        try:
            coef = np.polyfit(indices, peak_energies, 1)
            calibration = {
                'slope': coef[0],
                'intercept': coef[1],
                'equation': f"E = {coef[0]:.6f} * channel + {coef[1]:.6f}"
            }
        except Exception as e:
            log_warning(f"Failed to calculate energy calibration: {e}")

    # Compile analysis results
    analysis_results = {
        'spectrum_info': {
            'energy_min': energies[0],
            'energy_max': energies[-1],
            'energy_bins': len(energies),
            'energy_step': (energies[-1] - energies[0]) / (len(energies) - 1),
            'total_counts': total_counts,
            'mean_energy': mean_energy
        },
        '
        'peaks': {
            'num_peaks': len(peak_indices),
            'peak_energies': peak_energies.tolist(),
            'peak_heights': peak_heights.tolist()
        },
        'peak_fits': peak_fits,
        'calibration': calibration
    }

    return analysis_results


def compare_spectra(spectra_dict, output_file=None):
    """
    Compare multiple energy spectra.

    Parameters:
        spectra_dict: Dictionary mapping names to (energies, counts) tuples
        output_file: Optional file path to save the plot

    Returns:
        dict: Comparison results
    """
    plt.figure(figsize=(14, 10))

    # Plot each spectrum
    colors = plt.cm.tab10.colors
    comparison_results = {}
    legend_entries = []

    for i, (name, (energies, counts)) in enumerate(spectra_dict.items()):
        color = colors[i % len(colors)]

        # Normalize counts to facilitate comparison
        counts_norm = counts / counts.max() if counts.max() > 0 else counts

        plt.step(energies, counts_norm, where='mid', color=color, linewidth=1.5)
        legend_entries.append(name)

        # Calculate spectrum characteristics
        total_counts = np.sum(counts)
        mean_energy = np.sum(energies * counts) / total_counts if total_counts > 0 else 0

        # Find peaks
        peak_indices, _ = find_peaks(
            counts_norm,
            height=0.3,
            distance=len(energies) // 20,
            prominence=0.1
        )

        peak_energies = energies[peak_indices]

        # Store results
        comparison_results[name] = {
            'total_counts': total_counts,
            'mean_energy': mean_energy,
            'num_peaks': len(peak_indices),
            'peak_energies': peak_energies.tolist()
        }

    plt.xlabel('Energy (MeV)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Counts', fontsize=12, fontweight='bold')
    plt.title('Energy Spectra Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(legend_entries)

    # Save plot if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        log_info(f"Spectra comparison plot saved to {output_file}")

    return comparison_results


def analyze_streaming_effect_on_spectrum(before_spectrum, after_spectrum, output_file=None):
    """
    Analyze how radiation streaming affects the energy spectrum.

    Parameters:
        before_spectrum: (energies, counts) tuple for spectrum before streaming
        after_spectrum: (energies, counts) tuple for spectrum after streaming
        output_file: Optional file path to save the plot

    Returns:
        dict: Analysis results
    """
    before_energies, before_counts = before_spectrum
    after_energies, after_counts = after_spectrum

    # Ensure energy bins match
    if not np.array_equal(before_energies, after_energies):
        log_warning("Energy bins do not match. Interpolating after spectrum to match before spectrum.")
        from scipy.interpolate import interp1d
        after_interp = interp1d(after_energies, after_counts, bounds_error=False, fill_value=0)
        after_counts = after_interp(before_energies)
        after_energies = before_energies

    # Normalize both spectra for comparison
    before_norm = before_counts / before_counts.max() if before_counts.max() > 0 else before_counts
    after_norm = after_counts / after_counts.max() if after_counts.max() > 0 else after_counts

    # Calculate difference spectrum (shows energy-dependent streaming effect)
    diff_spectrum = after_norm - before_norm

    # Calculate streaming factor for each energy bin
    streaming_factors = np.zeros_like(before_energies)
    mask = before_counts > 0
    streaming_factors[mask] = after_counts[mask] / before_counts[mask]

    # Find energies with highest streaming factor
    max_streaming_indices = np.argsort(streaming_factors)[-5:]  # Top 5
    max_streaming_energies = before_energies[max_streaming_indices]
    max_streaming_factors = streaming_factors[max_streaming_indices]

    # Calculate overall streaming factor (ratio of total counts)
    total_before = np.sum(before_counts)
    total_after = np.sum(after_counts)
    overall_streaming_factor = total_after / total_before if total_before > 0 else float('inf')

    # Calculate average energy before and after
    avg_energy_before = np.sum(before_energies * before_counts) / total_before if total_before > 0 else 0
    avg_energy_after = np.sum(after_energies * after_counts) / total_after if total_after > 0 else 0

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot normalized spectra
    ax1.step(before_energies, before_norm, where='mid', color='blue', linewidth=1.5, label='Before Streaming')
    ax1.step(after_energies, after_norm, where='mid', color='red', linewidth=1.5, label='After Streaming')

    # Highlight highest streaming energy regions
    for energy, factor in zip(max_streaming_energies, max_streaming_factors):
        idx = np.abs(before_energies - energy).argmin()
        if factor > 1.5:  # Only mark significant differences
            ax1.axvspan(energy - 0.05, energy + 0.05, color='yellow', alpha=0.3)
            ax1.annotate(f"{energy:.2f} MeV\n({factor:.1f}x)",
                         xy=(energy, after_norm[idx]),
                         xytext=(energy, after_norm[idx] * 1.1),
                         arrowprops=dict(arrowstyle="->", color='black'),
                         ha='center')

    ax1.set_ylabel('Normalized Counts', fontsize=12, fontweight='bold')
    ax1.set_title('Spectra Before and After Streaming', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot energy-dependent streaming factor
    ax2.semilogy(before_energies, streaming_factors, color='green', linewidth=1.5)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=overall_streaming_factor, color='red', linestyle='-', linewidth=1,
                label=f'Overall: {overall_streaming_factor:.2f}x')

    ax2.set_xlabel('Energy (MeV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Streaming Factor', fontsize=12, fontweight='bold')
    ax2.set_title('Energy-Dependent Streaming Factor', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add summary text
    summary_text = (
        f"Overall streaming factor: {overall_streaming_factor:.2f}x\n"
        f"Average energy before: {avg_energy_before:.3f} MeV\n"
        f"Average energy after: {avg_energy_after:.3f} MeV\n"
        f"Energy shift: {(avg_energy_after - avg_energy_before):.3f} MeV"
    )

    plt.figtext(0.02, 0.02, summary_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()

    # Save plot if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        log_info(f"Streaming effect analysis plot saved to {output_file}")

    # Compile analysis results
    analysis_results = {
        'overall_streaming_factor': overall_streaming_factor,
        'avg_energy_before': avg_energy_before,
        'avg_energy_after': avg_energy_after,
        'energy_shift': avg_energy_after - avg_energy_before,
        'max_streaming_energies': max_streaming_energies.tolist(),
        'max_streaming_factors': max_streaming_factors.tolist(),
        'energy_dependent_streaming': {
            'energies': before_energies.tolist(),
            'factors': streaming_factors.tolist()
        }
    }

    return analysis_results
