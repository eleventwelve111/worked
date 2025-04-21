import numpy as np
import openmc


def get_mass_energy_absorption_coefficient(energy, material="tissue"):
    """
    Get mass energy absorption coefficient (μen/ρ) for a material at given energy.

    Parameters:
        energy (float): Photon energy in MeV
        material (str): Material name (tissue, air, water)

    Returns:
        float: Mass energy absorption coefficient in cm²/g
    """
    # Dictionary of energy (MeV) vs. μen/ρ (cm²/g) for various materials
    # Data from NIST XCOM database
    # https://physics.nist.gov/PhysRefData/XrayMassCoef/tab4.html

    # Energy points in MeV
    energy_points = np.array([
        0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
        0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5,
        2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0
    ])

    # Soft tissue coefficients (cm²/g)
    tissue_coeffs = np.array([
        4.62E+00, 1.33E+00, 5.41E-01, 1.56E-01, 7.80E-02, 5.10E-02, 4.09E-02, 3.42E-02, 3.22E-02,
        3.05E-02, 2.92E-02, 2.68E-02, 2.46E-02, 2.27E-02, 2.11E-02, 1.84E-02, 1.64E-02, 1.29E-02,
        1.06E-02, 7.92E-03, 6.53E-03, 5.68E-03, 5.13E-03, 4.49E-03, 4.17E-03, 3.88E-03, 3.86E-03
    ])

    # Air coefficients (cm²/g)
    air_coeffs = np.array([
        4.61E+00, 1.27E+00, 5.32E-01, 1.48E-01, 6.83E-02, 4.25E-02, 3.26E-02, 2.69E-02, 2.60E-02,
        2.72E-02, 2.80E-02, 2.79E-02, 2.68E-02, 2.55E-02, 2.42E-02, 2.17E-02, 1.97E-02, 1.59E-02,
        1.33E-02, 1.02E-02, 8.48E-03, 7.40E-03, 6.68E-03, 5.83E-03, 5.39E-03, 4.97E-03, 4.88E-03
    ])

    # Water coefficients (cm²/g)
    water_coeffs = np.array([
        4.87E+00, 1.34E+00, 5.48E-01, 1.54E-01, 7.63E-02, 4.94E-02, 3.91E-02, 3.31E-02, 3.21E-02,
        3.18E-02, 3.10E-02, 2.90E-02, 2.69E-02, 2.49E-02, 2.32E-02, 2.03E-02, 1.81E-02, 1.42E-02,
        1.17E-02, 8.76E-03, 7.22E-03, 6.28E-03, 5.67E-03, 4.95E-03, 4.58E-03, 4.25E-03, 4.21E-03
    ])

    # Select the appropriate coefficients based on material
    if material.lower() == "tissue":
        coeffs = tissue_coeffs
    elif material.lower() == "air":
        coeffs = air_coeffs
    elif material.lower() == "water":
        coeffs = water_coeffs
    else:
        raise ValueError(f"Mass energy absorption coefficients not available for material: {material}")

    # Interpolate to get coefficient at the requested energy
    mu_en_rho = np.interp(energy, energy_points, coeffs)

    return mu_en_rho


def calculate_kerma(energy, flux, material="tissue"):
    """
    Calculate KERMA (Kinetic Energy Released per unit MAss)

    Parameters:
        energy (float): Gamma-ray energy in MeV
        flux (float): Particle flux in photons/cm²-s
        material (str): Material name (tissue, air, water)

    Returns:
        float: KERMA rate in Gy/s (J/kg-s)
    """
    # Get mass energy absorption coefficient (cm²/g)
    mu_en_rho = get_mass_energy_absorption_coefficient(energy, material)

    # Convert photon energy from MeV to joules
    energy_joules = energy * 1.602e-13  # 1 MeV = 1.602e-13 J

    # Calculate KERMA (Gy/s = J/kg-s)
    # K = Φ * E * (μen/ρ) * conversion factors
    kerma = flux * energy_joules * mu_en_rho * 0.1  # 0.1 converts from (J/g-s) to (J/kg-s)

    return kerma


def kerma_to_equivalent_dose(kerma, radiation_weighting_factor=1.0):
    """
    Convert KERMA to equivalent dose

    Parameters:
        kerma (float): KERMA in Gy/s
        radiation_weighting_factor (float): Radiation weighting factor (1.0 for photons)

    Returns:
        float: Equivalent dose rate in Sv/s
    """
    # For photons, the radiation weighting factor is 1.0
    # For other radiation types, different factors would apply
    return kerma * radiation_weighting_factor


def convert_heating_to_dose_rate(heating, density, energy=None, flux=None):
    """
    Convert heating rate (energy deposition) to dose rate

    Parameters:
        heating (float): Heating rate in W/cm³
        density (float): Material density in g/cm³
        energy (float, optional): Photon energy in MeV (for spectrum correction)
        flux (float, optional): Particle flux in photons/cm²-s (for spectrum correction)

    Returns:
        tuple: (dose_rate_gy_per_s, dose_rate_sv_per_s, dose_rate_rem_per_hr)
    """
    # Convert W/cm³ to Gy/s (J/kg-s)
    # 1 W/cm³ = 1 J/s/cm³
    # Divide by density (g/cm³) to get J/s/g
    # Multiply by 1000 to convert J/s/g to J/s/kg (Gy/s)
    dose_rate_gy_per_s = heating / density * 1000

    # Apply spectrum correction if energy and flux are provided
    if energy is not None and flux is not None:
        # Calculate expected KERMA
        expected_kerma = calculate_kerma(energy, flux)

        # Calculate correction factor
        if expected_kerma > 0:
            correction = expected_kerma / dose_rate_gy_per_s
            dose_rate_gy_per_s *= correction

    # Convert to Sv/s (for photons, 1 Gy = 1 Sv)
    dose_rate_sv_per_s = dose_rate_gy_per_s

    # Convert to rem/hr
    # 1 Sv = 100 rem, 1 hr = 3600 s
    dose_rate_rem_per_hr = dose_rate_sv_per_s * 100 * 3600

    return (dose_rate_gy_per_s, dose_rate_sv_per_s, dose_rate_rem_per_hr)


def estimate_dose_from_heating_tally(heating_result, material="tissue", energy=None):
    """
    Estimate dose from a heating tally (e.g., from MCNP F6 or F7 tally)

    Parameters:
        heating_result (float): Heating tally result in MeV/g per source particle
        material (str): Material name
        energy (float, optional): Source photon energy in MeV

    Returns:
        float: Dose rate in rem/hr per source particle
    """
    # Material densities in g/cm³
    material_densities = {
        "tissue": 1.04,
        "water": 1.0,
        "air": 0.001205,
        "concrete": 2.3,
        "lead": 11.35,
        "iron": 7.87
    }

    # Get density for the material
    if material.lower() in material_densities:
        density = material_densities[material.lower()]
    else:
        raise ValueError(f"Material '{material}' density not defined")

    # Convert MeV/g to J/kg (Gy)
    # 1 MeV = 1.602e-13 J
    dose_gy = heating_result * 1.602e-13 * 1000  # *1000 to convert from J/g to J/kg

    # For photons and electrons, 1 Gy = 1 Sv
    dose_sv = dose_gy

    # Convert to rem
    # 1 Sv = 100 rem
    dose_rem = dose_sv * 100

    # Return dose in rem per source particle
    return dose_rem


def estimate_dose_from_flux(energy, flux):
    """
    Estimate dose using flux-to-dose conversion factors from ANS-6.1.1-1977

    Parameters:
        energy (float): Photon energy in MeV
        flux (float): Particle flux in photons/cm²-s

    Returns:
        float: Dose rate in rem/hr
    """
    # Get ANS-6.1.1-1977 flux-to-dose conversion factors
    from materials import get_dose_conversion_factors

    # Get the conversion factor for this energy
    conversion_factor = get_dose_conversion_factors(np.array([energy]))[0]

    # Calculate dose rate (rem/hr)
    dose_rate = flux * conversion_factor

    return dose_rate


def determine_most_accurate_dose_method(results, energy_kev):
    """
    Determine which dose calculation method is most accurate for the given scenario.

    Parameters:
        results (dict): Simulation results dictionary
        energy_kev (float): Source energy in keV

    Returns:
        tuple: (most_accurate_method, comparison_results)
    """
    energy_mev = energy_kev / 1000.0

    # Extract results for the reference position (30 cm, 0 degrees)
    if '30' in results['dose_data'] and '0' in results['dose_data']['30']:
        ref_position_data = results['dose_data']['30']['0']
    else:
        print("Reference position data not available")
        return None

    # Get different dose estimates
    methods = {}

    # 1. Heating tally
    if 'dose' in ref_position_data:
        methods['heating'] = ref_position_data['dose']

    # 2. Kerma tally
    if 'kerma' in ref_position_data:
        methods['kerma'] = ref_position_data['kerma']

    # 3. Flux-to-dose equivalent
    if 'dose_equiv' in ref_position_data:
        methods['flux_to_dose'] = ref_position_data['dose_equiv']

    # 4. Get flux spectrum if available, for theoretical calculation
    theoretical_dose = None
    if '30' in results['dose_data'] and 'spectrum' in results['dose_data']['30']:
        spectrum = results['dose_data']['30']['spectrum']
        energies = np.array(spectrum['energies'])
        flux = np.array(spectrum['flux'])

        # Integrate over spectrum to get total dose
        total_dose = 0
        for i in range(len(energies)):
            # Skip if flux is too low
            if flux[i] < 1e-10:
                continue

            # Calculate dose contribution using KERMA approach
            kerma = calculate_kerma(energies[i], flux[i], "tissue")
            dose_rate_sv_per_s = kerma_to_equivalent_dose(kerma)
            dose_rate_rem_per_hr = dose_rate_sv_per_s * 100 * 3600  # 100 rem/Sv, 3600 s/hr

            total_dose += dose_rate_rem_per_hr

        theoretical_dose = total_dose
        methods['theoretical'] = theoretical_dose

    # Compare methods to determine the most accurate
    # For photons, KERMA-based calculation is generally considered more accurate
    # as it accounts for energy deposition

    # Calculate relative differences
    comparisons = {}
    reference_method = 'kerma' if 'kerma' in methods else 'heating'
    reference_value = methods[reference_method]

    for method, value in methods.items():
        if method != reference_method:
            rel_diff = abs(value - reference_value) / reference_value
            comparisons[method] = {
                'value': value,
                'relative_diff': rel_diff
            }

    # Select the method with lowest relative difference to KERMA as most accurate
    # If KERMA is available, it's considered the reference
    most_accurate = reference_method
    min_diff = float('inf')

    for method, comp in comparisons.items():
        if method != reference_method and comp['relative_diff'] < min_diff:
            min_diff = comp['relative_diff']
            most_accurate = method

    # Add comparison to reference
    comparison_results = {
        'reference_method': reference_method,
        'reference_value': reference_value,
        'comparisons': comparisons,
        'most_accurate': most_accurate,
        'energy_mev': energy_mev
    }

    return most_accurate, comparison_results


# Add these functions to the existing dosimetry.py file

def compare_with_experimental_data(calculated_doses, experimental_data):
    """
    Compare calculated doses with experimental measurements.

    Parameters:
        calculated_doses (dict): Dictionary with calculated doses using different methods
        experimental_data (dict): Dictionary with experimental measurements
            Format: {
                'distance': [distances in cm],
                'angle': [angles in degrees],
                'dose_rate': [dose rates in rem/hr],
                'uncertainty': [uncertainties in percent]
            }

    Returns:
        dict: Comparison results with metrics for each method
    """
    if not experimental_data or 'dose_rate' not in experimental_data:
        return {"error": "No experimental data provided"}

    # Initialize comparison metrics for each method
    comparison = {}
    for method, dose_values in calculated_doses.items():
        comparison[method] = {
            'relative_error': [],
            'chi_square': 0,
            'mape': 0,  # Mean Absolute Percentage Error
        }

    # Calculate comparison metrics
    total_points = len(experimental_data['dose_rate'])
    for i in range(total_points):
        exp_dose = experimental_data['dose_rate'][i]
        exp_uncertainty = experimental_data['uncertainty'][i] if 'uncertainty' in experimental_data else 0.1 * exp_dose
        dist = experimental_data['distance'][i]
        angle = experimental_data['angle'][i]

        # Location key for accessing calculated data
        location_key = f"d{dist}_a{angle}"

        for method, dose_values in calculated_doses.items():
            if location_key in dose_values:
                calc_dose = dose_values[location_key]

                # Relative error: (calculated - experimental) / experimental
                rel_error = (calc_dose - exp_dose) / exp_dose
                comparison[method]['relative_error'].append(rel_error)

                # Chi-square contribution: ((calc - exp) / uncertainty)^2
                chi_sq_contrib = ((calc_dose - exp_dose) / exp_uncertainty) ** 2
                comparison[method]['chi_square'] += chi_sq_contrib

                # Mean Absolute Percentage Error contribution
                mape_contrib = abs(rel_error)
                comparison[method]['mape'] += mape_contrib / total_points

    # Calculate average and standard deviation of relative errors
    for method in comparison:
        if comparison[method]['relative_error']:
            errors = np.array(comparison[method]['relative_error'])
            comparison[method]['mean_rel_error'] = np.mean(errors)
            comparison[method]['std_rel_error'] = np.std(errors)

    return comparison


def determine_most_accurate_dose_method_with_experiments(calculated_doses, experimental_data=None):
    """
    Determine which dose calculation method is most accurate, considering experimental data if available.

    Parameters:
        calculated_doses (dict): Dictionary with calculated doses using different methods
            Format: {'method_name': {'d{dist}_a{angle}': dose_value, ...}, ...}
        experimental_data (dict, optional): Dictionary with experimental measurements

    Returns:
        tuple: (most_accurate_method, comparison_results)
    """
    # First, check if we have experimental data for comparison
    if experimental_data and 'dose_rate' in experimental_data and len(experimental_data['dose_rate']) > 0:
        # Compare with experimental data
        comparisons = compare_with_experimental_data(calculated_doses, experimental_data)

        # Determine most accurate method based on lowest MAPE
        min_mape = float('inf')
        most_accurate = None

        for method, metrics in comparisons.items():
            if 'mape' in metrics and metrics['mape'] < min_mape:
                min_mape = metrics['mape']
                most_accurate = method

        return most_accurate, comparisons

    # If no experimental data, fall back to theoretical comparison
    # For gamma radiation, KERMA-based calculation is generally most accurate
    if 'kerma' in calculated_doses:
        reference_method = 'kerma'
    elif 'heating' in calculated_doses:
        reference_method = 'heating'
    else:
        reference_method = next(iter(calculated_doses))

    # Use reference dose to compare other methods
    ref_doses = calculated_doses[reference_method]

    # Calculate relative differences for each location and method
    comparisons = {method: {'relative_diff': []} for method in calculated_doses}

    # Compare at each location
    for location in ref_doses:
        ref_value = ref_doses[location]

        # Skip if reference dose is too small
        if ref_value < 1e-10:
            continue

        for method, doses in calculated_doses.items():
            if method != reference_method and location in doses:
                rel_diff = abs(doses[location] - ref_value) / ref_value
                comparisons[method]['relative_diff'].append(rel_diff)

    # Calculate average relative difference for each method
    for method in comparisons:
        if method != reference_method and comparisons[method]['relative_diff']:
            comparisons[method]['avg_rel_diff'] = np.mean(comparisons[method]['relative_diff'])
        elif method == reference_method:
            comparisons[method]['avg_rel_diff'] = 0.0  # Reference method has zero difference from itself

    # Find method with minimum average relative difference (excluding reference)
    min_diff = float('inf')
    most_accurate = reference_method

    for method, metrics in comparisons.items():
        if method != reference_method and 'avg_rel_diff' in metrics and metrics['avg_rel_diff'] < min_diff:
            min_diff = metrics['avg_rel_diff']
            most_accurate = method

    # If no other method is better, use the reference method
    if most_accurate == reference_method or min_diff > 0.2:  # If difference is >20%, prefer reference
        most_accurate = reference_method

    return most_accurate, {
        'reference_method': reference_method,
        'comparisons': comparisons,
        'most_accurate': most_accurate
    }


def analyze_dose_accuracy(results, energy_kev, experimental_data=None):
    """
    Analyze the accuracy of different dose calculation methods.

    Parameters:
        results (dict): Simulation results dictionary
        energy_kev (float): Source energy in keV
        experimental_data (dict, optional): Dictionary with experimental measurements

    Returns:
        dict: Analysis results of dose accuracy
    """
    energy_mev = energy_kev / 1000.0

    # Extract calculated doses using different methods
    calculated_doses = {
        'heating': {},
        'kerma': {},
        'flux_to_dose': {}
    }

    # Extract doses for each position
    for distance, angle_data in results['dose_data'].items():
        for angle, dose_info in angle_data.items():
            if angle == 'spectrum':
                continue

            location_key = f"d{distance}_a{angle}"

            # Extract the different dose calculations
            if 'dose' in dose_info:
                calculated_doses['heating'][location_key] = dose_info['dose']

            if 'kerma' in dose_info:
                calculated_doses['kerma'][location_key] = dose_info['kerma']

            if 'dose_equiv' in dose_info:
                calculated_doses['flux_to_dose'][location_key] = dose_info['dose_equiv']

    # Calculate theoretical doses from scratch using spectrum if available
    if 'spectrum' in results['dose_data'].get('30', {}):
        calculated_doses['theoretical'] = {}

        # Loop through all positions
        for distance, angle_data in results['dose_data'].items():
            if 'spectrum' in angle_data:
                spectrum = angle_data['spectrum']
                energies = np.array(spectrum['energies'])
                flux = np.array(spectrum['flux'])

                # Integrate over spectrum to get total dose
                total_dose = 0
                for i in range(len(energies)):
                    if flux[i] < 1e-10:
                        continue

                    kerma = calculate_kerma(energies[i], flux[i], "tissue")
                    dose_rate_sv_per_s = kerma_to_equivalent_dose(kerma)
                    dose_rate_rem_per_hr = dose_rate_sv_per_s * 100 * 3600

                    total_dose += dose_rate_rem_per_hr

                # Store the theoretical dose
                calculated_doses['theoretical'][f"d{distance}_a0"] = total_dose

    # Determine most accurate method
    most_accurate, comparison_results = determine_most_accurate_dose_method_with_experiments(
        calculated_doses, experimental_data
    )

    # Format results
    analysis = {
        'energy_kev': energy_kev,
        'energy_mev': energy_mev,
        'calculated_doses': calculated_doses,
        'most_accurate_method': most_accurate,
        'comparison_results': comparison_results
    }

    # Add experimental data if provided
    if experimental_data:
        analysis['experimental_data'] = experimental_data

    return analysis


def analyze_dose_components(results, energy_kev, channel_diameter):
    """
    Analyze the relative importance of different dose components (primary, scattered, etc.)

    Args:
        results: Simulation results dictionary
        energy_kev: Energy in keV
        channel_diameter: Channel diameter in cm

    Returns:
        dict: Analysis of dose components
    """
    # Analyze spectral components if available
    spectrum_analysis = {}

    dose_data = results['dose_data']
    for distance, angle_data in dose_data.items():
        if 'spectrum' in angle_data:
            spectrum = angle_data['spectrum']
            energies = np.array(spectrum['energies'])
            flux = np.array(spectrum['flux'])

            if len(energies) > 0 and len(flux) > 0:
                # Determine primary vs scattered components
                # Primary photons at or near the source energy, scattered at lower energies
                source_energy_mev = energy_kev / 1000.0

                # Define energy windows
                energy_window = 0.05 * source_energy_mev  # 5% energy window

                # Primary component: within +/- energy_window of source energy
                primary_mask = np.abs(energies - source_energy_mev) <= energy_window

                # Scattered component: lower than (source_energy - energy_window)
                scattered_mask = energies < (source_energy_mev - energy_window)

                # Secondary radiation: higher than (source_energy + energy_window)
                secondary_mask = energies > (source_energy_mev + energy_window)

                # Calculate flux components
                primary_flux = np.sum(flux[primary_mask]) if any(primary_mask) else 0
                scattered_flux = np.sum(flux[scattered_mask]) if any(scattered_mask) else 0
                secondary_flux = np.sum(flux[secondary_mask]) if any(secondary_mask) else 0
                total_flux = np.sum(flux)

                # Calculate dose components
                from dosimetry import calculate_kerma, kerma_to_equivalent_dose

                primary_dose = 0
                scattered_dose = 0
                secondary_dose = 0
                total_dose = 0

                for i in range(len(energies)):
                    if flux[i] < 1e-10:
                        continue

                    # Convert flux to dose
                    kerma = calculate_kerma(energies[i], flux[i], "tissue")
                    dose_sv_per_s = kerma_to_equivalent_dose(kerma)
                    dose_rem_per_hr = dose_sv_per_s * 100 * 3600

                    # Accumulate by component
                    if primary_mask[i]:
                        primary_dose += dose_rem_per_hr
                    elif scattered_mask[i]:
                        scattered_dose += dose_rem_per_hr
                    elif secondary_mask[i]:
                        secondary_dose += dose_rem_per_hr

                    total_dose += dose_rem_per_hr

                # Store analysis
                spectrum_analysis[distance] = {
                    'flux_components': {
                        'primary': primary_flux,
                        'scattered': scattered_flux,
                        'secondary': secondary_flux,
                        'total': total_flux
                    },
                    'flux_fractions': {
                        'primary': primary_flux / total_flux if total_flux > 0 else 0,
                        'scattered': scattered_flux / total_flux if total_flux > 0 else 0,
                        'secondary': secondary_flux / total_flux if total_flux > 0 else 0
                    },
                    'dose_components': {
                        'primary': primary_dose,
                        'scattered': scattered_dose,
                        'secondary': secondary_dose,
                        'total': total_dose
                    },
                    'dose_fractions': {
                        'primary': primary_dose / total_dose if total_dose > 0 else 0,
                        'scattered': scattered_dose / total_dose if total_dose > 0 else 0,
                        'secondary': secondary_dose / total_dose if total_dose > 0 else 0
                    }
                }

    # Analyze trends with distance
    if len(spectrum_analysis) >= 2:
        # Extract distances and dose components
        distances = sorted([int(d) for d in spectrum_analysis.keys()])
        primary_fractions = [spectrum_analysis[str(d)]['dose_fractions']['primary'] for d in distances]
        scattered_fractions = [spectrum_analysis[str(d)]['dose_fractions']['scattered'] for d in distances]

        # Calculate rate of change with distance
        primary_trend = np.polyfit(distances, primary_fractions, 1)[0]  # Slope of best-fit line
        scattered_trend = np.polyfit(distances, scattered_fractions, 1)[0]

        # Determine dominant component at different distances
        dominant_components = {}
        for d in distances:
            components = spectrum_analysis[str(d)]['dose_fractions']
            dominant = max(components.items(), key=lambda x: x[1])
            dominant_components[d] = dominant[0]
    else:
        primary_trend = scattered_trend = None
        dominant_components = {}

    # Overall analysis
    analysis = {
        'energy_kev': energy_kev,
        'channel_diameter': channel_diameter,
        'spectrum_analysis': spectrum_analysis,
        'distance_trends': {
            'primary_trend': primary_trend,
            'scattered_trend': scattered_trend,
            'dominant_components': dominant_components
        }
    }

    return analysis

