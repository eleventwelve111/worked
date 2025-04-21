numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
openmc>=0.13.0
pytest>=6.2.5

"""
Requirements module for gamma-ray streaming simulation.
Contains functions to verify system requirements, check dependencies,
and validate inputs.
"""

import sys
import importlib
import logging
import os
import platform
import subprocess
from logging import log_info, log_error, log_warning


def check_python_version(min_version=(3, 6)):
    """
    Check if the Python version meets the minimum requirement.

    Parameters:
        min_version: Tuple with minimum required Python version

    Returns:
        bool: True if version requirement is met, False otherwise
    """
    current_version = sys.version_info[:2]

    if current_version >= min_version:
        log_info(f"Python version check passed: {platform.python_version()}")
        return True
    else:
        log_error(f"Python version check failed: {platform.python_version()} < {'.'.join(map(str, min_version))}")
        return False


def check_required_packages(required_packages):
    """
    Check if all required packages are installed.

    Parameters:
        required_packages: Dictionary mapping package names to minimum versions

    Returns:
        tuple: (bool indicating if all requirements are met, list of missing packages)
    """
    missing_packages = []

    for package, min_version in required_packages.items():
        try:
            # Try to import the package
            module = importlib.import_module(package)

            # Check version if provided
            if min_version:
                try:
                    # Get version from module
                    version = getattr(module, '__version__', None)

                    # If version not found in module, try pkg_resources
                    if version is None:
                        import pkg_resources
                        version = pkg_resources.get_distribution(package).version

                    # Split version string and compare
                    current_version = tuple(map(int, version.split('.')[:3]))
                    required_version = tuple(map(int, min_version.split('.')[:3]))

                    if current_version < required_version:
                        log_warning(f"Package version check failed: {package} {version} < {min_version}")
                        missing_packages.append(f"{package}>={min_version}")
                    else:
                        log_info(f"Package version check passed: {package} {version}")

                except (AttributeError, ImportError, pkg_resources.DistributionNotFound):
                    # If version checking fails, assume it's OK but log a warning
                    log_warning(f"Could not determine version for {package}")
            else:
                log_info(f"Package check passed: {package}")

        except ImportError:
            log_warning(f"Package check failed: {package} not found")
            missing_packages.append(package if not min_version else f"{package}>={min_version}")

    return len(missing_packages) == 0, missing_packages


def install_missing_packages(packages):
    """
    Install missing packages using pip.

    Parameters:
        packages: List of packages to install

    Returns:
        bool: True if all packages were installed successfully, False otherwise
    """
    if not packages:
        return True

    log_info(f"Attempting to install missing packages: {', '.join(packages)}")

    # Try installing with conda first
    try:
        # Check if conda is available
        subprocess.run(['conda', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Use conda to install packages
        log_info("Using conda to install packages")
        for package in packages:
            try:
                # Parse package name and version
                if '>=' in package:
                    pkg_name, pkg_version = package.split('>=')
                    install_cmd = ['conda', 'install', '-y', f"{pkg_name}={pkg_version}"]
                else:
                    install_cmd = ['conda', 'install', '-y', package]

                # Run conda install command
                result = subprocess.run(
                    install_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                log_info(f"Successfully installed {package} with conda")
            except subprocess.CalledProcessError as e:
                log_warning(f"Failed to install {package} with conda: {e}")
                # If conda fails, we'll try pip next
                return install_with_pip(packages)

        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        log_warning("Conda not available, falling back to pip")
        return install_with_pip(packages)


def install_with_pip(packages):
    """
    Install missing packages using pip.

    Parameters:
        packages: List of packages to install

    Returns:
        bool: True if all packages were installed successfully, False otherwise
    """
    try:
        # Use pip to install packages
        pip_cmd = [sys.executable, '-m', 'pip', 'install'] + packages
        result = subprocess.run(
            pip_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        log_info(f"Successfully installed packages with pip: {', '.join(packages)}")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install packages with pip: {e}")
        return False


def check_memory_requirements(min_memory_gb=4):
    """
    Check if the system has enough memory.

    Parameters:
        min_memory_gb: Minimum required memory in GB

    Returns:
        bool: True if memory requirement is met, False otherwise
    """
    try:
        # Check platform to determine how to get memory information
        if platform.system() == 'Windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]

            memory_status = MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))

            total_memory_gb = memory_status.ullTotalPhys / (1024 ** 3)

        elif platform.system() == 'Linux':
            # On Linux, read from /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                mem_info = f.read()

            # Extract total memory
            for line in mem_info.splitlines():
                if 'MemTotal' in line:
                    # MemTotal is in kB
                    total_memory_kb = int(line.split()[1])
                    total_memory_gb = total_memory_kb / (1024 ** 2)
                    break

        elif platform.system() == 'Darwin':  # macOS
            # Use sysctrl to get memory info
            result = subprocess.run(
                ['sysctl', 'hw.memsize'],
                check=True,
                stdout=subprocess.PIPE,
                text=True
            )
            total_memory_bytes = int(result.stdout.split()[1])
            total_memory_gb = total_memory_bytes / (1024 ** 3)

        else:
            # Unknown platform, assume requirement is met but log warning
            log_warning(f"Unknown platform {platform.system()}, cannot check memory requirements")
            return True

        # Check if memory meets minimum requirement
        if total_memory_gb >= min_memory_gb:
            log_info(f"Memory check passed: {total_memory_gb:.2f} GB available (>= {min_memory_gb} GB required)")
            return True
        else:
            log_warning(f"Memory check failed: {total_memory_gb:.2f} GB available (< {min_memory_gb} GB required)")
            return False

    except Exception as e:
        log_warning(f"Failed to check memory requirements: {e}")
        # In case of error, assume requirement is met but log warning
        return True


def check_disk_space(min_space_gb=1, output_dir='.'):
    """
    Check if there is enough disk space in the output directory.

    Parameters:
        min_space_gb: Minimum required disk space in GB
        output_dir: Directory where output will be written

    Returns:
        bool: True if disk space requirement is met, False otherwise
    """
    try:
        # Make sure output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Check available disk space
        if platform.system() == 'Windows':
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(output_dir),
                None,
                None,
                ctypes.pointer(free_bytes)
            )
            free_space_gb = free_bytes.value / (1024 ** 3)
        else:
            # Unix-based systems
            stat = os.statvfs(output_dir)
            free_space_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)

        # Check if disk space meets minimum requirement
        if free_space_gb >= min_space_gb:
            log_info(
                f"Disk space check passed: {free_space_gb:.2f} GB available in {output_dir} (>= {min_space_gb} GB required)")
            return True
        else:
            log_warning(
                f"Disk space check failed: {free_space_gb:.2f} GB available in {output_dir} (< {min_space_gb} GB required)")
            return False

    except Exception as e:
        log_warning(f"Failed to check disk space requirements: {e}")
        # In case of error, assume requirement is met but log warning
        return True


def check_system_requirements(config):
    """
    Check if the system meets all requirements for the simulation.

    Parameters:
        config: Configuration dictionary containing requirements

    Returns:
        bool: True if all requirements are met, False otherwise
    """
    requirements_met = True

    # Get requirements from config
    min_python_version = tuple(config.get('requirements', {}).get('python_version', (3, 6)))
    min_memory_gb = config.get('requirements', {}).get('min_memory_gb', 4)
    min_disk_space_gb = config.get('requirements', {}).get('min_disk_space_gb', 1)
    output_dir = config.get('output_dir', '.')

    required_packages = config.get('requirements', {}).get('packages', {
        'numpy': '1.16.0',
        'matplotlib': '3.0.0',
        'scipy': '1.2.0'
    })

    # Check Python version
    python_ok = check_python_version(min_python_version)
    requirements_met = requirements_met and python_ok

    # Check required packages
    packages_ok, missing_packages = check_required_packages(required_packages)
    if not packages_ok:
        # Try to install missing packages
        if config.get('auto_install_packages', False):
            packages_ok = install_missing_packages(missing_packages)
        requirements_met = requirements_met and packages_ok

    # Check memory
    memory_ok = check_memory_requirements(min_memory_gb)
    requirements_met = requirements_met and memory_ok

    # Check disk space
    disk_ok = check_disk_space(min_disk_space_gb, output_dir)
    requirements_met = requirements_met and disk_ok

    # Log overall result
    if requirements_met:
        log_info("All system requirements are met.")
    else:
        log_error("Some system requirements are not met.")

    return requirements_met


def validate_config(config):
    """
    Validate configuration for correctness.

    Parameters:
        config: Configuration dictionary

    Returns:
        tuple: (bool indicating if config is valid, list of validation errors)
    """
    errors = []

    # Check required configuration sections
    required_sections = ['simulation', 'geometry', 'materials', 'source', 'tally']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")

    # Check simulation parameters
    if 'simulation' in config:
        sim_config = config['simulation']

        # Check required simulation parameters
        required_sim_params = ['num_particles', 'max_energy']
        for param in required_sim_params:
            if param not in sim_config:
                errors.append(f"Missing required simulation parameter: {param}")

        # Check numeric parameters are positive
        for param in ['num_particles', 'max_energy', 'cutoff_energy']:
            if param in sim_config and not (isinstance(sim_config[param], (int, float)) and sim_config[param] > 0):
                errors.append(f"Invalid value for {param}: must be a positive number")

    # Check geometry parameters
    if 'geometry' in config:
        geo_config = config['geometry']

        # Check required geometry parameters
        required_geo_params = ['wall_thickness', 'channel_diameter']
        for param in required_geo_params:
            if param not in geo_config:
                errors.append(f"Missing required geometry parameter: {param}")

        # Check numeric parameters are positive
        for param in ['wall_thickness', 'channel_diameter']:
            if param in geo_config and not (isinstance(geo_config[param], (int, float)) and geo_config[param] > 0):
                errors.append(f"Invalid value for {param}: must be a positive number")

        # Check channel shape is valid
        if 'channel_shape' in geo_config and geo_config['channel_shape'] not in ['straight', 'stepped', 'curved']:
            errors.append(f"Invalid channel shape: {geo_config['channel_shape']}")

    # Check material parameters
    if 'materials' in config:
        mat_config = config['materials']

        # Check at least one material is defined
        if not mat_config:
            errors.append("No materials defined in configuration")

        # Check each material has required properties
        for name, material in mat_config.items():
            if not isinstance(material, dict):
                errors.append(f"Invalid material definition for {name}")
                continue

            required_mat_params = ['density', 'composition']
            for param in required_mat_params:
                if param not in material:
                    errors.append(f"Missing required material parameter for {name}: {param}")

            # Check density is positive
            if 'density' in material and not (
                    isinstance(material['density'], (int, float)) and material['density'] > 0):
                errors.append(f"Invalid density for {name}: must be a positive number")

            # Check composition format
            if 'composition' in material:
                comp = material['composition']
                if not isinstance(comp, dict):
                    errors.append(f"Invalid composition format for {name}")
                else:
                    # Check each element has a valid weight or atomic fraction
                    for element, fraction in comp.items():
                        if not (isinstance(fraction, (int, float)) and 0 <= fraction <= 1):
                            errors.append(f"Invalid fraction for element {element} in {name}: must be between 0 and 1")

    # Check source parameters
    if 'source' in config:
        src_config = config['source']

        # Check required source parameters
        required_src_params = ['type', 'energy']
        for param in required_src_params:
            if param not in src_config:
                errors.append(f"Missing required source parameter: {param}")

        # Check source type is valid
        if 'type' in src_config and src_config['type'] not in ['point', 'disc', 'beam', 'volume']:
            errors.append(f"Invalid source type: {src_config['type']}")

        # Check energy specification
        if 'energy' in src_config:
            energy = src_config['energy']
            if isinstance(energy, (int, float)):
                # Single energy value
                if energy <= 0:
                    errors.append(f"Invalid energy value: must be positive")
            elif isinstance(energy, dict):
                # Energy spectrum
                if 'distribution' not in energy:
                    errors.append("Missing 'distribution' in energy spectrum")
                elif energy['distribution'] not in ['mono', 'uniform', 'gaussian', 'file']:
                    errors.append(f"Invalid energy distribution: {energy['distribution']}")
            else:
                errors.append("Invalid energy specification format")

    return len(errors) == 0, errors


def validate_simulation_parameters(config):
    """
    Validate simulation-specific parameters.

    Parameters:
        config: Configuration dictionary

    Returns:
        tuple: (bool indicating if parameters are valid, list of validation errors)
    """
    errors = []

    # Check that source energy is within simulation energy range
    if 'source' in config and 'simulation' in config:
        source_energy = config['source'].get('energy', 0)
        max_energy = config['simulation'].get('max_energy', 0)

        # Handle different energy specifications
        if isinstance(source_energy, (int, float)):
            if source_energy > max_energy:
                errors.append(f"Source energy ({source_energy}) exceeds maximum simulation energy ({max_energy})")
        elif isinstance(source_energy, dict):
            # For energy distributions, check maximum
            if source_energy.get('distribution') == 'mono':
                if source_energy.get('value', 0) > max_energy:
                    errors.append(
                        f"Source mono energy ({source_energy.get('value')}) exceeds maximum simulation energy ({max_energy})")
            elif source_energy.get('distribution') == 'uniform':
                if source_energy.get('max', 0) > max_energy:
                    errors.append(
                        f"Source max energy ({source_energy.get('max')}) exceeds maximum simulation energy ({max_energy})")
            elif source_energy.get('distribution') == 'gaussian':
                # For Gaussian, check mean + 3*sigma as a reasonable maximum
                mean = source_energy.get('mean', 0)
                sigma = source_energy.get('sigma', 0)
                if mean + 3 * sigma > max_energy:
                    errors.append(f"Source energy distribution may exceed maximum simulation energy ({max_energy})")

    # Check that wall thickness is appropriate for the source energy
    if 'source' in config and 'geometry' in config:
        source_energy = 0
        if isinstance(config['source'].get('energy'), (int, float)):
            source_energy = config['source'].get('energy')
        elif isinstance(config['source'].get('energy'), dict):
            if config['source'].get('energy').get('distribution') == 'mono':
                source_energy = config['source'].get('energy').get('value', 0)
            elif config['source'].get('energy').get('distribution') == 'uniform':
                source_energy = config['source'].get('energy').get('max', 0)
            elif config['source'].get('energy').get('distribution') == 'gaussian':
                source_energy = config['source'].get('energy').get('mean', 0)

        wall_thickness = config['geometry'].get('wall_thickness', 0)

        # Very basic check - this is a simplification, real validation would be more complex
        # Consider a rough rule of thumb for gamma attenuation
        if source_energy > 0.5 and wall_thickness < 1.0:
            errors.append(
                f"Wall thickness ({wall_thickness} cm) may be insufficient for source energy ({source_energy} MeV)")

    # Check that number of particles is appropriate for the required statistical precision
    num_particles = config.get('simulation', {}).get('num_particles', 0)
    if num_particles < 10000:
        errors.append(f"Number of particles ({num_particles}) may be too low for good statistics")

    # Check that the channel diameter is physically meaningful
    if 'geometry' in config:
        channel_diameter = config['geometry'].get('channel_diameter', 0)
        wall_thickness = config['geometry'].get('wall_thickness', 0)

        if channel_diameter > wall_thickness:
            errors.append(f"Channel diameter ({channel_diameter} cm) exceeds wall thickness ({wall_thickness} cm)")

    return len(errors) == 0, errors


def suggest_optimization(config):
    """
    Suggest optimizations for the simulation based on configuration.

    Parameters:
        config: Configuration dictionary

    Returns:
        list: Suggestions for optimization
    """
    suggestions = []

    # Check if particle count can be optimized
    num_particles = config.get('simulation', {}).get('num_particles', 0)
    if num_particles > 1000000:
        suggestions.append("Consider using variance reduction techniques for better efficiency")
    elif num_particles < 50000:
        suggestions.append("Consider increasing particle count for better statistics")

    # Check if energy cutoff can be optimized
    max_energy = config.get('simulation', {}).get('max_energy', 0)
    cutoff_energy = config.get('simulation', {}).get('cutoff_energy', 0)

    if cutoff_energy < 0.01 and max_energy > 1.0:
        suggestions.append("Consider increasing cutoff energy for faster simulation")

    # Suggest parallel computation if particle count is high
    if num_particles > 500000:
        suggestions.append("Consider enabling parallel computation for faster results")

    # Suggest geometry optimizations
    channel_shape = config.get('geometry', {}).get('channel_shape', '')
    if channel_shape == 'straight':
        suggestions.append("Consider testing stepped or curved channel shapes for better shielding")

    return suggestions


def print_requirements_report():
    """
    Print a report of system requirements and package versions.
    """
    print("\n" + "=" * 50)
    print("SYSTEM REQUIREMENTS REPORT")
    print("=" * 50)

    # Python version
    print(f"\nPython Version: {platform.python_version()}")
    print(f"Python Implementation: {platform.python_implementation()}")

    # Operating system
    print(f"\nOperating System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")

    # Memory information
    try:
        if platform.system() == 'Windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]

            memory_status = MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))

            total_memory_gb = memory_status.ullTotalPhys / (1024 ** 3)
            avail_memory_gb = memory_status.ullAvailPhys / (1024 ** 3)

        elif platform.system() == 'Linux':
            # On Linux, read from /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                mem_info = f.read()

            # Extract memory information
            mem_info_dict = {}
            for line in mem_info.splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    mem_info_dict[key.strip()] = value.strip()

            total_memory_kb = int(mem_info_dict.get('MemTotal', '0').split()[0])
            avail_memory_kb = int(mem_info_dict.get('MemAvailable', '0').split()[0])

            total_memory_gb = total_memory_kb / (1024 ** 2)
            avail_memory_gb = avail_memory_kb / (1024 ** 2)

        else:  # macOS or other
            # Use sysctrl to get memory info on macOS
            try:
                result = subprocess.run(
                    ['sysctl', 'hw.memsize'],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True
                )
                total_memory_bytes = int(result.stdout.split()[1])
                total_memory_gb = total_memory_bytes / (1024 ** 3)
                avail_memory_gb = None  # Not easily available on macOS
            except:
                total_memory_gb = None
                avail_memory_gb = None

        if total_memory_gb is not None:
            print(f"\nTotal Memory: {total_memory_gb:.2f} GB")
        if avail_memory_gb is not None:
            print(f"Available Memory: {avail_memory_gb:.2f} GB")

    except Exception as e:
        print(f"\nCould not determine memory information: {e}")

    # Package versions
    print("\nPackage Versions:")
    required_packages = [
        "numpy", "scipy", "matplotlib", "pandas",
        "scikit-learn", "h5py", "tqdm"
    ]

    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  {package}: {version}")
        except ImportError:
            print(f"  {package}: Not installed")

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    # Print system requirements report when module is run directly
    print_requirements_report()