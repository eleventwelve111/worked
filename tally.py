import openmc
import numpy as np
from config import DETECTOR_DISTANCES, DETECTOR_ANGLES, WALL_THICKNESS


def create_tallies(energy_kev, channel_diameter):
    """
    Create a set of tallies for the simulation.

    Args:
        energy_kev: Energy of the source in keV (for naming)
        channel_diameter: Diameter of the channel in cm (for naming)

    Returns:
        openmc.Tallies: The configured tallies
    """
    tallies = openmc.Tallies()

    # Create flux tally for each phantom position
    for distance in DETECTOR_DISTANCES:
        for angle in DETECTOR_ANGLES:
            # Skip some angle-distance combinations as specified
            if distance in [100, 150] and angle > 0:
                continue

            # Create a cell filter for the phantom
            phantom_filter = openmc.CellFilter(
                openmc.Cell.get_cell_by_name(f'phantom_d{distance}_a{angle}')
            )

            # Create energy filter with many groups for spectral analysis
            energy_bins = np.logspace(-2, 1, 500)  # 0.01 to 10 MeV with 500 bins
            energy_filter = openmc.EnergyFilter(energy_bins)

            # Create a tally for phantom flux
            tally = openmc.Tally(name=f'flux_d{distance}_a{angle}')
            tally.filters = [phantom_filter, energy_filter]
            tally.scores = ['flux']
            tallies.append(tally)

            # Add a dose tally (using heating for gamma deposit)
            dose_tally = openmc.Tally(name=f'dose_d{distance}_a{angle}')
            dose_tally.filters = [phantom_filter]
            dose_tally.scores = ['heating']
            tallies.append(dose_tally)

            # Add kerma tally for dosimetry analysis
            kerma_tally = openmc.Tally(name=f'kerma_d{distance}_a{angle}')
            kerma_tally.filters = [phantom_filter, energy_filter]
            kerma_tally.scores = ['kerma-photon']  # Photon kerma
            tallies.append(kerma_tally)

            # Add a flux-to-dose-equivalent tally using ANS-6.1.1-1977 conversion factors
            # This uses the flux and multipliers to convert directly to dose equivalent
            dose_equiv_tally = openmc.Tally(name=f'dose_equiv_d{distance}_a{angle}')
            dose_equiv_tally.filters = [phantom_filter, energy_filter]
            dose_equiv_tally.scores = ['flux']

            # Use the ANS-6.1.1-1977 flux-to-dose conversion factors as energy-dependent multipliers
            # We'll define this multiplier function in the materials module
            from materials import get_dose_conversion_factors
            dose_factors = get_dose_conversion_factors(energy_bins[:-1])  # Use lower bin edges
            dose_equiv_tally.multiplier = dose_factors

            tallies.append(dose_equiv_tally)

    # Create mesh tallies for visualization
    # 2D Cartesian mesh for radiation through and around the shield
    mesh_xy = openmc.RegularMesh()
    mesh_xy.dimension = [200, 200, 1]
    mesh_xy.lower_left = [-100, -100, WALL_THICKNESS]
    mesh_xy.upper_right = [100, 100, WALL_THICKNESS + 1]

    mesh_xz = openmc.RegularMesh()
    mesh_xz.dimension = [200, 1, 200]
    mesh_xz.lower_left = [-100, -0.5, -10]
    mesh_xz.upper_right = [100, 0.5, WALL_THICKNESS + 190]

    # Create mesh filters
    mesh_xy_filter = openmc.MeshFilter(mesh_xy)
    mesh_xz_filter = openmc.MeshFilter(mesh_xz)

    # Create mesh tallies
    mesh_xy_tally = openmc.Tally(name='flux_xy_mesh')
    mesh_xy_tally.filters = [mesh_xy_filter]
    mesh_xy_tally.scores = ['flux']
    tallies.append(mesh_xy_tally)

    # Add kerma for mesh as well
    mesh_xy_kerma_tally = openmc.Tally(name='kerma_xy_mesh')
    mesh_xy_kerma_tally.filters = [mesh_xy_filter]
    mesh_xy_kerma_tally.scores = ['kerma-photon']
    tallies.append(mesh_xy_kerma_tally)

    mesh_xz_tally = openmc.Tally(name='flux_xz_mesh')
    mesh_xz_tally.filters = [mesh_xz_filter]
    mesh_xz_tally.scores = ['flux']
    tallies.append(mesh_xz_tally)

    mesh_xz_kerma_tally = openmc.Tally(name='kerma_xz_mesh')
    mesh_xz_kerma_tally.filters = [mesh_xz_filter]
    mesh_xz_kerma_tally.scores = ['kerma-photon']
    tallies.append(mesh_xz_kerma_tally)

    # Create finer mesh around the channel exit for detailed analysis
    fine_mesh = openmc.RegularMesh()
    fine_mesh.dimension = [100, 100, 1]
    fine_mesh.lower_left = [-10, -10, WALL_THICKNESS]
    fine_mesh.upper_right = [10, 10, WALL_THICKNESS + 0.1]

    fine_mesh_filter = openmc.MeshFilter(fine_mesh)
    fine_mesh_tally = openmc.Tally(name='fine_exit_mesh')
    fine_mesh_tally.filters = [fine_mesh_filter]
    fine_mesh_tally.scores = ['flux']
    tallies.append(fine_mesh_tally)

    fine_mesh_kerma_tally = openmc.Tally(name='fine_exit_kerma_mesh')
    fine_mesh_kerma_tally.filters = [fine_mesh_filter]
    fine_mesh_kerma_tally.scores = ['kerma-photon']
    tallies.append(fine_mesh_kerma_tally)

    # Export tallies to XML
    tallies.export_to_xml()

    return tallies


import numpy as np
import os
import pickle
import h5py
from logging import log_info, log_error, log_warning


class Tally:
    """Base class for all tallies."""

    def __init__(self, name, region=None):
        """
        Initialize a tally.

        Parameters:
            name: Name of the tally
            region: Region where the tally is applied
        """
        self.name = name
        self.region = region
        self.results = {}
        self.is_active = True

    def score(self, particle):
        """
        Score a particle in the tally.

        Parameters:
            particle: Particle object

        Returns:
            bool: True if scored successfully, False otherwise
        """
        raise NotImplementedError("Must be implemented by subclass")

    def reset(self):
        """Reset tally results."""
        self.results = {}

    def get_results(self):
        """
        Get tally results.

        Returns:
            dict: Tally results
        """
        return self.results

    def save(self, directory, file_format='pickle'):
        """
        Save tally results to a file.

        Parameters:
            directory: Directory to save the file
            file_format: Format to save the file ('pickle' or 'hdf5')

        Returns:
            str: Path to saved file
        """
        os.makedirs(directory, exist_ok=True)

        if file_format == 'pickle':
            filename = os.path.join(directory, f"{self.name}_tally.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(self.results, f)
        elif file_format == 'hdf5':
            filename = os.path.join(directory, f"{self.name}_tally.h5")
            with h5py.File(filename, 'w') as f:
                self._save_to_hdf5(f)
        else:
            log_error(f"Unsupported file format: {file_format}")
            return None

        log_info(f"Saved tally {self.name} to {filename}")
        return filename

    def _save_to_hdf5(self, h5file):
        """
        Save tally results to an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        raise NotImplementedError("Must be implemented by subclass")

    def load(self, filename):
        """
        Load tally results from a file.

        Parameters:
            filename: Path to file

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if filename.endswith('.pkl'):
                with open(filename, 'rb') as f:
                    self.results = pickle.load(f)
            elif filename.endswith('.h5'):
                with h5py.File(filename, 'r') as f:
                    self._load_from_hdf5(f)
            else:
                log_error(f"Unsupported file format: {filename}")
                return False

            log_info(f"Loaded tally {self.name} from {filename}")
            return True

        except Exception as e:
            log_error(f"Error loading tally from {filename}: {e}")
            return False

    def _load_from_hdf5(self, h5file):
        """
        Load tally results from an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        raise NotImplementedError("Must be implemented by subclass")


class FluxTally(Tally):
    """Tally for particle flux."""

    def __init__(self, name, region=None, energy_bins=None, spatial_bins=None):
        """
        Initialize a flux tally.

        Parameters:
            name: Name of the tally
            region: Region where the tally is applied
            energy_bins: Energy bin edges (MeV)
            spatial_bins: Spatial bin edges (cm)
        """
        super().__init__(name, region)

        # Default energy bins if none provided
        if energy_bins is None:
            energy_bins = np.logspace(-3, 1, 50)  # 0.001 to 10 MeV

        # If single value is provided, interpret as number of bins
        if isinstance(energy_bins, int):
            energy_bins = np.logspace(-3, 1, energy_bins)

        self.energy_bins = np.asarray(energy_bins)
        self.spatial_bins = None if spatial_bins is None else np.asarray(spatial_bins)

        # Initialize flux arrays
        if self.spatial_bins is None:
            # Energy-only binning
            self.flux = np.zeros(len(self.energy_bins) - 1)
            self.error = np.zeros(len(self.energy_bins) - 1)
        else:
            # Energy and spatial binning
            self.flux = np.zeros((len(self.energy_bins) - 1, len(self.spatial_bins) - 1))
            self.error = np.zeros((len(self.energy_bins) - 1, len(self.spatial_bins) - 1))

        # Initialize counts for error calculation
        self.counts = np.zeros_like(self.flux)

        # Store results
        self.results = {
            'energy_bins': self.energy_bins,
            'spatial_bins': self.spatial_bins,
            'flux': self.flux,
            'error': self.error,
            'counts': self.counts
        }

    def score(self, particle):
        """
        Score a particle in the flux tally.

        Parameters:
            particle: Particle object

        Returns:
            bool: True if scored successfully, False otherwise
        """
        try:
            # Check if particle is in the tally region
            if self.region is not None and not self.region.contains(particle.position):
                return False

            # Get particle energy and position
            energy = particle.energy
            position = particle.position

            # Find energy bin
            e_bin = np.digitize(energy, self.energy_bins) - 1

            # If energy is outside bins, ignore
            if e_bin < 0 or e_bin >= len(self.energy_bins) - 1:
                return False

            # If no spatial binning
            if self.spatial_bins is None:
                self.flux[e_bin] += particle.weight
                self.counts[e_bin] += 1
            else:
                # Get distance or appropriate spatial coordinate
                if hasattr(particle, 'distance'):
                    spatial_value = particle.distance
                else:
                    # Default to radial distance from origin
                    spatial_value = np.sqrt(np.sum(np.array(position) ** 2))

                # Find spatial bin
                s_bin = np.digitize(spatial_value, self.spatial_bins) - 1

                # If position is outside bins, ignore
                if s_bin < 0 or s_bin >= len(self.spatial_bins) - 1:
                    return False

                self.flux[e_bin, s_bin] += particle.weight
                self.counts[e_bin, s_bin] += 1

            return True

        except Exception as e:
            log_error(f"Error scoring particle in flux tally: {e}")
            return False

    def normalize(self, norm_factor):
        """
        Normalize flux by a factor.

        Parameters:
            norm_factor: Normalization factor

        Returns:
            bool: True if normalized successfully, False otherwise
        """
        try:
            # Normalize flux
            self.flux = self.flux * norm_factor

            # Calculate relative error
            with np.errstate(divide='ignore', invalid='ignore'):
                self.error = np.zeros_like(self.flux)
                mask = self.counts > 0
                self.error[mask] = 1.0 / np.sqrt(self.counts[mask])

            # Update results
            self.results['flux'] = self.flux
            self.results['error'] = self.error

            return True

        except Exception as e:
            log_error(f"Error normalizing flux tally: {e}")
            return False

    def get_total_flux(self):
        """
        Get total integrated flux.

        Returns:
            float: Total flux
        """
        return np.sum(self.flux)

    def get_mean_energy(self):
        """
        Get flux-weighted mean energy.

        Returns:
            float: Mean energy
        """
        # Calculate energy bin centers
        e_centers = 0.5 * (self.energy_bins[1:] + self.energy_bins[:-1])

        # Calculate mean energy
        if self.spatial_bins is None:
            total_flux = np.sum(self.flux)
            if total_flux > 0:
                return np.sum(e_centers * self.flux) / total_flux
        else:
            total_flux = np.sum(self.flux)
            if total_flux > 0:
        # Sum over spatial
        # Sum over spatial bins first
        energy_flux = np.sum(self.flux, axis=1)
        return np.sum(e_centers * energy_flux) / total_flux

        return 0.0

        def _save_to_hdf5(self, h5file):
            """
                Save flux tally results to an HDF5 file.

                Parameters:
                    h5file: Open HDF5 file object
                """
            try:
                # Create a group for this tally
                tally_group = h5file.create_group(self.name)

                # Add basic attributes
                tally_group.attrs['type'] = 'flux'
                if self.region is not None:
                    tally_group.attrs['region'] = self.region.name

                # Add datasets
                tally_group.create_dataset('energy_bins', data=self.energy_bins)
                tally_group.create_dataset('flux', data=self.flux)
                tally_group.create_dataset('error', data=self.error)
                tally_group.create_dataset('counts', data=self.counts)

                if self.spatial_bins is not None:
                    tally_group.create_dataset('spatial_bins', data=self.spatial_bins)

            except Exception as e:
                log_error(f"Error saving flux tally to HDF5: {e}")

        def _load_from_hdf5(self, h5file):
            """
                Load flux tally results from an HDF5 file.

                Parameters:
                    h5file: Open HDF5 file object
                """
            try:
                # Get group for this tally
                if self.name in h5file:
                    tally_group = h5file[self.name]
                else:
                    # If tally name not found, try to use the first group
                    tally_group = list(h5file.values())[0]
                    log_warning(f"Tally {self.name} not found in HDF5 file, using first group instead")

                # Load datasets
                self.energy_bins = tally_group['energy_bins'][...]
                self.flux = tally_group['flux'][...]
                self.error = tally_group['error'][...]
                self.counts = tally_group['counts'][...]

                if 'spatial_bins' in tally_group:
                    self.spatial_bins = tally_group['spatial_bins'][...]
                else:
                    self.spatial_bins = None

                # Update results dictionary
                self.results = {
                    'energy_bins': self.energy_bins,
                    'spatial_bins': self.spatial_bins,
                    'flux': self.flux,
                    'error': self.error,
                    'counts': self.counts
                }

            except Exception as e:
                log_error(f"Error loading flux tally from HDF5: {e}")

    class EnergyDepositionTally(Tally):
        """Tally for energy deposition."""

        def __init__(self, name, region=None, spatial_bins=None):
            """
                Initialize an energy deposition tally.

                Parameters:
                    name: Name of the tally
                    region: Region where the tally is applied
                    spatial_bins: Spatial bin edges (cm)
                """
            super().__init__(name, region)

            self.spatial_bins = None if spatial_bins is None else np.asarray(spatial_bins)

            # Initialize deposition arrays
            if self.spatial_bins is None:
                # Single bin for total deposition
                self.deposition = 0.0
                self.squared_deposition = 0.0
                self.count = 0
            else:
                # Spatial binning
                self.deposition = np.zeros(len(self.spatial_bins) - 1)
                self.squared_deposition = np.zeros(len(self.spatial_bins) - 1)
                self.count = np.zeros(len(self.spatial_bins) - 1)

            # Store results
            self.results = {
                'spatial_bins': self.spatial_bins,
                'deposition': self.deposition,
                'error': None,  # Will be calculated during normalization
                'count': self.count
            }

        def score(self, particle):
            """
                Score a particle's energy deposition in the tally.

                Parameters:
                    particle: Particle object

                Returns:
                    bool: True if scored successfully, False otherwise
                """
            try:
                # Check if particle is in the tally region
                if self.region is not None and not self.region.contains(particle.position):
                    return False

                # Get particle energy deposition and position
                if not hasattr(particle, 'energy_deposition') or particle.energy_deposition <= 0:
                    return False

                e_dep = particle.energy_deposition * particle.weight
                position = particle.position

                # If no spatial binning
                if self.spatial_bins is None:
                    self.deposition += e_dep
                    self.squared_deposition += e_dep ** 2
                    self.count += 1
                else:
                    # Get distance or appropriate spatial coordinate
                    if hasattr(particle, 'distance'):
                        spatial_value = particle.distance
                    else:
                        # Default to radial distance from origin
                        spatial_value = np.sqrt(np.sum(np.array(position) ** 2))

                    # Find spatial bin
                    s_bin = np.digitize(spatial_value, self.spatial_bins) - 1

                    # If position is outside bins, ignore
                    if s_bin < 0 or s_bin >= len(self.spatial_bins) - 1:
                        return False

                    self.deposition[s_bin] += e_dep
                    self.squared_deposition[s_bin] += e_dep ** 2
                    self.count[s_bin] += 1

                return True

            except Exception as e:
                log_error(f"Error scoring particle in energy deposition tally: {e}")
                return False

        def normalize(self, norm_factor):
            """
                Normalize energy deposition by a factor.

                Parameters:
                    norm_factor: Normalization factor

                Returns:
                    bool: True if normalized successfully, False otherwise
                """
            try:
                # Normalize deposition
                if self.spatial_bins is None:
                    self.deposition = self.deposition * norm_factor

                    # Calculate error (standard error of the mean)
                    if self.count > 1:
                        variance = (self.squared_deposition - self.deposition ** 2 / self.count) / (self.count - 1)
                        self.error = np.sqrt(variance / self.count) * norm_factor
                    else:
                        self.error = 0.0
                else:
                    self.deposition = self.deposition * norm_factor

                    # Calculate error for each bin
                    self.error = np.zeros_like(self.deposition)
                    for i in range(len(self.deposition)):
                        if self.count[i] > 1:
                            variance = (self.squared_deposition[i] - self.deposition[i] ** 2 / self.count[i]) / (
                                        self.count[i] - 1)
                            self.error[i] = np.sqrt(variance / self.count[i]) * norm_factor

                # Update results
                self.results['deposition'] = self.deposition
                self.results['error'] = self.error

                return True

            except Exception as e:
                log_error(f"Error normalizing energy deposition tally: {e}")
                return False

        def get_total_deposition(self):
            """
                Get total energy deposition.

                Returns:
                    float: Total energy deposition
                """
            if self.spatial_bins is None:
                return self.deposition
            else:
                return np.sum(self.deposition)

        def _save_to_hdf5(self, h5file):
            """
                Save energy deposition tally results to an HDF5 file.

                Parameters:
                    h5file: Open HDF5 file object
                """
            try:
                # Create a group for this tally
                tally_group = h5file.create_group(self.name)

                # Add basic attributes
                tally_group.attrs['type'] = 'energy_deposition'
                if self.region is not None:
                    tally_group.attrs['region'] = self.region.name

                # Add datasets
                tally_group.create_dataset('deposition', data=self.deposition)
                if self.error is not None:
                    tally_group.create_dataset('error', data=self.error)
                tally_group.create_dataset('count', data=self.count)

                if self.spatial_bins is not None:
                    tally_group.create_dataset('spatial_bins', data=self.spatial_bins)

            except Exception as e:
                log_error(f"Error saving energy deposition tally to HDF5: {e}")

        def _load_from_hdf5(self, h5file):
            """
                Load energy deposition tally results from an HDF5 file.

                Parameters:
                    h5file: Open HDF5 file object
                """
            try:
                # Get group for this tally
                if self.name in h5file:
                    tally_group = h5file[self.name]
                else:
                    # If tally name not found, try to use the first group
                    tally_group = list(h5file.values())[0]
                    log_warning(f"Tally {self.name} not found in HDF5 file, using first group instead")

                # Load datasets
                self.deposition = tally_group['deposition'][...]
                if 'error' in tally_group:
                    self.error = tally_group['error'][...]
                else:
                    self.error = None
                self.count = tally_group['count'][...]

                if 'spatial_bins' in tally_group:
                    self.spatial_bins = tally_group['spatial_bins'][...]
                else:
                    self.spatial_bins = None

                # Update results dictionary
                self.results = {
                    'spatial_bins': self.spatial_bins,
                    'deposition': self.deposition,
                    'error': self.error,
                    'count': self.count
                }

            except Exception as e:
                log_error(f"Error loading energy deposition tally from HDF5: {e}")

    class PulseHeightTally(Tally):
        """Tally for pulse height distribution."""

        def __init__(self, name, region=None, energy_bins=None):
            """
                Initialize a pulse height tally.

                Parameters:
                    name: Name of the tally
                    region: Region where the tally is applied
                    energy_bins: Energy bin edges (MeV)
                """
            super().__init__(name, region)

            # Default energy bins if none provided
            if energy_bins is None:
                energy_bins = np.logspace(-3, 1, 100)  # 0.001 to 10 MeV

            # If single value is provided, interpret as number of bins
            if isinstance(energy_bins, int):
                energy_bins = np.logspace(-3, 1, energy_bins)

            self.energy_bins = np.asarray(energy_bins)

            # Initialize pulse height array
            self.pulse_height = np.zeros(len(self.energy_bins) - 1)

            # Store results
            self.results = {
                'energy_bins': self.energy_bins,
                'pulse_height': self.pulse_height
            }

        def score(self, particle):
            """
                Score a particle in the pulse height tally.

                Parameters:
                    particle: Particle object

                Returns:
                    bool: True if scored successfully, False otherwise
                """
            try:
                # Check if particle is in the tally region
                if self.region is not None and not self.region.contains(particle.position):
                    return False

                # Get particle energy deposition
                if not hasattr(particle, 'energy_deposition') or particle.energy_deposition <= 0:
                    return False

                e_dep = particle.energy_deposition

                # Find energy bin
                e_bin = np.digitize(e_dep, self.energy_bins) - 1

                # If energy is outside bins, ignore
                if e_bin < 0 or e_bin >= len(self.energy_bins) - 1:
                    return False

                self.pulse_height[e_bin] += particle.weight

                return True

            except Exception as e:
                log_error(f"Error scoring particle in pulse height tally: {e}")
                return False

        def normalize(self, norm_factor):
            """
                Normalize pulse height distribution by a factor.

                Parameters:
                    norm_factor: Normalization factor

                Returns:
                    bool: True if normalized successfully, False otherwise
                """
            try:
                # Normalize pulse height
                self.pulse_height = self.pulse_height * norm_factor

                # Update results
                self.results['pulse_height'] = self.pulse_height

                return True

            except Exception as e:
                log_error(f"Error normalizing pulse height tally: {e}")
                return False

        def _save_to_hdf5(self, h5file):
            """
                Save pulse height tally results to an HDF5 file.

                Parameters:
                    h5file: Open HDF5 file object
                """
            try:
                # Create a group for this tally
                tally_group = h5file.create_group(self.name)

                # Add basic attributes
                tally_group.attrs['type'] = 'pulse_height'
                if self.region is not None:
                    tally_group.attrs['region'] = self.region.name

                # Add datasets
                tally_group.create_dataset('energy_bins', data=self.energy_bins)
                tally_group.create_dataset('pulse_height', data=self.pulse_height)

            except Exception as e:
                log_error(f"Error saving pulse height tally to HDF5: {e}")

        def _load_from_hdf5(self, h5file):
            """
                Load pulse height tally results from an HDF5 file.

                Parameters:
                    h5file: Open HDF5 file object
                """
            try:
                # Get group for this tally
                if self.name in h5file:
                    tally_group = h5file[self.name]
                else:
                    # If tally name not found, try to use the first group
                    tally_group = list(h5file.values())[0]
                    log_warning(f"Tally {self.name} not found in HDF5 file, using first group instead")

                # Load datasets
                self.energy_bins = tally_group['energy_bins'][...]
                self.pulse_height = tally_group['pulse_height'][...]

                # Update results dictionary
                self.results = {
                    'energy_bins': self.energy_bins,              # Sum over spatial bins first
                energy_flux = np.sum(self.flux, axis=1)
                return np.sum(e_centers * energy_flux) / total_flux

        return 0.0

    def _save_to_hdf5(self, h5file):
        """
        Save flux tally results to an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Create a group for this tally
            tally_group = h5file.create_group(self.name)

            # Add basic attributes
            tally_group.attrs['type'] = 'flux'
            if self.region is not None:
                tally_group.attrs['region'] = self.region.name

            # Add datasets
            tally_group.create_dataset('energy_bins', data=self.energy_bins)
            tally_group.create_dataset('flux', data=self.flux)
            tally_group.create_dataset('error', data=self.error)
            tally_group.create_dataset('counts', data=self.counts)

            if self.spatial_bins is not None:
                tally_group.create_dataset('spatial_bins', data=self.spatial_bins)

        except Exception as e:
            log_error(f"Error saving flux tally to HDF5: {e}")

    def _load_from_hdf5(self, h5file):
        """
        Load flux tally results from an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Get group for this tally
            if self.name in h5file:
                tally_group = h5file[self.name]
            else:
                # If tally name not found, try to use the first group
                tally_group = list(h5file.values())[0]
                log_warning(f"Tally {self.name} not found in HDF5 file, using first group instead")

            # Load datasets
            self.energy_bins = tally_group['energy_bins'][...]
            self.flux = tally_group['flux'][...]
            self.error = tally_group['error'][...]
            self.counts = tally_group['counts'][...]

            if 'spatial_bins' in tally_group:
                self.spatial_bins = tally_group['spatial_bins'][...]
            else:
                self.spatial_bins = None

            # Update results dictionary
            self.results = {
                'energy_bins': self.energy_bins,
                'spatial_bins': self.spatial_bins,
                'flux': self.flux,
                'error': self.error,
                'counts': self.counts
            }

        except Exception as e:
            log_error(f"Error loading flux tally from HDF5: {e}")

class EnergyDepositionTally(Tally):
    """Tally for energy deposition."""

    def __init__(self, name, region=None, spatial_bins=None):
        """
        Initialize an energy deposition tally.

        Parameters:
            name: Name of the tally
            region: Region where the tally is applied
            spatial_bins: Spatial bin edges (cm)
        """
        super().__init__(name, region)

        self.spatial_bins = None if spatial_bins is None else np.asarray(spatial_bins)

        # Initialize deposition arrays
        if self.spatial_bins is None:
            # Single bin for total deposition
            self.deposition = 0.0
            self.squared_deposition = 0.0
            self.count = 0
        else:
            # Spatial binning
            self.deposition = np.zeros(len(self.spatial_bins) - 1)
            self.squared_deposition = np.zeros(len(self.spatial_bins) - 1)
            self.count = np.zeros(len(self.spatial_bins) - 1)

        # Store results
        self.results = {
            'spatial_bins': self.spatial_bins,
            'deposition': self.deposition,
            'error': None,  # Will be calculated during normalization
            'count': self.count
        }

    def score(self, particle):
        """
        Score a particle's energy deposition in the tally.

        Parameters:
            particle: Particle object

        Returns:
            bool: True if scored successfully, False otherwise
        """
        try:
            # Check if particle is in the tally region
            if self.region is not None and not self.region.contains(particle.position):
                return False

            # Get particle energy deposition and position
            if not hasattr(particle, 'energy_deposition') or particle.energy_deposition <= 0:
                return False

            e_dep = particle.energy_deposition * particle.weight
            position = particle.position

            # If no spatial binning
            if self.spatial_bins is None:
                self.deposition += e_dep
                self.squared_deposition += e_dep**2
                self.count += 1
            else:
                # Get distance or appropriate spatial coordinate
                if hasattr(particle, 'distance'):
                    spatial_value = particle.distance
                else:
                    # Default to radial distance from origin
                    spatial_value = np.sqrt(np.sum(np.array(position)**2))

                # Find spatial bin
                s_bin = np.digitize(spatial_value, self.spatial_bins) - 1

                # If position is outside bins, ignore
                if s_bin < 0 or s_bin >= len(self.spatial_bins) - 1:
                    return False

                self.deposition[s_bin] += e_dep
                self.squared_deposition[s_bin] += e_dep**2
                self.count[s_bin] += 1

            return True

        except Exception as e:
            log_error(f"Error scoring particle in energy deposition tally: {e}")
            return False

    def normalize(self, norm_factor):
        """
        Normalize energy deposition by a factor.

        Parameters:
            norm_factor: Normalization factor

        Returns:
            bool: True if normalized successfully, False otherwise
        """
        try:
            # Normalize deposition
            if self.spatial_bins is None:
                self.deposition = self.deposition * norm_factor

                # Calculate error (standard error of the mean)
                if self.count > 1:
                    variance = (self.squared_deposition - self.deposition**2 / self.count) / (self.count - 1)
                    self.error = np.sqrt(variance / self.count) * norm_factor
                else:
                    self.error = 0.0
            else:
                self.deposition = self.deposition * norm_factor

                # Calculate error for each bin
                self.error = np.zeros_like(self.deposition)
                for i in range(len(self.deposition)):
                    if self.count[i] > 1:
                        variance = (self.squared_deposition[i] - self.deposition[i]**2 / self.count[i]) / (self.count[i] - 1)
                        self.error[i] = np.sqrt(variance / self.count[i]) * norm_factor

            # Update results
            self.results['deposition'] = self.deposition
            self.results['error'] = self.error

            return True

        except Exception as e:
            log_error(f"Error normalizing energy deposition tally: {e}")
            return False

    def get_total_deposition(self):
        """
        Get total energy deposition.

        Returns:
            float: Total energy deposition
        """
        if self.spatial_bins is None:
            return self.deposition
        else:
            return np.sum(self.deposition)

    def _save_to_hdf5(self, h5file):
        """
        Save energy deposition tally results to an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Create a group for this tally
            tally_group = h5file.create_group(self.name)

            # Add basic attributes
            tally_group.attrs['type'] = 'energy_deposition'
            if self.region is not None:
                tally_group.attrs['region'] = self.region.name

            # Add datasets
            tally_group.create_dataset('deposition', data=self.deposition)
            if self.error is not None:
                tally_group.create_dataset('error', data=self.error)
            tally_group.create_dataset('count', data=self.count)

            if self.spatial_bins is not None:
                tally_group.create_dataset('spatial_bins', data=self.spatial_bins)

        except Exception as e:
            log_error(f"Error saving energy deposition tally to HDF5: {e}")

    def _load_from_hdf5(self, h5file):
        """
        Load energy deposition tally results from an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Get group for this tally
            if self.name in h5file:
                tally_group = h5file[self.name]
            else:
                # If tally name not found, try to use the first group
                tally_group = list(h5file.values())[0]
                log_warning(f"Tally {self.name} not found in HDF5 file, using first group instead")

            # Load datasets
            self.deposition = tally_group['deposition'][...]
            if 'error' in tally_group:
                self.error = tally_group['error'][...]
            else:
                self.error = None
            self.count = tally_group['count'][...]

            if 'spatial_bins' in tally_group:
                self.spatial_bins = tally_group['spatial_bins'][...]
            else:
                self.spatial_bins = None

            # Update results dictionary
            self.results = {
                'spatial_bins': self.spatial_bins,
                'deposition': self.deposition,
                'error': self.error,
                'count': self.count
            }

        except Exception as e:
            log_error(f"Error loading energy deposition tally from HDF5: {e}")

class PulseHeightTally(Tally):
    """Tally for pulse height distribution."""

    def __init__(self, name, region=None, energy_bins=None):
        """
        Initialize a pulse height tally.

        Parameters:
            name: Name of the tally
            region: Region where the tally is applied
            energy_bins: Energy bin edges (MeV)
        """
        super().__init__(name, region)

        # Default energy bins if none provided
        if energy_bins is None:
            energy_bins = np.logspace(-3, 1, 100)  # 0.001 to 10 MeV

        # If single value is provided, interpret as number of bins
        if isinstance(energy_bins, int):
            energy_bins = np.logspace(-3, 1, energy_bins)

        self.energy_bins = np.asarray(energy_bins)

        # Initialize pulse height array
        self.pulse_height = np.zeros(len(self.energy_bins) - 1)

        # Store results
        self.results = {
            'energy_bins': self.energy_bins,
            'pulse_height': self.pulse_height
        }

    def score(self, particle):
        """
        Score a particle in the pulse height tally.

        Parameters:
            particle: Particle object

        Returns:
            bool: True if scored successfully, False otherwise
        """
        try:
            # Check if particle is in the tally region
            if self.region is not None and not self.region.contains(particle.position):
                return False

            # Get particle energy deposition
            if not hasattr(particle, 'energy_deposition') or particle.energy_deposition <= 0:
                return False

            e_dep = particle.energy_deposition

            # Find energy bin
            e_bin = np.digitize(e_dep, self.energy_bins) - 1

            # If energy is outside bins, ignore
            if e_bin < 0 or e_bin >= len(self.energy_bins) - 1:
                return False

            self.pulse_height[e_bin] += particle.weight

            return True

        except Exception as e:
            log_error(f"Error scoring particle in pulse height tally: {e}")
            return False

    def normalize(self, norm_factor):
        """
        Normalize pulse height distribution by a factor.

        Parameters:
            norm_factor: Normalization factor

        Returns:
            bool: True if normalized successfully, False otherwise
        """
        try:
            # Normalize pulse height
            self.pulse_height = self.pulse_height * norm_factor

            # Update results
            self.results['pulse_height'] = self.pulse_height

            return True

        except Exception as e:
            log_error(f"Error normalizing pulse height tally: {e}")
            return False

    def _save_to_hdf5(self, h5file):
        """
        Save pulse height tally results to an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Create a group for this tally
            tally_group = h5file.create_group(self.name)

            # Add basic attributes
            tally_group.attrs['type'] = 'pulse_height'
            if self.region is not None:
                tally_group.attrs['region'] = self.region.name

            # Add datasets
            tally_group.create_dataset('energy_bins', data=self.energy_bins)
            tally_group.create_dataset('pulse_height', data=self.pulse_height)

        except Exception as e:
            log_error(f"Error saving pulse height tally to HDF5: {e}")

    def _load_from_hdf5(self, h5file):
        """
        Load pulse height tally results from an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Get group for this tally
            if self.name in h5file:
                tally_group = h5file[self.name]
            else:
                # If tally name not found, try to use the first group
                tally_group = list(h5file.values())[0]
                log_warning(f"Tally {self.name} not found in HDF5 file, using first group instead")

            # Load datasets
            self.energy_bins = tally_group['energy_bins'][...]
            self.pulse_height = tally_group['pulse_height'][...]

            # Update results dictionary
            self.results = {
                'energy_bins': self.energy_bins,                 'pulse_height': self.pulse_height
            }

        except Exception as e:
            log_error(f"Error loading pulse height tally from HDF5: {e}")

class AngleTally(Tally):
    """Tally for angular distribution."""

    def __init__(self, name, region=None, angle_bins=None):
        """
        Initialize an angle tally.

        Parameters:
            name: Name of the tally
            region: Region where the tally is applied
            angle_bins: Angle bin edges (degrees, 0-180)
        """
        super().__init__(name, region)

        # Default angle bins if none provided
        if angle_bins is None:
            angle_bins = np.linspace(0, 180, 37)  # 5-degree bins

        # If single value is provided, interpret as number of bins
        if isinstance(angle_bins, int):
            angle_bins = np.linspace(0, 180, angle_bins + 1)

        self.angle_bins = np.asarray(angle_bins)

        # Initialize angle distribution array
        self.distribution = np.zeros(len(self.angle_bins) - 1)

        # Store results
        self.results = {
            'angle_bins': self.angle_bins,
            'distribution': self.distribution
        }

    def score(self, particle):
        """
        Score a particle in the angle tally.

        Parameters:
            particle: Particle object

        Returns:
            bool: True if scored successfully, False otherwise
        """
        try:
            # Check if particle is in the tally region
            if self.region is not None and not self.region.contains(particle.position):
                return False

            # Get particle direction
            if not hasattr(particle, 'direction'):
                return False

            # Calculate angle with respect to z-axis (forward direction)
            z_axis = np.array([0, 0, 1])
            direction = np.array(particle.direction)

            # Normalize direction vector
            direction = direction / np.linalg.norm(direction)

            # Calculate cosine of angle
            cos_angle = np.dot(direction, z_axis)

            # Convert to angle in degrees
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi

            # Find angle bin
            a_bin = np.digitize(angle, self.angle_bins) - 1

            # If angle is outside bins, ignore
            if a_bin < 0 or a_bin >= len(self.angle_bins) - 1:
                return False

            self.distribution[a_bin] += particle.weight

            return True

        except Exception as e:
            log_error(f"Error scoring particle in angle tally: {e}")
            return False

    def normalize(self, norm_factor):
        """
        Normalize angular distribution by a factor.

        Parameters:
            norm_factor: Normalization factor

        Returns:
            bool: True if normalized successfully, False otherwise
        """
        try:
            # Normalize distribution
            self.distribution = self.distribution * norm_factor

            # Update results
            self.results['distribution'] = self.distribution

            return True

        except Exception as e:
            log_error(f"Error normalizing angle tally: {e}")
            return False

    def _save_to_hdf5(self, h5file):
        """
        Save angle tally results to an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Create a group for this tally
            tally_group = h5file.create_group(self.name)

            # Add basic attributes
            tally_group.attrs['type'] = 'angle'
            if self.region is not None:
                tally_group.attrs['region'] = self.region.name

            # Add datasets
            tally_group.create_dataset('angle_bins', data=self.angle_bins)
            tally_group.create_dataset('distribution', data=self.distribution)

        except Exception as e:
            log_error(f"Error saving angle tally to HDF5: {e}")

    def _load_from_hdf5(self, h5file):
        """
        Load angle tally results from an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Get group for this tally
            if self.name in h5file:
                tally_group = h5file[self.name]
            else:
                # If tally name not found, try to use the first group
                tally_group = list(h5file.values())[0]
                log_warning(f"Tally {self.name} not found in HDF5 file, using first group instead")

            # Load datasets
            self.angle_bins = tally_group['angle_bins'][...]
            self.distribution = tally_group['distribution'][...]

            # Update results dictionary
            self.results = {
                'angle_bins': self.angle_bins,
                'distribution': self.distribution
            }

        except Exception as e:
            log_error(f"Error loading angle tally from HDF5: {e}")

class PathLengthTally(Tally):
    """Tally for particle path length distribution."""

    def __init__(self, name, region=None, length_bins=None):
        """
        Initialize a path length tally.

        Parameters:
            name: Name of the tally
            region: Region where the tally is applied
            length_bins: Length bin edges (cm)
        """
        super().__init__(name, region)

        # Default length bins if none provided
        if length_bins is None:
            length_bins = np.logspace(-1, 2, 50)  # 0.1 to 100 cm

        # If single value is provided, interpret as number of bins
        if isinstance(length_bins, int):
            length_bins = np.logspace(-1, 2, length_bins)

        self.length_bins = np.asarray(length_bins)

        # Initialize path length array
        self.path_length = np.zeros(len(self.length_bins) - 1)

        # Store results
        self.results = {
            'length_bins': self.length_bins,
            'path_length': self.path_length
        }

    def score(self, particle):
        """
        Score a particle in the path length tally.

        Parameters:
            particle: Particle object

        Returns:
            bool: True if scored successfully, False otherwise
        """
        try:
            # Check if particle is in the tally region
            if self.region is not None and not self.region.contains(particle.position):
                return False

            # Get particle path length
            if not hasattr(particle, 'path_length') or particle.path_length <= 0:
                return False

            path_length = particle.path_length

            # Find length bin
            l_bin = np.digitize(path_length, self.length_bins) - 1

            # If length is outside bins, ignore
            if l_bin < 0 or l_bin >= len(self.length_bins) - 1:
                return False

            self.path_length[l_bin] += particle.weight

            return True

        except Exception as e:
            log_error(f"Error scoring particle in path length tally: {e}")
            return False

    def normalize(self, norm_factor):
        """
        Normalize path length distribution by a factor.

        Parameters:
            norm_factor: Normalization factor

        Returns:
            bool: True if normalized successfully, False otherwise
        """
        try:
            # Normalize path length
            self.path_length = self.path_length * norm_factor

            # Update results
            self.results['path_length'] = self.path_length

            return True

        except Exception as e:
            log_error(f"Error normalizing path length tally: {e}")
            return False

    def _save_to_hdf5(self, h5file):
        """
        Save path length tally results to an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Create a group for this tally
            tally_group = h5file.create_group(self.name)

            # Add basic attributes
            tally_group.attrs['type'] = 'path_length'
            if self.region is not None:
                tally_group.attrs['region'] = self.region.name

            # Add datasets
            tally_group.create_dataset('length_bins', data=self.length_bins)
            tally_group.create_dataset('path_length', data=self.path_length)

        except Exception as e:
            log_error(f"Error saving path length tally to HDF5: {e}")

    def _load_from_hdf5(self, h5file):
        """
        Load path length tally results from an HDF5 file.

        Parameters:
            h5file: Open HDF5 file object
        """
        try:
            # Get group for this tally
            if self.name in h5file:
                tally_group = h5file[self.name]
            else:
                # If tally name not found, try to use the first group
                tally_group = list(h5file.values())[0]
                log_warning(f"Tally {self.name} not found in HDF5 file, using first group instead")

            # Load datasets
            self.length_bins = tally_group['length_bins'][...]
            self.path_length = tally_group['path_length'][...]

            # Update results dictionary
            self.results = {
                'length_bins': self.length_bins,
                'path_length': self.path_length
            }

        except Exception as e:
            log_error(f"Error loading path length tally from HDF5: {e}")

def create_tallies(config):
    """
    Create tally objects based on configuration.

    Parameters:
        config: Tally configuration dictionary

    Returns:
        dict: Dictionary of tally objects
    """
    tallies = {}

    try:
        if not config:
            log_warning("No tally configuration provided, using default flux tally")
            tallies['default_flux'] = FluxTally(name='default_flux')
            return tallies

        for tally_name, tally_config in config.items():
            tally_type = tally_config.get('type', 'flux')
            region = tally_config.get('region', None)

            if tally_type == 'flux':
                energy_bins = tally_config.get('energy_bins', None)
                spatial_bins = tally_config.get('spatial_bins', None)
                tallies[tally_name] = FluxTally(
                    name=tally_name,
                    region=region,
                    energy_bins=energy_bins,
                    spatial_bins=spatial_bins
                )

            elif tally_type == 'energy_deposition':
                spatial_bins = tally_config.get('spatial_bins', None)
                tallies[tally_name] = EnergyDepositionTally(
                    name=tally_name,
                    region=region,
                    spatial_bins=spatial_bins
                )

            elif tally_type == 'pulse_height':
                energy_bins = tally_config.get('energy_bins', None)
                tallies[tally_name] = PulseHeightTally(
                    name=tally_name,
                    region=region,
                    energy_bins=energy_bins
                )

            elif tally_type == 'angle':
                angle_bins = tally_config.get('angle_bins', None)
                tallies[tally_name] = AngleTally(
                    name=tally_name,
                    region=region,
                    angle_bins=angle_bins
                )

            elif tally_type == 'path_length':
                length_bins = tally_config.get('length_bins', None)
                tallies[tally_name] = PathLengthTally(
                    name=tally_name,
                    region=region,
                    length_bins=length_bins
                )

            else:
                log_warning(f"Unknown tally type: {tally_type}, skipping tally {tally_name}")

        return tallies

    except Exception as e:
        log_error(f"Error creating tallies: {e}")
        # Return at least one default tally to allow simulation to proceed
        tallies['default_flux'] = FluxTally(name='default_flux')
        return tallies

def merge_tally_results(tallies_list):
    """
    Merge results from multiple tallies of the same type.

    Parameters:
        tallies_list: List of tally objects of the same type

    Returns:
        Tally: A new tally object with merged results
    """
    if not tallies_list:
        raise ValueError("Empty tally list")

    # Check that all tallies are of the same type
    tally_type = type(tallies_list[0])
    if not all(isinstance(t, tally_type) for t in tallies_list):
        raise ValueError("All tallies must be of the same type")

    # Create a new tally with the same parameters as the first one
    first_tally = tallies_list[0]
    merged_tally = None

    if isinstance(first_tally, FluxTally):
        merged_tally = FluxTally(
            name=f"merged_{first_tally.name}",
            region=first_tally.region,
            energy_bins=first_tally.energy_bins,
            spatial_bins=first_tally.spatial_bins
        )
        # Add flux and counts from all tallies
        for tally in tallies_list:
            merged_tally.flux += tally.flux
            merged_tally.counts += tally.counts

    elif isinstance(first_tally, EnergyDepositionTally):
        merged_tally = EnergyDepositionTally(
            name=f"merged_{first_tally.name}",
            region=first_tally.region,
            spatial_bins=first_tally.spatial_bins
        )
        # Add deposition and counts from all tallies
        for tally in tallies_list:
            merged_tally.deposition += tally.deposition
            merged_tally.squared_deposition += tally.squared_deposition
            merged_tally.count += tally.count

    elif isinstance(first_tally, PulseHeightTally):
        merged_tally = PulseHeightTally(
            name=f"merged_{first_tally.name}",
            region=first_tally.region,
            energy_bins=first_tally.energy_bins
        )
        # Add pulse height from all tallies
        for tally in tallies_list:
            merged_tally.pulse_height += tally.pulse_height

    elif isinstance(first_tally, AngleTally):
        merged_tally = AngleTally(
            name=f"merged_{first_tally.name}",
            region=first_tally.region,
            angle_bins=first_tally.angle_bins
        )
        # Add distribution from all tallies
        for tally in tallies_list:
            merged_tally.distribution += tally.distribution

    elif isinstance(first_tally, PathLengthTally):
        merged_tally = PathLengthTally(
            name=f"merged_{first_tally.name}",
            region=first_tally.region,
            length_bins=first_tally.length_bins
        )
        # Add path length from all tallies
        for tally in tallies_list:
            merged_tally.path_length += tally.path_length

        # Update the results dictionary
    merged_tally.results = merged_tally.get_results()

    return merged_tally


def save_tally_results(tallies, directory, file_format='hdf5'):
    """
    Save all tally results to files.

    Parameters:
        tallies: Dictionary of tally objects
        directory: Directory to save the files
        file_format: Format to save the files ('pickle' or 'hdf5')

    Returns:
        dict: Dictionary of saved file paths
    """
    saved_files = {}

    try:
        os.makedirs(directory, exist_ok=True)

        if file_format == 'hdf5':
            # Save all tallies to a single HDF5 file
            filename = os.path.join(directory, 'tally_results.h5')
            with h5py.File(filename, 'w') as f:
                for tally_name, tally in tallies.items():
                    tally._save_to_hdf5(f)

            saved_files['all'] = filename

        else:
            # Save each tally to a separate file
            for tally_name, tally in tallies.items():
                filename = tally.save(directory, file_format)
                if filename:
                    saved_files[tally_name] = filename

        return saved_files

    except Exception as e:
        log_error(f"Error saving tally results: {e}")
        return saved_files


def load_tally_results(tallies, filename):
    """
    Load tally results from a file.

    Parameters:
        tallies: Dictionary of tally objects
        filename: Path to file

    Returns:
        bool: True if loaded successfully, False otherwise
    """
    try:
        if filename.endswith('.h5'):
            # Load tallies from an HDF5 file
            with h5py.File(filename, 'r') as f:
                for tally_name, tally in tallies.items():
                    if tally_name in f:
                        tally._load_from_hdf5(f)
                    else:
                        log_warning(f"Tally {tally_name} not found in HDF5 file")
        else:
            # Load tallies from individual files
            for tally_name, tally in tallies.items():
                tally_file = os.path.join(os.path.dirname(filename), f"{tally_name}_tally.pkl")
                if os.path.exists(tally_file):
                    tally.load(tally_file)
                else:
                    log_warning(f"Tally file {tally_file} not found")

        return True

    except Exception as e:
        log_error(f"Error loading tally results: {e}")
        return False


def print_tally_summary(tallies):
    """
    Print a summary of all tally results.

    Parameters:
        tallies: Dictionary of tally objects
    """
    print("\n" + "=" * 50)
    print("TALLY RESULTS SUMMARY")
    print("=" * 50)

    for tally_name, tally in tallies.items():
        print(f"\nTally: {tally_name}")

        if isinstance(tally, FluxTally):
            total_flux = tally.get_total_flux()
            mean_energy = tally.get_mean_energy()
            print(f"  Type: Flux")
            print(f"  Total Flux: {total_flux:.6e}")
            print(f"  Mean Energy: {mean_energy:.6f} MeV")

        elif isinstance(tally, EnergyDepositionTally):
            total_deposition = tally.get_total_deposition()
            print(f"  Type: Energy Deposition")
            print(f"  Total Energy Deposition: {total_deposition:.6e} MeV")

        elif isinstance(tally, PulseHeightTally):
            total_counts = np.sum(tally.pulse_height)
            print(f"  Type: Pulse Height")
            print(f"  Total Counts: {total_counts:.6e}")

        elif isinstance(tally, AngleTally):
            total_counts = np.sum(tally.distribution)
            print(f"  Type: Angular Distribution")
            print(f"  Total Counts: {total_counts:.6e}")

        elif isinstance(tally, PathLengthTally):
            total_counts = np.sum(tally.path_length)
            print(f"  Type: Path Length")
            print(f"  Total Counts: {total_counts:.6e}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Example usage when module is run directly
    import matplotlib.pyplot as plt

    # Create a flux tally
    flux_tally = FluxTally(
        name="test_flux",
        energy_bins=np.logspace(-2, 1, 30)  # 0.01 to 10 MeV
    )


    # Create a simple particle for testing
    class TestParticle:
        def __init__(self, energy, weight=1.0):
            self.energy = energy
            self.weight = weight
            self.position = (0, 0, 0)


    # Score some particles
    for i in range(1000):
        # Random energy between 0.1 and 5 MeV
        energy = 0.1 + 4.9 * np.random.random()
        particle = TestParticle(energy)
        flux_tally.score(particle)

    # Normalize results
    flux_tally.normalize(1.0)

    # Get results
    results = flux_tally.get_results()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(results['energy_bins'][:-1], results['flux'], 'o-')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Flux')
    plt.title('Test Flux Tally')
    plt.grid(True)
    plt.savefig('test_flux_tally.png')
    plt.close()

    print("Created test plot: test_flux_tally.png")


