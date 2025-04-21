import openmc
import numpy as np
from config import SOURCE_TO_SHIELD_DISTANCE, CM_TO_FT


def create_point_source(energy_kev, channel_diameter):
    """
    Create a gamma-ray point source with biased angular distribution.

    Args:
        energy_kev: Energy of the source in keV
        channel_diameter: Diameter of the channel in cm

    Returns:
        openmc.Source: The configured source
    """
    # Convert energy to MeV
    energy_mev = energy_kev / 1000.0

    # Define source position (6 ft in front of shield)
    source_position = [0, 0, -SOURCE_TO_SHIELD_DISTANCE]

    # Calculate solid angle for biasing
    channel_radius = channel_diameter / 2.0
    distance_to_shield = SOURCE_TO_SHIELD_DISTANCE

    # Solid angle calculation
    theta_max = np.arctan(channel_radius / distance_to_shield)

    # Create distribution for energy
    energy_dist = openmc.stats.Discrete([energy_mev], [1.0])

    # Create spatial distribution (point)
    spatial_dist = openmc.stats.Point(source_position)

    # Create angular distribution
    # Use a focused conical distribution to ensure particles go through the channel
    # Calculate the cone half-angle (in radians)
    mu_cutoff = np.cos(theta_max)  # Cosine of the maximum angle

    # Create an angular distribution focused on the forward direction (z-axis)
    # with particles only emitted within the cone that will hit the channel
    angle_dist = openmc.stats.Monodirectional([0, 0, 1])

    # Create a biased source distribution using the solid angle
    # This ensures 100% of particles go through the channel
    source = openmc.Source(space=spatial_dist, angle=angle_dist, energy=energy_dist)

    # Add a conical spatial filter to increase efficiency
    source.particle = 'photon'

    # Define angle parameters for biasing
    mu = openmc.stats.PowerLaw(mu_cutoff, 1.0, 3)  # Power=3 gives more focus at channel center
    phi = openmc.stats.Uniform(0.0, 2 * np.pi)

    # Override angle distribution with biased distribution
    source.angle = openmc.stats.PolarAzimuthal(mu, phi, reference_uvw=[0, 0, 1])

    return source