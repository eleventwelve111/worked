import openmc
import numpy as np
from config import (WALL_THICKNESS, SOURCE_TO_SHIELD_DISTANCE,
                    DETECTOR_INITIAL_DISTANCE, PHANTOM_DIAMETER)


def create_geometry(channel_diameter):
    """
    Create the geometry for the simulation.

    Args:
        channel_diameter: Diameter of the air channel in cm

    Returns:
        openmc.Geometry: The complete geometry
    """
    # Define dimensions
    wall_thickness = WALL_THICKNESS
    world_size = 500.0  # Large enough to contain everything

    # Create materials dictionary
    materials = {}
    materials['concrete'] = openmc.Material.get_material_by_name('concrete')
    materials['air'] = openmc.Material.get_material_by_name('air')
    materials['void'] = openmc.Material.get_material_by_name('void')
    materials['tissue'] = openmc.Material.get_material_by_name('tissue')

    # Define the outer boundary (world)
    world_min = -world_size / 2
    world_max = world_size / 2
    world_box = openmc.Box(world_min, world_min, world_min,
                           world_max, world_max, world_max)
    world_region = -world_box
    world_cell = openmc.Cell(name='world')
    world_cell.region = world_region
    world_cell.fill = materials['void']

    # Define the concrete wall
    wall_min_z = 0
    wall_max_z = wall_thickness
    wall_box = openmc.Box(-world_size / 4, -world_size / 4, wall_min_z,
                          world_size / 4, world_size / 4, wall_max_z)
    wall_region = -wall_box
    wall_cell = openmc.Cell(name='wall')
    wall_cell.region = wall_region
    wall_cell.fill = materials['concrete']

    # Create air channel through the wall
    channel_radius = channel_diameter / 2
    channel_cylinder = openmc.ZCylinder(x0=0, y0=0, r=channel_radius)
    channel_z_bounds = openmc.Box(-world_size / 2, -world_size / 2, wall_min_z - 1,
                                  world_size / 2, world_size / 2, wall_max_z + 1)
    channel_region = -channel_cylinder & -channel_z_bounds
    channel_cell = openmc.Cell(name='channel')
    channel_cell.region = channel_region
    channel_cell.fill = materials['air']

    # Remove the channel from the wall
    wall_cell.region = wall_region & ~channel_region

    # Create universe
    universe = openmc.Universe(cells=[world_cell, wall_cell, channel_cell])

    # Create geometry
    geometry = openmc.Geometry(universe)
    geometry.export_to_xml()

    return geometry


def create_phantom(distance, angle=0.0):
    """
    Create a phantom detector cell at the specified distance and angle.

    Args:
        distance: Distance behind the wall in cm
        angle: Angle off the central axis in degrees

    Returns:
        openmc.Cell: The phantom cell
    """
    materials = {}
    materials['tissue'] = openmc.Material.get_material_by_name('tissue')

    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Calculate center position of the phantom
    wall_exit_z = WALL_THICKNESS
    phantom_radius = PHANTOM_DIAMETER / 2

    # Position at angle
    x_position = distance * np.sin(angle_rad)
    z_position = wall_exit_z + distance * np.cos(angle_rad)

    # Create sphere for phantom
    phantom_sphere = openmc.Sphere(x0=x_position, y0=0, z0=z_position, r=phantom_radius)
    phantom_cell = openmc.Cell(name=f'phantom_d{distance}_a{angle}')
    phantom_cell.region = -phantom_sphere
    phantom_cell.fill = materials['tissue']

    return phantom_cell


def add_phantom_to_geometry(geometry, distance, angle=0.0):
    """
    Add a phantom to an existing geometry.

    Args:
        geometry: The existing geometry
        distance: Distance behind the wall in cm
        angle: Angle off the central axis in degrees

    Returns:
        openmc.Geometry: The updated geometry
    """
    phantom_cell = create_phantom(distance, angle)

    # Get the cells from the root universe
    cells = list(geometry.root_universe.get_all_cells().values())

    # Add the phantom cell
    cells.append(phantom_cell)

    # Create new universe and geometry
    universe = openmc.Universe(cells=cells)
    new_geometry = openmc.Geometry(universe)

    return new_geometry