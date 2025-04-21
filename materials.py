mport
openmc
import numpy as np


def create_materials():
    """Create all materials needed for the simulation."""
    materials = {}

    # Define concrete according to ANSI/ANS-6.4-2006
    concrete = openmc.Material(name='concrete')
    concrete.set_density('g/cm3', 2.3)
    concrete.add_element('H', 0.01, 'wo')
    concrete.add_element('C', 0.001, 'wo')
    concrete.add_element('O', 0.529, 'wo')
    concrete.add_element('Na', 0.016, 'wo')
    concrete.add_element('Mg', 0.002, 'wo')
    concrete.add_element('Al', 0.034, 'wo')
    concrete.add_element('Si', 0.337, 'wo')
    concrete.add_element('K', 0.013, 'wo')
    concrete.add_element('Ca', 0.044, 'wo')
    concrete.add_element('Fe', 0.014, 'wo')
    materials['concrete'] = concrete

    # Define air
    air = openmc.Material(name='air')
    air.set_density('g/cm3', 0.001205)
    air.add_element('N', 0.7553, 'wo')
    air.add_element('O', 0.2318, 'wo')
    air.add_element('Ar', 0.0128, 'wo')
    air.add_element('C', 0.0001, 'wo')
    materials['air'] = air

    # Define ICRU tissue (for phantom)
    tissue = openmc.Material(name='tissue')
    tissue.set_density('g/cm3', 1.0)
    tissue.add_element('H', 0.101, 'wo')
    tissue.add_element('C', 0.111, 'wo')
    tissue.add_element('N', 0.026, 'wo')
    tissue.add_element('O', 0.762, 'wo')
    materials['tissue'] = tissue

    # Create material for void/vacuum
    void = openmc.Material(name='void')
    void.set_density('g/cm3', 1e-10)
    void.add_element('H', 1.0)
    void.set_density('g/cm3', 1e-10)
    materials['void'] = void

    # Create material collection
    material_collection = openmc.Materials(materials.values())
    material_collection.export_to_xml()

    return materials


def get_dose_conversion_factors(energy_groups):
    """
    Get flux-to-dose conversion factors based on NCRP-38, ANS-6.1.1-1977.

    Args:
        energy_groups: Energy groups in MeV

    Returns:
        Dictionary with conversion factors
    """
    # ANS-6.1.1-1977 flux-to-dose conversion factors (rem/hr)/(p/cm²-s)
    # Energy (MeV) and corresponding conversion factors
    energy_points = np.array([
        0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
        0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5,
        2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0
    ])

    conversion_factors = np.array([
        3.96E-6, 5.82E-6, 8.28E-6, 1.98E-5, 3.42E-5, 5.22E-5, 7.20E-5, 1.22E-4, 1.73E-4,
        2.88E-4, 3.96E-4, 6.12E-4, 8.28E-4, 1.04E-3, 1.22E-3, 1.58E-3, 1.94E-3, 2.70E-3,
        3.42E-3, 4.68E-3, 5.76E-3, 6.84E-3, 7.92E-3, 1.00E-2, 1.22E-2
    ])

    # Interpolate to get conversion factors for our energy groups
    result = np.interp(energy_groups, energy_points, conversion_factors)

    return result