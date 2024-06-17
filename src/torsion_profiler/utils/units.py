"""
Defines some default units and conversions.
"""
import pint
from scipy import constants

unit_registry = pint.UnitRegistry()
unit_registry.default_format = "~"

# Recommended Units
## Energies
kJ = unit_registry.kJ
kcal = unit_registry.kcal
hatree = unit_registry.E_h
ev = unit_registry.eV


## Temperature
K = unit_registry.K

## Geometrie: Distance
nm = unit_registry.nm
a = unit_registry.angstrom
deg = unit_registry.degree
bohr = unit_registry.bohr

## Chem
mol = unit_registry.avogadro_number

# Constants
avogadro_constant = constants.Avogadro
k_b = constants.k

# Conversions
hatree_to_kcal_per_mol = (1 * hatree).to(kcal / mol).magnitude
hatree_to_kJ_per_mol = (1 * hatree).to(kJ / mol).magnitude
kcal_to_kJ = (1 * kcal).to(kJ).magnitude
kJ_to_kcal = (1 * kJ).to(kcal).magnitude
ev_to_hatree = (1 * ev).to(hatree).magnitude
ev_to_kcal_per_mol = (1 * ev).to(kcal / mol).magnitude
bohr_to_angstrom = (1 * bohr).to(a).magnitude
