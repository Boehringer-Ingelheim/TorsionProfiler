"""
Test Paramter mol
"""

import importlib

import pytest
import click
from rdkit import Chem

from ...parameters.mol import get_molecule


def test_get_molecule_smiles():
    """
    test parsing smiles
    """
    mol = get_molecule("CC")
    assert isinstance(mol, Chem.Mol)


def test_get_molecule_sdf():
    """
    test parsing sdf
    """
    with importlib.resources.path(
        "openfe.tests.data.serialization", "ethane_template.sdf"
    ) as filename:
        # Note: the template doesn't include a valid version, but it loads
        # anyway. In the future, we may need to create a temporary file with
        # template substitutions done, but that seemed like overkill now.
        mol = get_molecule(filename)
        assert isinstance(mol, Chem.Mol)


def test_get_molecule_mol2():
    """
    test parsing mol2
    """
    with importlib.resources.path("openfe.tests.data.lomap_basic", "toluene.mol2") as f:
        mol = get_molecule(str(f))

        assert isinstance(mol, Chem.Mol)


def test_get_molecule_error():
    """
    test error with nonparsable
    """
    with pytest.raises(click.BadParameter):
        get_molecule("foobar")
