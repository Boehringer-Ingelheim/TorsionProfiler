"""
Test Paramter mol_dir
"""

import os
import importlib

import pytest
import click

from ...parameters.mol_dir import get_molecules


def test_get_dir_molecules_sdf():
    """
    test parsing mol dir with sdfs
    """
    with importlib.resources.path("openfe.tests.data.serialization", "__init__.py") as file_path:
        # Note: the template doesn't include a valid version, but it loads
        # anyway. In the future, we may need to create a temporary file with
        # template substitutions done, but that seemed like overkill now.
        dir_path = os.path.dirname(file_path)
        mols = get_molecules(dir_path)

        assert len(mols) == 1
        assert mols[0].smiles == "CC"
        assert mols[0].name == "ethane"


def test_get_dir_molecules_mol2():
    """
    test parsing mol dir with mol2s
    """
    with importlib.resources.path("openfe.tests.data.lomap_basic", "__init__.py") as file_path:
        # Note: the template doesn't include a valid version, but it loads
        # anyway. In the future, we may need to create a temporary file with
        # template substitutions done, but that seemed like overkill now.
        dir_path = os.path.dirname(file_path)
        mols = get_molecules(dir_path)

        assert len(mols) == 8
        assert mols[0].smiles == "Cc1cc(C)c2cc(C)ccc2c1"
        assert mols[0].name == "*****"


def test_get_molecule_error():
    """
    test error with nonparsable
    """
    with pytest.raises(click.BadParameter):
        get_molecules("foobar")
