"""
Providing some test data
"""

import pytest
import numpy as np
from rdkit import Chem
from torsion_profiler.utils import bash


def mol_from_smiles(smiles: str)->Chem.Mol:
    """
        Get rdkit mol form a given smiles, wiht hydrogens and 3D Coords

    Parameters
    ----------
    smiles:str
        Smiles

    Returns
    -------
    Chem.Mol
        rdkit molecule to smiles

    """
    rdmols = [Chem.MolFromSmiles(s) for s in smiles]
    rdmols = [Chem.AddHs(m, addCoords=True) for m in rdmols]

    for m in rdmols:
        Chem.rdDistGeom.EmbedMolecule(m, useRandomCoords=False, randomSeed=0)

    return rdmols


@pytest.fixture(scope="session")
def butan()->Chem.Mol:
    """
    A test Butan

    Returns
    -------
    Chem.Mol
        butan mol.
    """
    smiles = ["CCCC"]

    rdmols = mol_from_smiles(smiles)
    rdmol: Chem.Mol = rdmols[0]
    rdmol.SetProp("ta", "1 2 3 4")
    rdmol.SetProp("tb", "2 3 4 5")
    rdmol.SetProp("_Name", "Budan")
    return rdmol


@pytest.fixture(scope="session")
def nmols()->list[Chem.Mol]:
    """
        multiple test mols

    Returns
    -------
    list[Chem.Mol]
        list of rd mols
    """
    smiles = [
        "CCCC",
        "CCCO",
        "CCCN",
    ]

    rdmols = mol_from_smiles(smiles)
    for rdmol in rdmols:
        rdmol.SetProp("ta", "1 2 3 4")
        rdmol.SetProp("tb", "2 3 4 5")

    return rdmols


@pytest.fixture(scope="session")
def butan_optimization_results()->dict[str, list[float]]:
    """
    expected coord optimization results for testing, according to different methods

    Returns
    -------
    dict[str, list[float]]
    """
    expected_results = {
        "Mmff94Calculator": np.array([5.98586389, 0.0]),
        "OffCalculator": np.array([4.49406, 0.0]),
        "AniCalculator": np.array([7.124276, 0.0]),
        "XtbCalculator": np.array([0.0, 0.0]),
        "Psi4Calculator": np.array([6.71147, 0.0]),
        "QuickCalculator": np.array([6.71147, 0.0]),
        "GaussianCalculator": np.array([3.0, 0.0]),
    }
    return expected_results


@pytest.fixture(scope="session")
def butan_tp_1d_5n_results():
    """
    expected torsionprofiles optimization results for testing, according to different methods

    Returns
    -------
    dict[str, list[float]]
    """
    n_measurements = 37
    expected_results = {
        "Mmff94Calculator": np.array(
            [
                0.0,
                0.25285,
                0.96221,
                1.95392,
                2.96923,
                3.71617,
                3.95713,
                3.61735,
                2.85551,
                1.95468,
                1.21607,
                0.82609,
                0.8383,
                1.25727,
                2.03578,
                3.04602,
                4.08658,
                4.8949,
                5.20962,
                4.89493,
                4.0861,
                3.04667,
                2.03569,
                1.25704,
                0.8384,
                0.82594,
                1.21608,
                1.95468,
                2.85529,
                3.61755,
                3.95716,
                3.7159,
                2.96947,
                1.95399,
                0.96218,
                0.25302,
                0.0,
            ]
        ),
        "angles": np.linspace(-180, 180, n_measurements),
        "n_measurements": n_measurements,
    }
    return expected_results


@pytest.fixture(scope="session")
def butan_tp_1d_5n_no_opt_results():
    """
    expected results for noOpts

    """
    n_measurements = 37
    expected_results = {
        "angles": np.linspace(-180, 180, n_measurements),
        "n_measurements": n_measurements,
    }
    return expected_results


@pytest.fixture(scope="session")
def butan_tp_2d_5n_results():
    """
    expected results for 2D prof

    """
    n_measurements = 37
    expected_results = {
        "Mmff94Calculator": np.array([0.0, 1.95460, 5.20961, 1.95460, 0.0]),
        "angles": np.linspace(-180, 180, n_measurements),
        "n_measurements": n_measurements,
    }
    return expected_results


@pytest.fixture(autouse=False)
def skip_slurm():
    """
    if no slurm present skip!
    """
    if bash.system("sinfo"):
        pytest.skip("No Slurm Detected (searched for SlurmInfo)")
