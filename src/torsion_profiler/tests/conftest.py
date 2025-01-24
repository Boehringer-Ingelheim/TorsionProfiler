"""
Build some test data
"""

import pandas as pd
import pytest
import numpy as np
from rdkit import Chem
from torsion_profiler.utils import bash

from torsion_profiler.tests.test_data import (
    butan_torsion_profile_sdf_path,
    butan_torsion_profile_tsv_path,
)


def mol_from_smiles(smiles: str):
    """
    Get a molecule with 3D coords and hydrogens
    """
    rdmols = [Chem.MolFromSmiles(s) for s in smiles]
    rdmols = [Chem.AddHs(m, addCoords=True) for m in rdmols]

    for m in rdmols:
        Chem.rdDistGeom.EmbedMolecule(m, useRandomCoords=False, randomSeed=0)

    return rdmols


@pytest.fixture(scope="session", name="butan_df")
def fixture_butan_df():
    """
    Butan test torsion  profile
    """
    return pd.read_csv(butan_torsion_profile_tsv_path, sep="\t", index_col=0)


@pytest.fixture(scope="session", name="butan_mol")
def fixture_butan_mol():
    """
    Butan torsion profile result molecule
    """

    spp = Chem.SDMolSupplier(butan_torsion_profile_sdf_path, removeHs=False)
    first = True
    for mol in spp:
        if first:
            rmol = mol
            first = False
        else:
            rmol.AddConformer(mol.GetConformer())

    return rmol


@pytest.fixture(scope="session", name="butan")
def fixture_butan():
    """
    butan test rdkit mol
    """
    smiles = ["CCCC"]

    rdmols = mol_from_smiles(smiles)
    rdmol = rdmols[0]
    rdmol.SetProp("ta", "0 1 2 3")
    rdmol.SetProp("tb", "1 2 3 11")

    return rdmol

@pytest.fixture(scope="session", name="hexan")
def fixture_hexan():
    """
    butan test rdkit mol
    """
    smiles = ["CCCCCC"]

    rdmols = mol_from_smiles(smiles)
    rdmol = rdmols[0]
    rdmol.SetProp("ta", "0 1 2 3")
    rdmol.SetProp("tb", "1 2 3 11")

    return rdmol

@pytest.fixture(scope="session", name="hexan_1d_alltors_tps")
def fixture_hexan_1d_alltors_tps():
    """
    butan test rdkit mol
    """
    n_measurements=5
    expected_results = {
        "Mmff94Calculator": np.array([0.0, 1.952, 5.332, 1.952, 0.0, 0.0, 1.96, 5.459, 1.96, 0.0, 0.0, 1.953, 5.332, 1.952, 0.0]),
        "OffCalculator": np.array([4.49406, 0.0]),
        "AniCalculator": np.array([7.124276, 0.0]),
        "MACECalculator": np.array([7.124276, 0.0]),
        "XtbCalculator": np.array([0.0, 0.0]),
        "Psi4Calculator": np.array([6.71147, 0.0]),
        "QuickCalculator": np.array([6.71147, 0.0]),
        "GaussianCalculator": np.array([3.0, 0.0]),
        "angles":np.array([-180.0, -89.99, 0.0, 89.99, 179.99,
                           -179.99, -89.99, -0.03, 89.99, 179.99,
                           -179.99, -89.99, 0.01, 89.99, 179.99]),
        "n_measurements": n_measurements,
    }
    return expected_results



@pytest.fixture(scope="session", name="butan_optimization_results")
def fixture_butan_optimization_results():
    """
    Expected butan results
    """

    expected_results = {
        "Mmff94Calculator": np.array([5.98586389, 0.0]),
        "OffCalculator": np.array([4.49406, 0.0]),
        "AniCalculator": np.array([7.124276, 0.0]),
        "MACECalculator": np.array([7.124276, 0.0]),
        "XtbCalculator": np.array([0.0, 0.0]),
        "Psi4Calculator": np.array([6.71147, 0.0]),
        "QuickCalculator": np.array([6.71147, 0.0]),
        "GaussianCalculator": np.array([3.0, 0.0]),
    }
    return expected_results


@pytest.fixture(scope="session", name="butan_sp_results")
def fixture_butan_sp_results()->dict[str, list[float]]:
    """
    example result for 1D Torsion profile 5 measurments, for multiple methods.

    Returns
    -------
    dict[str, list[float]]
    """
    expected_results = {
        "Mmff94Calculator": 0.909891,
        "OffCalculator": 24.101171,
        "AniCalculator": -99397.10338,
        "AltAniCalculator": -415877.5378,
        "MACECalculator": -1293.985105,
        "XtbCalculator": -8568.60625,
        "Psi4Calculator": 0.909891,
        "QuickCalculator": 0.909891,
        "GaussianCalculator": -99398.35256,
    }
    return expected_results

@pytest.fixture(scope="session", name="butan_opt_results")
def fixture_butan_opt_results()->dict[str, list[float]]:
    """
    example result for 1D Torsion profile 5 measurments, for multiple methods.

    Returns
    -------
    dict[str, list[float]]
    """
    expected_results = {
        "Mmff94Calculator": -5.075973,
        "OffCalculator": 19.1311,
        "AniCalculator": -99404.229347,
        "AltAniCalculator": -99404.229347,
        "MACECalculator": -1301.324994,
        "XtbCalculator": -8574.99666,
        "Psi4Calculator": 0.909891,
        "QuickCalculator": 0.909891,
        "GaussianCalculator": -99405.10624,
    }
    return expected_results

@pytest.fixture(scope="session", name="butan_mc_opt_results")
def fixture_butan_mc_opt_results()->dict[str, list[float]]:
    """
    example result for 1D Torsion profile 5 measurments, for multiple methods.

    Returns
    -------
    dict[str, list[float]]
    """
    n_measurements = 5
    expected_results = {
        "Mmff94Calculator": np.array([-5.07597, -1.60765, -4.20325, -2.13496, -2.13496, -4.2034, -1.60765, -5.07597]),
        "OffCalculator": np.array([19.24667, 22.25631, 20.06346, 22.04812, 22.04812, 20.06339, 22.25615, 19.24668]),
        "AniCalculator": np.array([-99404.07068, -99400.93564, -99403.50138, -99401.47815,
                  -99401.47815, -99403.38063, -99400.93562, -99404.07069]),
        "AltAniCalculator": np.array([-99404.07068, -99400.93564, -99403.50138, -99401.47815,
                  -99401.47815, -99403.38063, -99400.93562, -99404.07069]),
        "MACECalculator":  np.array([-1301.22225, -1298.18298, -1300.56762, -1298.56935, -1298.56935,
                   -1300.46506, -1298.18299, -1301.22225]),
        "XtbCalculator": np.array([-8574.97327, -8572.72894, -8574.3569 , -8572.15088, -8572.15088,
                   -8574.30985, -8572.72894, -8574.97327]),
        "Psi4Calculator": np.array([0.00, 1.64593, 5.28014, 1.64433, 0.00]),
        "QuickCalculator": np.array([0.00, 1.64593, 5.28014, 1.64433, 0.00]),
        "GaussianCalculator": np.array([[-99405.10622, -99402.18764, -99404.51212, -99402.3838 ,
                   -99402.3838 , -99404.51211, -99402.18764, -99405.10622]]),
        "angles": np.linspace(-180, 180, n_measurements),
        "n_measurements": n_measurements,
    }
    return expected_results


@pytest.fixture(scope="session", name="butan_tp_1d_5n_results")
def fixture_butan_tp_1d_5n_results()->dict[str, list[float]]:
    """
    example result for 1D Torsion profile 5 measurments, for multiple methods.

    Returns
    -------
    dict[str, list[float]]
    """
    n_measurements = 5
    expected_results = {
        "Mmff94Calculator": np.array([0.0, 1.95460, 5.20961, 1.95460, 0.0]),
        "OffCalculator": np.array([471776.46297, 0.0, 3.1073, 0.0, 471771.89198]),  # WEIRDD!
        "AniCalculator": np.array([0.0, 1.79321, 5.13169, 2.28893, 0.0]),
        "MACECalculator": np.array([0,  1.84979, 10.14636,  1.84962,  0]),
        "Psi4Calculator": np.array([0.00, 1.64593, 5.28014, 1.64433, 0.00]),
        "QuickCalculator": np.array([0.00, 1.64593, 5.28014, 1.64433, 0.00]),
        "GaussianCalculator": np.array([0.00, 1.66898, 5.32143, 1.66932, 0.00]),
        "angles": np.linspace(-180, 180, n_measurements),
        "n_measurements": n_measurements,
    }
    return expected_results


@pytest.fixture(scope="session", name="butan_tp_1d_5n_no_opt_results")
def fixture_butan_tp_1d_5n_no_opt_results()->dict[str, list[float]]:
    """
    example result for 1D Torsion profile 5 measurments, for multiple methods. no coord
    optimization

    Returns
    -------
    dict[str, list[float]]
    """
    n_measurements = 5
    expected_results = {
        "Mmff94Calculator": np.array([0.0, 1.28124, 5.1287, 1.28114, 0.0]),
        "OffCalculator": np.array([471778.12978, 1.66727, 4.81775, 3.23117, 0.0]),
        "AniCalculator": np.array([0.0, 1.81953, 5.05441, 2.30406, 0.0]),
        "MACECalculator": np.array([0,  2.1042 , 10.65911,  2.10331,  0]),
        "XtbCalculator": np.array([0.0, 1.33399, 5.03578, 2.16645, 0.0]),
        "Psi4Calculator": np.array([0.0, 1.73797, 5.14703, 2.07297, 0.0]),
        "QuickCalculator": np.array([0.04602, 2.09562, 5.67456, 2.09079, 0.0]),
        "GaussianCalculator": np.array([0.0, 1.76466, 5.19185, 2.080810, 0.0]),
        "angles": np.linspace(-180, 180, n_measurements),
        "n_measurements": n_measurements,
    }
    return expected_results


@pytest.fixture(scope="session", name="butan_tp_2d_5n_results")
def fixture_butan_tp_2d_5n_results()->dict[str, list[float]]:
    """
    example result for 2D Torsion profile 5 measurments, for multiple methods. no coord
    optimization

    Returns
    -------
    dict[str, list[float]]
    """
    n_measurements = 5
    expected_results = {
        "Mmff94Calculator": np.array([0.0, 1.5495799999999997, 3.41857, 1.5495799999999997, 0.0, 0.0, 2.7682900000000004, 4.579630000000001, 1.0704000000000002, 0.0, 4.37361, 1.0699699999999996, 9.12536, 6.49266, 0.0, 0.0, 1.0704000000000002, 3.2384600000000003, 2.7682900000000004, 0.0, 0.0, 1.5495799999999997, 3.41857, 1.5495799999999997, 0.0]),
        "OffCalculator": np.array([85.511087, 7.808516, 0.000000, 126.160421, 472263.944296]),
        "AniCalculator": np.array([104.014874, 0.0, 59.992542, 85.222875, 30.415095]),
        "MACECalculator": np.array([-99404.07068, -99400.93564, -99403.50138, -99401.47815,
                                    -99401.47815, -99403.38063, -99400.93562, -99404.07069]),
        "XtbCalculator": np.array([29.153546, 157.986322, 121.170269, 0.000000, 58.962322]),
        "Psi4Calculator": np.array([3.0, 0.0]),
        "GaussianCalculator": np.array([3.0, 0.0]),
        "angles": np.array([-179.99683356253036, -89.9899956254692, -1.656052458586812, 89.98999519006097, 179.99313847131413, -179.98999974420877, -89.98999559009438, -0.00832186787684234, 89.98999557791655, 179.99000712747292, 179.99285513594572, -89.99000096872456, 5.00890905864129, 89.98999087731757, 179.99000282866993, 179.99001728368867, -89.99001675357812, 0.010000323522573969, 90.00000000000003, 179.98999974396116, 179.9973382555329, -89.98999586798325, -1.656052068780273, 89.99002459748974, 179.9901226400802]),
        "n_measurements": n_measurements,
    }
    return expected_results


@pytest.fixture(autouse=False, name="skip_slurm")
def fixture_skip_slurm():
    """
    check if slurm is present.
    """
    if bash.system("sinfo"):
        pytest.skip("No Slurm Detected (searched for SlurmInfo)")
