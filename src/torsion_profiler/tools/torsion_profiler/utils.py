"""
Torsion profile utils
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
from scipy import stats

from ...utils import bash


def _read_tp_pdb(in_file_path, torsion_atom_ids ) -> Chem.Mol:

    if not in_file_path.endswith(".pdb"):
        raise IOError("File has no .pdb ending")
    if not bash.path.isfile(in_file_path):
        raise IOError("File has no .pdb ending")

    mol_0 = Chem.MolFromPDBFile(in_file_path, removeHs=False)
    mol_0.SetProp("torsion_atom_ids", " ".join(map(str, torsion_atom_ids)))
    empty_mol = Chem.Mol(mol_0)
    empty_mol.RemoveAllConformers()

    pdb_mols = []
    print("Mol: ", mol_0.GetNumConformers())
    for i in range(mol_0.GetNumConformers()):
        tmol = Chem.Mol(empty_mol)
        c = mol_0.GetConformer(i)
        angle = np.round(Chem.rdMolTransforms.GetDihedralDeg(c, *torsion_atom_ids), 2)
        tmol.SetProp("torsion_angle", str(angle))
        tmol.AddConformer(c)
        pdb_mols.append(tmol)
    return pdb_mols


def get_angles_of_torsion_profile_mol(
    mols: pd.DataFrame, torsion_atom_ids: tuple[int, int, int, int]
) -> list[float]:
    """

    Parameters
    ----------
    mol
    torsion_atom_ids

    Returns
    -------

    """
    torsion_angles = []
    for mol in mols["ROMol"]:
        conf = mol.GetConformer()
        angle = np.round(Chem.rdMolTransforms.GetDihedralDeg(conf, *torsion_atom_ids), 2)
        torsion_angles.append(angle)
    torsion_angles = np.round(torsion_angles)
    return torsion_angles


def is_a_torsion_profile(
    mol,
    torsion_atom_ids: tuple[int, int, int, int],
    min_angle=-160,
    max_angle: float = 160,
    test_uniform: bool = True,
) -> bool:
    """
    This function tests if a molecule is a Torsion profile regarding to max and min angle and
    uniformity of the angle distribution given by conformers in mol.

    Parameters
    ----------
    mol : _type_
        rdkit mol containing the profile mols
    torsion_atom_ids : tuple[int, int, int, int]
        ids for the torsion atoms to investigate.
    min_angle : int, optional
        at least expect this minimal angle for a range (if < angle more no problem), by default -160
    max_angle : float, optional
        at least expect this maximal angle for a range (if > angle more no problem), by default 160
    test_uniform : bool, optional
        are the angle measurements unifromly distributed?, by default True

    Returns
    -------
    bool
        result of the test
    """

    angles = []
    for i in range(mol.GetNumConformers()):
        angles.append(GetDihedralDeg(mol.GetConformer(i), *torsion_atom_ids))

    if test_uniform:
        test_uniform = lambda x: stats.kstest(
            x, stats.uniform(loc=np.min(x), scale=np.max(x) - np.min(x)).cdf
        ).pvalue
    else:
        test_uniform = lambda x: 1.0

    is_a_torsion_profiles = (
        lambda x: np.min(x) < min_angle and np.max(x) > max_angle and test_uniform(x) > 0.95
    )

    return is_a_torsion_profiles(angles)
