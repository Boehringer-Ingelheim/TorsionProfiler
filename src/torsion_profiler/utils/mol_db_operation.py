import ast

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

from torsion_profiler.utils.metrics import get_boltzman_ensemble_p
from torsion_profiler.utils.units import kcal_to_kJ


def read_mol_db(
    in_file_path: str, expected_cols = ["ROMol"]
) -> pd.DataFrame:
    """
    This function is reading an sdf or pdb file and merges all conformers into one rd-mol.
    Additional 4 atom ids can be provided, to calculate for each conformere the torsion angle
    between them.

    Parameters
    ----------
    sdf_path : str
        path to pdb or sdf file.
    expected_cols : tuple[int, int, int, int], optional
        atom ids, that should be stored in the mol and used to calculate the torsion angle,
        by default None

    Returns
    -------
    Chem.Mol
        resulting merged molecule

    Raises
    ------
    ValueError
        if the path is not matching the .pdb or .sdf pattern.
    """

    if in_file_path.endswith(".sdf"):
        mol_db = PandasTools.LoadSDF(in_file_path, idName='molID', molColName='ROMol',
                                    includeFingerprints=False, smilesName='smiles', embedProps=True,
                                    removeHs=False)
    else:
        raise ValueError(f"Can not translate to anything! here: {in_file_path}")

    if mol_db is None:
        raise ValueError(f"Can not translate to anything! here: {in_file_path}")

    elif(any(c not in mol_db.columns for c in expected_cols)):
        col_pres = "\n".join([f"{c} -> {c not in mol_db.columns}" for c in expected_cols])
        raise ValueError(f"The given profile is missing: the following columns:\n {col_pres}")

    # General
    if "potential_energy" in mol_db.columns:
        mol_db["potential_energy"] = mol_db["potential_energy"].astype(float)

    if "rel_potential_energy" in mol_db.columns:
        mol_db["rel_potential_energy"] = mol_db["rel_potential_energy"].astype(float)

    if "boltzmann_ensemble_weight" in mol_db.columns:
        mol_db["boltzmann_ensemble_weight"] = mol_db["boltzmann_ensemble_weight"].astype(float)

    if "conf_id" in mol_db.columns:
        mol_db["conf_id"] = mol_db["conf_id"].astype(int)

    # Torsion Profile Related
    if "torsion_atom_ids" in mol_db.columns:
        if any(c in mol_db["torsion_atom_ids"].iloc[0] for c in ["(", "["]):
            mol_db["torsion_atom_ids"] = mol_db["torsion_atom_ids"].apply(
                lambda x: tuple(ast.literal_eval(x)))
        else:
            mol_db["torsion_atom_ids"] = mol_db["torsion_atom_ids"].apply(
                lambda x: tuple(map(int, x.split()))
            )

    if "torsion_atom_ids_b" in mol_db.columns:
        if any(c in mol_db["torsion_atom_ids_b"].iloc[0] for c in ["(", "["]):
            mol_db["torsion_atom_ids_b"] = mol_db["torsion_atom_ids_b"].apply(
                lambda x: tuple(ast.literal_eval(x)))
        else:
            mol_db["torsion_atom_ids_b"] = mol_db["torsion_atom_ids_b"].apply(
                lambda x: tuple(map(int, x.split()))
            )

    if "torsion_angle" in mol_db.columns:
        mol_db["torsion_angle"] = mol_db["torsion_angle"].astype(float)

    if "torsion_angle_b" in mol_db.columns:
        mol_db["torsion_angle_b"] = mol_db["torsion_angle_b"].astype(float)

    return mol_db


def store_mol_db(
    df_mols: pd.DataFrame,
    out_sdf_path: str,
    _conf_range=None,
) -> str:
    """
    This function does store a torsion profile mol from this package in a sdf file with all
    conformers.

    Parameters
    ----------
    mol : Chem.Mol
        mol containing multiple conformers in the rdkit mol
    out_sdf_path : str
        out file path for the molecule structures
    torsion_atom_ids : tuple[int, int, int, int], optional
        ids of the torison of interest. if provided the dihedral angles will be added.,
        by default None


    Returns
    -------
    str
        out_sdf_path
    """

    if "potential_energy" in df_mols.columns:
        df_mols = add_potential_energy_terms(df_mols)

    ref_mol = df_mols["ROMol"].iloc[0]
    for i, r in df_mols.iterrows():
        mol = r["ROMol"]

        #torsion_profile related
        if "torsion_atom_ids" in df_mols.columns:
            torsion_atom_ids = r["torsion_atom_ids"]

            atom_map = [(i, i) for i in torsion_atom_ids[:3]]
            Chem.rdMolAlign.AlignMol(prbMol=mol, refMol=ref_mol, atomMap=atom_map)

            angle = np.round(
                Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *torsion_atom_ids),
                2)

            mol.SetProp("torsion_angle", str(angle))
            mol.SetProp("torsion_atom_ids", str(torsion_atom_ids))
            r["torsion_angle"] = angle
            r["torsion_atom_ids"] = str(torsion_atom_ids)
        else:
            Chem.rdMolAlign.AlignMol(prbMol=mol, refMol=ref_mol)

    PandasTools.WriteSDF(df_mols, out_sdf_path, molColName='ROMol', idName="molID",
                         properties=df_mols.columns,
                         )

    return out_sdf_path


def mols_to_moldf(mols) -> pd.DataFrame:
    if isinstance(mols, Chem.Mol):
        mols = [mols]
    mol_db = pd.DataFrame([m.GetPropsAsDict() for m in mols])
    mol_db["ROMol"] = mols
    mol_db["smiles"] = mol_db["ROMol"].apply(lambda m: Chem.MolToSmiles(m))

    mol_names = []
    for i, mol in enumerate(mols):
        if mol.HasProp("molID") and mol.GetProp("molID") != "":
            mol_names.append(mol.GetProp("molID"))
        elif mol.HasProp("_Name") and mol.GetProp("_Name") != "":
            mol_names.append(mol.GetProp("_Name"))
        else:
            mol_names.append(f"mol_{i}")
    mol_db["molID"] = mol_names

    if "torsion_atom_ids" in mol_db.columns:
        if any(c in mol_db["torsion_atom_ids"].iloc[0] for c in ["(", "["]):
            mol_db["torsion_atom_ids"] = mol_db["torsion_atom_ids"].apply(
                lambda x: tuple(ast.literal_eval(x)))
        elif isinstance(mol_db["torsion_atom_ids"].iloc[0], str):
            mol_db["torsion_atom_ids"] = mol_db["torsion_atom_ids"].apply(
                lambda x: tuple(map(int, x.split()))
            )

    if "torsion_atom_ids_b" in mol_db.columns:
        if any(c in mol_db["torsion_atom_ids_b"].iloc[0] for c in ["(", "["]):
            mol_db["torsion_atom_ids_b"] = mol_db["torsion_atom_ids_b"].apply(
                lambda x: tuple(ast.literal_eval(x)))
        elif isinstance(mol_db["torsion_atom_ids_b"].iloc[0], str):
            mol_db["torsion_atom_ids_b"] = mol_db["torsion_atom_ids_b"].apply(
                lambda x: tuple(map(int, x.split()))
            )

    if "potential_energy" in mol_db.columns:
        mol_db["potential_energy"] = mol_db["potential_energy"].astype(float)

    if "rel_potential_energy" in mol_db.columns:
        mol_db["rel_potential_energy"] = mol_db["rel_potential_energy"].astype(float)

    if "botlzmann_weight" in mol_db.columns:
        mol_db["botlzmann_weight"] = mol_db["botlzmann_weight"].astype(float)

    return mol_db


def add_potential_energy_terms(mol_db:pd.DataFrame)->pd.DataFrame:
    arr = mol_db["potential_energy"].astype(float).to_numpy()
    arr = np.nan_to_num(arr, nan=20)
    mol_db["potential_energy"] = np.round(arr, 5)
    min_v = min(mol_db["potential_energy"])
    mol_db["rel_potential_energy"] = mol_db["potential_energy"] - min_v
    b_weights = np.round(
        get_boltzman_ensemble_p((mol_db["rel_potential_energy"] - min_v) * kcal_to_kJ,
                                n=len(mol_db["rel_potential_energy"])), 5)
    mol_db["botlzmann_weight"] = b_weights
    return mol_db
