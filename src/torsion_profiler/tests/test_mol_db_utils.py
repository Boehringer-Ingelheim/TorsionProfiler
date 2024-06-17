"""
Test utilities here
"""
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem

from torsion_profiler.tests.conftest import butan_torsion_profile_sdf_path
from torsion_profiler.utils import bash
from torsion_profiler.utils import read_mol_db, store_mol_db


def test_read_mol():
    """
    Test reading a torsion profile
    """
    df_tp = read_mol_db(butan_torsion_profile_sdf_path)

    assert df_tp.shape[0] == 37
    assert df_tp["mol_name"][0] == "Budan"
    assert df_tp["torsion_atom_ids"][0] == (0, 1, 2, 3)

    angles = np.linspace(-180, 180, 37)
    got_angles =df_tp["torsion_angle"].to_numpy()
    np.testing.assert_almost_equal(got_angles, angles, decimal=2)
    assert isinstance(df_tp["ROMol"][0], Chem.Mol)


def test_write_mol():
    """
    Test writing a torsion profile with a roundtrip.
    """

    df_tp = read_mol_db(butan_torsion_profile_sdf_path)

    with tempfile.NamedTemporaryFile(suffix=".sdf") as tf:
        out_sdf_path = store_mol_db(df_mols=df_tp, out_sdf_path=tf.name)

        assert tf.name == out_sdf_path
        assert bash.path.isfile(out_sdf_path)

        # Roundtrip test
        df_tp2 = read_mol_db(out_sdf_path)

        df_tp_short = df_tp[['Magic_Prop', 'torsion_atom_ids', 'mol_name', 'torsion_angle',
       'conf_id', 'potential_energy', 'relative_potential_energy',
       'boltzmann_ensemble_weight', 'smiles', 'molID',]]
        df_tp2_short = df_tp2[['Magic_Prop', 'torsion_atom_ids', 'mol_name', 'torsion_angle',
       'conf_id', 'potential_energy', 'relative_potential_energy',
       'boltzmann_ensemble_weight', 'smiles', 'molID',]]
        pd.testing.assert_frame_equal(df_tp_short, df_tp2_short)
