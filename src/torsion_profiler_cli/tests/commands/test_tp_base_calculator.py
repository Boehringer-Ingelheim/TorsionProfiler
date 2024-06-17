"""
This Module test the torsion profile cli
"""
import tempfile
import numpy as np
import pandas as pd

from torsion_profiler.engines import Mmff94Calculator
from torsion_profiler.utils import bash
from torsion_profiler.utils import read_mol_db

from ...tests import tmp_root_path
from ...commands._private_functions._base_torsion_profile_cmds import _base_torsion_profile
from .conftest import butan, butan_tp_1d_5n_results

CalculatorClass = Mmff94Calculator


def test_base_torsion_profile_cli(butan, butan_tp_1d_5n_results):
    """
    test torsion profile calculation

    Parameters
    ----------
    butan
    butan_tp_1d_5n_results

    Returns
    -------

    """
    expected_rel_energy_results = butan_tp_1d_5n_results[CalculatorClass.__name__]
    expected_angles = butan_tp_1d_5n_results["angles"]
    n_measurements = butan_tp_1d_5n_results["n_measurements"]

    calculator = CalculatorClass()
    tmp_dir = tempfile.mkdtemp(
        prefix="tmp_test_tp_cli_1d_local_" + calculator.__class__.__name__, dir=tmp_root_path
    )
    torsion_atom_ids = butan.GetProp("ta")
    butan.ClearProp("ta")
    butan.ClearProp("tb")

    magic_prop = "FUN"
    butan.SetProp("Magic_Prop", magic_prop)

    _base_torsion_profile(
        in_mol=butan,
        out_folder=tmp_dir,
        torsion_atom_ids=torsion_atom_ids,
        calculator=calculator,
        n_measurements=n_measurements,
        submit_to_queue=False,
        n_tasks_parallel=1,
        n_processors=5,
        all_torsion_atoms=False,
        _conda_env=None,
        _wait_for_jobs=True,
        torsion_atom_ids_b=None,
    )
    out_file_prefix = tmp_dir + "/" + bash.path.basename(tmp_dir)
    out_files_tsv = out_file_prefix + "_torsion_profiles.tsv"
    out_files_plot = out_file_prefix + "_torsion_profile.png"
    out_files_sdf = out_file_prefix + "_optimized_torsion_profile.sdf"

    assert bash.path.isfile(out_files_tsv)
    assert bash.path.isfile(out_files_plot)
    assert bash.path.isfile(out_files_sdf)

    # Check Output Files:
    # Check CSV
    df = pd.read_csv(out_files_tsv, sep="\t")

    assert butan_tp_1d_5n_results["n_measurements"] == len(df.rel_potential_energies)
    assert np.allclose(
        a=expected_rel_energy_results, b=df.rel_potential_energies.to_numpy(), atol=10**-3
    )
    assert np.allclose(a=expected_angles, b=df.angles.to_numpy(), atol=10**-2)

    # Check SDF
    mol = read_mol_db(out_files_sdf)

    assert mol.GetProp("_Name") == butan.GetProp("_Name")
    assert mol.GetProp("torsion_atom_ids") == " ".join(
        map(lambda x: str(int(x) - 1), torsion_atom_ids.split())
    )
    assert mol.GetProp("Magic_Prop") == magic_prop

    for c in mol.GetConformers():
        print(c.GetPropsAsDict())

    # bash.rmtree(tmp_dir)
