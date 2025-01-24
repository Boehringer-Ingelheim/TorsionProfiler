"""
Optimization Test
"""

import tempfile
import numpy as np
import pandas as pd

from torsion_profiler.engines import Mmff94Calculator
from torsion_profiler.utils import bash

from .. import tmp_root_path
from ..commands.conftest import butan, butan_optimization_results
from ...commands._private_functions._base_optimizer_cmds import _base_geom_optimizer

CalculatorClass = Mmff94Calculator


def test_base_optimization_cli_(butan, butan_optimization_results):
    """
    test simple butan optimization cli

    Parameters
    ----------
    butan
    butan_optimization_results

    Returns
    -------

    """
    expected_rel_energy_results = butan_optimization_results[CalculatorClass.__name__]

    calculator = CalculatorClass()
    tmp_dir = tempfile.mkdtemp(
        prefix="tmp_test_opt_cli_1d_local_" + calculator.__class__.__name__, dir=tmp_root_path
    )

    _base_geom_optimizer(
        in_mol=butan,
        out_folder=tmp_dir,
        _calculator=calculator,
        submit_to_queue=False,
        _conda_env=None,
    )

    out_files_tsv = str(tmp_dir) + "/work_optimization.tsv"
    out_files_tab = tmp_dir + "/work_0/work_0.tsv"
    out_files_sdf = tmp_dir + "/work_0/work_0_emin.sdf"

    assert bash.path.isfile(out_files_tsv)
    assert bash.path.isfile(out_files_tab)
    assert bash.path.isfile(out_files_sdf)

    df = pd.read_csv(out_files_tsv, sep="\t")
    print(list(np.round(df.rel_potential_energies.to_numpy(), 5)))

    assert len(expected_rel_energy_results) == len(df.rel_potential_energies)
    assert np.allclose(
        a=expected_rel_energy_results, b=df.rel_potential_energies.to_numpy(), atol=10**-2
    )

    bash.rmtree(tmp_dir)
