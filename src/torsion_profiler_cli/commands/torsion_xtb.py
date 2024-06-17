# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""GFN2-xTB Torsion Profiler:
    this is the command for calculating an GFN2-xTB Torsion Profile
"""

import click
from pandas import DataFrame

from torsion_profiler import conf
from typing import Union
from torsion_profiler.engines import XtbCalculator


from torsion_profiler_cli.plugins import TorsionProfilerCommandPlugin
from torsion_profiler_cli.parameters import (
    MOL,
    TORSIONATOMS,
    TORSIONATOMSB,
    OUTPUT_DIR,
    ALLTORSIONATOMS,
    ALLFRAGMENTEDTORSIONATOMS,
    SUBMIT,
    NMEASUREMENT,
)
from torsion_profiler_cli.commands._private_functions._base_torsion_profile_cmds import (
    _base_torsion_profile,
)


@click.command(
    "tp-xtb",
    short_help="Run a torsionProfile with XTB with GFN2-xTB",
)
@MOL.parameter(required=True, help=MOL.kwargs["help"] + " One sdf file with 1 mol.")
@TORSIONATOMS.parameter(
    required=False, help=TORSIONATOMS.kwargs["help"] + " One tuple of length 4."
)
@TORSIONATOMSB.parameter(
    help=TORSIONATOMSB.kwargs["help"]
    + " One tuple of length 4. Only good in combination with torsionatoms 1."
)
@ALLTORSIONATOMS.parameter(required=False)
@ALLFRAGMENTEDTORSIONATOMS.parameter(required=False)
@OUTPUT_DIR.parameter(
    help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./torsion_profile_cxtb`.",
    default="torsion_profile_GFN2-xTBxtb",
)
@NMEASUREMENT.parameter()
@SUBMIT.parameter()
def xtb_torsion_profile(
    mol: str,
    torsionids: tuple[int, int, int, int],
    torsionidsb: tuple[int, int, int, int],
    all_torsions: bool = False,
    all_fragmented_torsions: bool = False,
    output_dir: str = None,
    n_measurements: bool = 24,
    submit_cluster: bool = False,
) -> Union[dict, DataFrame]:
    """
    This function is here to implement the XTB-torsion profiler
    """
    if submit_cluster:
        n_task_parallel = 8
    else:
        n_task_parallel = 1
    return _base_torsion_profile(
        in_mol=mol,
        out_folder=output_dir,
        torsion_atom_ids=torsionids,
        torsion_atom_ids_b=torsionidsb,
        calculator=XtbCalculator(),
        n_measurements=n_measurements,
        n_tasks_parallel=n_task_parallel,
        submit_to_queue=submit_cluster,
        all_torsion_atoms=all_torsions,
        all_fragmented_torsions=all_fragmented_torsions,
        _conda_env=conf["conda_calculator_envs"]["XtbCalculator"],
    )


PLUGIN = TorsionProfilerCommandPlugin(
    command=xtb_torsion_profile,
    section="TorsionProfiles",
)
