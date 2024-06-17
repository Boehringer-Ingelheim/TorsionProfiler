# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""MMFF94 Torsion Profiler:
    this is the command for calculating an MMFF94 Torsion Profile
"""

import click
from pandas import DataFrame

from torsion_profiler import conf
from typing import Union
from torsion_profiler.engines import Mmff94Calculator

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
    "tp-mmff94",
    short_help="Run a torsionProfile with MMFF94",
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
    help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./torsion_profile_mmff94`.",
    default="torsion_profile_mmff94",
)
@NMEASUREMENT.parameter()
@SUBMIT.parameter()
def mmff94_torsion_profile(
    mol: str,
    torsionids: tuple[int, int, int, int],
    torsionidsb: tuple[int, int, int, int],
    all_torsions: bool = False,
    all_fragmented_torsions: bool = False,
    output_dir: str = None,
    n_measurements: bool = 24,
    n_processors: int = 5,
    submit_cluster: bool = False,
    ) -> Union[dict, DataFrame]:
    """
    This function is here to implement the MMFF94-torsion profiler
    """
    return _base_torsion_profile(
        in_mol=mol,
        out_folder=output_dir,
        torsion_atom_ids=torsionids,
        torsion_atom_ids_b=torsionidsb,
        n_measurements=n_measurements,
        submit_to_queue=submit_cluster,
        all_torsion_atoms=all_torsions,
        all_fragmented_torsions=all_fragmented_torsions,
        n_processors=n_processors,
        calculator=Mmff94Calculator(),
        _conda_env=conf["conda_calculator_envs"]["Mmff94Calculator"],
    )


PLUGIN = TorsionProfilerCommandPlugin(
    command=mmff94_torsion_profile,
    section="TorsionProfiles",
)
