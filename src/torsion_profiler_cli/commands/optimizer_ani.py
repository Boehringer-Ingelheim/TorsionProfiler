# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""ANI Coordinate Optimization::
    this is the command for calculating an ANI optimization
"""

import click
from pandas import DataFrame

from torsion_profiler import conf
from typing import Union
from torsion_profiler.engines import AniCalculator

from torsion_profiler_cli.plugins import TorsionProfilerCommandPlugin
from torsion_profiler_cli.parameters import MOL, OUTPUT_DIR, SUBMIT
from torsion_profiler_cli.commands._private_functions._base_optimizer_cmds import (
    _base_geom_optimizer,
)


@click.command(
    "opt-ani",
    short_help="Run an molecule optimization with ANI2x",
)
@MOL.parameter(required=True, help=MOL.kwargs["help"])
@OUTPUT_DIR.parameter(
    help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./geom_optimization_ANI2x`.",
    default="geom_optimization_ANI2x",
)
@SUBMIT.parameter()
def ani_geom_optimization(
    mol: str, output_dir: str = None, submit_cluster: bool = False
) -> Union[dict, DataFrame]:
    """
    This function is here to implement the ANI-coordinate optimization
    """
    return _base_geom_optimizer(
        in_mol=mol,
        out_folder=output_dir,
        submit_to_queue=submit_cluster,
        _calculator=AniCalculator(),
        _conda_env=conf["conda_calculator_envs"]["AniCalculator"],
    )


PLUGIN = TorsionProfilerCommandPlugin(
    command=ani_geom_optimization,
    section="Geometry Optimization",
)
