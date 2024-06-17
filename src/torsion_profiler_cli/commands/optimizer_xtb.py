#!/home/riesbenj/.conda/envs/aniEnvFrag/bin/python

"""GFN2 Coordinate Optimization:
    this is the command for calculating an GFM2 optimization
"""


import click
from pandas import DataFrame

from torsion_profiler import conf
from typing import Union
from torsion_profiler.engines import XtbCalculator

from torsion_profiler_cli.plugins import TorsionProfilerCommandPlugin
from torsion_profiler_cli.parameters import MOL, OUTPUT_DIR, SUBMIT
from torsion_profiler_cli.commands._private_functions._base_optimizer_cmds import (
    _base_geom_optimizer,
)


@click.command(
    "opt-GFN2",
    short_help="Run an molecule optimization with GFN2 - xTB",
)
@MOL.parameter(required=True, help=MOL.kwargs["help"])
@OUTPUT_DIR.parameter(
    help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./geom_optimization_GFN2`.",
    default="geom_optimization_GFN2",
)
@SUBMIT.parameter()
def xtb_geom_optimization(
    mol: str, output_dir: str = None, submit_cluster: bool = False
) -> Union[dict, DataFrame]:
    """
    This function is here to implement the GFN2-coordinate optimization
    """
    return _base_geom_optimizer(
        in_mol=mol,
        out_folder=output_dir,
        submit_to_queue=submit_cluster,
        _calculator=XtbCalculator(),
        _conda_env=conf["conda_calculator_envs"]["XtbCalculator"],
    )


PLUGIN = TorsionProfilerCommandPlugin(
    command=xtb_geom_optimization,
    section="Geometry Optimization",
)
