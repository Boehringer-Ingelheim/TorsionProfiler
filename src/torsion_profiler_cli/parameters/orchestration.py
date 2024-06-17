# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
This file implements the submit to slurm queue flag.
"""

from plugcli.params import Option

SUBMITSLURM = Option(
    "-s",
    "--submit-cluster",
    help=("submit to cluster."),
    count=True,
    required=False,
    default=False,
)
