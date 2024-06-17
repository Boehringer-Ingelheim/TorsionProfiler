# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
This file implements output dir parameter.
"""

import pathlib

import click
from plugcli.params import Option


def get_dir(user_input, context):
    """
    Get output directory
    """
    dir_path = pathlib.Path(user_input)

    return dir_path


OUTPUT_DIR = Option(
    "-o",
    "--output-dir",
    help="Path to the output directory. ",
    getter=get_dir,
    type=click.Path(file_okay=False, resolve_path=True),
)
