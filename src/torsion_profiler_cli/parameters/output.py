# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
This file implements the output parameter.
"""

import click
from plugcli.params import Option

def get_file_and_extension(user_input, context):
    """
    split the extension from the file path
    """
    file = user_input
    ext = file.name.split(".")[-1] if file else None
    return file, ext


OUTPUT_FILE_AND_EXT = Option(
    "-o",
    "--output",
    help="output file",
    getter=get_file_and_extension,
    type=click.File(mode="wb"),
)
