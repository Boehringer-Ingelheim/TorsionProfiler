# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
This Module implements the CLI tool for Torsion_Profiler
"""

from importlib.metadata import version

from .plugins import TorsionProfilerCommandPlugin
from . import commands
