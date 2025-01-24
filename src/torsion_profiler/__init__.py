"""
This is the top hierarchy level of torsion_profiler a package for geometry optimizations and
torsion profiles
"""
# Useful Namespace
import json

from . import engines
from .utils import bash
from .orchestration import submission_systems
from .tools import TorsionProfiler, GeomOptimizer
from .visualize import visualize_torsion_profiles

root_path = bash.path.abspath(bash.path.dirname(__file__))

conf = json.load(open(root_path + "/conf.json", "r"))
