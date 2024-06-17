"""
all the test torsionprofile are in here.
"""

from .toy_test import ToyTestSet
from .biaryl_set import RowleyBiarylSet
from .torsionnet_set import TorsionNetSet
from .openff_benchmark import OpenFFFullDataSet

from .systematic_datasets import SystematicSet

root_path = __path__[0]
data_sets = [k for k in sorted(locals().keys()) if (k[0].isupper() and "Set" in k)]
