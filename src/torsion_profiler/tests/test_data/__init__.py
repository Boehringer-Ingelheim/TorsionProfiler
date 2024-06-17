"""
Here are a few test files
"""

from torsion_profiler.utils import bash

test_data_dir = bash.path.dirname(__file__)

butan_torsion_profile_sdf_path = test_data_dir + "/butan_tp.sdf"
butan_torsion_profile_tsv_path = test_data_dir + "/butan_tp.tsv"
