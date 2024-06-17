"""
Test module of torsion_profiler
"""
# test import
import tempfile
from ..utils import bash

tmp_root_path = tempfile.mkdtemp(prefix="tmp_out_torsion_profiler_tests",
                                 dir=f"{bash.path.dirname(__file__)}/test_data")
