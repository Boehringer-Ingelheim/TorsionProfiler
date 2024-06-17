"""
Test Paramter out dir
"""

import os
import pathlib
import importlib

from ...parameters.output_dir import get_dir


def test_get_output_dir():
    """
    test output dir functionality
    """
    with importlib.resources.path("torsion_profiler.tests", "__init__.py") as file_path:
        dir_path = os.path.dirname(file_path)
        out_dir = get_dir(dir_path, None)

        assert isinstance(out_dir, pathlib.Path)
        assert out_dir.parent.exists()
