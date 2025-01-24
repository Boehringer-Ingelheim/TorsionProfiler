"""
Test Paramter out path
"""

import pytest

from ...parameters.output import get_file_and_extension


@pytest.mark.parametrize(
    "fname,expected_ext",
    [
        ("foo.bar", "bar"),
        ("foo.bar.bz", "bz"),
    ],
)
def test_get_file_and_extension(tmpdir, fname, expected_ext):
    """
    test output file
    """
    with open(tmpdir / fname, mode="w") as file:
        outfile, ext = get_file_and_extension(file, {})
        assert outfile is file
        assert ext == expected_ext
