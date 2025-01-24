# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
this file defines how a plugin has to looklike for our CLI
"""

from plugcli.plugin_management import CommandPlugin


class TorsionProfilerCommandPlugin(CommandPlugin):
    """
    Plugin base class
    """

    def __init__(
        self,
        command,
        section,
    ):  # requires_ofe
        super().__init__(command=command, section=section, requires_lib=None, requires_cli=None)
