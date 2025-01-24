# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
CLI builds the torsion_profiler main command, allowing acess to sub tools.
"""

import pathlib
import logging
import logging.config

import click
from plugcli.cli import CLI, CONTEXT_SETTINGS
from plugcli.plugin_management import FilePluginLoader

from torsion_profiler_cli.plugins import TorsionProfilerCommandPlugin


class TorsionProfilerCLI(CLI):
    """
    The Torsion Profiler CLI class is the core construct of the CLI.
    """

    COMMAND_SECTIONS = ["TorsionProfiles", "Geometry Optimization", "Analysis"]

    def get_loaders(self):
        """
        get plugin loaders
        """
        commands = str(pathlib.Path(__file__).parent.resolve() / "commands")
        loader = FilePluginLoader(commands, TorsionProfilerCommandPlugin)
        return [loader]

    def get_installed_plugins(self):
        """
        load all installed plugins
        """
        loader = self.get_loaders()[0]
        return list(loader())


_MAIN_HELP = """
This is the command line tool to provide easy access to functionality from
the torsion_profiler Python library.
"""


@click.command(
    cls=TorsionProfilerCLI,
    name="torsion_profiler_cli",
    help=_MAIN_HELP,
    context_settings=CONTEXT_SETTINGS,
)
@click.version_option(version=1.0)
@click.option(
    "--log", type=click.Path(exists=True, readable=True), help="logging configuration file"
)
def main(log):
    """
    this is the main hook
    """
    # Subcommand runs after this is processed.
    # set logging if provided
    if log:
        logging.config.fileConfig(log, disable_existing_loggers=False)


if __name__ == "__main__":  # -no-cov- (useful in debugging)
    main()
