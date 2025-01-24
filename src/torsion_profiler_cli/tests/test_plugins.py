
"""
This Module tests the plugin implementatoin
"""
import click
from ..plugins import TorsionProfilerCommandPlugin


@click.command("fake")
def fake():
    """
    test fake command
    """
    pass  # -no-cov-  a fake placeholder click subcommand


class TestCommandPlugin:
    """
    Test the Plugin
    """
    def __init__(self):
        self.plugin = None

    def setup(self):
        """
        Build the Plugin
        """
        self.plugin = TorsionProfilerCommandPlugin(
            command=fake, section="Some Section"
        )

    def test_plugin_setup(self):
        """
        test the Plugin
        """
        assert self.plugin.command is fake
        assert isinstance(self.plugin.command, click.Command)
        assert self.plugin.section == "Some Section"
        assert self.plugin.requires_lib == self.plugin.requires_cli
        assert self.plugin.requires_lib == (0, 0, 1)
        assert self.plugin.requires_cli == (0, 0, 1)
