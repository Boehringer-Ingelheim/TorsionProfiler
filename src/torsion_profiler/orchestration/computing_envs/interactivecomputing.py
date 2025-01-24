"""
interactive computing, allows to use the tools directly in the same python session
"""
from ._computing_envs import _ComputingEnv


class InteractiveComputing(_ComputingEnv):
    """
    An executer for the python function inside the python session

    """
    def run(
        self,
        in_target_function: callable,
        in_function_parameters: dict[str, any],
        out_root_dir: str = None,
        out_file_prefix: str = None,
    ) -> any:
        """
        This is an interactive computing environment interface layer.
        it allows easy switching between script base executions and interactive ones.

        Parameters
        ----------
        in_target_function : callable
            function, that should be executed.
        in_function_parameters : dict[str, any]
            parameters for the function as a dict
        out_root_dir : str, optional
            not used, by default None
        out_file_prefix : str, optional
            not used, by default None

        Returns
        -------
        any
            whatever is returned
        """

        return in_target_function(**in_function_parameters)
