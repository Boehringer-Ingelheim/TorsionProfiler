"""
this class is a interface abstract class
Todo: consider refactoring
"""
import abc


class _ComputingEnv(abc.ABC):
    def __init__(self, verbose: bool = False) -> None:
        """
        Abstract class/ interface

        Parameters
        ----------
        verbose : bool, optional
            blubb blubb, by default False
        """
        self.verbose = verbose

    def run(
        self,
        in_target_function: callable,
        in_function_parameters: dict[str, any],
        out_root_dir: str = None,
        out_file_prefix: str = None,
        *args,
        **kwargs
    ) -> any:
        """
        needs to be implemented in every child!

        Parameters
        ----------
        in_target_function : callable
            a function, that should be executed
        in_function_parameters : dict[str, any]
            the parameters for the function as kwargs

        Returns
        -------
        any
            result of execution

        Raises
        ------
        NotImplementedError
            Needs to be implemented in child classes
        """
        raise NotImplementedError("This function needs to be implemented!")
