"""
This module allows the execution of commands in scripts.
"""
from ..submission_systems.submissionsystem import SubmissionSystem
from ..submission_systems.submissionjob import SubmissionJob
from ..submission_systems.local import Local

from ...utils import bash
from ...utils.utils import write_job_script

from ._computing_envs import _ComputingEnv


class ScriptComputing(_ComputingEnv):
    """
    Script Computing gives access to execute files form the os environment.
    """
    submission_system: SubmissionSystem
    n_tasks: int

    def __init__(
        self,
        submission_system: SubmissionSystem = Local(),
        n_tasks: int = 1,
        verbose: bool = False,
    ) -> None:
        """
        Construct the class

        Parameters
        ----------
        submission_system : _Submission_system, optional
            submission system under the hood, submitting the job, by default Local()
        n_tasks : int, optional
            number of tasks in parallel/jobs, by default 1
        verbose : bool, optional
            blubb, by default False
        """
        super().__init__(verbose=verbose)

        self.submission_system = submission_system
        self.n_tasks = n_tasks

    def _setup_calculation(
        self,
        out_root_dir: str,
        out_file_prefix: str,
        in_target_function: callable,
        in_function_parameters: dict[str, any],
    ) -> str:
        """
            This function writes out the required scripts and makes folders for code execution

        Parameters
        ----------
        out_root_dir : str
            output folder for code execution
        out_file_prefix : str
            prefixes for files
        in_target_function : callable
            function, that shall be executed
        in_function_parameters : dict[str, any]
            parameters for that featueres

        Returns
        -------
        str
            job script path for execution
        """
        # setup out folder
        if not bash.path.isdir(out_root_dir):
            bash.makedirs(out_root_dir)

        # setup job script
        out_job_path = write_job_script(
            out_script_path=out_root_dir + "/" + out_file_prefix + "_calculate.py",
            target_function=in_target_function,
            variable_dict=in_function_parameters,
            out_rdkitMol_prefix=out_file_prefix,
        )

        return out_job_path

    # execute:

    def run(
        self,
        in_target_function: callable,
        in_function_parameters: dict[str, any],
        out_root_dir: str,
        out_file_prefix: str,
        queue_after_jobID: int = None,
        verbose: bool = False,
    ) -> int:
        """
        run the function in the way, that scripts are written out and executed.

        Parameters
        ----------
        out_root_dir : str
            path to the dir, in which the code is written to and executed
        out_file_prefix : str
            filename prefixes
        in_target_function : callable
            function to be executed
        in_function_parameters : dict[str, any]
            fuynction parameters kargs
        queue_after_jobID : int, optional
            queue the execution of the job after this id, by default None
        verbose : bool, optional
            nananana, by default False

        Returns
        -------
        int
            job-id or any other int.
        """
        if verbose:
            print("prepare folder")
        in_script_path = self._setup_calculation(
            out_root_dir=out_root_dir,
            out_file_prefix=out_file_prefix,
            in_target_function=in_target_function,
            in_function_parameters=in_function_parameters,
        )

        if verbose:
            print("prepare submission")
        script_name = bash.path.basename(in_script_path)
        if self.verbose and hasattr(self.submission_system, "verbose"):
            self.submission_system.verbose = self.verbose

        job = SubmissionJob(
            command=in_script_path,
            job_name=out_file_prefix,
            out_log=out_root_dir + "/" + script_name.replace(".py", ".log"),
            err_log=out_root_dir + "/" + script_name.replace(".py", ".err"),
            submit_from_dir=out_root_dir,
            submit_from_file=True,
            queue_after_job_id=queue_after_jobID,
        )

        job_id = self.submission_system.submit_to_queue(job)

        return job_id
