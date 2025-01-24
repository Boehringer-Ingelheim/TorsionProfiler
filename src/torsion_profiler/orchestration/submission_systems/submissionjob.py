"""
In this module all job types for submission are defined.
"""

class SubmissionJob:
    """
    Description:
    This class stores parameters for the submission of jobs. It is used by the submission_systems:
    - submission dummy
    - submission local
    - submission lsf
    It should handle all the information required for a single job, while the submission_systems
    handles more permanent settings.
    It should provide an easy way to modify jobs, even from high level modules (e.g. the simulation
    module like TI, EMIN, ...).
    Author: Marc Lehner
    """

    def __init__(
        self,
        command: str,
        job_name: str,
        out_log: str = None,
        err_log: str = None,
        queue_after_job_id: int = None,
        post_execution_command: str = None,
        submit_from_dir: str = None,
        submit_from_file: bool = True,
        job_group: str = None,
        job_id=None,
    ) -> None:
        self._command = command
        self._job_name = job_name
        self._out_log = out_log
        self._err_log = err_log
        self._queue_after_job_id = queue_after_job_id
        self._post_execution_command = post_execution_command
        self._submit_from_dir = submit_from_dir
        self._submit_from_file = submit_from_file
        self._job_group = job_group
        self._job_id = job_id

    def __repr__(self) -> str:
        msg = self.__class__.__name__ + "("
        for var in vars(self):
            msg += " " + var + "=" + str(getattr(self, var)) + ","
        msg += ")"
        return msg

    @property
    def command(self) -> str:
        """
        getter for command to be executed.
        """
        if self._command is None:
            raise ValueError("command not set")
        return self._command

    @command.setter
    def command(self, command: str) -> None:
        """
        setter for command to be executed.
        """
        if isinstance(command, str):
            self._command = command
        else:
            raise ValueError("command must be a string")

    @property
    def job_name(self) -> str:
        """
        getter for job_name
        """
        if self._job_name is None:
            return "test"
        return self._job_name

    @property
    def out_log(self) -> str:
        """
        getter for out log path
        """
        return self._out_log

    @property
    def err_log(self) -> str:
        """
        getter for err log path
        """
        return self._err_log

    @property
    def queue_after_job_id(self) -> int:
        """
        getter queue after this job
        """
        return self._queue_after_job_id

    @property
    def post_execution_command(self) -> str:
        """
        getter for post execution command str
        """
        return self._post_execution_command

    @property
    def submit_from_file(self) -> bool:
        """
        getter if the job shall be submitted form a file
        """
        return self._submit_from_file

    @property
    def submit_from_dir(self) -> str:
        """
        getter for path where to submit form
        """
        return self._submit_from_dir

    @property
    def job_group(self) -> str:
        """
        getter for the job group str
        """
        return self._job_group

    @property
    def job_id(self) -> int:
        """
        getter for the job id
        """
        return self._job_id

    @job_id.setter
    def job_id(self, jobID: int) -> None:
        """
        setter for the jobid
        """
        if isinstance(jobID, int):
            self._job_id = jobID
        else:
            raise ValueError("jobID must be an int")


class ArraySubmissionJob(SubmissionJob):
    """
    Implementation for Array jobs.
    """
    def __init__(
        self,
        commands: list[str] = None,
        job_name: str = None,
        out_log: str = None,
        err_log: str = None,
        start_job: int = None,
        end_job: int = None,
        job_lim: int = None,
        queue_after_job_id: int = None,
        post_execution_command: str = None,
        submit_from_dir: str = None,
        submit_from_file: bool = True,
        job_group: str = None,
        job_id=None,
    ) -> None:
        """
        Instantiate a Job Array job.

        Parameters
        ----------
        commands: str
        job_name: str
        out_log: str
        err_log: str
        start_job: int
        end_job: int
        job_lim: int
        job_array: list
        queue_after_job_id: int
        post_execution_command: str
        submit_from_dir: str
        submit_from_file: str
        job_group: str
        job_id: int
        """
        super().__init__(
            command=commands,
            job_name=job_name,
            out_log=out_log,
            err_log=err_log,
            queue_after_job_id=queue_after_job_id,
            submit_from_dir=submit_from_dir,
            submit_from_file=submit_from_file,
            job_group=job_group,
            job_id=job_id,
        )

        self._start_job = start_job
        self._end_job = end_job
        self._job_lim = job_lim

        self._post_execution_command = post_execution_command

    @property
    def start_job(self) -> int:
        """
        getter for index of the first job
        """
        return self._start_job

    @property
    def end_job(self) -> int:
        """
        getter for index of the last job
        """
        return self._end_job

    @property
    def commands(self) -> list[str]:
        """
        getter the list of jobs
        """
        return list(self._command)

    @property
    def jobLim(self) -> int:
        """
        getter for  how many jobs at the same timepoint
        """
        return self._job_lim

    @jobLim.setter
    def jobLim(self, jobLim: int) -> None:
        """
        setter for  how many jobs at the same timepoint
        """
        if isinstance(jobLim, int):
            self._job_lim = jobLim
        else:
            raise ValueError("jobLim must be an int")
