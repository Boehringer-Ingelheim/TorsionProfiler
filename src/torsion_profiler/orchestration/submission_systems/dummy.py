"""
This is a dummy submission system for test purposes
"""
import warnings

from .submissionsystem import SubmissionSystem
from .submissionjob import SubmissionJob
from typing import Union


class Dummy(SubmissionSystem):
    """DUMMY
    This SubmissionSystem is for testing submission systems. It basically prints everything out.
    """

    def __init__(
        self,
        verbose: bool = False,
        nomp: int = 1,
        nmpi: int = 1,
        ngpu: int = 0,
        job_duration: str = "24:00",
        submission: bool = True,
        environment=None,
        conda_env_path: str = None,
    ):
        super().__init__(
            verbose=verbose,
            nmpi=nmpi,
            nomp=nomp,
            ngpu=ngpu,
            job_duration=job_duration,
            submission=submission,
            environment=environment,
            conda_env_path=conda_env_path,
        )

    def submit_to_queue(self, sub_job: SubmissionJob) -> Union[int, None]:
        """submit_to_queue
            This function submits a str command to the submission system.
        Parameters
        ----------
        sub_job : SubmissionJob
            submission job parameters
        Returns
        -------
        int, None
            if a job was submitted the jobID is returned else None.
        """
        if self.submission:
            print("\n", sub_job.command, "\n")
            return 0
        print("did not submit")
        return None

    def submit_job_array_to_queue(self, sub_job: SubmissionJob) -> Union[int, None]:
        """submit_jobAarray_to_queue
            this function is submitting
        Parameters
        ----------
        sub_job : SubmissionJob
            submission job parameters
        Returns
        -------
        int, None
            if a job was submitted the job_id is returned else None.
        """
        if self.submission:
            print()
            for job_id in range(sub_job.start_job, sub_job.end_job + 1):
                print("Job " + str(job_id) + ":", sub_job.command, "\n")
            print()
            return 0
        print("did not submit")
        return None


    def search_queue_for_jobname(self, job_name: str, **kwargs) -> list[str]:
        """search_queue_for_jobname
            this jobs searches the job queue for a certain job id.
        Parameters
        ----------
        job_name :  str
        regex:  bool, optional
            if the string is a Regular Expression
        Returns
        -------
        list[str]
            the output of the queue containing the jobname
        """
        if self.verbose:
            print("Searching job Name: ", job_name)
        warnings.warn("Queue search was called, but no queue present!")
        return []
