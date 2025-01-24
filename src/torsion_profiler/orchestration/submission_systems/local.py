"""
Implements a "local" submission system, that smiply executes the commands directly in the shell.
"""
import os
import warnings
import pandas as pd
from tqdm import tqdm
from .submissionsystem import SubmissionSystem
from .submissionjob import SubmissionJob
from ...utils import bash


class Local(SubmissionSystem):
    """
    This class handles local submission without a queueing system
    """

    _array_index_variable: str = "JOBID"

    def __init__(
        self,
        submission: bool = True,
        nomp: int = 1,
        nmpi: int = 1,
        ngpu: int = 0,
        job_duration: str = "24:00",
        verbose: bool = False,
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
        self.command_file_path = None
        self.prefix_file_list = None

    def submit_to_queue(self, sub_job: SubmissionJob) -> int:
        """
        submitt a local job
        """

        orig_dir = os.getcwd()

        if isinstance(sub_job.submit_from_dir, str) and os.path.isdir(sub_job.submit_from_dir):
            os.chdir(sub_job.submit_from_dir)
            self.command_file_path = (
                    sub_job.submit_from_dir + "/job_" + str(sub_job.job_name) + ".sh"
            )
        else:
            self.command_file_path = "./job_" + str(sub_job.job_name) + ".sh"

        if self.conda_env_path is not None:
            conda_pref = "conda run --prefix " + self.conda_env_path
        else:
            conda_pref = ""

        sub_job.command = sub_job.command.strip()  # remove trailing linebreaks

        if self._nomp >= 1:
            command = (
                "export OMP_NUM_THREADS="
                + str(self._nomp)
                + ";\n"
                + conda_pref
                + " "
                + sub_job.command
                + ""
            )
        else:
            command = conda_pref + " " + sub_job.command

        if sub_job.submit_from_file:
            with open(self.command_file_path, "w") as command_file:
                command_file.write("#!/bin/bash\n")
                command_file.write(command.replace("&& ", ";\n") + ";\n")
                command_file.close()

            command = self.command_file_path
            bash.chmod(self.command_file_path, bash.chmod_755)

        if self.conda_env_path is not None:
            command = "conda run --prefix " + self.conda_env_path + " " + command

        # finalize string
        if sub_job.out_log is not None:
            command = command + f"> {sub_job.out_log}"

        if self.verbose:
            print("Submission Command: \t", " ".join(command))
        if self.submission:
            try:
                process = bash.execute(command=command, catch_std=True, env=self.environment)
                std_out_buff = map(str, process.stdout.readlines())
                std_out = "\t" + "\n\t".join(std_out_buff)

                # next sopt_job is queued with id:
                if self.verbose:
                    print("STDOUT: \n\t" + std_out + "\nEND\n")
                if os.path.exists(orig_dir):
                    os.chdir(orig_dir)

                return 0
            except ChildProcessError as err:
                raise ChildProcessError(
                    "command failed: \n" + str(command)) from err
        else:
            # print("Did not submit: ", command)
            return -1

    def submit_job_array_to_queue(self, sub_job: SubmissionJob) -> int:
        """
        submitt a local job array
        """

        # generate submission_string:
        submission_string = ""

        if isinstance(sub_job.submit_from_dir, str) and os.path.isdir(sub_job.submit_from_dir):
            submission_string += "cd " + sub_job.submit_from_dir + " && "

        command = submission_string
        if self._nomp > 1:
            command += " export OMP_NUM_THREADS=" + str(self._nomp) + " && "

        self.prefix_file_list = "tasks=(" + " ".join(sub_job.command) + ")"
        sub_job_cmd = "eval ${tasks[${" + self._array_index_variable + "}]}"
        command += sub_job_cmd
        self.command_file_path = sub_job.command

        if self.submission:
            try:
                iterator = list(range(sub_job.start_job, sub_job.end_job + 1))
                magnitude_mol = 10 ** len(str(len(iterator)))
                for job_id in tqdm(
                    iterator, desc="Executing jobs", mininterval=len(iterator) // 10**magnitude_mol
                ):
                    sub_cmd = (
                        f"export {self._array_index_variable}={job_id}; {self.prefix_file_list};"
                        + command
                    )
                    if self.verbose:
                        print("Submission Command: \t", command)
                    p = bash.execute(command=sub_cmd, env=self.environment, catch_std=True)
                    if self.verbose:
                        lines = "\n".join(map(lambda x: x.decode(), p.stdout.readlines())).strip()
                        print("sdtout : " + str(lines))
                return 0
            except ChildProcessError as exc:
                raise ChildProcessError(
                    'could not submit this command: \n' + submission_string) from exc
        else:
            # print("Did note submit: ", command)
            return -1

    def search_queue_for_jobname(self, job_name: str, **kwargs) -> list[str]:
        """search_queue_for_jobname
            this jobs searches the job queue for a certain job name.
            DUMMY FUNCTION!
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

    def search_queue_for_jobid(self, job_id: int, **kwargs) -> pd.DataFrame:
        """search_queue_for_jobid
            this jobs searches the job queue for a certain job id.
            DUMMY FUNCTION!
        Parameters
        ----------
        job_id :  int
            id of the job
        Raises
        -------
        NotImplemented
            Needs to be implemented in subclasses
        """
        if self.verbose:
            print("Searching job ID: ", job_id)
            warnings.warn("Queue search was called, but no queue present!")
        return []
