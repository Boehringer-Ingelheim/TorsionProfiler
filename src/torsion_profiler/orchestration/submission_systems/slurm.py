"""
Implements a "slurm" submission system, that smiply submits jobs to a slurm queue.
"""
import os
import tempfile
import time
from datetime import datetime
from collections import OrderedDict

import tqdm
import pandas as pd



from typing import Union
from ...utils import bash
from .submissionsystem import SubmissionSystem
from .submissionjob import SubmissionJob, ArraySubmissionJob

class Slurm(SubmissionSystem):
    """
    This class handles slurm submission without a queueing system
    """

    _dummy: bool = False
    _refresh_job_queue_list_all_s: int = 60  # update the job-queue list every x seconds
    _wait_job_frequency: int = 30  # s for refreshing:
    _job_queue_time_stamp: datetime
    _array_index_variable = "SLURM_ARRAY_TASK_ID"

    def _type_line(self, x:str)->tuple:
        """
        translates the slurm job lines
        """
        return  (str(x[0]), x[1], x[2], x[3], x[4], x[5], x[6], int(x[7]), x[8])

    def __init__(
        self,
        submission: bool = True,
        nomp: int = 1,
        nmpi: int = 1,
        ngpu: int = 0,
        job_duration: str = "04:00:00",
        verbose: bool = False,
        environment=None,
        conda_env_path: str = None,
        partition: str = None,
    ):
        """

        Parameters
        ----------
        submission: bool
        nomp: int
        nmpi: int
        ngpu: int
        job_duration: str
        verbose: bool
        environment
        conda_env_path: str
        partition:str
        """
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
        self.partition = partition
        self.force_job_after = False

        self._special_array_str = None
        self.prefix_file_list = None
        self._job_queue_list = None
        self._finished_jobs = None
        self._all_job_list = None

    def _solv_sbatch_options(self, sub_job):
        """

        Parameters
        ----------
        sub_job

        Returns
        -------

        """
        # Costruct slurm job
        sbatch_options = OrderedDict()
        sbatch_options["time"] = self._job_duration
        sbatch_options["job-name"] = sub_job.job_name

        if self.partition is not None and isinstance(self.partition, str):
            sbatch_options["partition"] = self.partition
        elif (self.nmpi > 1 or self._nomp > 1) and self.ngpu == 0:
            sbatch_options["partition"] = "cpu"
        elif self.ngpu > 1:
            sbatch_options["partition"] = "gpu"
        else:
            sbatch_options["partition"] = "cpu"

        if self.max_storage is not None:
            sbatch_options["mem"] = self.max_storage

        sbatch_options["cpus-per-task"] = self.nmpi + self._nomp
        if self._nomp > 1:
            sbatch_options["ntasks-per-node"] = self._nomp

        if self.ngpu > 0:
            sbatch_options["gpus"] = self.ngpu
        # sbatch_options["cpus-per-gpu"] = self.ngpu

        if sub_job.queue_after_job_id is not None and not self.force_job_after:
            sbatch_options["dependency"] = "afterok:" + str(sub_job.queue_after_job_id)
        if sub_job.queue_after_job_id is not None and self.force_job_after:
            sbatch_options["dependency"] = "after:" + str(sub_job.queue_after_job_id)

        if sub_job.out_log is not None:
            sbatch_options["output"] = sub_job.out_log

        if sub_job.err_log is not None:
            sbatch_options["error"] = sub_job.err_log
        if self.end_mail:
            sbatch_options["mail-type"] = "END"
        if self.fail_mail:
            sbatch_options["mail-type"] = "FAIL"
        if self.begin_mail:
            sbatch_options["mail-type"] = "BEGIN"
        return sbatch_options

    def submit_to_queue(self, sub_job: SubmissionJob) -> int:
        """
        submitt a job
        """

        # QUEUE checking to not double submit
        if self._block_double_submission and self._submission:
            if self.verbose:
                print("check queue")
            ids = list(self.search_queue_for_jobname(sub_job.job_name).JOBID)

            if len(ids) > 0:
                if self.verbose:
                    print(
                        "\tSKIP - FOUND JOB: \t\t"
                        + "\n\t\t".join(map(str, ids))
                        + "\n\t\t with jobname: "
                        + sub_job.job_name
                    )
                return ids[0]

        # BUild COmmand
        sub_job.command = sub_job.command.strip()  # remove trailing linebreaks

        if self.conda_env_path is not None:
            conda_pref = "conda run --prefix " + self.conda_env_path
        else:
            conda_pref = ""

        if self._nomp >= 1:
            command = (
                f"export OMP_NUM_THREADS={self._nomp};\n"
                + f"{conda_pref}  {sub_job.command}"
            )
        else:
            command = conda_pref + " " + sub_job.command

        # Check Slurm Options
        sbatch_options = self._solv_sbatch_options(sub_job)

        # build submissionFilepath
        orig_dir = os.getcwd()

        if isinstance(sub_job.submit_from_dir, str) and os.path.isdir(sub_job.submit_from_dir):
            os.chdir(sub_job.submit_from_dir)
            self.command_file_path = (
                    sub_job.submit_from_dir + "/job_" + str(sub_job.job_name) + ".sh"
            )
        else:
            self.command_file_path = "./job_" + str(sub_job.job_name) + ".sh"

        submission_string = "sbatch "
        if sub_job.submit_from_file:
            command_file = open(self.command_file_path, "w")
            command_file.write("#!/bin/bash\n")
            for option, value in sbatch_options.items():
                command_file.write("#SBATCH --" + option + "=" + str(value) + "\n")

            if self._special_array_str is not None:
                command_file.write("#SBATCH " + self._special_array_str + "\n")
            if self.prefix_file_list  is not None and self.prefix_file_list is not None:
                command_file.write(self.prefix_file_list + "\n")

            command_file.write("\n")

            command_file.write(command.replace("&& ", ";\n") + ";\n")
            command_file.close()

            command = self.command_file_path
            bash.execute("chmod +x " + self.command_file_path, env=self.environment)
            submission_string += command + " "
        else:
            for option, value in sbatch_options.items():
                submission_string += "--" + str(option) + "=" + str(value) + " "

            if  self._special_array_str is not None:
                submission_string += "#SBATCH " + self._special_array_str + ""

        # finalize string
        if self.verbose:
            print("Submission Command: \t", "".join(command))
        if self.submission:
            try:
                process = bash.execute(
                    command=submission_string, catch_std=True, env=self.environment
                )

                # prefix to jobID
                search_for = "Submitted batch job "
                str_out_lines = process.stdout.readlines()
                std_out = "\n".join(map(lambda x: x.decode(), str_out_lines)).strip()
                start_id = str(std_out).find(search_for) + len(search_for)
                job_id = std_out[start_id:].replace("'\\n", "")

                if str(job_id) == "" and job_id.isalnum():
                    raise ValueError("Did not get at job ID! Got: <" + str(job_id) + ">")
                job_id = int(job_id)

                # next sopt_job is queued with id:
                if self.verbose:
                    print("Process was submitted with ID: \n\t" + str(job_id) + "\n")
                if os.path.exists(orig_dir):
                    os.chdir(orig_dir)

                sub_job.job_id = job_id
                return job_id
            except Exception as e:
                raise ChildProcessError(
                    "could not submit this command: \n" + str(submission_string)) from e
        if self.verbose:
            print("Did not submit: ", command)
        sub_job.job_id = -1
        return -1

    def submit_job_array_to_queue(self, sub_job: ArraySubmissionJob) -> int:
        """
        submitt a job array to slurm
        """
        if sub_job.start_job is not None and sub_job.end_job is not None:
            self._special_array_str = (
                "--array=" + str(sub_job.start_job) + "-" + str(sub_job.end_job)
            )
        elif sub_job.job_array is not None:
            self._special_array_str = "--array=" + str(list(sub_job.start_job))
        else:
            raise ValueError("no fitting array found in array_submission job.")

        if sub_job.jobLim is not None:
            self._special_array_str += "%" + str(sub_job.jobLim)

        self.prefix_file_list = "tasks=(" + " ".join(sub_job.command) + ")"
        sub_job.command = "${tasks[${" + self._array_index_variable + "}]}"
        ret = self.submit_to_queue(sub_job=sub_job)
        # self.prefix_file_list = None
        return ret

    def wait_for_jobs(self, job_ids: list[int], _wait_job_frequency: int = None) -> bool:
        """
        slurm submitter will wait for a job to finish.

        Parameters
        ----------
        job_ids: list[int]
            a list of jobs to wait for
        _wait_job_frequency: int
            how often shall the job list be called in order to retrieve job status.

        Returns
        -------
        bool
            result.

        """
        if isinstance(_wait_job_frequency, int):
            self._wait_job_frequency = _wait_job_frequency

        ori_list_refresh = self._refresh_job_queue_list_all_s
        self._refresh_job_queue_list_all_s = (
            self._wait_job_frequency - 5 if (self._wait_job_frequency - 5 > 0) else 0
        )

        if isinstance(job_ids, int):
            job_ids = [job_ids]

        total_job_num = len(job_ids)
        job_ids = list(set(job_ids))  # only uniques
        self.get_queued_jobs(force_update=True)
        with tqdm.tqdm(list(range(total_job_num)), desc="Waiting for Jobs") as pbar:
            progress = 0
            while True:
                nqueued_jobs = 0
                for job_id in job_ids:
                    jlist = self.search_queue_for_jobid(job_id)
                    nqueued_jobs += len(jlist)
                if nqueued_jobs == 0:
                    pbar.update(total_job_num)
                    time.sleep(2)
                    break

                if progress != total_job_num - nqueued_jobs:
                    progress = total_job_num - nqueued_jobs
                    pbar.update(progress)
                time.sleep(self._wait_job_frequency - 5)

        self._refresh_job_queue_list_all_s = ori_list_refresh
        return True

    def get_queued_jobs(self, force_update: bool = False) -> pd.DataFrame:
        """
            return the queued jobs

        Parameters
        ----------
        force_update: bool

        Returns
        -------
        pd.DataFrame

        """
        # Do we need an update of the job list?
        check_job_list = True
        if hasattr(self, "_job_queue_time_stamp"):
            last_update = datetime.now() - self._job_queue_time_stamp
            check_job_list = last_update.seconds > self._refresh_job_queue_list_all_s

        if check_job_list or force_update:  # if timestamp is to old, update joblist
            self._job_queue_list = pd.DataFrame(
                columns=[
                    "JOBID",
                    "PARTITION",
                    "NAME",
                    "USER",
                    "STATE",
                    "TIME",
                    "START_TIME",
                    "NODES",
                    "NODElist(REASON)",
                ]
            )
            if self._submission:
                try:
                    with tempfile.NamedTemporaryFile() as file:
                        slurm_format = "%i %P %j %u %T %M %S %D %R".split()
                        _ = bash.execute(
                            'squeue -u ${USER} --array -o "'
                            + "|".join(slurm_format)
                            + '" > '
                            + file.name,
                            catch_std=False,
                        )
                        process_out = list(map(lambda x: x.decode(), file.readlines()))
                        file.close()

                    header = process_out[0].strip().split("|")
                    if len(process_out) > 2:  # job list not empty
                        jobs_list = list(
                            map(lambda x: self._type_line(x.strip().split("|")), process_out[1:])
                        )

                        self._job_queue_list = pd.DataFrame(jobs_list, columns=header)
                        self._job_queue_time_stamp = datetime.now()
                except Exception as err:
                    raise OSError("Could not get job_list!") from err

        return self._job_queue_list

    def get_finished_jobs(self) -> pd.DataFrame:
        """
        returns all finished jobs in the sinfo

        Returns
        -------
        pd.DataFrame
        """

        if self._submission:
            try:
                # get output:
                with tempfile.NamedTemporaryFile() as file:
                    _ = bash.execute(
                        "sacct -u ${USER} -o jobid,partition,jobname,user,state,start,end,AllocNodes,"
                        "nodelist -p --truncate --starttime now-2weeks > "
                        + file.name
                    )
                    process_out = list(map(lambda x: x.decode(), file.readlines()))
                    file.close()

                # parse list:
                out_str = list(
                    map(
                        lambda x: x.strip().split("|"),
                        filter(
                            lambda x: not (".extern||" in x or ".batch||" in x or ".0|" in x),
                            process_out,
                        ),
                    )
                )

                header = out_str[0][:-1]
                jobs_list = list(map(lambda x: self._type_line(x), out_str[1:]))
                self._finished_jobs = pd.DataFrame(jobs_list, columns=header)
                self._job_queue_time_stamp = datetime.now()

            except Exception as err:
                raise OSError("Could not get old jobs- job_list!") from err
        else:
            header = []
            self._finished_jobs = pd.DataFrame({}, columns=header)

        return self._finished_jobs

    def get_all_job_info(self) -> pd.DataFrame:
        """
        get all job infos, independent of status.
        Returns
        -------

        """
        # add also old_jobs from x time
        _queued_jobs = self.get_queued_jobs()
        _old_jobs = self.get_finished_jobs()
        _old_jobs.columns = self._job_queue_list.columns
        self._all_job_list = pd.concat([_queued_jobs, _old_jobs])

    def search_queue_for_jobname(self, job_name: str, regex: bool = False) -> list[str]:
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
        self.get_queued_jobs()
        if regex:
            return self._job_queue_list.where(
                self._job_queue_list.NAME.str.match(job_name)
            ).dropna()
        return self._job_queue_list.where(self._job_queue_list.NAME == job_name).dropna()

    def search_queue_for_jobid(self, job_id: int) -> pd.DataFrame:
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
        self.get_queued_jobs()
        job_list = self._job_queue_list.where(self._job_queue_list.JOBID == job_id).dropna()
        if len(job_list) == 0:
            job_list = self._job_queue_list.where(
                self._job_queue_list.JOBID == str(job_id)
            ).dropna()
            if len(job_list) == 0:  # save for job_arrays
                tmpjob_ids = self._job_queue_list.JOBID.apply(lambda x: x.split("_")[0])
                tmp_jobs_list = pd.DataFrame(self._job_queue_list)
                tmp_jobs_list.JOBID = tmpjob_ids
                job_list = tmp_jobs_list.where(tmp_jobs_list.JOBID == str(job_id)).dropna()

        return job_list

    def kill_jobs(
        self, job_name: str = None, regex: bool = False, job_ids: Union[list[int], int] = None
    ):
        """
            this function can be used to terminate or remove pending jobs from the queue.
        Parameters
        ----------
        job_name : str
            name of the job to be killed
        regex : bool
            if true, all jobs matching job_name get killed!
        job_ids : Union[list[int], int]
            job Ids to be killed
        """

        if job_name is not None:
            job_ids = list(self.search_queue_for_jobname(job_name, regex=regex).index)
        elif job_ids is not None:
            if isinstance(job_ids, int):
                job_ids = [job_ids]
        else:
            raise ValueError("Please provide either job_name or job_ids!")

        if self.verbose:
            print("Stopping: " + ", ".join(map(str, job_ids)))
        try:
            bash.execute("scancel " + " ".join(map(str, job_ids)))
        except ChildProcessError as err:
            if any("Job has already finished" in x for x in err.args):
                print("Job has already finished")
            else:
                raise ChildProcessError("could not execute this command:") from err
