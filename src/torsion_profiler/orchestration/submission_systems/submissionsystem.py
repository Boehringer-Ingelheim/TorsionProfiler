"""
This module implements the base class for Submission systems
"""
import inspect
import pandas as pd

from typing import Union

from .submissionjob import SubmissionJob


class SubmissionSystem:
    verbose: bool
    submission: bool
    nmpi: int
    nomp: int
    max_storage: str
    job_duration: str
    environment: dict
    block_double_submission: bool
    chain_prefix: str
    begin_mail: bool
    end_mail: bool
    job_queue_list: pd.DataFrame  # contains all jobs in the queue (from this users)
    conda_env_path: str

    def __init__(
        self,
        submission: bool = True,
        nmpi: int = 1,
        nomp: int = 1,
        ngpu: int = 0,
        max_storage: str = "5GB",
        job_duration: str = "24:00",
        verbose: bool = False,
        environment=None,
        block_double_submission: bool = True,
        chain_prefix: str = "done",
        begin_mail: bool = False,
        end_mail: bool = False,
        fail_mail: bool = False,
        conda_env_path: str = None,
    ):
        """
            Construct a submission system with required parameters.
        Parameters
        ----------
        submission : bool, optional
            if the job should be submitted? or only a dry run?
        nmpi : int, optional
            number of desired nmpi threads  (default: 1 == No MPI)
        nomp : int, optional
            number of desired omp threads (default: 1 == No OMP)
        max_storage : str, optional
            maximally required storage for a job in mb (default, 1GB)
        job_duration : str, optional
            the duration of the job as str("HHH:MM")  (default: "24:00")
        verbose : bool, optional
            let me write you a book!  (default: False)
        environment: dict, optional
            here you can pass environment variables as dict{varname: value} (default: None)
        block_double_submission: bool, optional
            if a job with the same name is already in the queue, it will not be submitted again.
            (default: True)
        chain_prefix: str, optional
            the mode with witch jobs are chained together (default: "done")
            (options: "done", "exit", "ended", "started", "post_done", "post_err")
        begin_mail: bool, optional
            determines if a mail is sent when job starts
        end_mail: bool, optional
            determines if a mail is sent when job ends
        fail_mail: bool, optional
            determines if a mail is sent when job fails
        """

        self._submission = submission
        self._nmpi = nmpi
        self._nomp = nomp
        self._ngpu = ngpu
        self._max_storage = max_storage
        self._job_duration = job_duration
        self.verbose = verbose
        self._environment = environment
        self._conda_env_path = conda_env_path
        self._block_double_submission = block_double_submission
        self._chain_prefix = chain_prefix
        self._begin_mail = begin_mail
        self._end_mail = end_mail
        self._fail_mail = fail_mail

    def __repr__(self) -> str:
        msg = self.__class__.__name__ + "("
        for var in vars(self):
            msg += " " + var + "=" + str(getattr(self, var)) + ","
        msg += ")"
        return msg

    def submit_to_queue(self, sub_job: SubmissionJob) -> int:
        """
        Stub for submitting a job to the system
        """
        raise NotImplementedError("This function is not implemented!")

    def submit_job_array_to_queue(self, sub_job: SubmissionJob) -> int:
        """
        Stub for submitting a jobArray to the system
        """
        raise NotImplementedError("This function is not implemented!")

    def get_script_generation_command(self, var_name: str = None, var_prefixes: str = "") -> str:
        """
            This command is generating a script, that is used to submit a given command.

        Parameters
        ----------
        var_name
        var_prefixes

        Returns
        -------
        str
            command as a string
        """
        name = self.__class__.__name__
        if var_name is None:
            var_name = var_prefixes + name

        params = []
        for key in inspect.signature(self.__init__).parameters:
            if hasattr(self, key):
                param = getattr(self, key)
                if isinstance(param, str):
                    params.append(key + "='" + str(getattr(self, key)) + "'")
                else:
                    params.append(key + "=" + str(getattr(self, key)))
            elif hasattr(self, "_" + key):
                param = getattr(self, key)
                if isinstance(param, str):
                    params.append(key + "='" + str(getattr(self, "_" + key)) + "'")
                else:
                    params.append(key + "=" + str(getattr(self, "_" + key)))
        parameters_str = ", ".join(params)

        gen_cmd = "#Generate " + name + "\n"
        gen_cmd += "from " + self.__module__ + " import " + name + " as " + name + "_class" + "\n"
        gen_cmd += var_name + " = " + name + "_class(" + parameters_str + ")\n\n"
        return gen_cmd

    def search_queue_for_jobname(
        self, job_name: str, regex: bool = False, **kwargs
    ) -> pd.DataFrame:
        """search_queue_for_jobname
            this function searches the job queue for a certain job name.
        Parameters
        ----------
        job_name :  str
            text part of the job_line
        regex:  bool, optional
            if the string is a Regular Expression
        Raises
        -------
        NotImplementedError
            Needs to be implemented in subclasses
        """

        raise NotImplementedError("Do is not implemented for: " + self.__class__.__name__)


    def search_queue_for_jobid(self, job_id: int, **kwargs) -> pd.DataFrame:
        """search_queue_for_jobid
            this jobs searches the job queue for a certain job id.
        Parameters
        ----------
        job_id :  int
            id of the job
        Raises
        -------
        NotImplementedError
            Needs to be implemented in subclasses
        """

        raise NotImplementedError(
            "search_queue_for_jobID is not implemented for: " + self.__class__.__name__
        )

    def is_job_in_queue(
        self, job_name: str = None, job_id: int = None, _only_runpend: bool = True
    ) -> bool:
        """
        checks wether a function is still in the lsf queue
        Parameters
        ----------
        job_name : str
            name of the job.
        job_id : int
            id of the job.
        verbose : bool, optional
            extra prints, by default False
        Returns
        -------
        bool
            is the job in the lsf queue?
        """
        if job_name is not None:
            if _only_runpend:
                queued_job_ids = self.search_queue_for_jobname(job_name=job_name)
                queued_job_ids = queued_job_ids.where(
                    queued_job_ids.STAT.isin(["RUN", "PEND"])
                ).dropna()
                return len(queued_job_ids) > 0
            return len(self.search_queue_for_jobname(job_name=job_name)) > 0
        elif job_id is not None:
            if _only_runpend:
                queued_job_ids = self.search_queue_for_jobid(job_id=job_id)
                queued_job_ids = queued_job_ids.where(
                    queued_job_ids.STAT.isin(["RUN", "PEND"])
                ).dropna()
                return len(queued_job_ids) > 0
            return len(self.search_queue_for_jobid(job_id=job_id)) > 0
        else:
            raise ValueError("Please provide either the job_name or the job_id!")

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
        raise NotImplementedError("kill_jobs is not implemented for: " + self.__class__.__name__)

    @property
    def submission(self)->bool:
        """
        getter
        bool, if a job should be submitted. or is this a dry run?
        """
        return self._submission

    @submission.setter
    def submission(self, submission_value):
        """
        setter
        bool, if a job should be submitted. or is this a dry run?
        """
        self._submission = submission_value

    @property
    def nmpi(self) -> int:
        """
        getter
        int, how many processes shall be used with mpi
        """
        return self._nmpi

    @nmpi.setter
    def nmpi(self, nmpi_value: int):
        """
        setter
        int, how many processes shall be used with mpi
        """
        self._nmpi = int(nmpi_value)

    @property
    def nomp(self) -> int:
        """
        getter
        int, how many processes shall be used with omp
        """
        return self._nomp

    @nomp.setter
    def nomp(self, nomp_value: int):
        """
        setter
        int, how many processes shall be used with omp
        """
        self._nomp = int(nomp_value)

    @property
    def ngpu(self) -> int:
        """
        getter
        int, how many gpus accesable
        """
        return self._ngpu

    @ngpu.setter
    def ngpu(self, ngpu_value: int):
        """
        setter
        int, how many gpus accesable
        """
        self._ngpu = int(ngpu_value)

    @property
    def max_storage(self) -> str:
        """
        getter
        str, how much max storage
        """
        return self._max_storage

    @max_storage.setter
    def max_storage(self, max_storage_value: str):
        """
        setter
        str, how much max storage
        """
        self._max_storage = str(max_storage_value)

    @property
    def job_duration(self) -> str:
        """
        getter
        str, expected job duration
        """
        return self._job_duration

    @job_duration.setter
    def job_duration(self, job_duration_value: str):
        """
        getter
        str, expected job duration
        """
        self._job_duration = str(job_duration_value)

    # no verbose setter since verbose is not private

    @property
    def environment(self) -> str:
        """
        getter
        str, which environment shall be used?
        """
        return self._environment

    @environment.setter
    def environment(self, environment_value: str):
        """
        setter
        str, which environment shall be used?
        """
        self._environment = str(environment_value)

    @property
    def block_double_submission(self) -> bool:
        """
        getter
        bool, can we doubly submit
        """
        return self._block_double_submission

    @block_double_submission.setter
    def block_double_submission(self, block_double_submission_value: bool):
        """
        setter
        bool, can we doubly submit
        """
        self._block_double_submission = bool(block_double_submission_value)

    @property
    def chain_prefix(self) -> str:
        """
        getter
        str, prefix for chaining
        """
        return self._chain_prefix

    @chain_prefix.setter
    def chain_prefix(self, chain_prefix_value: str):
        """
        setter
        str, prefix for chaining
        """
        self._chain_prefix = str(chain_prefix_value)

    @property
    def begin_mail(self) -> str:
        """
        getter
        str, send a job starts mail to
        """
        return self._begin_mail

    @begin_mail.setter
    def begin_mail(self, begin_mail_value: str):
        """
        setter
        str, send a job starts mail to
        """
        self._begin_mail = str(begin_mail_value)

    @property
    def end_mail(self) -> str:
        """
        getter
        str, send a job ends mail to
        """
        return self._end_mail

    @end_mail.setter
    def end_mail(self, end_mail_value: str):
        """
        setter
        str, send a job end mail to
        """
        self._end_mail = str(end_mail_value)

    @property
    def fail_mail(self) -> str:
        """
        getter
        str, send a job fail mail to
        """
        return self._fail_mail

    @fail_mail.setter
    def fail_mail(self, fail_mail_value: str):
        """
        getter
        str, send a job fail mail to
        """
        self._fail_mail = str(fail_mail_value)

    @property
    def conda_env_path(self) -> str:
        """
        getter
        str, conda environment path
        """
        return self._conda_env_path

    @conda_env_path.setter
    def conda_env_path(self, conda_env_path_value: str):
        """
        setter
        str, conda environment path
        """
        self._conda_env_path = str(conda_env_path_value)
