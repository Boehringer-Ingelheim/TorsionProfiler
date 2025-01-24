"""
Base Class for tool classes
"""
import abc
import inspect
import pandas as pd
from typing import Callable
from types import FunctionType
import logging

from ..orchestration.computing_envs import ScriptComputing, InteractiveComputing
from ..orchestration.submission_systems import submissionsystem, Slurm, Local
from ..orchestration.submission_systems.submissionjob import ArraySubmissionJob

from . import bash

log = logging.getLogger()

class SubmitterClass:
    """
    this class contains the base implementation for executing/submitting a job.
    """
    _submission_system: submissionsystem.SubmissionSystem
    n_tasks: int = 1
    remove_tmp_calculation_path = False

    def _check(self, out_folder_path: str = None, approach_name: str = None,
               collect_only=False, force_run=False, out_suffix="tp") -> bool:
        """
        check if job is done
        """

        # Decide for execution mode:
        if self._submission_system is None:
            self._interactive = True
            self.out_result_path =  self.tmp_calculation_path= None
            final_result_file_present = all_subjob_results_present = False
        else:
            self._interactive = False

            # construct output files:
            self.out_result_path = f"{out_folder_path}/{approach_name}_{out_suffix}.sdf"
            self.tmp_calculation_path = f"{out_folder_path}/{approach_name}"
            self._job_status_file_path = f"{out_folder_path}/{approach_name}_jobs.json"

            # job status
            if bash.path.exists(self._job_status_file_path) and not force_run:
                self.load_job_status(job_status_file_path=self._job_status_file_path)

            if self._job_status is None:
                self._job_status = {
                    approach_name: {
                        "sub_jobs": {},
                        "work_folder": self.tmp_calculation_path,
                        "root_out_folder": out_folder_path,
                        "approach_name": approach_name,
                    }
                }
            elif approach_name not in self._job_status:
                self._job_status[approach_name] = {
                    "sub_jobs": {},
                    "work_folder": self.tmp_calculation_path,
                    "root_out_folder": out_folder_path,
                    "approach_name": approach_name,
                }

            # Folder generation if
            if not collect_only:
                bash.makedirs(out_folder_path, exist_ok=True)
                bash.makedirs(self.tmp_calculation_path, exist_ok=True)

            # do calculation or read results
            # File Based
            final_result_file_present, all_subjob_results_present = (
                self._previous_results_present(out_result_file_path=self.out_result_path,
                                               approach_name=approach_name))
            if (final_result_file_present and bash.path.isdir(self.tmp_calculation_path) and
                    self.remove_tmp_calculation_path):
                bash.rmtree(self.tmp_calculation_path)

            # submission system: is a job still running?
            if self._submission_system is not None:
                jobs_running = {}
                for taskID in range(self.n_tasks_parallel):
                    job_name = str(approach_name) + "_" + str(taskID)
                    found_jobs = self._submission_system.search_queue_for_jobname(job_name)
                    if len(found_jobs) > 0:
                        still_running = True
                    else:
                        still_running = False

                    jobs_running[taskID] = still_running

            log.info(
                f"Found final results / all subjobResults -->"
                f" run_finished {all_subjob_results_present}, "
                f"finished subjobs {len(self._sub_results_opt_structures)}/{self.n_tasks_parallel},"
            )

        return (final_result_file_present or all_subjob_results_present)

    def _do_calc(self, func:Callable, func_options: dict, _out_folder_path: str,
                 _out_file_prefix: str):
        """
        Calculate the funciton with func options in the given context.

        Parameters
        ----------
        func: Callable
        func_options: dict
        _out_folder_path: str
        _out_file_prefix: str

        Returns
        -------
        object
            the return value of func
        """
        if self._submission_system is None:
            self._computing = InteractiveComputing()
            res = self._computing.run(in_target_function=func, in_function_parameters=func_options)
        elif isinstance(self._submission_system, (Local, Slurm)):
            self._computing = ScriptComputing(
                submission_system=self._submission_system, n_tasks=self.n_tasks
            )

            res = self._computing.run(
                out_root_dir=_out_folder_path,
                out_file_prefix=_out_file_prefix,
                in_target_function=func,
                in_function_parameters=func_options,
            )
        else:
            raise ValueError("Unknown submission system: " + str(self._submission_system))

        return res

    def _split_tasks(self, ntasks: int, n_tasks_parallel: int) -> list[list[int]]:
        """
        Parallelize the task execution.

        Parameters
        ----------
        ntasks: int
            number of tasks
        n_tasks_parallel: int
            parallelization of the tasks

        Returns
        -------
        list[list[int]]
            chunked job list.
        """
        task_lengths = ntasks // n_tasks_parallel
        tasks_offset = ntasks % n_tasks_parallel
        load_per_task = [
            task_lengths + 1 if (i < tasks_offset) else task_lengths
            for i in range(n_tasks_parallel)
        ]
        task_splits = [
            list(range(sum(load_per_task[:n]), sum(load_per_task[: n + 1])))
            for n in range(n_tasks_parallel)
        ]
        return task_splits

    def generate_job_array(
        self,
        collect_only:bool,
        job_end,
        job_files,
        job_lim,
        root_out_dir,
        root_out_prefix,
        root_out_result_file_path,
        sub_job_results,
        sub_job_tasks,
        submission_batch_size,
    ) -> list :
        """
        build a job array for a set of tasks.

        Parameters
        ----------
        collect_only
        job_end
        job_files
        job_lim
        root_out_dir
        root_out_prefix
        root_out_result_file_path
        sub_job_results
        sub_job_tasks
        submission_batch_size

        Returns
        -------

        """
        if not collect_only and len(job_files) > 0:
            batch_suffix = "array_batch_"

            njobs = len(job_files) - 1
            chunks = [
                job_files[i : i + submission_batch_size]
                for i in range(0, njobs, submission_batch_size)
            ]
            print(
                "submiting chunks: " + str([len(chunk) for chunk in chunks]),
                "from: " + root_out_dir,
            )

            log_dir = root_out_dir + "/slurm_logs"
            if not bash.path.exists(log_dir):
                bash.mkdir(log_dir)

            # remove old submitters:
            old_submission_files = bash.glob(f"{root_out_prefix}{batch_suffix}*.sh")
            for p in old_submission_files:
                bash.remove(p)

            job_ids = []
            for i, chunk in enumerate(chunks):
                batch_suffix_nr = batch_suffix + str(i)
                if job_end is None:
                    job_end = len(chunk) - 1

                job = ArraySubmissionJob(
                    chunk,
                    job_name=root_out_prefix + batch_suffix_nr,
                    submit_from_dir=root_out_dir,
                    submit_from_file=True,
                    job_lim=job_lim,
                    start_job=0,
                    end_job=job_end,
                    out_log=log_dir + "/" + batch_suffix_nr + "-%a_%j.out",
                    err_log=log_dir + "/" + batch_suffix_nr + "-%a_%j.err",
                )

                job_id = self._submission_system.submit_job_array_to_queue(job)
                job_ids.append(job_id)

            if len(job_ids) == 1:
                job_ids = str(job_ids[0]).split("_", maxsplit=1)[0]
                print("Submit Job: ", job_ids)

                for job_name, job_details in sub_job_tasks.items():
                    for sub_job_id, sub_job in job_details["sub_jobs"].items():
                        sub_job["jobID"] = job_ids
            else:
                print("Submit Job: ", job_ids)

            self._job_status = sub_job_tasks
            self.save_job_status(job_status_file_path=self._root_job_status_file_path)
            sub_job_results.update(self._job_status)

        else:  # final result
            if collect_only and len(job_files) > 0:
                pass
            else:
                print("Write out csv to : ", root_out_result_file_path)
                sub_job_results = pd.concat(sub_job_results.values())
                sub_job_results.to_csv(root_out_result_file_path, sep="\t")
                sub_job_results.to_pickle(root_out_result_file_path.replace("tsv", "obj"))

        self._out_folder_path = root_out_dir
        self.out_result_path = root_out_result_file_path
        return sub_job_results


class BaseClass(abc.ABC):
    """
    Abstract class, building  a scaffold for all package classes.
    """
    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(map(lambda x: str(x[0]) + "=" + str(x[1]), vars(self).items()))
            + ")"
        )

    def _string_constructor(self, obj_name="target_obj") -> str:
        """
        translate this class status into a string reconstruction.

        Parameters
        ----------
        obj_name
            name of the obj, that should be generated.

        Returns
        -------
        str
            a string variant on how to instatiate this class in a str.

        """
        init_signature = inspect.getfullargspec(self.__init__)

        suppl_str = ""
        kargs = []
        for key in init_signature.args:
            if key in ["self", "return"]:
                continue
            elif hasattr(self, key):
                value = getattr(self, key)
                if hasattr(value, "__module__"):
                    if "torsion_profiler" in value.__module__:
                        if isinstance(value, FunctionType):
                            suppl_str += (
                                "from " + value.__module__ + " import " + value.__name__ + "\n"
                            )
                            value = value.__name__
                        else:
                            suppl_str += (
                                value._string_constructor(obj_name="sub_" + obj_name) + "\n"
                            )
                            value = "sub_" + obj_name
                    else:
                        if hasattr(value, "__name__"):
                            suppl_str += (
                                "from " + value.__module__ + " import " + value.__name__ + "\n"
                            )
                            value = value.__name__
                elif isinstance(value, str):
                    value = '"' + str(value) + '"'

                kargs.append((key, str(value)))
            else:
                raise ValueError("constructor parameter is not an attribute! Make it so! --> ", key)

        suppl_str += (
            "from " + self.__class__.__module__ + " import " + self.__class__.__name__ + "\n"
        )
        instanciate_str = suppl_str
        instanciate_str += (
            f"{obj_name} = {self.__class__.__name__}("
            + ", ".join(map(lambda x: "=".join(x), kargs))
            + ")\n"
        )

        return instanciate_str
