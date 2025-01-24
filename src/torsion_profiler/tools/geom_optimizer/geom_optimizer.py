"""
Geometry Optimizer
"""
import os
import copy
import tempfile
from typing import Iterable, Union
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from ...engines._abstractcalculator import _AbstractCalculator
from ...orchestration.computing_envs import ScriptComputing
from ...orchestration.submission_systems import submissionsystem, Local, Slurm
from ...utils import bash, store_mol_db, read_mol_db
from ...utils.baseclass import BaseClass, SubmitterClass
from ...utils.mol_db_operation import add_potential_energy_terms

log = logging.getLogger()
class GeomOptimizer(BaseClass, SubmitterClass):
    """
    Geometry Optimizer Class
    Implements the modular optimization approaches.
    """
    calculator: _AbstractCalculator
    _submission_system: submissionsystem.SubmissionSystem
    job_status_file_path: str

    # private:
    _quiet: bool = False

    def __init__(
        self,
        calculators: Union[_AbstractCalculator, Iterable[_AbstractCalculator]],
        submission_system: submissionsystem.SubmissionSystem = None,
        _force: bool = False,
    ) -> None:
        """
        This class is the base for calculating torsion profiles.

        It can be used in different contexts
        *  use calculators like gaussian, psi4, torch-ani etc.
        *  calculate interactively in a jupyter notebook or scale on the cluster via slurm.

        In Script mode:
            The class checks if results with are already present and does not overwrite them
            (except _force is True)

        Parameters
        ----------
        calculator : _abstract_calculator, optional
            the method of choice to calculate the torsion profile, by default ani_calculator()
        out_folder_path : str, optional
            if files are written out, its their general root path., by default None
        out_file_prefix : str, optional
            if files are written out, its their general prefix., by default None
        submission_system : _submission_system, optional
            submit jobs to the cluster, if None interactive mode is active. If you want to
            subparallel each task, please add cores via the submission system, by default None

        _force : bool, optional
            force the recalculation of all present results.
        """
        if isinstance(calculators, _AbstractCalculator):
            self.calculators = [calculators]
        else:
            self.calculators = calculators
        if isinstance(submission_system, Slurm):
            raise NotImplementedError("Optimization for Slurm not implemented")
        else:
            self._submission_system = submission_system
        self._force = _force
        self._job_status = {}
        self._collect_only = False
        self.n_tasks_parallel =1
        self._sub_results_opt_structures = []
        self._sub_results_in_structures = None
        self._sub_results_finished = None

        # Initialize vars:
        self._initial_path = None
        self._job_status_file_path = None
        self.interactive = None
        self.out_result_path = None
        self._partial_done = None



    """
        property
    """

    @property
    def out_file_prefix(self) -> str:
        return self._out_file_prefix

    @out_file_prefix.setter
    def out_file_prefix(self, name: str):
        self._out_file_prefix = name

    def optimize(
        self,
        mol: Chem.Mol,
        out_dir: str = None,
        approach_name: str = None,
        _additional_rest: list[int] = None,
        collect_only:bool=False
    ):
        self._in_mol = Chem.Mol(mol)
        self._initial_path = bash.getcwd()  # more a symptom fix!
        self._additional_rest = _additional_rest

        # Do Check
        log.info(
                "#" * 250
                + "\n > Check if run done\n result_path: "
                + str(out_dir)
            )

        result_files_present = self._check(out_folder_path=out_dir, approach_name=approach_name,
                                   collect_only=collect_only, force_run=self._force,
                                   out_suffix="opt")

        # Calculate
        run_calculations = (not (result_files_present or collect_only) or self._force)
        res = None
        log.info(f"run calculations: {run_calculations}")
        if run_calculations:
            log.info("\n" + "#" * 250 + "\n > Run ")
            res = self._run(mol, approach_name=approach_name,
                            _additional_pos_res=_additional_rest)

            # Recheck if everything done
            if isinstance(self._submission_system, Local) and not self._interactive:
                result_files_present = self._check(out_folder_path=out_dir,
                                                   approach_name=approach_name, force_run=False)

        # Gather
        # are there result files present?
        result_files_present=True
        if self._interactive: #finalize the interactive results.
            res = pd.concat(res.values())
            res = add_potential_energy_terms(res)
        elif result_files_present or collect_only:  # gather results
            log.info("\n" + "#" * 250 + "\n > Gather ")
            res = self._gather_results()

        os.chdir(self._initial_path)
        return res

    # Utils
    def wait(self) -> pd.DataFrame:
        """
        wait for job to be finished.

        Returns
        -------

        """
        raise NotImplementedError()

    # Private
    def _run(self, mol:Chem.Mol, approach_name:str, _additional_pos_res=None)->pd.DataFrame:
        """
        Optimize molecule.

        Parameters
        ----------
        mol: Chem.Mol

        Returns
        -------
        dict
        """
        # Create an instance of the model
        log.info("Stage 1: Optimize and calculate Potential")

        # subjob already done?
        check_subjob_done = lambda task_id: False
        if not self._interactive and approach_name in self._job_status:
            if (
                len(self._job_status[approach_name]["sub_jobs"])
                == self.n_tasks_parallel
            ):
                check_subjob_done = (
                    lambda taskID: self._job_status[approach_name]["sub_jobs"][taskID][
                        "status"
                    ]
                    == "DONE"
                )
            elif len(self._job_status[approach_name]["sub_jobs"]) != 0:
                raise ValueError(
                    "task number changed! got job_statuses: "
                    + str(len(self._job_status[approach_name]["sub_jobs"]))
                    + " \t got n_tasks-parallel: "
                    + str(self.n_tasks_parallel)
                )


        res = self._job_status
        iterators = list(enumerate(self.calculators))
        if not self._quiet:
            iterator = tqdm(iterators)
        else:
            iterator = iterators
        func_options = {"mol": mol}

        for i, calculator in iterator:
            if hasattr(calculator, "calculate_conformer_potentials"):
                func_options.update({"out_file_path": self.out_result_path})

                #task already done?
                task_id = calculator.name + "_" + str(i)
                if check_subjob_done(task_id):
                    log.info("\t\tsubjob " + str(task_id) + " already finished! - SKIP")
                    continue

                tmp_out_dir = tempfile.mkdtemp(
                        prefix=f"tmp_{approach_name}",
                        suffix=f"_{task_id}",
                    )

                if _additional_pos_res is not None:
                    func_options["_additional_pos_res"] = _additional_pos_res

                if not self.interactive:
                    log.info("\t\tsubjob " + str(task_id) + " submit")
                elif  self.interactive:
                    log.info("\t\tsubjob " + str(task_id) + " calculation")

                calculator._optimize_structure = False

                if self._interactive:
                    tmp_out_file = None
                else:
                    tmp_out_dir = tempfile.mkdtemp(
                            prefix=f"tmp_{approach_name}_sp",
                            suffix=f"_{task_id}",
                            dir=self.tmp_calculation_path,
                        )
                    single_point_file = bash.path.basename(
                        self.out_result_path.replace(
                            ".sdf", f"_{task_id}_singlepoint.sdf"
                        ))
                    tmp_out_file = f"{tmp_out_dir}/{single_point_file}"

                func_options["out_file_path"] = tmp_out_file
                out_file_prefix = str(f"{approach_name}_singlepoint"
                    if (approach_name is not None)
                    else "singlepoint"
                )
                tres_start = self._do_calc(
                    func=calculator.calculate_conformer_potentials,
                    func_options=func_options,
                    _out_folder_path=tmp_out_dir,
                    _out_file_prefix=out_file_prefix,
                )

                if func_options["out_file_path"] is not None:  # chek why!
                    if not self._interactive:
                        out_file = bash.path.basename(self.out_result_path.replace(".sdf",
                                                                                   f"_{task_id}_opt.sdf"))
                        tmp_out_dir = tempfile.mkdtemp(
                            prefix=f"tmp_{approach_name}_opt",
                            suffix=f"_{task_id}",
                            dir=self.tmp_calculation_path,
                        )
                        tmp_out_file = f"{tmp_out_dir}/{out_file}"
                    else:
                        tmp_out_file=None
                    func_options["out_file_path"] = tmp_out_file

                out_file_prefix = f"{approach_name}_optimized" if (approach_name is not None) \
                    else "optimized"

                calculator._optimize_structure = True
                tres_opt = self._do_calc(
                    func=calculator.calculate_conformer_potentials,
                    func_options=func_options,
                    _out_folder_path=tmp_out_dir,
                    _out_file_prefix=out_file_prefix,
                )
                if func_options["out_file_path"] is not None:
                    func_options["mol"] = func_options["out_file_path"].replace(
                        ".sdf", "_opt.sdf"
                    )

                if isinstance(tres_start, pd.DataFrame) and isinstance(tres_opt, pd.DataFrame):
                    tres_start["status"] = calculator.name + "_start"
                    tres_opt["status"] = calculator.name + "_opt"
                    df_res = pd.concat([tres_start, tres_opt])
                    df_res["calculator"] = calculator.name
                    res[approach_name] = df_res
                elif isinstance(self._computing, ScriptComputing):
                    tres = {
                        "jobID": tres_opt,
                        "status": "PD",
                        "output_dir": tmp_out_dir,
                        "out_opt": tmp_out_file,
                    }
                    res[approach_name]["sub_jobs"][task_id] = tres

            else:
                raise ValueError(
                    "Calculator has no calculate_conformer_potentials function, but this is needed!"
                )
        return res

    def _gather_results(
        self,
    ) -> pd.DataFrame:
        """
            wrap results.

        Returns
        -------

        """

        if self.out_result_path is not None and bash.path.isfile(self.out_result_path):
            log.info(f"\t Loading final results: {self.out_result_path}", stacklevel=2)
            self.out_tp = read_mol_db(self.out_result_path)
        else:
            log.info("\t Building final results from sub_results")

            # try to calculate the torsion angles from the output structures.
            sub_res = []
            for p in self._sub_results_opt_structures:
                tres = read_mol_db(p)
                tres["approach"] = bash.path.dirname(p).split("_")[-2]
                tres["chain_step"] = bash.path.dirname(p).split("_")[-1]
                tres["approach_stage"] = (
                    "in" if ("singlepoint" in bash.path.basename(p)) else "optimized"
                )
                ref_mol = tres["ROMol"].to_list()[0]
                rms = [
                    Chem.rdMolAlign.CalcRMS(prbMol=m, refMol=ref_mol,) for m in tres["ROMol"]
                ]
                tres["rms"] = np.round(rms, 2)
                tres = add_potential_energy_terms(tres)
                sub_res.append(tres)

            res = pd.concat(sub_res, ignore_index=True)

        return res

    def _previous_results_present(self, out_result_file_path,
                                        approach_name) -> (bool, bool):
        """
        Check previous results.

        Returns
        -------

        """
        # are there final results?
        fin_results = not self._interactive and bash.path.exists(out_result_file_path)
        log.info("Found Final Result File:", fin_results)

        # subjobs?
        self._sub_results_opt_structures = []
        self._sub_results_in_structures = []
        self._sub_results_finished = {}

        sub_job_paths = [
            f"{self.tmp_calculation_path}/{a}"
            for a in bash.listdir(self.tmp_calculation_path)
            if (
                bash.path.isdir(f"{self.tmp_calculation_path}/{a}")
                and a.startswith("tmp_" + approach_name)
            )
        ]

        nsubjobs = len(sub_job_paths) * 2
        for d in sub_job_paths:
            sub_job_number = int(str(d.split("_")[-1]))
            s_results = (bash.glob(f"{d}/*opt.sdf") +
                         bash.glob(f"{d}/*singlepoint.sdf"))

            in_structure = bash.glob(f"{d}/in_*.sdf")
            self._sub_results_in_structures.append(in_structure)
            sub_job_finished = False
            if len(s_results) > 0:
                self._sub_results_opt_structures.extend(s_results)
                sub_job_finished = True

            elif len(s_results) > 2:
                raise ValueError("oh oh to many pdb files in subjob folder: " + d)

            self._sub_results_finished[int(sub_job_number)] = sub_job_finished

        all_subjobs_present = False
        self._partial_done = False
        if len(self._sub_results_in_structures) > 0:
            all_subjobs_present = nsubjobs == len(self._sub_results_in_structures)

            if not all_subjobs_present and len(self._sub_results_in_structures) / float(nsubjobs) > 0.55:
                self._partial_done = True
                print("PARTIAL DONE")

        log.info(
            f"Found all_subjobs Result File:{all_subjobs_present}",
            f"found sub_results: {len(self._sub_results_in_structures)}",
        )

        return fin_results, all_subjobs_present
