"""
Torsion profiler Implementatoins
"""

import os
import time
import json
import copy
import logging
from datetime import datetime
import multiprocessing as mult
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from ...engines._abstractcalculator import _AbstractCalculator
from ...orchestration.computing_envs import ScriptComputing
from ...orchestration.submission_systems import submissionsystem, Local, Slurm
from ...utils.molecule_attribs import get_all_torsion_atoms_idx, fragment_mol_complete
from ...utils.baseclass import BaseClass, SubmitterClass
from ...utils import bash

from .torsion_profile_generators import (_AbstractTorsionProfileGenerator,
                                         LandscaperTorsionProfileGenerator, )
from ...utils.mol_db_operation import add_potential_energy_terms

from ...utils import read_mol_db, store_mol_db

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def thread_prepare_input(job_id: int, mols: list[Chem.Mol],
                         torsions: list[tuple[int, int, int,int]],
                         out_dir:str, tmp_out_file_prefixes:str,
                         mol_range: Iterable[int], tp_ks: dict[str, any]):
    """
    Helper Function
    This thread is preparing the input structure a folder
    
    Parameters
    ----------
    job_id: int
    mols: list[Chem.Mol]
    torsions> list[tuple[int, int, int, int]]
    tmp_out_file_prefixes: str
    mol_range: Iterable[int]
    tp_ks: dict[str, any]

    Returns
    -------
    list[dict[str, any]], list, list[str]
        sub_job_results, sub_job_tasks, job_files
    """

    job_files = []
    sub_job_tasks = []
    sub_job_results = []
    tp = TorsionProfiler(calculator=None)

    for k, v in tp_ks.items():
        setattr(tp, k, v)
    _initial_path = bash.getcwd()

    for i in mol_range:
        bash.chdir(_initial_path)
        approach_name=tmp_out_file_prefixes[i]
        mol = mols[i]
        torsion = torsions[i]
        try:
            res = tp.calculate_torsion_profile(mol=mol, torsion_atom_ids=torsion,
                                               out_dir=out_dir, approach_name=approach_name)

            # get job_files
            if not isinstance(res, pd.DataFrame):
                job = res[approach_name]
                if tp._submission_system.command_file_path is not None and isinstance(job, dict):
                    for k, sub_job in job["sub_jobs"].items():
                        tout_dir = sub_job["output_dir"]
                        job_file = [
                            tout_dir + "/" + file_name
                            for file_name in os.listdir(tout_dir)
                            if (file_name.startswith("job_") and file_name.endswith(".sh"))
                        ][0]
                        job_files.append(job_file)

                sub_job_tasks.append((approach_name, job))
                sub_job_results.append((approach_name, None))

            else:
                res["torsion_molID"] = (f"{approach_name}_ta_" + "_".join(map(str,
                                                                              res.torsion_atom_ids.iloc[0]))
                )
                res["molID"] = approach_name
                sub_job_results.append((approach_name, res))

        except RuntimeError as err:
            print("Error: ", err.args)
            if "bad direction in linearSearch" not in err.args[0]:
                raise err
        except ValueError as err:
            print("skip mol " + str(i) + "because of ring problem!")
            if "bond (j,k) must not belong" not in err.args[0]:
                raise err

    # raise err
    return sub_job_results, sub_job_tasks, job_files


def thread_prepare_all_tors_input(job_id: int, mols: list[Chem.Mol],
                                  tmp_out_file_prefixes: str, mol_range: Iterable[int],
                                  tp_ks: dict[str, any]):
    """
    Helper Function
    This thread is preparing the input structure a folder calculating all torsion profiles
    
    Parameters
    ----------
    job_id: int
    mols: list[Chem.Mol]
    tmp_out_file_prefixes: str
    mol_range: Iterable[int]
    tp_ks: dict[str, any]
    
    Returns
    -------
    dict, dict, list[str]
        sub_job_results, sub_job_tasks, job_files
    """
    job_files = []
    sub_job_tasks = []
    sub_job_results = []
    tp = TorsionProfiler(calculator=None)

    for k, v in tp_ks.items():
        setattr(tp, k, v)

    for i in mol_range:  # tqdm(mol_range, desc="Job" + str(job_id) + ": ", mininterval=10):
        mol = mols[i]
        try:
            for res in tp.calculate_all_mol_torsions_profiles(mol=mol, out_dir=out_dir,
                approach_name= tmp_out_file_prefixes[i]):
                # get job_files
                if not isinstance(res, pd.DataFrame):
                    job = res[tp._out_file_prefix]
                    if tp._submission_system.command_file_path is not None and isinstance(
                        job, dict
                    ):
                        for k, sub_job in job["sub_jobs"].items():
                            out_dir = sub_job["output_dir"]
                            job_file = [
                                out_dir + "/" + file_name
                                for file_name in os.listdir(out_dir)
                                if (file_name.startswith("job_") and file_name.endswith(".sh"))
                            ][0]
                            job_files.append(job_file)

                    sub_job_tasks.append((tmp_out_file_prefixes[i], job))
                    sub_job_results.append((tmp_out_file_prefixes[i], None))

                else:
                    res["torsion_molID"] = (
                        tp._out_file_prefix
                        + "_ta_"
                        + "_".join(map(str, res.torsion_atom_ids.iloc[0]))
                    )
                    res["molID"] = tp._out_file_prefix
                    sub_job_results.append((tmp_out_file_prefixes[i], res))

        except RuntimeError as err:
            print("Error: ", err.args)
            if "bad direction in linearSearch" not in err.args[0]:
                raise err
        except ValueError as err:
            print("skip mol " + str(i) + "because of ring problem!")
            if "bond (j,k) must not belong" not in err.args[0]:
                raise err

    # raise err
    return sub_job_results, sub_job_tasks, job_files


def thread_prepare_2d_angle_input(
    job_id: int,
    mol: Chem.Mol,
    torsion_angle2: float,
    torsion1: tuple[int, int, int, int],
    torsion2: tuple[int, int, int, int],
    out_dir: str,
    tmp_out_file_prefixes: str,
    angle2_range: Iterable[float],
    tp_ks: dict,
    _additional_pos_res: list[int],
    _additional_torsions: list[tuple[int, int, int, int, float]],
):
    """
    Helper Function
    This thread is preparing the input structure a folder calculating all torsion profiles

    Parameters
    ----------
    job_id: int
    mol: Chem.Mol,
    torsion_angle2: float,
    torsion1: tuple[int, int, int, int],
    torsion2: tuple[int, int, int, int],
    tmp_out_file_prefixes: str,
    angle2_range: Iterable[float],
    tp_ks: dict,
    _additional_pos_res: list[int],
    _additional_torsions: list[tuple[int, int, int, int, float]],


    Returns
    -------
    dict, dict, list[str]
        sub_job_results, sub_job_tasks, job_files

    """
    job_files = []
    sub_job_tasks = []
    sub_job_results = []
    tp = TorsionProfiler(calculator=None)

    for k, v in tp_ks.items():
        setattr(tp, k, v)

    tp_angle1 = tp.initial_torsion_profile_generator.generate(
        mol,
        torsion2,
        _additional_pos_res=_additional_pos_res,
        _additional_torsions=_additional_torsions,
    )

    for i in angle2_range:
        approach_name = tmp_out_file_prefixes[i]
        tmp_mol = Chem.Mol(tp_angle1["ROMol"].iloc[i])
        angle2 = torsion_angle2[i]
        try:
            torsion2_angle = [torsion2[0], torsion2[1], torsion2[2], torsion2[3], angle2]
            const_torsions = [torsion2_angle]
            if _additional_torsions is not None:
                const_torsions.extend(_additional_torsions)
            res = tp.calculate_torsion_profile(
                mol=tmp_mol,
                torsion_atom_ids=torsion1,
                out_dir=out_dir,
                approach_name=approach_name,
                additional_const=_additional_pos_res,
                additional_torsions=const_torsions,
            )
            if isinstance(res, pd.DataFrame):
                res["torsion_angle_b"] = [angle2 for _ in range(res.shape[0])]
        except Exception as err:
            print("\n".join(err.args))
            continue
            # raise err

        # get job_files
        if not isinstance(res, pd.DataFrame):
            job = res[approach_name]
            if (tp._submission_system.command_file_path is not None and
                    isinstance(job, dict)):
                for k, sub_job in job["sub_jobs"].items():
                    out_dir = sub_job["output_dir"]
                    job_file = [
                        out_dir + "/" + file_name
                        for file_name in os.listdir(out_dir)
                        if file_name.startswith("job_") and file_name.endswith(".sh")
                    ][0]
                    job_files.append(job_file)
            sub_job_tasks.append((approach_name[i], job))
            sub_job_results.append((approach_name[i], None))
        else:
            res["torsion_molID"] = (
                f"{approach_name}_ta_"
                + "_".join(map(str, res.torsion_atom_ids.iloc[0]))
                + "_tb_"
                + "_".join(map(str, torsion2))
            )
            res["torsion_atom_ids_2"] = [torsion2 for _ in range(len(res))]
            res["molID"] = approach_name
            res["torsion_angle_b"] = angle2
            sub_job_results.append((approach_name, res))
    return sub_job_results, sub_job_tasks, job_files


class TorsionProfiler(BaseClass, SubmitterClass):
    """
    Implement Torsion Profiles
    """
    calculator: _AbstractCalculator
    initial_torsion_profile_generator: _AbstractTorsionProfileGenerator
    n_measurements: int
    n_tasks_parallel: int
    n_processes: int
    rd_force_constant: float = 10**4
    job_status_file_path: str

    def __init__(
        self,
        calculator: _AbstractCalculator,
        n_measurements: int = 37,
        submission_system: submissionsystem.SubmissionSystem = None,
        initial_tors_profile_generator: _AbstractTorsionProfileGenerator = None,
        n_tasks: int = 1,
        n_processes: int = 1,
        _force: bool = False,
    ):
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
        n_measurements : int, optional
            how many calculations should be done along the torsion angle?(will split evenly),
            by default 37
        out_folder_path : str, optional
            if files are written out, its their general root path., by default None
        out_file_prefix : str, optional
            if files are written out, its their general prefix., by default None
        submission_system : _submission_system, optional
            submit jobs to the cluster, if None interactive mode is active. If you want to
            subparallel each task, please add cores via the submission system, by default None
        initial_tors_profile_generator : _AbstractTorsionProfileGenerator
            Not usable currently, but a future expansion point.
        n_tasks : int, optional
            if n_tasks is large > 1 the code will try to parallelize the workload (conformer
            calculations) evenly on several process (if available).
        _force : bool, optional
            force the recalculation of all present results.
        """

        self.n_measurements = n_measurements

        if initial_tors_profile_generator is None:
            self.initial_torsion_profile_generator = LandscaperTorsionProfileGenerator(
                n_measurements=self.n_measurements, force_constant=self.rd_force_constant
            )
        elif isinstance(initial_tors_profile_generator, _AbstractTorsionProfileGenerator):
            self.initial_torsion_profile_generator = initial_tors_profile_generator
        else:
            raise ValueError("The provided initial torsionProfile generator was not recognized!")

        self.calculator = calculator
        self._submission_system = submission_system
        self._submission_system = submission_system
        self.n_tasks_parallel = int(n_tasks)
        self.n_processes = int(n_processes)
        self._force = _force
        self._job_status = {}

        # init
        self._interactive = True
        self.out_result_file_path = None
        self.tmp_calculation_path = None # Todo: Add deleting tmp_path
        self.remove_tmp_calculation_path =False
        self._out_root_file_prefix = None
        self._job_status_file_path = None
        self._root_job_status_file_path = None

        self.fin_results = None
        self.all_subjobs_present = None

        self.out_result_path = None
        self.out_tp = None

        # private tmp vars
        self._tmp_input_profile_mol = None

        self._sub_results_opt_structures = None
        self._sub_results_in_structures = None
        self._sub_results_finished = None


    @staticmethod
    def read_torsion_profile_mol(
        in_file_path: str, torsion_atom_ids: tuple[int, int, int, int] = None
    ) -> Chem.Mol:
        """
        read a torsion profile.

        Parameters
        ----------
        in_file_path: str
        torsion_atom_ids: tuple[int, int, int, int]
            define the expected torsion atoms, optional

        Returns
        -------
        Chem.Mol
            torsion profile molecule
        """
        return read_mol_db(
            in_file_path=in_file_path, torsion_atom_ids=torsion_atom_ids
        )

    @staticmethod
    def store_torsion_profile_mol(
        mol: Chem.Mol, out_sdf_path: str, torsion_atom_ids: tuple[int, int, int, int] = None
    ) -> str:
        """
        store the given torsion profile.

        Parameters
        ----------
        mol: Chem.Mol
            rdkit mol
        out_sdf_path: str
            out file path
        torsion_atom_ids:  tuple[int, int, int, int]
            define the expected torsion atoms, optional

        Returns
        -------
        str
            out path
        """
        return store_mol_db(
            mol=mol, out_sdf_path=out_sdf_path, torsion_atom_ids=torsion_atom_ids
        )

    def calculate_torsion_profile(
        self,
        mol: Chem.Mol,
        torsion_atom_ids: tuple[int, int, int, int],
        out_dir: str = None,
        approach_name: str = None,
        additional_const: list[int] = None,
        additional_torsions: list[tuple[int, int, int, int, float]] = None,
        collect_only: bool = False,
    ) -> pd.DataFrame:
        """
            Calculate the Torsion profile potentials for a single selected torsion.

        Parameters
        ----------
        mol : Chem.Mol
            molecule of interest
        torsion_atom_ids : list[int]
            torsion indices
        out_dir : str, optional
            path where the results should be stored.
        approach_name
        additional_const
        additional_torsions
        collect_only

        Returns
        -------
        pd.DataFrame
            result containing the potential Energies
        """

        in_mol = Chem.Mol(mol)
        in_torsion_atom_ids = copy.deepcopy(torsion_atom_ids)
        _initial_path = bash.getcwd()  # more a symptom fix!

        if approach_name is None:
            approach_name = "mol"

        # Check
        # test if all required result files are present in any form in order to short cut calcs.
        log.info(
         "%s\n > Check if run done\n result_path: %s".format("#" * 250,
                                                             out_dir)
        )
        result_files_present = self._check(out_folder_path=out_dir, approach_name=approach_name,
                                           collect_only=collect_only, force_run=self._force,
                                           out_suffix="tp")
        # Calculate
        # if required results files were not found, do some calculations
        run_calculations = (not (result_files_present or collect_only) or self._force)
        res = None
        log.info(f"run calculations: {run_calculations}")
        if run_calculations:
            log.info("\n" + "#" * 250 + "\n > Run ")
            res = self._run(in_mol, in_torsion_atom_ids,
                            _additional_pos_res=additional_const,
                            _additional_torsions=additional_torsions,
                            approach_name=approach_name)

            # Recheck if everything done
            if isinstance(self._submission_system, Local) and not self._interactive:
                result_files_present = self._check(out_folder_path=out_dir,
                                                   approach_name=approach_name, force_run=False)

        # Gather
        # are there result files present?
        if result_files_present or collect_only:  # gather results
            log.info("\n" + "#" * 250 + "\n > Gather ")
            res = self._gather_results()
        elif self._interactive: #finalize the interactive results.
            res = add_potential_energy_terms(res)
        if isinstance(res, pd.DataFrame):
            Chem.PandasTools.ChangeMoleculeRendering(res)

            if (len(res.loc[res["confID"] == self.n_measurements-1]) == 1
                and float(res.loc[res["confID"] == self.n_measurements-1, "torsion_angle"]) < 0):
                res.loc[res["confID"] == self.n_measurements-1, "torsion_angle"] *= -1
            if (len(res.loc[res["confID"] == 0]) == 1
                and float(res.loc[res["confID"] == 0, "torsion_angle"]) > 0):
                res.loc[res["confID"] == 0, "torsion_angle"] *= -1

        bash.chdir(_initial_path)

        return res

    def calculate_all_torsions_profiles(
        self, mol: Chem.Mol, out_dir: str = None,
        approach_name: str = None, collect_only: bool = False, fragment_mol: bool=False,
        _start_ind:  int = 0
    ) -> Iterable[pd.DataFrame]:
        """
        This function calulates all Torsion profiles found for all molecules provided with
        the ANI force field.

        Parameters
        ----------
        mols : Union[list, Molecule]
            molecules, that should be considered.

        Returns
        -------
        pd.DataFrame
            returns the results and timings of the calculation.

        Raises
        ------
        ValueError
            if no rotatable bond can be found.
        """
        root_job_status = {}
        if out_dir is None:
            out_dir = "."
        if approach_name is None:
            approach_name = "mol"
        if self._submission_system is None:
            self._interactive = True
        else:
            self._interactive = False

            # build out_rootdir
            if not bash.path.isdir(out_dir):
                bash.makedirs(out_dir)

        # Get all torsion atom ids
        if fragment_mol:
            fragment_tasks = fragment_mol_complete(mol=mol)
            parent_mols = {frag_tors:parent_mol for _, frag_tors, _, parent_mol in fragment_tasks}
            parent_tors_ids = {frag_tors:parent_tors_ids for _, frag_tors, parent_tors_ids,
            _ in fragment_tasks}
            print(fragment_tasks)
            tors_tasks = [(frag_mol, frag_tors) for frag_mol, frag_tors, _, _ in fragment_tasks]
        else:
            all_tors = get_all_torsion_atoms_idx(mol)
            tors_tasks = [(mol, tors) for tors in all_tors]

        print(tors_tasks)
        iterations = list(enumerate(tors_tasks, start=_start_ind))
        iterator = tqdm(iterations, desc="Calculate Torsions")

        # Start Calculations
        collected_res = []
        for ind, (tors_mol, torsion_atom_ids) in iterator:
            sub_approach_name = f"{approach_name}_torsion{ind}_"+ "_".join(map(str, torsion_atom_ids))

            res = self.calculate_torsion_profile(
                mol=tors_mol,
                torsion_atom_ids=torsion_atom_ids,
                out_dir=out_dir,
                approach_name=sub_approach_name,
                collect_only=collect_only,
            )
            if isinstance(res, dict):
                root_job_status.update(res)
            else:
                root_job_status.update(self.job_status)
                res["torsion_atom_ids"] = [torsion_atom_ids for _ in range(res.shape[0])]

            # work around.
            collected_res.append(res)

        self._job_status = root_job_status

        dfs = list(filter(lambda f: isinstance(f, pd.DataFrame), collected_res))

        if len(dfs) > 0:
            res = pd.concat(dfs)
            if fragment_mol:
                #print("frag tasks: ", tors_tasks)
                #print("parent_ids: ", parent_tors_ids)
                #print("parenting: ", parent_mols)
                #print(res["torsion_atom_ids"].to_list())
                res["parent_mol"] = [parent_mols[frag_tors] for frag_tors in res["torsion_atom_ids"]]
                res["parent_torsion_atom_ids"] = [parent_tors_ids[frag_tors] for frag_tors in res[
                    "torsion_atom_ids"]]
            store_mol_db(df_mols=res,
                         out_sdf_path=f"{out_dir}/{approach_name}_all_tps.sdf")
        else:
            res = collected_res

        return res

    def calculate_torsions_iter_mols(
        self,
        mols: Iterable[Chem.Mol],
        torsions: list[tuple[int, int, int, int]],
        out_dir: str = None,
        approach_name: str = None,
        collect_only: bool = False,
        jobarray_end: Iterable[int] = None,
        jobarray_lim: int = 60,
        jobarray_junk_size: int = 5000,
    ) -> Iterable[pd.DataFrame]:
        """
        calcualte torsions for multiple mols.

        Parameters
        ----------
        mols: Iterable[Chem.Mol]
        torsions: list[tuple[int, int, int, int]]
        jobarray_end: Iterable[int]
        jobarray_lim: int
        collect_only: bool
        jobarray_junk_size: int

        Returns
        -------
        Iterable[pd.DataFrame]
            list of pd.Dataframes containing the results.
        """


        # Check:
        if len(mols) != len(torsions):
            raise ValueError("The length of mols and torsion must be equal!")
        if any(
                v is None for v in [out_dir, approach_name, self._submission_system]
        ):
            raise IOError(
                "iter mols can not be used interactivly (write a for loop you sloth)! Please "
                "provide out_folder_path, out_file_prefix and submission_system."
            )

        new_torsions = []
        for tors in torsions:
            new_torsions.append(tuple(map(int, tors)))
        torsions = new_torsions

        # Settings
        if approach_name is not None:
            root_out_prefix = approach_name
            tmp_out_file_prefixes = [root_out_prefix + "_" + str(i) for i in range(len(mols) + 1)]
        elif isinstance(approach_name, Iterable):
            root_out_prefix = "mol"
            tmp_out_file_prefixes = [
                root_out_prefix + "_" + mid_prefix + "_" + str(i)
                for i, mid_prefix in enumerate(approach_name)
            ]
        else:
            root_out_prefix = "mol"
            tmp_out_file_prefixes = [root_out_prefix + "_" + str(i) for i in range(len(mols) + 1)]

        root_out_dir = out_dir
        root_out_result_file_path = root_out_dir + "/" + root_out_prefix + ".sdf"

        if not bash.path.exists(root_out_dir):
            bash.makedirs(root_out_dir)

        self._job_status_file_path = self._root_job_status_file_path = (
            root_out_dir + "/" + root_out_prefix + "_jobs.json"
        )
        if bash.path.exists(self._root_job_status_file_path):
            self.load_job_status(job_status_file_path=self._root_job_status_file_path)

        #=======================================
        # BUILD
        #=======================================
        # Build sub-folders & and read results if possible
        orig_submission_setting = self._submission_system.submission
        self._submission_system.submission = False

        # batching - Parallelization!
        nbatches = self.n_processes * 10 if (self.n_processes * 10 < len(mols)) else len(mols)
        nmols = len(mols) - 1
        batch_size = nmols // nbatches
        n_rest = nmols % nbatches
        batch_sizes = [batch_size + 1 if (i < n_rest) else batch_size for i in range(nbatches)]
        batch_mol_ranges = [
            list(range(sum(batch_sizes[:i]), sum(batch_sizes[:i]) + batch_size + 1))
            for i in range(len(batch_sizes))
        ]

        with mult.Pool(self.n_processes) as p:
            tp_ks = vars(self)
            distribute_jobs = [
                (n, mols, torsions, out_dir, tmp_out_file_prefixes, batch_mol_ranges[n], tp_ks)
                for n in range(nbatches)
            ]

            p_job_res = p.starmap(
                thread_prepare_input,
                tqdm(distribute_jobs, total=nbatches, desc="Prepare Mols (batched): ",
                leave=False),
            )
            p.close()
            p.join()

        sub_job_tasks = []
        sub_job_results = []
        job_files = []
        for j_sub_job_results, j_sub_job_tasks, j_job_files in p_job_res:
            sub_job_tasks.extend(j_sub_job_tasks)
            sub_job_results.extend(j_sub_job_results)
            job_files.extend(j_job_files)

        sub_job_tasks = dict(sub_job_tasks)
        sub_job_results = dict(sub_job_results)
        self._submission_system.submission = orig_submission_setting

        # =================================
        # DO
        # =================================
        sub_job_results = self.generate_job_array(
            collect_only,
            jobarray_end,
            job_files,
            jobarray_lim,
            root_out_dir,
            root_out_prefix,
            root_out_result_file_path,
            sub_job_results,
            sub_job_tasks,
            jobarray_junk_size,
        )

        if isinstance(self._submission_system, Local):
            sub_jobs_iterator = list(enumerate(zip(mols, torsions)))
            tmp_sub_job_results = []
            for i, (mol, torsion) in tqdm(sub_jobs_iterator, desc="Collecting Results"):
                approach_name = tmp_out_file_prefixes[i]
                res = self.calculate_torsion_profile(mol=mol, torsion_atom_ids=torsion, out_dir=
                out_dir, approach_name=approach_name, collect_only=True)
                res["torsion_molID"] = approach_name+ "_ta_" + "_".join(map(str, torsion))
                res["molID"] = approach_name
                tmp_sub_job_results.append(res)
            sub_job_results = pd.concat(tmp_sub_job_results)
        return sub_job_results

    def calculate_2D_torsion_mol(
        self,
        mol: Chem.Mol,
        torsion1: tuple[int, int, int, int],
        torsion2: tuple[int, int, int, int],
        out_dir:str=None,
        approach_name:str=None,
        _additional_pos_res=None,
        _additional_torsions=None,
        job_end: Iterable[int] = None,
        job_lim=60,
        collect_only: bool = False,
        submission_batch_size: int = 5000,
    ) -> Iterable[pd.DataFrame]:
        """

        Parameters
        ----------
        mol
        torsion1
        torsion2
        job_end
        _additional_pos_res
        _additional_torsions
        job_lim
        collect_only
        submission_batch_size

        Returns
        -------

        """
        # Settings
        if self._submission_system is not None:
            root_out_prefix = approach_name
            tmp_out_file_prefixes = [
                root_out_prefix + "_angle1_" + str(i) for i in range(self.n_measurements)
            ]
            root_out_dir = out_dir
            root_out_result_file_path = root_out_dir + "/" + root_out_prefix + ".sdf"
            self._job_status_file_path = self._root_job_status_file_path = (
                root_out_dir + "/" + root_out_prefix + "_jobs.json"
            )

            if bash.path.exists(self._root_job_status_file_path):
                self.load_job_status(job_status_file_path=self._root_job_status_file_path)

            orig_submission_setting = self._submission_system.submission
            self._submission_system.submission = False
            self._interactive = False
        else:
            self._interactive = True
            str_t1 = "_".join(map(str, torsion1))
            str_t2 = "_".join(map(str, torsion2))
            root_out_prefix = f"Mol_ta_{str_t1}_tb_{str_t2}"
            tmp_out_file_prefixes = [
                root_out_prefix + "_angle2_" + str(i) for i in range(self.n_measurements)
            ]

        # Build sub-folders & and read results if possible
        torsion_angle2 = list(np.linspace(-180, 180, self.n_measurements))

        # batching - PARALLELIZATION
        nbatches = (
            self.n_processes * 5 if (self.n_processes * 5 < len(torsion_angle2)) else len(torsion_angle2)
        )  # more batches then processes allow faster processes to thrive
        nprofiles = len(torsion_angle2)

        batch_size = nprofiles // nbatches
        n_rest = nprofiles % nbatches
        batch_sizes = [batch_size + 1 if (i < n_rest) else batch_size for i in range(nbatches)]
        batch_angles_ranges = [
            list(range(sum(batch_sizes[:i]), sum(batch_sizes[:i]) + batch_sizes[i]))
            for i in range(len(batch_sizes))
        ]

        with mult.Pool(self.n_processes) as p:
            tp_ks = vars(self)
            distribute_jobs = [
                (
                    n,
                    mol,
                    torsion_angle2,
                    torsion1,
                    torsion2,
                    out_dir,
                    tmp_out_file_prefixes,
                    batch_angles_ranges[n],
                    tp_ks,
                    _additional_pos_res,
                    _additional_torsions,
                )
                for n in range(nbatches)
            ]

            p_job_res = p.starmap(
                thread_prepare_2d_angle_input,
                tqdm(distribute_jobs, total=nbatches, desc="Prepare Mols (batched): "),
            )
            p.close()
            p.join()

        if self._interactive:
            res_dfs = []
            for j_sub_job_results, j_sub_job_tasks, j_job_files in p_job_res:
                res_dfs.extend([jdf[1] for jdf in j_sub_job_results])

            res_df = pd.concat(res_dfs)
            str_t1 = "_".join(map(str, res_df.torsion_atom_ids.iloc[0]))
            str_t2 = "_".join(map(str, torsion2))
            res_df["torsion_molID"] = f"{approach_name}_ta_{str_t1}_tb_{str_t2}"
            res_df["torsion_atom_ids_2"] = [torsion2 for _ in range(len(res_df))]
            res_df["molID"] = approach_name

        else:
            sub_job_tasks = []
            sub_job_results = []
            job_files = []
            for j_sub_job_results, j_sub_job_tasks, j_job_files in p_job_res:
                sub_job_tasks.extend(j_sub_job_tasks)
                sub_job_results.extend(j_sub_job_results)
                job_files.extend(j_job_files)

            sub_job_tasks = dict(sub_job_tasks)
            sub_job_results = dict(sub_job_results)
            if self._submission_system is not None:
                self._submission_system.submission = orig_submission_setting

                sub_job_results = self.generate_job_array(
                    collect_only,
                    job_end,
                    job_files,
                    job_lim,
                    root_out_dir,
                    root_out_prefix,
                    root_out_result_file_path,
                    sub_job_results,
                    sub_job_tasks,
                    submission_batch_size,
                )

            if not isinstance(sub_job_results, pd.DataFrame) and len(
                [v for v in sub_job_results.values() if isinstance(v, pd.DataFrame)]
            ) != len(sub_job_results):
                tmp_sub_job_results = {}
                sub_jobs_iterator = list(enumerate(torsion_angle2))
                for i, angle2 in tqdm(sub_jobs_iterator, desc="Collecting Results"):
                    torsion2_angle = [torsion2[0], torsion2[1], torsion2[2], torsion2[3], angle2]

                    approach_name = tmp_out_file_prefixes[i]
                    const_torsions = [torsion2_angle]
                    if _additional_torsions is not None:
                        const_torsions.extend(_additional_torsions)

                    res = self.calculate_torsion_profile(
                        mol=mol,
                        torsion_atom_ids=torsion1,
                        out_dir=out_dir,
                        approach_name=approach_name,
                        additional_torsions=const_torsions,
                        collect_only=True,
                    )

                    if isinstance(res, pd.DataFrame):
                        str_t1 = "_".join(map(str, res.torsion_atom_ids.iloc[0]))
                        str_t2 = "_".join(map(str, torsion2))
                        res["torsion_molID"] =f"{approach_name}_ta_{str_t1}_tb_{str_t2}"
                        res["torsion_atom_ids_2"] = [torsion2 for _ in range(len(res))]
                        res["molID"] = approach_name
                        res["torsion_angle_b"] = angle2

                    tmp_sub_job_results.update({tmp_out_file_prefixes[i]: res})
                sub_job_results = tmp_sub_job_results
            if all(isinstance(df, pd.DataFrame) for df in sub_job_results.values()):
                dfs = sub_job_results.values()
                res_df = pd.concat(dfs)
                store_mol_db(res_df, f"{out_dir}/{root_out_prefix}_2dtp.sdf")

        return res_df

    def wait(self, no_collect:bool=False):
        """
            wait for finishing of the job.

        Parameters
        ----------
        no_collect: bool
            don't trigger collect.

        Returns
        -------
        dict
            or None if no_collect.

        """
        if self._submission_system is not None and isinstance(self._submission_system, Slurm):
            start_time = datetime.now()
            job_ids = []

            for job in self._job_status:
                for _, inf in self._job_status[job]["sub_jobs"].items():
                    job_ids.append(inf["jobID"])
            log.info(f"waiting for jobs: {job_ids}")
            log.info(f"START: {start_time}")
            self._submission_system.wait_for_jobs(job_ids)

            if not no_collect:
                res = self.collect_results()
            else:
                res = None

            end_time = datetime.now()
            log.info(f"END: {end_time}")
            log.info(f"Duration:  {end_time-start_time}")
            return res
        return self.collect_results()

    def collect_results(self) -> dict:
        """
        collect results of the triggered functions.

        Returns
        -------
        dict
            results.
        """
        iter_dict = copy.deepcopy(self._job_status)
        res = []
        for job in iter_dict:
            approach_name = iter_dict[job]["approach_name"]
            out_folder_path = iter_dict[job]["root_out_folder"]
            self.tmp_calculation_path = iter_dict[job]["work_folder"]

            self._check(out_folder_path=out_folder_path, approach_name=approach_name,
                        out_suffix="tp",  force_run=False, collect_only=True)

            if (
                self._sub_results_opt_structures is None or len(self._sub_results_opt_structures) == 0
            ):
                self._sub_results_opt_structures = [
                    values["out_tp"]
                    for k, values in sorted(iter_dict[job]["sub_jobs"].items())
                ]
            res.append(self._gather_results())

        if(len(res)>0):
            res = pd.concat(res)
        return res

    # props:
    @property
    def job_status(self) -> dict:
        """
        getter status of the slurm jobs.
        """
        return self._job_status

    # Utils
    def load_job_status(self, job_status_file_path: str):
        """
        load job status from file.
        """
        def parse_json_helper(d: dict)->dict:
            """
                this helper, gives correct typing to the json load command.

            Parameters
            ----------
            d: dict
                loaded dict

            Returns
            -------
            dict
                parsed and typed
            """
            res_d = {}
            for k, v in d.items():
                n_k = k
                n_v = v
                if isinstance(k, str) and k.lstrip("-").isdigit():
                    n_k = int(k)

                if isinstance(v, str) and v.lstrip("-").isdigit():
                    n_v = int(v)
                elif isinstance(v, dict):
                    n_v = parse_json_helper(v)

                res_d[n_k] = n_v

            return res_d

        with open(job_status_file_path, "r") as fc:
            stat_d = json.load(fc, object_hook=parse_json_helper)
            if self._job_status is None:
                self._job_status =stat_d
            else:
                self._job_status.update(stat_d)

    def save_job_status(self, job_status_file_path: str):
        """
        load job status to file.
        """
        json.dump(self._job_status, open(job_status_file_path, "w"), indent="    ")

    # Privates
    def _run(self, mol, torsion_atom_ids, _additional_pos_res, _additional_torsions, approach_name):
        """

        Parameters
        ----------
        mol
        torsion_atom_ids

        Returns
        -------

        """

        # Generate initial torsion profiles
        log.info("\t Stage 1: Generate Torsion Profile", stacklevel=2)
        self._tmp_input_profile_mol = self.initial_torsion_profile_generator.generate(
            mol=mol,
            torsion_atom_ids=torsion_atom_ids,
            _additional_pos_res=_additional_pos_res,
            _additional_torsions=_additional_torsions,
        )

        # Create an instance of the model
        log.info("\t Stage 2: Optimize and calculate Potential", stacklevel=2)
        if hasattr(self.calculator, "calculate_conformer_potentials"):
            func_options = {
                "mol": self._tmp_input_profile_mol,
                "torsion_atom_ids": torsion_atom_ids,
                "out_file_path": self.out_result_path,
            }

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

            task_splits = self._split_tasks(
                ntasks=self._tmp_input_profile_mol.shape[0],
                n_tasks_parallel=self.n_tasks_parallel
            )
            for task_id, task_load in enumerate(task_splits):
                if (self.tmp_calculation_path is not None):
                    tmp_out_dir = f"{self.tmp_calculation_path}/tmp_{approach_name}_{task_id}"
                    bash.makedirs(tmp_out_dir, exist_ok=True)
                else:
                    tmp_out_dir=None

                if (self.tmp_calculation_path is not None):
                    tmp_name = bash.path.basename(self.out_result_path.replace(".sdf", f"_{task_id}.sdf"))
                    tmp_out_file = f"{tmp_out_dir}/{tmp_name}"
                else:
                    tmp_out_file = None

                tmp_func_options = copy.deepcopy(func_options)
                tmp_func_options["out_file_path"] = tmp_out_file
                tmp_func_options["_conf_range"] = task_load

                # Optionals
                if hasattr(self, "_additional_torsions") and _additional_torsions is not None:
                    tmp_func_options["_additional_torsions"] = _additional_torsions
                if hasattr(self, "_additional_pos_res") and _additional_pos_res is not None:
                    tmp_func_options["_additional_pos_res"] = _additional_pos_res

                if check_subjob_done(task_id):
                    log.info("\t\tsubjob " + str(task_id) + " already finished! - SKIP")
                    continue

                if self._interactive or isinstance(self._submission_system, Local):
                    log.info("\t\tsubjob " + str(task_id) + " calculate")
                else:
                    log.info("\t\tsubjob " + str(task_id) + " submit")

                tres = self._do_calc(
                    func=self.calculator.calculate_conformer_potentials,
                    func_options=tmp_func_options,
                    _out_folder_path=tmp_out_dir,
                    _out_file_prefix=str(approach_name) + "_" + str(task_id),
                )

                if isinstance(self._computing, ScriptComputing):
                    job_desc = {
                        "jobID": tres,
                        "status": "PD",
                        "output_dir": tmp_out_dir,
                        "out_tp": tmp_out_file,
                    }

                    if task_id in res[approach_name]["sub_jobs"]:
                        res[approach_name]["sub_jobs"][task_id].update(job_desc)
                    else:
                        res[approach_name]["sub_jobs"][task_id] = job_desc
                else:
                    res = tres

        else:
            raise ValueError(
                "Calculator has no calculate_conformer_potentials function, but this is needed!"
            )
        return res

    def _gather_results(self) -> pd.DataFrame:
        """

        Parameters
        ----------
        torsion_atom_ids

        Returns
        -------

        """
        # Found final results
        if self.out_result_path is not None and bash.path.isfile(self.out_result_path):
            log.info(f"\t Loading final results: {self.out_result_path}", stacklevel=2)
            self.out_tp = read_mol_db(self.out_result_path)
        else: # Piece together sub results
            log.info("\t Building final results from sub_results", stacklevel=2)

            # Structure information
            # try to readin all input structures
            if (self._sub_results_in_structures is not None
                  and len(self._sub_results_in_structures)> 0): # try to
                self._tmp_input_profile_mol  = read_mol_db(self._sub_results_in_structures[0])

            # try to calculate the torsion angles from the output structures.
            if (
                    self._sub_results_opt_structures is not None
                    and len(self._sub_results_opt_structures) > 0
            ):  # try reading in strucutre information of final results.
                bash.wait_for_file_system(self._sub_results_opt_structures)
                mols = [read_mol_db(f) for f in self._sub_results_opt_structures]
                self.out_tp = pd.concat(mols)

                store_mol_db(
                    self.out_tp,
                    self.out_result_path,
                )

        return self.out_tp

    def _previous_results_present(self, out_result_file_path, approach_name) -> tuple[bool, bool]:
        """

        Returns
        -------
        tuple[bool, bool]
             fin_results, all_subjobs_present
        """
        # are there final results?
        fin_results = not self._interactive and bash.path.exists(out_result_file_path)
        log.info(f"Found Final Result File: {fin_results}", stacklevel=2)

        # subjobs?
        self._sub_results_opt_structures = []
        self._sub_results_in_structures = []
        self._sub_results_finished = {}

        sub_job_dirs = [
            f"{self.tmp_calculation_path}/{a}"
            for a in bash.listdir(self.tmp_calculation_path)
            if (
                bash.path.isdir(f"{self.tmp_calculation_path}/{a}")
                and a.startswith(f"tmp_{approach_name}")
            )
        ]

        nsubjobs = len(sub_job_dirs)
        for d in sub_job_dirs:
            sub_job_number = int(str(d.split("_")[-1]))

            in_structure = bash.glob(d + f"/in*{approach_name}*.sdf")
            s_results = bash.glob(f"{d}/{approach_name}*tp*.sdf")

            if len(s_results) == 0:
                s_results = bash.glob(f"{d}/{approach_name}*emin.pdb")

            self._sub_results_in_structures.extend(in_structure)

            sub_job_finished = False
            if len(s_results) == 1:
                self._sub_results_opt_structures.append(s_results[0])
                if len(self._job_status[approach_name]["sub_jobs"]) > 0:
                    self._job_status[approach_name]["sub_jobs"][sub_job_number][
                        "status"
                    ] = "DONE"
                sub_job_finished = True
            elif len(s_results) > 1:
                raise ValueError("oh oh to many pdb files in subjob folder: " + d)

            self._sub_results_finished[int(sub_job_number)] = sub_job_finished

        all_subjobs_present = False
        self._partial_done = False
        if len(self._sub_results_opt_structures) > 0:
            all_subjobs_present = nsubjobs == len(self._sub_results_opt_structures)

            if not all_subjobs_present and len(self._sub_results_opt_structures) / float(nsubjobs) > 0.55:
                self._partial_done = True

        log.info(
            f"Found all_subjobs Result File:{all_subjobs_present}, found sub_results:"
            f" {len(self._sub_results_opt_structures)}", stacklevel=2,
        )

        return fin_results, all_subjobs_present
