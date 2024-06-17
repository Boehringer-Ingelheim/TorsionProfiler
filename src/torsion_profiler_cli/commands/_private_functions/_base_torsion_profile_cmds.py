# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
This Module implements the functionality for all torsion profiling CLI-commands.
"""

import os
import ast
import time
from typing import Union
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
from rdkit import Chem

from torsion_profiler import conf
from torsion_profiler.utils import bash
from torsion_profiler.orchestration.submission_systems import Slurm, Local
from torsion_profiler.engines.openmm_ml_calculator import AniCalculator

from torsion_profiler.tools.torsion_profiler import TorsionProfiler
from torsion_profiler.engines._abstractcalculator import _AbstractCalculator
from torsion_profiler.visualize.visualize_torsion_profiles import plot_torsion_profile
from torsion_profiler.tools.torsion_profiler.torsion_profile_generators import LandscaperTorsionProfileGenerator
from torsion_profiler.utils.mol_db_operation import store_mol_db

from torsion_profiler_cli.parameters import (
    MOL,
    TORSIONATOMS,
    TORSIONATOMSB,
    OUTPUT_DIR,
)


def _base_torsion_profile(
    in_mol: str,
    out_folder: str,
    torsion_atom_ids: tuple[int, int, int, int],
    calculator: _AbstractCalculator,
    n_measurements: int = 24,
    submit_to_queue: bool = False,
    n_tasks_parallel: int = 1,
    n_processors: int = 5,
    all_torsion_atoms=False,
    all_fragmented_torsions=False,
    _conda_env: str = None,
    _wait_for_jobs: bool = True,
    torsion_atom_ids_b: tuple[int, int, int, int] = None,
    _debug: bool = False,
) -> Union[dict, DataFrame]:
    """
    This function contains the functionality for all torsion profiling commands in the CLI.

    Parameters
    ----------
    in_mol: str
        path to the input molecule
    out_folder: str
        path to the output folder
    torsion_atom_ids: tuple[int, int, int, int]
        atom ids describing the torsion with four atoms. (starts with 1)
    calculator: _AbstractCalculator
        torsion_profiler calculator class for level of theory
    n_measurements: int
        number of mesearuments along rotation axis. (default: 37)
    submit_to_queue: bool
        if True, will try to submit job to a Slurm queue.
    n_tasks_parallel: int
        number of tasks, that will executed. This will split the optimizations of
        n_measurements by n_tasks_parallel.
    n_processors: int
        number of available processes
    all_torsion_atoms: bool
        should all torsions of the molecule be calculated?
    _conda_env: str
        path to the conda env, that should be used.
    _wait_for_jobs: bool
        if True, the CLI will wait till job is finished.
    torsion_atom_ids_b: tuple[int, int, int, int]
        can be used to calculate a 2D torsion profile
    _debug: bool
        if true, will leave all tmp files present.

    Returns
    -------
    Union[dict, DataFrame]
        if the job was done, return a pandas dataframe. if the job is stil running, it is
         returning a dict, with the job description.
    """
    start_datetime = datetime.now()
    header = """
    =================================================================
        TorsionProfiler
    =================================================================\n
    """
    print(header)
    print("START: ", start_datetime)

    if _conda_env is None:
        _conda_env = conf["conda_calculator_envs"][calculator.__class__.__name__]

    # PARSE
    print("\n" + "#" * 120 + "\nInput: ")
    print("\tin_mol:", in_mol)
    out_folder = OUTPUT_DIR.get(out_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    mol = MOL.get(in_mol)

    if all_torsion_atoms or all_fragmented_torsions:
        torsion_atom_ids = None
    elif torsion_atom_ids is not None:
        torsion_atom_ids = TORSIONATOMS.get(torsion_atom_ids)
    elif not isinstance(mol, list) and mol.HasProp("torsion_atom_ids"):
        torsion_atom_ids = ast.literal_eval(mol.GetProp("torsion_atom_ids"))
    elif isinstance(mol, list):
        torsion_atom_ids = None
    else:
        raise ValueError(
            "You need to provide option all torsions or torsion atom ids either via the "
            "flag or as a field in the sdf named: 'torsion_atom_ids'"
        )

    print("\ttorsion_atom_ids:", torsion_atom_ids)

    if torsion_atom_ids_b is not None:
        torsion_atom_ids_b = TORSIONATOMSB.get(torsion_atom_ids_b)
        print("\ttorsion_atom_ids 2:", torsion_atom_ids_b)

    tpg = LandscaperTorsionProfileGenerator() #FastProfileGenerator()


    print("\tout_folder:", out_folder)
    print("\tn_points:", n_measurements)
    print("\tinit_tp_generator: ", tpg.__class__.__name__)
    print("\tcalculator: ", calculator.__class__.__name__)
    print("\tn_processes:", n_processors)

    print()

    # Setup machinery
    if submit_to_queue:
        submission_system = Slurm(
            nmpi=6, job_duration="24:00:00", conda_env_path=_conda_env,
            partition="cpu-mockgiant"
        )

        if isinstance(calculator, AniCalculator):
            submission_system.partition = "cpu-mockgiant" #"gpu"

        out_prefix = os.path.basename(out_folder)
        out_folder = str(out_folder)
    else:

        submission_system = Local(conda_env_path=_conda_env)
        out_prefix = os.path.basename(out_folder)
        out_folder = str(out_folder)
    tmp_dir = f"{out_folder}/{out_prefix}"

    # leads to much nicer profiles, compared to easy profiler.

    tp = TorsionProfiler(
        calculator=calculator,
        initial_tors_profile_generator=tpg,
        n_measurements=n_measurements,
        submission_system=submission_system,
        n_processes=n_processors,
    )

    tp.n_tasks_parallel = n_tasks_parallel
    tp.n_tasks_parallel = n_measurements


    # Calculate
    print("\n" + "#" * 120 + "\nCalculate: ")

    if isinstance(mol, list) and len(mol) == 1:
        mol = mol[0]

    if isinstance(mol, Chem.Mol) :  # 1/2D 1 Mol 1 Tors
        if all_torsion_atoms or all_fragmented_torsions:
            print("Running all 1D Profiles")
            df = tp.calculate_all_torsions_profiles(mol=mol,
                                                    collect_only=False,
                                                    out_dir=out_folder,
                                                    fragment_mol=all_fragmented_torsions,
                                                    approach_name=out_prefix)
            if submit_to_queue and _wait_for_jobs and not isinstance(df, pd.DataFrame):
                df = tp.wait()

        elif torsion_atom_ids_b is not None:
            print("Running 2D Profile")
            df = tp.calculate_2D_torsion_mol(
                mol=mol, torsion1=torsion_atom_ids, torsion2=torsion_atom_ids_b
            )
            if submit_to_queue and _wait_for_jobs and not isinstance(df, pd.DataFrame):  # Post ana
                tp.wait(no_collect=True)
                df = tp.calculate_2D_torsion_mol(
                    mol=mol,
                    torsion1=torsion_atom_ids,
                    torsion2=torsion_atom_ids_b,
                    collect_only=True,
                )
        else:
            print("Running 1D Profile")
            df = tp.calculate_torsion_profile(
                mol=mol,
                approach_name=out_prefix,
                out_dir=out_folder,
                torsion_atom_ids=torsion_atom_ids,
            )

            # Post ana
            if submit_to_queue and _wait_for_jobs and not isinstance(df, pd.DataFrame):
                df = tp.wait(no_collect=False)
            elif tp.n_tasks_parallel > 1:
                df = tp.calculate_torsion_profile(
                    mol=mol, torsion_atom_ids=torsion_atom_ids, collect_only=True
                )
            print("")
        print()
    elif isinstance(mol, list) and len(mol) > 1:  # Nmol, all tors
        tp.verbose = False
        df = tp.calculate_all_torsions_iter_mols(mols=mol, collect_only=False,
                                                  approach_name=out_prefix,
                                                  out_folder_path=out_folder,
                                                  )
        if submit_to_queue and _wait_for_jobs and not isinstance(df, pd.DataFrame):
            df = tp.wait()

    # Post File Generation
    if isinstance(df, pd.DataFrame):
        print("\n" + "#" * 120 + "\nOUTPUT: ")
        # Add final inf.
        df["Calculator"] = calculator.__class__.__name__
        df["TPG"] = tpg.__class__.__name__
        #clip:
        df["rel_potential_energy"] =df["rel_potential_energy"].clip(lower=0, upper=30)


        # plot
        mpl.use("Agg")
        print(f"\tWriting plot to: ")
        tdf=pd.DataFrame(df)
        tdf["torsion_atom_ids"] = tdf["torsion_atom_ids"].apply(lambda x: "_".join(map(str, x)))
        tais = np.unique(tdf["torsion_atom_ids"].to_list(), axis=0)
        for tai in tais:
            ttdf = tdf.loc[tdf["torsion_atom_ids"] == tai]
            ttdf["torsion_atom_ids"] = ttdf["torsion_atom_ids"].apply(lambda x: tuple(map(int, x.split("_"))))
            ttdf["rel_potential_energy"] = ttdf["potential_energy"] - ttdf["potential_energy"].min()

            out_plot_path = f"{out_folder}/{out_prefix}_torsion_{tai}_profile.png"
            print(f"\t\t{out_plot_path}")
            plot_torsion_profile(
                tp_df=ttdf, out_fig_path=out_plot_path, title_prefix=out_prefix, ylim=[0,20],
            )

        # Write out .csv
        out_csv_path = f"{out_folder}/{out_prefix}_tp.csv"
        print(f"\tWriting df-csv to: {out_csv_path}")
        exclude = ["ROMol"]
        df[[c for c in df.columns if c not in exclude]].to_csv(out_csv_path)

        # store df
        out_sdf_path = f"{out_folder}/{out_prefix}_tp.sdf"
        print(f"\tWriting df-sdf to: {out_sdf_path}")

        store_mol_db(df, out_sdf_path=out_sdf_path)

        if not _debug:
            if bash.path.isdir(tmp_dir) :
                print(f"\tdelete tmp_dir: {tmp_dir}")
                bash.rmtree(tmp_dir)

    end_datetime = datetime.now()
    duration = end_datetime - start_datetime
    print()
    print("OUT DF:")
    print(df)
    print()
    print("END: ", end_datetime)
    print("DURATION: ", duration)
    print()
    return df
