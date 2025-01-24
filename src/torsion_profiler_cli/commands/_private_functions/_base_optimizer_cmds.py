# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
Here the basic functionality for the optimizers is defined.
"""
import os
from datetime import datetime
import pandas as pd
from pandas import DataFrame


from torsion_profiler import conf
from torsion_profiler.engines._abstractcalculator import _AbstractCalculator
from torsion_profiler.tools.geom_optimizer import GeomOptimizer
from torsion_profiler.utils import bash
from typing import Union
from torsion_profiler.orchestration.submission_systems import Slurm, Local

from torsion_profiler_cli.parameters import MOL, OUTPUT_DIR


def _base_geom_optimizer(
    in_mol: str,
    out_folder: str,
    _calculator: _AbstractCalculator,
    submit_to_queue: bool = False,
    _conda_env: str = None,
    _debug:bool=True,
) -> Union[dict, DataFrame]:
    """
     This function is implementing the optimizer core functionality.

     Parameters
     ----------
     in_mol: str
         path to the input molecule
     out_folder: str
         path to the output folder
     _calculator: _AbstractCalculator
         torsion_profiler calculator class for level of theory
     submit_to_queue: bool
         if True, will try to submit job to a Slurm queue.
     _conda_env: str
         path to the conda env, that should be used.


     Returns
     -------
    Union[dict, DataFrame]
         if the job was done, return a pandas dataframe. if the job is stil running,
         it is returning a dict, with the job description.
    """
    start_datetime = datetime.now()
    print("START: ", start_datetime)

    if _conda_env is None:
        _conda_env = conf["conda_calculator_envs"][_calculator.__class__.__name__]

    # PARSE
    print("INPUT: ")
    print("\tin_mol:", in_mol)
    print("\tout_folder:", out_folder)

    out_folder = OUTPUT_DIR.get(out_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    mols = MOL.get(in_mol)

    print()
    # Setup machinery
    if submit_to_queue:
        submission_system = Slurm(nmpi=4, job_duration="24:00:00", conda_env_path=_conda_env)
        out_prefix = "work"
        out_folder = str(out_folder)
    else:
        submission_system = Local(conda_env_path=_conda_env)
        out_prefix = "work"
        out_folder = str(out_folder)

    opt = GeomOptimizer(
        calculators=[_calculator],
        submission_system=submission_system,
    )

    # Calculate
    dfs = []
    if len(mols) > 0:
        for i, mol in enumerate(mols):
            opt.out_file_prefix = out_prefix + "_" + str(i)
            df = opt.optimize(mol=mol, approach_name=out_prefix, out_dir=out_folder)
            dfs.append(df)
    else:
        df = opt.optimize(mol=mols[0])
        dfs.append(df)

    # Post ana
    if submit_to_queue:
        df = opt.wait()

    if isinstance(df, pd.DataFrame):
        out_path = out_folder + "/" + out_prefix + "_optimization.sdf"
        print("Writing results out to :", out_path)
        from torsion_profiler.utils import store_mol_db
        store_mol_db(df, out_path)
        if(bash.path.exists(out_folder+"/work") and not _debug):
            bash.system("rm -r "+out_folder+"/work")

    end_datetime = datetime.now()
    duration = end_datetime - start_datetime
    print()
    print("END: ", start_datetime)
    print("DURATION: ", duration)
    print()
    print("Result:")
    return df
