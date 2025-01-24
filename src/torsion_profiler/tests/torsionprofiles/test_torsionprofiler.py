"""
Test torsion profile tool.
"""
import ast
import tempfile
import pytest
import numpy as np
import pandas as pd

from ... import conf
from ... import TorsionProfiler
from ...utils import bash
from ...orchestration.submission_systems import Slurm, Local
from ...engines import (
    Mmff94Calculator,
    OffCalculator,
    AniCalculator,
    MACECalculator,
    Psi4Calculator,
    XtbCalculator,
    GaussianCalculator,
)

from .. import tmp_root_path

test_local = True
test_slurm = True
slurm_avail = bash.program_binary_found("sbatch")

test_off = True
test_ani = True
test_mace = True
test_xtb = False
test_gaus = False
test_2d_MMFF94_only = True

test_calculators = [Mmff94Calculator,
                    OffCalculator,
                    AniCalculator,
                    MACECalculator,
                    XtbCalculator,
                    GaussianCalculator,
                    #Psi4Calculator,
                    ]
not_interactive = [XtbCalculator, GaussianCalculator]

test_interactive_calculator = [c for c in test_calculators if c not in not_interactive]


# Test Control:
test_job_splitting = [1, 2, 5]

test_submitters = []
if test_local:
    test_submitters.append(Local)
if test_slurm and slurm_avail:
    test_submitters.append(Slurm)

if test_2d_MMFF94_only:
    test_calculators_2D = [
        Mmff94Calculator,
    ]
else:
    test_calculators_2D = test_calculators

test_2d_interactive_calculator = [c for c in test_calculators_2D if c not in not_interactive]

# Do Tests
@pytest.mark.parametrize(
    "calculator_class",
    test_interactive_calculator,
)
def test_tp_1d_interactive(calculator_class, butan, butan_tp_1d_5n_results):
    """
    test 1D interactive exec TP
    """
    expected_rel_energy_results = butan_tp_1d_5n_results[Mmff94Calculator.__name__]
    expected_angles = butan_tp_1d_5n_results["angles"]
    n_measurements = int(butan_tp_1d_5n_results["n_measurements"])

    torsion_atom_ids = tuple(map(int, butan.GetProp("ta").split(" ")))

    calculator = calculator_class()
    tp = TorsionProfiler(calculator=calculator, n_measurements=n_measurements)

    # Calculate
    df = tp.calculate_torsion_profile(mol=butan, torsion_atom_ids=torsion_atom_ids)

    # Verify
    np.allclose(a=expected_rel_energy_results, b=df["rel_potential_energy"].to_numpy(),
                rtol=10 ** -2)
    np.allclose(a=expected_angles, b=df["torsion_angle"].to_numpy(), rtol=10 ** -3)


@pytest.mark.parametrize(
    "calculator_class",
    test_calculators
)
@pytest.mark.parametrize(
    "submitter_class",
    test_submitters
)
def test_tp_1d_without_opt(calculator_class, submitter_class, butan,
                           butan_tp_1d_5n_no_opt_results):
    """
    test 1D local exec TP without opt
    """
    expected_rel_energy_results = butan_tp_1d_5n_no_opt_results[calculator_class.__name__]
    expected_angles = butan_tp_1d_5n_no_opt_results["angles"]
    n_measurements = butan_tp_1d_5n_no_opt_results["n_measurements"]

    torsion_atom_ids = tuple(map(int, butan.GetProp("ta").split(" ")))
    conda_env = conf["conda_calculator_envs"][calculator_class.__name__]
    tmp_dir = tempfile.mkdtemp(
        prefix="tmp_test_tp_1d_local_" + calculator_class.__name__, dir=tmp_root_path
    )

    submitter = submitter_class(conda_env_path=conda_env)
    calculator = calculator_class()
    tp = TorsionProfiler(
        calculator=calculator,
        submission_system=submitter,
        n_measurements=n_measurements,
    )
    tp.calculator._optimize_structure = False

    # Calculate
    df = tp.calculate_torsion_profile(mol=butan, torsion_atom_ids=torsion_atom_ids,
                                      out_dir=tmp_dir, approach_name="tmp_test")

    if isinstance(submitter, Slurm):
        df = tp.wait()

    print(df["rel_potential_energy"].to_list())
    np.allclose(a=expected_angles, b=df["torsion_angle"].to_numpy(), atol=10 ** 0)
    np.allclose(
        a=expected_rel_energy_results, b=df["rel_potential_energy"].to_numpy(), atol=10 ** -3
    )

    bash.rmtree(tmp_dir)


@pytest.mark.parametrize(
    "calculator_class",
    test_calculators
)
@pytest.mark.parametrize(
    "submitter_class",
    test_submitters
)
@pytest.mark.parametrize("splits", test_job_splitting)
def test_tp_1d_script(calculator_class, submitter_class, splits, butan, butan_tp_1d_5n_results):
    """
    test 1D local exec TP
    """
    expected_rel_energy_results = butan_tp_1d_5n_results[calculator_class.__name__]
    expected_angles = butan_tp_1d_5n_results["angles"]
    n_measurements = int(butan_tp_1d_5n_results["n_measurements"])

    torsion_atom_ids = list(map(int, butan.GetProp("ta").split(" ")))
    conda_env = conf["conda_calculator_envs"][calculator_class.__name__]
    tmp_dir = tempfile.mkdtemp(
        prefix=f"tmp_test_tp_1d_{submitter_class.__name__}_nsplits_{splits}_"
               f"{calculator_class.__name__}",
        dir=tmp_root_path
    )

    submitter = submitter_class(conda_env_path=conda_env)
    calculator = calculator_class()
    tp = TorsionProfiler(
        calculator=calculator,
        submission_system=submitter,
        n_measurements=n_measurements,
    )
    tp.calculator._optimize_structure = True

    # Calculate
    df = tp.calculate_torsion_profile(mol=butan, torsion_atom_ids=torsion_atom_ids,
                                      out_dir=tmp_dir, approach_name="tmp_test")

    if isinstance(submitter, Slurm):
        df = tp.wait()

    print(df["rel_potential_energy"].to_list())
    np.allclose(
        a=expected_rel_energy_results, b=df["rel_potential_energy"].to_numpy(), atol=10 ** -3
    )
    np.allclose(a=expected_angles, b=df["torsion_angle"].to_numpy(), atol=10 ** 0)

    bash.rmtree(tmp_dir)


@pytest.mark.optional
@pytest.mark.parametrize(
    "calculator_class",
    test_2d_interactive_calculator
)
def test_tp_2d_interactive(calculator_class, butan, butan_tp_2d_5n_results):
    """
    test 2D interactive exec TP
    """
    expected_rel_energy_results = butan_tp_2d_5n_results[calculator_class.__name__]
    expected_angles = butan_tp_2d_5n_results["angles"]
    n_measurements = butan_tp_2d_5n_results["n_measurements"]

    torsion_atom_ids_a = list(map(int, butan.GetProp("tb").split(" ")))
    torsion_atom_ids_b = list(map(int, butan.GetProp("ta").split(" ")))

    calculator = calculator_class()
    tp = TorsionProfiler(
        calculator=calculator,
        n_measurements=n_measurements,
    )

    # Calculate
    df = tp.calculate_2D_torsion_mol(
        mol=butan, torsion1=torsion_atom_ids_a, torsion2=torsion_atom_ids_b,
        approach_name="tmp_test_2D",
    )
    print("<", df, ">")
    print(df["rel_potential_energy"].to_list())
    np.allclose(a=expected_rel_energy_results, b=df["rel_potential_energy"].to_numpy())
    print(df["torsion_angle"].round(2).to_list())
    np.allclose(a=expected_angles, b=df["torsion_angle"].to_numpy(), atol=10 ** -2)


@pytest.mark.optional
@pytest.mark.parametrize(
    "calculator_class",
    test_calculators_2D
)
def test_tp_2d_script(calculator_class, butan, butan_tp_2d_5n_results):
    """
    test 2D local exec TP
    """
    expected_rel_energy_results = butan_tp_2d_5n_results[calculator_class.__name__]
    expected_angles = butan_tp_2d_5n_results["angles"]
    n_measurements = butan_tp_2d_5n_results["n_measurements"]

    torsion_atom_ids_a = list(map(int, butan.GetProp("tb").split(" ")))
    torsion_atom_ids_b = list(map(int, butan.GetProp("ta").split(" ")))
    conda_env = conf["conda_calculator_envs"][calculator_class.__name__]
    submitter = Local(conda_env_path=conda_env)
    tmp_dir = tempfile.mkdtemp(prefix="tmp_test_2D_prof_", dir=tmp_root_path)

    calculator = calculator_class()
    tp = TorsionProfiler(
        calculator=calculator,
        submission_system=submitter,
        n_measurements=n_measurements,
    )

    # Calculate
    df = tp.calculate_2D_torsion_mol(
        mol=butan, torsion1=torsion_atom_ids_a, torsion2=torsion_atom_ids_b,
        approach_name="tmp_test", out_dir=tmp_dir,
    )

    print(df["rel_potential_energy"].to_list())
    print(df["torsion_angle"].to_list())

    np.allclose(a=expected_rel_energy_results, b=df["rel_potential_energy"].to_numpy())
    np.allclose(a=expected_angles, b=df["torsion_angle"].to_numpy(), atol=10 ** -2)
    #bash.rmtree(tmp_dir)


@pytest.mark.parametrize(
    "calculator_class",
    test_calculators,
)
@pytest.mark.parametrize(
    "fragment",
    [False, True]
)
def test_tp_1d_all_tors_interactive(calculator_class, fragment, hexan, hexan_1d_alltors_tps):
    """
    test 1D interactive exec TP
    """
    expected_rel_energy_results = hexan_1d_alltors_tps[Mmff94Calculator.__name__]
    expected_angles = hexan_1d_alltors_tps["angles"]
    n_measurements = int(hexan_1d_alltors_tps["n_measurements"])

    calculator = calculator_class()
    tp = TorsionProfiler(calculator=calculator, n_measurements=n_measurements)

    # Calculate
    df = tp.calculate_all_torsions_profiles(mol=hexan, approach_name="test_tp_1d_alltors",
                                            fragment_mol=fragment)

    # Verify
    print(df["rel_potential_energy"].round(3).to_list())
    print(df["torsion_angle"].round(2).to_list())
    assert len(expected_rel_energy_results) == len(df["rel_potential_energy"])
    np.allclose(a=expected_rel_energy_results, b=df["rel_potential_energy"].to_numpy(),
                rtol=10 ** -2)

    assert len(expected_angles) == len(df["torsion_angle"])
    np.allclose(a=expected_angles, b=df["torsion_angle"].to_numpy(), rtol=10 ** -2)
    
    #bash.rmtree(tmp_dir)


@pytest.mark.parametrize(
    "calculator_class",
    test_calculators,
)
@pytest.mark.parametrize(
    "submitter_class",
    test_submitters
)
@pytest.mark.parametrize(
    "fragment",
    [False, True]
)
def test_tp_1d_all_tors_script(calculator_class, submitter_class, fragment,
                              hexan, hexan_1d_alltors_tps):
    """
    test 1D interactive exec TP
    """
    expected_rel_energy_results = hexan_1d_alltors_tps[Mmff94Calculator.__name__]
    expected_angles = hexan_1d_alltors_tps["angles"]
    n_measurements = int(hexan_1d_alltors_tps["n_measurements"])
    tmp_dir = tempfile.mkdtemp(prefix="tmp_test_tp_1d_all_tors_local", dir=tmp_root_path)

    submitter = submitter_class()
    calculator = calculator_class()
    tp = TorsionProfiler(
        calculator=calculator,
        submission_system=submitter,
        n_measurements=n_measurements,
    )

    # Calculate
    df = tp.calculate_all_torsions_profiles(mol=hexan,
                                            out_dir=tmp_dir,
                                            approach_name="test_tp_1d_alltors",
                                            fragment_mol=fragment)

    if isinstance(submitter, Slurm):
        df = tp.wait()

    # Verify
    print(df["rel_potential_energy"].to_list())
    print(df["torsion_angle"].to_list())

    np.allclose(a=expected_rel_energy_results, b=df["rel_potential_energy"].to_numpy(),
                rtol=10 ** -2)
    np.allclose(a=expected_angles, b=df["torsion_angle"].to_numpy(), rtol=10 ** -3)
    #bash.rmtree(tmp_dir)

@pytest.mark.parametrize(
    "calculator_class",
    test_calculators,
)
@pytest.mark.parametrize(
    "submitter_class",
    test_submitters
)
@pytest.mark.parametrize("splits", test_job_splitting)
def test_tp_1d_multiplemols(calculator_class, submitter_class, splits,
                            butan, butan_tp_1d_5n_results, ):
    """
    test 1D interactive exec TP
    """
    expected_rel_energy_results = np.vstack([butan_tp_1d_5n_results[Mmff94Calculator.__name__]
                                             for _ in range(3)]).flatten()
    expected_angles = np.vstack([butan_tp_1d_5n_results["angles"]
                                 for _ in range(3)]).flatten()
    n_measurements = int(butan_tp_1d_5n_results["n_measurements"])
    tai = list(map(float, butan.GetProp("ta").split()))
    tmp_dir = tempfile.mkdtemp(prefix="tmp_test_iter_mols", dir=tmp_root_path)

    submitter = submitter_class()
    calculator = calculator_class()
    tp = TorsionProfiler(
        calculator=calculator,
        submission_system=submitter,
        n_measurements=n_measurements,
    )
    tp.n_tasks_parallel = splits
    # Calculate
    df = tp.calculate_torsions_iter_mols(mols=[butan, butan, butan], torsions=[tai, tai, tai],
                                         approach_name="test_tp_iter_mols", out_dir=tmp_dir)

    if isinstance(submitter, Slurm):
        print(df)
        df = tp.wait()

    # Verify
    np.allclose(a=expected_rel_energy_results, b=df["rel_potential_energy"].to_numpy(),
                rtol=10 ** -2)
    np.allclose(a=expected_angles, b=df["torsion_angle"].to_numpy(), rtol=10 ** -3)
    #bash.rmtree(tmp_dir)


def test_empty_wait():
    """
    test 2D local exec TP
    """
    submitter = Slurm()

    calculator = Mmff94Calculator()
    tp = TorsionProfiler(
        calculator=calculator,
        submission_system=submitter,
        n_measurements=37,
    )

    df = tp.wait()

    assert len(df) == 0
