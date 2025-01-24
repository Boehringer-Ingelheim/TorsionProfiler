"""
Test single optimizations
"""
import tempfile
import pytest
import numpy as np

from torsion_profiler import conf
from torsion_profiler import GeomOptimizer
from torsion_profiler.utils import bash
from torsion_profiler.orchestration.submission_systems import Slurm, Local
from torsion_profiler.engines import (
    Mmff94Calculator,
    OffCalculator,
    AniCalculator,
    Psi4Calculator,
    XtbCalculator,
    GaussianCalculator,
)

from .. import tmp_root_path

test_local = True

test_off = False
test_ani = False
test_xtb = False
test_gaus = False
test_2d_MMFF94_only = True

# Test Control:
test_job_splitting = [1, 2, 4, 5, 6]

test_submitters = []
if test_local:
    test_submitters.append(Local)

test_calculators = [Mmff94Calculator]
if test_off:
    test_calculators.append(OffCalculator)
if test_ani:
    test_calculators.append(AniCalculator)
if test_xtb:
    test_calculators.append(XtbCalculator)
if test_gaus:
    test_calculators.append(GaussianCalculator)

if test_2d_MMFF94_only:
    test_calculators_2D = [
        Mmff94Calculator,
    ]
else:
    test_calculators_2D = test_calculators


@pytest.mark.parametrize(
    "calculator_class",
    test_calculators,
)
def test_go_interactive(calculator_class, butan, butan_optimization_results):
    """
    Test interative optimization
    """
    expected_energies = butan_optimization_results[Mmff94Calculator.__name__]

    calculator = Mmff94Calculator()
    go = GeomOptimizer(calculators=[calculator])

    # Calculate
    df = go.optimize(mol=butan)

    # Verify
    assert np.allclose(a=expected_energies, b=df["rel_potential_energy"].to_numpy(), atol=10**-2)


@pytest.mark.parametrize(
    "calculator_class",
    test_calculators,
)
@pytest.mark.parametrize(
    "submitter_class",
    test_submitters
)
def test_go_local(calculator_class, submitter_class, butan, butan_optimization_results):
    """
    Test local optimization
    """
    expected_energies = butan_optimization_results[calculator_class.__name__]

    conda_env = conf["conda_calculator_envs"][calculator_class.__name__]
    tmp_dir = tempfile.mkdtemp(
        prefix="tmp_test_go_local_" + calculator_class.__name__, dir=tmp_root_path
    )

    submitter = submitter_class(conda_env_path=conda_env)
    calculator = calculator_class()
    go = GeomOptimizer(
        calculators=calculator,
        submission_system=submitter,
    )

    # Calculate
    df = go.optimize(mol=butan,
        approach_name="tmp_test_go",
        out_dir=tmp_dir,)

    print(df["rel_potential_energy"].to_list())
    np.allclose(a=expected_energies, b=df["rel_potential_energy"].to_numpy(), atol=10**-2)
    bash.rmtree(tmp_dir)

