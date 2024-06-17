"""
test Calculators
"""
import pandas as pd
import pytest
import numpy as np


from ..tools.torsion_profiler import FastProfileGenerator

from ..engines import (
    Mmff94Calculator,
    OffCalculator,
    AniCalculator,
    MACECalculator,
    AltAniCalculator,
    Psi4Calculator,
    XtbCalculator,
    GaussianCalculator,
)

from .conftest import fixture_butan_sp_results, fixture_butan_opt_results

test_calculators= [
        Mmff94Calculator,
        OffCalculator,
        AniCalculator,
        MACECalculator,
        AltAniCalculator,
        #XtbCalculator,
        #Psi4Calculator,
        #GaussianCalculator,
    ]

@pytest.mark.parametrize(
    "calculator_class",
   test_calculators,
)
def test_calculator_single_point_structure(butan, butan_sp_results, calculator_class):
    """
    test 1D interactive exec TP
    """
    expected_V = butan_sp_results[calculator_class.__name__]

    calculator = calculator_class(optimize_structure=False)
    result_df = calculator.calculate_conformer_potentials(mol=butan)

    assert isinstance(result_df, pd.DataFrame)
    pd.testing.assert_series_equal(result_df["molID"], pd.Series(["mol_0"], name="molID"))
    np.testing.assert_allclose(result_df["potential_energy"],  expected_V,
                               rtol=10**-5)


@pytest.mark.parametrize(
    "calculator_class",
    test_calculators,
)
def test_calculator_optimize_structure(butan, butan_opt_results, calculator_class):
    """
    test 1D interactive exec TP
    """
    expected_V = butan_opt_results[calculator_class.__name__]
    calculator = calculator_class(optimize_structure=True)
    result_df = calculator.calculate_conformer_potentials(mol=butan)

    assert isinstance(result_df, pd.DataFrame)
    pd.testing.assert_series_equal(result_df["molID"], pd.Series(["mol_0"], name="molID"))
    np.testing.assert_allclose(result_df["potential_energy"], expected_V, rtol=10**-1)

@pytest.mark.parametrize(
    "calculator_class",
    test_calculators,
)
def test_calculator_optimize_multiple_structures(butan, butan_mc_opt_results, calculator_class):
    """
    test 1D interactive exec TP
    """
    expected_V = butan_mc_opt_results[calculator_class.__name__]
    n_measurements = 8
    tpg = FastProfileGenerator(n_measurements=n_measurements)
    torsion_atom_ids = tuple(map(int, butan.GetProp("ta").split()))
    tp_df = tpg.generate(butan, torsion_atom_ids=torsion_atom_ids)

    calculator = calculator_class()
    result_df = calculator.calculate_conformer_potentials(mol=tp_df,
                                                          torsion_atom_ids=torsion_atom_ids)

    print(result_df["potential_energy"].round(5)-result_df["potential_energy"].round(5).min())
    assert isinstance(result_df, pd.DataFrame)
    pd.testing.assert_series_equal(result_df["molID"], pd.Series([f"mol_{i}" for i in range(n_measurements)],
                                                                 name="molID"))
    np.testing.assert_allclose(result_df["potential_energy"].round(5), expected_V, rtol=10**-5)
