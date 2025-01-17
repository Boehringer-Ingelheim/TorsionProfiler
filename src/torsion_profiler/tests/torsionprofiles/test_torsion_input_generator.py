"""
Here initial Torsion profile algorithms are tested.
"""
import pandas as pd
import pytest

import numpy as np
from torsion_profiler.tools.torsion_profiler.torsion_profile_generators import (
    FastProfileGenerator,
    LandscaperTorsionProfileGenerator,
)
from ...utils import store_mol_db
from ..conftest import fixture_butan

@pytest.mark.parametrize(
    "tpg_class",
    [
        FastProfileGenerator,
        LandscaperTorsionProfileGenerator,
    ],
)
def test_tpg_1d(butan, tpg_class):
    """
    Test the fast tp generator
    """
    n_measurements = 37
    torsion_atom_ids = tuple(map(int, butan.GetProp("ta").split(" ")))

    tpg = tpg_class(n_measurements=n_measurements)
    torsion_profile = tpg.generate(butan, torsion_atom_ids=torsion_atom_ids)

    assert isinstance(torsion_profile, pd.DataFrame)
    assert torsion_profile.shape[0] == n_measurements

    all_angles = np.linspace(-180, 180, 37)
    np.testing.assert_allclose(torsion_profile["torsion_angle"].to_numpy(), all_angles,
                               rtol=10**-1)



def test_tpg_2d_landscape(butan):
    """
    Test the 2D landscaper tp generator
    """
    # expected_rel_energy_results = butan_tp_1d_5n_results[Mmff94Calculator.__name__]
    # expected_angles = butan_tp_1d_5n_results["angles"]
    n_measurements = 12  # butan_tp_1d_5n_results["n_measurements"]
    torsion_atom_ids = tuple(map(int, butan.GetProp("ta").split(" ")))
    torsion_atom_ids_b = tuple(map(int, butan.GetProp("tb").split(" ")))

    tpg = LandscaperTorsionProfileGenerator(n_measurements=n_measurements)
    torsion_profile, _ = tpg.landscape_exploration_2d(
        in_mol=butan, t1=torsion_atom_ids, t2=torsion_atom_ids_b
    )

    assert isinstance(torsion_profile, pd.DataFrame)
    assert torsion_profile.shape[0] == n_measurements * n_measurements

    one_row = np.round(np.linspace(-180, 180, n_measurements))
    all_angles = np.hstack([one_row for i in range(n_measurements)])
    np.testing.assert_allclose(torsion_profile["torsion_angle"].to_numpy(), all_angles,
                                   rtol=10**-1)
    path= f"{tpg.__class__.__name__}_2d.sdf"
    store_mol_db(torsion_profile, path)