"""
Torsion Profile Analyzer tests
"""
import numpy as np
import pandas as pd

from ...tools.torsion_profiler.torsion_profile_analyzer import TorsionProfileAnalyzer
from ...utils.metrics import interpolate_function

def test_ta_constructore():
    """
    Torsion Profile Analyzer Constructor test
    """
    TorsionProfileAnalyzer()


def test_generate_decoy_data(butan_df):
    """
    Torsion Profile Analyzer decoyGen test
    """
    ta = TorsionProfileAnalyzer()
    print(butan_df.angles)

    decoy_df = ta._build_decoy_data(butan_df)

    print(butan_df)
    df = decoy_df
    print(df)


def test_generate_decoy_data_metrics(butan_df):
    """
    Torsion Profile Analyzer metrics calc test
    """
    ta = TorsionProfileAnalyzer()

    decoy_metric, decoy_df = ta._build_decoy_dist(butan_df)

    perfect_df = [d for d in decoy_df if "perfect" in d.approach.iloc[0]][0]
    shift_dfs = [d for d in decoy_df if "shift" in d.approach.iloc[0]]
    scale_dfs = [d for d in decoy_df if "scale" in d.approach.iloc[0]]

    assert all(c in perfect_df.columns for c in butan_df.columns)

    check_columns = [c for c in butan_df.columns if c not in ["approach"]]
    pd.testing.assert_frame_equal(butan_df[check_columns], perfect_df[check_columns])

    assert len(shift_dfs) == 17
    msdf = pd.concat(shift_dfs)
    assert isinstance(msdf, pd.DataFrame)
    assert len(msdf.loc[(msdf["angles"] > 180) & (msdf["angles"] < -180)]) == 0

    assert len(scale_dfs) == 21
    msdf = pd.concat(scale_dfs)
    assert isinstance(msdf, pd.DataFrame)
    assert len(msdf.loc[(msdf["angles"] > 180) & (msdf["angles"] < -180)]) == 0


def test_generate_build_desc(butan_df):
    """
    Torsion Profile Analyzer descriptor test
    """
    ta = TorsionProfileAnalyzer()
    decoy_dfs = ta._build_decoy_data(butan_df)

    ds = []
    for df in decoy_dfs:
        d = ta.build_comparison_descriptors(method_profiles=df, ref_profiles=butan_df)
        ds.append(d)
    metric_df = pd.concat(ds)
    print(metric_df)


def test_interpolation():
    """
    Torsion Profile Analyzer Constructor test
    """
    new_angles = np.linspace(-160, 180, 37)
    old_angles = np.linspace(-180, 180, 37)
    old_energies = np.linspace(-180, 180, 37)
    fitter_f = interpolate_function(x_angles=old_angles, y_energies=old_energies)
    fitter_f(new_angles)


def test_interpolation2(butan_df):
    """
    Torsion Profile Analyzer test interpolation 2
    """
    ta = TorsionProfileAnalyzer()
    decoy_dfs = ta._build_decoy_data(butan_df)
    df = decoy_dfs[0]

    # Same Size
    print(butan_df)
    print(df)

    print("equal sized")
    _, _, _ = ta.deal_with_diff_angle_ranges(
        method_profile=df, ref_profile=butan_df, min_angle=-178, max_angle=178
    )

    print("truncated right")
    sub_butan = butan_df.loc[(butan_df.angles < 105)]
    _, _, _ = ta.deal_with_diff_angle_ranges(
        method_profile=df, ref_profile=sub_butan, min_angle=-178, max_angle=178
    )

    print("truncated both sides")
    sub_butan = butan_df.loc[(butan_df.angles < 105) & (butan_df.angles > -160)]
    _, _, _ = ta.deal_with_diff_angle_ranges(
        method_profile=df, ref_profile=sub_butan, min_angle=-178, max_angle=178
    )

    print("truncated left")
    sub_butan = butan_df.loc[(butan_df.angles > -160)]
    _, _, _= ta.deal_with_diff_angle_ranges(
        method_profile=df, ref_profile=sub_butan, min_angle=-178, max_angle=178
    )
