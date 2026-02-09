#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#  https://github.com/thfmn/xjtu-sy-bearing
#
#  This work is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the condition that the above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  For more information, visit: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   xjtu-sy-bearing onset and RUL prediction ML Pipeline

"""Tests for temporal feature enrichment.

Tests the temporal feature module including:
- Lag features computed correctly
- Rolling statistics computed correctly
- No cross-bearing leakage
- Rate of change (diff) features
- NaN handling at bearing boundaries
- Output shape validation
- Custom base feature selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.temporal import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    TEMPORAL_BASE_FEATURES,
    enrich_temporal_features,
    get_temporal_feature_names,
)


@pytest.fixture
def two_bearing_df() -> pd.DataFrame:
    """Create a small DataFrame with two bearings for testing."""
    return pd.DataFrame(
        {
            "bearing_id": ["1_1"] * 5 + ["1_2"] * 5,
            "file_idx": list(range(5)) + list(range(5)),
            "h_kurtosis": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "v_kurtosis": [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0],
            "h_rms": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
            "rul": [100, 80, 60, 40, 20, 100, 80, 60, 40, 20],
        }
    )


@pytest.fixture
def single_bearing_df() -> pd.DataFrame:
    """Create a DataFrame with a single bearing for simple validation."""
    return pd.DataFrame(
        {
            "bearing_id": ["1_1"] * 6,
            "file_idx": list(range(6)),
            "h_kurtosis": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "v_kurtosis": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "rul": [50, 40, 30, 20, 10, 0],
        }
    )


class TestLagFeatures:
    """Test lag feature computation."""

    def test_lag_1_values(self, single_bearing_df):
        """Verify lag-1 values match manual calculation."""
        result = enrich_temporal_features(
            single_bearing_df,
            base_features=["h_kurtosis"],
            lags=[1],
            rolling_windows=[],
        )
        expected = [np.nan, 1.0, 2.0, 4.0, 8.0, 16.0]
        actual = result["h_kurtosis_lag_1"].tolist()
        assert pd.isna(actual[0])
        assert actual[1:] == expected[1:]

    def test_lag_5_values(self, single_bearing_df):
        """Verify lag-5 has NaN for first 5 rows."""
        result = enrich_temporal_features(
            single_bearing_df,
            base_features=["h_kurtosis"],
            lags=[5],
            rolling_windows=[],
        )
        lag5 = result["h_kurtosis_lag_5"]
        assert lag5.isna().sum() == 5
        assert lag5.iloc[5] == 1.0

    def test_multiple_lags(self, single_bearing_df):
        """Verify multiple lag columns are created."""
        result = enrich_temporal_features(
            single_bearing_df,
            base_features=["h_kurtosis"],
            lags=[1, 2, 3],
            rolling_windows=[],
        )
        assert "h_kurtosis_lag_1" in result.columns
        assert "h_kurtosis_lag_2" in result.columns
        assert "h_kurtosis_lag_3" in result.columns


class TestRollingFeatures:
    """Test rolling statistics computation."""

    def test_rolling_mean_window_3(self, single_bearing_df):
        """Verify rolling mean with window 3."""
        result = enrich_temporal_features(
            single_bearing_df,
            base_features=["h_kurtosis"],
            lags=[],
            rolling_windows=[3],
        )
        rmean = result["h_kurtosis_rmean_3"]
        # min_periods=1, so first value is just the value itself
        assert rmean.iloc[0] == pytest.approx(1.0)
        # Second: mean(1, 2) = 1.5
        assert rmean.iloc[1] == pytest.approx(1.5)
        # Third: mean(1, 2, 4) = 7/3
        assert rmean.iloc[2] == pytest.approx(7.0 / 3.0)
        # Fourth: mean(2, 4, 8) = 14/3
        assert rmean.iloc[3] == pytest.approx(14.0 / 3.0)

    def test_rolling_std_window_3(self, single_bearing_df):
        """Verify rolling std with window 3."""
        result = enrich_temporal_features(
            single_bearing_df,
            base_features=["h_kurtosis"],
            lags=[],
            rolling_windows=[3],
        )
        rstd = result["h_kurtosis_rstd_3"]
        # First element: std of single value = NaN (ddof=1)
        assert pd.isna(rstd.iloc[0])
        # Third: std([1, 2, 4])
        expected_std = np.std([1.0, 2.0, 4.0], ddof=1)
        assert rstd.iloc[2] == pytest.approx(expected_std)

    def test_rolling_columns_created(self, single_bearing_df):
        """Verify rolling mean and std columns are created for each window."""
        result = enrich_temporal_features(
            single_bearing_df,
            base_features=["h_kurtosis"],
            lags=[],
            rolling_windows=[5, 10],
        )
        assert "h_kurtosis_rmean_5" in result.columns
        assert "h_kurtosis_rstd_5" in result.columns
        assert "h_kurtosis_rmean_10" in result.columns
        assert "h_kurtosis_rstd_10" in result.columns


class TestNoCrossBearingLeakage:
    """Test that temporal features don't leak across bearings."""

    def test_lag_nan_at_bearing_boundary(self, two_bearing_df):
        """Verify first row of each bearing has NaN for lag-1."""
        result = enrich_temporal_features(
            two_bearing_df,
            base_features=["h_kurtosis"],
            lags=[1],
            rolling_windows=[],
        )
        # Get rows for bearing 1_2
        b2 = result[result["bearing_id"] == "1_2"]
        assert pd.isna(b2["h_kurtosis_lag_1"].iloc[0]), (
            "First row of bearing 1_2 should have NaN lag-1"
        )
        # Lag-1 of second row of 1_2 should be 10.0, not 5.0 (from bearing 1_1)
        assert b2["h_kurtosis_lag_1"].iloc[1] == 10.0

    def test_diff_nan_at_bearing_boundary(self, two_bearing_df):
        """Verify diff is NaN at the start of each bearing."""
        result = enrich_temporal_features(
            two_bearing_df,
            base_features=["h_kurtosis"],
            lags=[],
            rolling_windows=[],
        )
        b1 = result[result["bearing_id"] == "1_1"]
        b2 = result[result["bearing_id"] == "1_2"]
        assert pd.isna(b1["h_kurtosis_diff"].iloc[0])
        assert pd.isna(b2["h_kurtosis_diff"].iloc[0])
        # Diff in bearing 1_2: 20-10=10, not 20-5=15
        assert b2["h_kurtosis_diff"].iloc[1] == 10.0

    def test_rolling_resets_at_bearing_boundary(self, two_bearing_df):
        """Verify rolling stats reset at bearing boundaries."""
        result = enrich_temporal_features(
            two_bearing_df,
            base_features=["h_kurtosis"],
            lags=[],
            rolling_windows=[3],
        )
        b2 = result[result["bearing_id"] == "1_2"]
        # First row of bearing 1_2: rolling mean with min_periods=1 should be 10.0
        assert b2["h_kurtosis_rmean_3"].iloc[0] == pytest.approx(10.0)
        # Second row: mean(10, 20) = 15.0 (not including bearing 1_1 data)
        assert b2["h_kurtosis_rmean_3"].iloc[1] == pytest.approx(15.0)


class TestDiffFeatures:
    """Test rate of change (diff) features."""

    def test_diff_values(self, single_bearing_df):
        """Verify diff values match manual calculation."""
        result = enrich_temporal_features(
            single_bearing_df,
            base_features=["h_kurtosis"],
            lags=[],
            rolling_windows=[],
        )
        diff = result["h_kurtosis_diff"]
        assert pd.isna(diff.iloc[0])
        # Differences: 2-1=1, 4-2=2, 8-4=4, 16-8=8, 32-16=16
        expected = [1.0, 2.0, 4.0, 8.0, 16.0]
        assert diff.iloc[1:].tolist() == expected


class TestNaNHandling:
    """Test NaN handling at start of each bearing's timeline."""

    def test_nan_count_lag_features(self, two_bearing_df):
        """Verify correct NaN count for lag features."""
        result = enrich_temporal_features(
            two_bearing_df,
            base_features=["h_kurtosis"],
            lags=[1, 3],
            rolling_windows=[],
        )
        # Lag-1: 1 NaN per bearing = 2 total
        assert result["h_kurtosis_lag_1"].isna().sum() == 2
        # Lag-3: 3 NaN per bearing = 6 total (but each bearing has only 5 rows,
        # so 3 NaN per bearing = 6)
        assert result["h_kurtosis_lag_3"].isna().sum() == 6

    def test_nan_count_diff(self, two_bearing_df):
        """Verify diff produces 1 NaN per bearing."""
        result = enrich_temporal_features(
            two_bearing_df,
            base_features=["h_kurtosis"],
            lags=[],
            rolling_windows=[],
        )
        assert result["h_kurtosis_diff"].isna().sum() == 2


class TestOutputShape:
    """Test output shape and column count."""

    def test_correct_num_new_columns(self, two_bearing_df):
        """Verify correct number of new columns with default params."""
        result = enrich_temporal_features(
            two_bearing_df,
            base_features=["h_kurtosis", "v_kurtosis"],
            lags=[1, 5],
            rolling_windows=[5],
        )
        # Per feature: 2 lags + 1 rmean + 1 rstd + 1 diff = 5
        # 2 features * 5 = 10 new columns
        original_cols = len(two_bearing_df.columns)
        new_cols = len(result.columns) - original_cols
        assert new_cols == 10

    def test_row_count_preserved(self, two_bearing_df):
        """Verify row count is preserved."""
        result = enrich_temporal_features(
            two_bearing_df, base_features=["h_kurtosis"]
        )
        assert len(result) == len(two_bearing_df)

    def test_original_columns_preserved(self, two_bearing_df):
        """Verify original columns are unchanged."""
        result = enrich_temporal_features(
            two_bearing_df, base_features=["h_kurtosis"]
        )
        for col in two_bearing_df.columns:
            assert col in result.columns

    def test_default_temporal_feature_count(self):
        """Verify default config produces 80 temporal features."""
        names = get_temporal_feature_names()
        # 10 base features * (3 lags + 2 rmean + 2 rstd + 1 diff) = 80
        assert len(names) == 80


class TestCustomBaseFeatures:
    """Test configurable feature selection."""

    def test_subset_of_features(self, two_bearing_df):
        """Verify enrichment with a subset of base features."""
        result = enrich_temporal_features(
            two_bearing_df,
            base_features=["h_kurtosis"],
            lags=[1],
            rolling_windows=[],
        )
        assert "h_kurtosis_lag_1" in result.columns
        assert "v_kurtosis_lag_1" not in result.columns

    def test_invalid_feature_raises(self, two_bearing_df):
        """Verify error for missing base features."""
        with pytest.raises(ValueError, match="Base features not found"):
            enrich_temporal_features(
                two_bearing_df,
                base_features=["nonexistent_feature"],
            )

    def test_get_temporal_feature_names_matches_output(self, two_bearing_df):
        """Verify get_temporal_feature_names matches actual output columns."""
        base = ["h_kurtosis", "v_kurtosis"]
        lags = [1, 5]
        windows = [3]

        names = get_temporal_feature_names(
            base_features=base, lags=lags, rolling_windows=windows
        )
        result = enrich_temporal_features(
            two_bearing_df,
            base_features=base,
            lags=lags,
            rolling_windows=windows,
        )

        new_cols = [c for c in result.columns if c not in two_bearing_df.columns]
        assert sorted(names) == sorted(new_cols)
