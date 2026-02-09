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

"""Tests for health indicator computation module.

Tests the health indicator aggregation including:
- Loading bearing health series from features dataframe
- Computing composite health indicators
- Savitzky-Golay smoothing
- Edge cases: NaN values, constant signals, single samples
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.onset.health_indicators import (
    BearingHealthSeries,
    _minmax_normalize,
    compute_composite_hi,
    get_all_bearing_ids,
    load_all_bearings_health_series,
    load_bearing_health_series,
    smooth_health_indicator,
)


# ============================================================================
# Fixtures for synthetic health indicator data
# ============================================================================


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """Create synthetic features DataFrame mimicking real structure.

    Creates data for 2 bearings with 50 samples each, simulating:
    - Bearing1_1: gradual degradation (kurtosis increases over time)
    - Bearing2_1: sudden degradation at sample 30
    """
    np.random.seed(42)

    data = []

    # Bearing1_1: gradual degradation pattern
    for i in range(50):
        # Kurtosis starts ~3 (healthy) and increases gradually
        base_kurtosis = 3.0 + (i / 50) * 10  # 3 -> 13
        data.append(
            {
                "bearing_id": "Bearing1_1",
                "condition": "35Hz12kN",
                "file_idx": i,
                "h_kurtosis": base_kurtosis + np.random.randn() * 0.5,
                "v_kurtosis": base_kurtosis * 0.9 + np.random.randn() * 0.5,
                "h_rms": 0.1 + (i / 50) * 0.3 + np.random.randn() * 0.01,
                "v_rms": 0.08 + (i / 50) * 0.25 + np.random.randn() * 0.01,
            }
        )

    # Bearing2_1: sudden degradation at sample 30
    for i in range(50):
        if i < 30:
            base_kurtosis = 3.0 + np.random.randn() * 0.3
            base_rms = 0.1 + np.random.randn() * 0.01
        else:
            base_kurtosis = 15.0 + np.random.randn() * 2.0
            base_rms = 0.5 + np.random.randn() * 0.05
        data.append(
            {
                "bearing_id": "Bearing2_1",
                "condition": "37.5Hz11kN",
                "file_idx": i,
                "h_kurtosis": base_kurtosis,
                "v_kurtosis": base_kurtosis * 0.95,
                "h_rms": base_rms,
                "v_rms": base_rms * 0.9,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def single_sample_df() -> pd.DataFrame:
    """Create DataFrame with single sample for edge case testing."""
    return pd.DataFrame(
        [
            {
                "bearing_id": "Bearing_single",
                "condition": "35Hz12kN",
                "file_idx": 0,
                "h_kurtosis": 5.0,
                "v_kurtosis": 4.5,
                "h_rms": 0.15,
                "v_rms": 0.12,
            }
        ]
    )


@pytest.fixture
def constant_signal_df() -> pd.DataFrame:
    """Create DataFrame with constant values for edge case testing."""
    data = []
    for i in range(20):
        data.append(
            {
                "bearing_id": "Bearing_const",
                "condition": "35Hz12kN",
                "file_idx": i,
                "h_kurtosis": 3.0,  # Constant
                "v_kurtosis": 3.0,  # Constant
                "h_rms": 0.1,  # Constant
                "v_rms": 0.1,  # Constant
            }
        )
    return pd.DataFrame(data)


@pytest.fixture
def nan_values_df() -> pd.DataFrame:
    """Create DataFrame with NaN values for edge case testing."""
    data = []
    for i in range(20):
        h_kurt = 3.0 + i * 0.5
        v_kurt = 2.8 + i * 0.45
        # Insert NaN at specific indices
        if i in [5, 10, 15]:
            h_kurt = np.nan
        if i in [7, 12]:
            v_kurt = np.nan

        data.append(
            {
                "bearing_id": "Bearing_nan",
                "condition": "35Hz12kN",
                "file_idx": i,
                "h_kurtosis": h_kurt,
                "v_kurtosis": v_kurt,
                "h_rms": 0.1 + i * 0.01,
                "v_rms": 0.09 + i * 0.01,
            }
        )
    return pd.DataFrame(data)


# ============================================================================
# Tests for load_bearing_health_series()
# ============================================================================


class TestLoadBearingHealthSeries:
    """Tests for load_bearing_health_series function."""

    def test_returns_correct_shape(self, synthetic_features_df: pd.DataFrame):
        """Test that returned arrays have expected shape for known bearing."""
        result = load_bearing_health_series("Bearing1_1", synthetic_features_df)

        assert isinstance(result, BearingHealthSeries)
        assert result.bearing_id == "Bearing1_1"
        assert result.condition == "35Hz12kN"
        assert len(result.file_indices) == 50
        assert len(result.kurtosis_h) == 50
        assert len(result.kurtosis_v) == 50
        assert len(result.rms_h) == 50
        assert len(result.rms_v) == 50
        assert len(result.composite) == 50

    def test_returns_time_ordered_data(self, synthetic_features_df: pd.DataFrame):
        """Test that file_indices are sorted in ascending order."""
        result = load_bearing_health_series("Bearing1_1", synthetic_features_df)

        # Verify time ordering
        assert np.all(np.diff(result.file_indices) >= 0), "file_indices not sorted"

    def test_raises_on_unknown_bearing(self, synthetic_features_df: pd.DataFrame):
        """Test ValueError raised for non-existent bearing."""
        with pytest.raises(ValueError, match="Bearing 'Unknown' not found"):
            load_bearing_health_series("Unknown", synthetic_features_df)

    def test_single_sample_bearing(self, single_sample_df: pd.DataFrame):
        """Test loading bearing with single sample."""
        result = load_bearing_health_series("Bearing_single", single_sample_df)

        assert len(result.file_indices) == 1
        assert len(result.composite) == 1

    def test_different_bearings_independent(self, synthetic_features_df: pd.DataFrame):
        """Test that different bearings return independent data."""
        result1 = load_bearing_health_series("Bearing1_1", synthetic_features_df)
        result2 = load_bearing_health_series("Bearing2_1", synthetic_features_df)

        assert result1.bearing_id != result2.bearing_id
        assert result1.condition != result2.condition
        # Values should be different (different degradation patterns)
        assert not np.allclose(result1.kurtosis_h, result2.kurtosis_h)


# ============================================================================
# Tests for compute_composite_hi()
# ============================================================================


class TestComputeCompositeHI:
    """Tests for compute_composite_hi function."""

    def test_output_in_zero_one_range(self):
        """Test that composite HI is always in [0, 1] range."""
        np.random.seed(42)
        # Random data with varying scales
        kurtosis_h = np.random.randn(100) * 10 + 5
        kurtosis_v = np.random.randn(100) * 8 + 4
        rms_h = np.random.rand(100) * 2
        rms_v = np.random.rand(100) * 1.5

        composite = compute_composite_hi(kurtosis_h, kurtosis_v, rms_h, rms_v)

        assert composite.min() >= 0.0, f"Min value {composite.min()} < 0"
        assert composite.max() <= 1.0, f"Max value {composite.max()} > 1"

    def test_output_spans_full_range(self):
        """Test that normalized output actually uses [0, 1] range."""
        # Monotonically increasing data
        n = 50
        kurtosis_h = np.linspace(3, 15, n)
        kurtosis_v = np.linspace(2.8, 14, n)
        rms_h = np.linspace(0.1, 0.5, n)
        rms_v = np.linspace(0.08, 0.4, n)

        composite = compute_composite_hi(kurtosis_h, kurtosis_v, rms_h, rms_v)

        # Should span approximately [0, 1] for monotonic input
        assert composite[0] == pytest.approx(0.0, abs=1e-10)
        assert composite[-1] == pytest.approx(1.0, abs=1e-10)

    def test_preserves_array_length(self):
        """Test that output has same length as input."""
        n = 73  # Odd number to catch indexing bugs
        kurtosis_h = np.random.randn(n) + 5
        kurtosis_v = np.random.randn(n) + 5
        rms_h = np.random.rand(n)
        rms_v = np.random.rand(n)

        composite = compute_composite_hi(kurtosis_h, kurtosis_v, rms_h, rms_v)

        assert len(composite) == n

    def test_raises_on_mismatched_lengths(self):
        """Test ValueError raised when arrays have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            compute_composite_hi(
                kurtosis_h=np.array([1, 2, 3]),
                kurtosis_v=np.array([1, 2]),  # Different length
                rms_h=np.array([1, 2, 3]),
                rms_v=np.array([1, 2, 3]),
            )

    def test_custom_weights(self):
        """Test that custom weights affect the intermediate result.

        Note: Final normalization to [0,1] is applied, so we test that
        different weights produce different intermediate weighted sums.
        """
        n = 20
        # Create data where all features vary differently
        kurtosis_h = np.linspace(0, 10, n)
        kurtosis_v = np.linspace(10, 0, n)  # Opposite direction
        rms_h = np.linspace(0, 1, n)
        rms_v = np.linspace(1, 0, n)  # Opposite direction

        # With high weight on kurtosis_h (increasing)
        composite_high_kh = compute_composite_hi(
            kurtosis_h, kurtosis_v, rms_h, rms_v, weights=(0.8, 0.1, 0.05, 0.05)
        )
        # With high weight on kurtosis_v (decreasing)
        composite_high_kv = compute_composite_hi(
            kurtosis_h, kurtosis_v, rms_h, rms_v, weights=(0.1, 0.8, 0.05, 0.05)
        )

        # With opposing weights, the composites should have opposite trends
        # First half of high_kh should be lower than second half (increasing)
        # First half of high_kv should be higher than second half (decreasing)
        assert np.mean(composite_high_kh[:n // 2]) < np.mean(composite_high_kh[n // 2:])
        assert np.mean(composite_high_kv[:n // 2]) > np.mean(composite_high_kv[n // 2:])

    def test_constant_signal_returns_zeros(self):
        """Test that constant input returns all zeros (no variation)."""
        n = 20
        constant = np.ones(n) * 5

        composite = compute_composite_hi(constant, constant, constant, constant)

        # All constant inputs -> all zeros after normalization
        np.testing.assert_array_equal(composite, np.zeros(n))


# ============================================================================
# Tests for smooth_health_indicator()
# ============================================================================


class TestSmoothHealthIndicator:
    """Tests for Savitzky-Golay smoothing function."""

    def test_preserves_array_length(self):
        """Test that smoothing preserves array length."""
        hi = np.random.randn(100) + 5

        smoothed = smooth_health_indicator(hi, window_length=11, polyorder=3)

        assert len(smoothed) == len(hi)

    def test_reduces_noise(self):
        """Test that smoothing reduces noise while preserving trend."""
        np.random.seed(42)
        n = 100
        # Create noisy signal with clear trend
        trend = np.linspace(0, 1, n)
        noise = np.random.randn(n) * 0.1
        noisy = trend + noise

        smoothed = smooth_health_indicator(noisy, window_length=11, polyorder=3)

        # Smoothed should be closer to true trend
        mse_noisy = np.mean((noisy - trend) ** 2)
        mse_smoothed = np.mean((smoothed - trend) ** 2)

        assert mse_smoothed < mse_noisy, "Smoothing did not reduce noise"

    def test_short_series_handled(self):
        """Test that very short series are handled gracefully."""
        # Single sample
        hi_single = np.array([5.0])
        smoothed_single = smooth_health_indicator(hi_single)
        np.testing.assert_array_equal(smoothed_single, hi_single)

        # Two samples
        hi_two = np.array([3.0, 5.0])
        smoothed_two = smooth_health_indicator(hi_two)
        np.testing.assert_array_equal(smoothed_two, hi_two)

        # Three samples (minimum for smoothing)
        hi_three = np.array([3.0, 5.0, 4.0])
        smoothed_three = smooth_health_indicator(hi_three, window_length=3, polyorder=1)
        assert len(smoothed_three) == 3

    def test_window_adjusted_for_short_series(self):
        """Test that window_length is adjusted for series shorter than window."""
        hi = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 samples
        # Request window larger than series
        smoothed = smooth_health_indicator(hi, window_length=11, polyorder=2)

        assert len(smoothed) == len(hi)

    def test_different_window_sizes(self):
        """Test smoothing with different window sizes."""
        np.random.seed(42)
        hi = np.random.randn(50) + 5

        smoothed_small = smooth_health_indicator(hi, window_length=5, polyorder=2)
        smoothed_large = smooth_health_indicator(hi, window_length=15, polyorder=2)

        # Both should have same length
        assert len(smoothed_small) == len(hi)
        assert len(smoothed_large) == len(hi)

        # Larger window should produce smoother result (lower variance)
        assert np.var(np.diff(smoothed_large)) < np.var(np.diff(smoothed_small))


# ============================================================================
# Tests for handling NaN values
# ============================================================================


class TestNaNHandling:
    """Tests for handling missing/NaN values."""

    def test_minmax_normalize_with_nan(self):
        """Test that _minmax_normalize handles NaN values."""
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        normalized = _minmax_normalize(arr)

        # Should normalize non-NaN values, preserve NaN positions
        assert normalized[0] == pytest.approx(0.0)
        assert normalized[-1] == pytest.approx(1.0)
        # NaN handling: np.nanmin/nanmax ignore NaNs

    def test_composite_hi_with_nan_propagation(self, nan_values_df: pd.DataFrame):
        """Test that composite HI computation handles NaN in input."""
        result = load_bearing_health_series("Bearing_nan", nan_values_df)

        # NaN values may propagate or be handled depending on implementation
        # Key is that the function doesn't crash
        assert len(result.composite) == 20

    def test_minmax_normalize_all_nan(self):
        """Test _minmax_normalize with all NaN values."""
        arr = np.array([np.nan, np.nan, np.nan])

        normalized = _minmax_normalize(arr)

        # Should return zeros when all NaN
        np.testing.assert_array_equal(normalized, np.zeros(3))

    def test_minmax_normalize_single_value(self):
        """Test _minmax_normalize with single non-NaN value."""
        arr = np.array([5.0])

        normalized = _minmax_normalize(arr)

        # Single value means zero range -> return zeros
        np.testing.assert_array_equal(normalized, np.zeros(1))


# ============================================================================
# Tests for utility functions
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_all_bearing_ids(self, synthetic_features_df: pd.DataFrame):
        """Test get_all_bearing_ids returns sorted unique IDs."""
        ids = get_all_bearing_ids(synthetic_features_df)

        assert isinstance(ids, list)
        assert len(ids) == 2
        assert ids == sorted(ids)  # Should be sorted
        assert "Bearing1_1" in ids
        assert "Bearing2_1" in ids

    def test_load_all_bearings_health_series(self, synthetic_features_df: pd.DataFrame):
        """Test loading all bearings at once."""
        all_series = load_all_bearings_health_series(synthetic_features_df)

        assert isinstance(all_series, dict)
        assert len(all_series) == 2
        assert "Bearing1_1" in all_series
        assert "Bearing2_1" in all_series

        # Each should be a BearingHealthSeries
        for bearing_id, series in all_series.items():
            assert isinstance(series, BearingHealthSeries)
            assert series.bearing_id == bearing_id

    def test_load_all_bearings_with_smoothing(
        self, synthetic_features_df: pd.DataFrame
    ):
        """Test loading all bearings with smoothing enabled."""
        all_series = load_all_bearings_health_series(
            synthetic_features_df, smooth=True, smooth_window=7
        )

        # Should not crash and return same structure
        assert len(all_series) == 2

        # Composite should be smoothed (lower variance than unsmoothed)
        unsmoothed = load_all_bearings_health_series(synthetic_features_df, smooth=False)

        for bearing_id in all_series:
            smoothed_var = np.var(np.diff(all_series[bearing_id].composite))
            unsmoothed_var = np.var(np.diff(unsmoothed[bearing_id].composite))
            # Smoothing should reduce variance of differences
            assert smoothed_var <= unsmoothed_var


# ============================================================================
# Tests with real data (skipped if not available)
# ============================================================================


class TestWithRealData:
    """Tests using real features_v2.csv data."""

    def test_load_real_bearing(self, features_df: pd.DataFrame):
        """Test loading a real bearing from features_v2.csv."""
        # Get first available bearing
        bearing_ids = get_all_bearing_ids(features_df)
        assert len(bearing_ids) > 0, "No bearings in features_df"

        result = load_bearing_health_series(bearing_ids[0], features_df)

        assert isinstance(result, BearingHealthSeries)
        assert result.bearing_id == bearing_ids[0]
        assert len(result.composite) > 0

    def test_all_15_bearings_loadable(self, features_df: pd.DataFrame):
        """Test that all 15 bearings can be loaded."""
        bearing_ids = get_all_bearing_ids(features_df)

        # Should have 15 bearings (5 per condition)
        assert len(bearing_ids) == 15, f"Expected 15 bearings, got {len(bearing_ids)}"

        # Each should load without error
        for bearing_id in bearing_ids:
            result = load_bearing_health_series(bearing_id, features_df)
            assert result.bearing_id == bearing_id

    def test_composite_hi_range_real_data(self, features_df: pd.DataFrame):
        """Test composite HI is in [0, 1] for all real bearings."""
        all_series = load_all_bearings_health_series(features_df)

        for bearing_id, series in all_series.items():
            assert series.composite.min() >= 0.0, (
                f"{bearing_id}: composite min {series.composite.min()} < 0"
            )
            assert series.composite.max() <= 1.0, (
                f"{bearing_id}: composite max {series.composite.max()} > 1"
            )
