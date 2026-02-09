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

"""Health indicator computation for bearing degradation onset detection.

This module aggregates time-domain features (kurtosis, RMS) into health indicator
time series that can be used by onset detection algorithms. Health indicators
should increase monotonically as bearing degradation progresses.

Reference: Kurtosis and RMS are widely used in bearing health monitoring because:
- Kurtosis captures impulsiveness (healthy ~3, degraded >> 3)
- RMS captures overall vibration energy (increases with wear)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BearingHealthSeries:
    """Container for a bearing's health indicator time series.

    Attributes:
        bearing_id: Unique identifier for the bearing (e.g., "Bearing1_1").
        condition: Operating condition (e.g., "35Hz12kN").
        file_indices: Array of file indices (time ordering).
        kurtosis_h: Horizontal channel kurtosis values.
        kurtosis_v: Vertical channel kurtosis values.
        rms_h: Horizontal channel RMS values.
        rms_v: Vertical channel RMS values.
        composite: Combined health indicator (normalized to [0, 1]).
    """

    bearing_id: str
    condition: str
    file_indices: np.ndarray
    kurtosis_h: np.ndarray
    kurtosis_v: np.ndarray
    rms_h: np.ndarray
    rms_v: np.ndarray
    composite: np.ndarray


def load_bearing_health_series(
    bearing_id: str,
    features_df: pd.DataFrame,
) -> BearingHealthSeries:
    """Load health indicator time series for a specific bearing.

    Extracts kurtosis and RMS features from the features dataframe,
    orders them by file index, and computes a composite health indicator.

    Args:
        bearing_id: Bearing identifier (e.g., "Bearing1_1").
        features_df: DataFrame with columns: bearing_id, file_idx,
            h_kurtosis, v_kurtosis, h_rms, v_rms.

    Returns:
        BearingHealthSeries with time-ordered health indicators.

    Raises:
        ValueError: If bearing_id not found in features_df.
    """
    # Filter to specific bearing
    mask = features_df["bearing_id"] == bearing_id
    if not mask.any():
        raise ValueError(
            f"Bearing '{bearing_id}' not found in features dataframe. "
            f"Available bearings: {features_df['bearing_id'].unique().tolist()}"
        )

    bearing_df = features_df[mask].sort_values("file_idx")

    # Extract arrays
    file_indices = bearing_df["file_idx"].to_numpy()
    kurtosis_h = bearing_df["h_kurtosis"].to_numpy()
    kurtosis_v = bearing_df["v_kurtosis"].to_numpy()
    rms_h = bearing_df["h_rms"].to_numpy()
    rms_v = bearing_df["v_rms"].to_numpy()

    # Get condition (should be same for all rows of this bearing)
    condition = bearing_df["condition"].iloc[0]

    # Compute composite HI
    composite = compute_composite_hi(kurtosis_h, kurtosis_v, rms_h, rms_v)

    return BearingHealthSeries(
        bearing_id=bearing_id,
        condition=condition,
        file_indices=file_indices,
        kurtosis_h=kurtosis_h,
        kurtosis_v=kurtosis_v,
        rms_h=rms_h,
        rms_v=rms_v,
        composite=composite,
    )


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1] range.

    Args:
        arr: Input array.

    Returns:
        Normalized array in [0, 1] range. Returns zeros if range is zero.
    """
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    range_val = arr_max - arr_min

    if range_val == 0 or np.isnan(range_val):
        return np.zeros_like(arr)

    return (arr - arr_min) / range_val


def compute_composite_hi(
    kurtosis_h: np.ndarray,
    kurtosis_v: np.ndarray,
    rms_h: np.ndarray,
    rms_v: np.ndarray,
    weights: tuple[float, float, float, float] = (0.4, 0.4, 0.1, 0.1),
) -> np.ndarray:
    """Compute composite health indicator from individual features.

    Combines normalized kurtosis and RMS from both channels using
    weighted sum. Default weights prioritize kurtosis (more sensitive
    to early-stage defects) over RMS.

    Args:
        kurtosis_h: Horizontal kurtosis values.
        kurtosis_v: Vertical kurtosis values.
        rms_h: Horizontal RMS values.
        rms_v: Vertical RMS values.
        weights: Tuple of (w_kurtosis_h, w_kurtosis_v, w_rms_h, w_rms_v).
            Must sum to 1.0 for proper normalization.

    Returns:
        Composite health indicator in [0, 1] range.

    Raises:
        ValueError: If input arrays have different lengths.
    """
    # Validate input shapes
    lengths = [len(kurtosis_h), len(kurtosis_v), len(rms_h), len(rms_v)]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All input arrays must have same length. Got lengths: {lengths}"
        )

    w_kh, w_kv, w_rh, w_rv = weights

    # Normalize each indicator to [0, 1]
    norm_kurtosis_h = _minmax_normalize(kurtosis_h)
    norm_kurtosis_v = _minmax_normalize(kurtosis_v)
    norm_rms_h = _minmax_normalize(rms_h)
    norm_rms_v = _minmax_normalize(rms_v)

    # Weighted combination
    composite = (
        w_kh * norm_kurtosis_h
        + w_kv * norm_kurtosis_v
        + w_rh * norm_rms_h
        + w_rv * norm_rms_v
    )

    # Final normalization to ensure [0, 1] range
    return _minmax_normalize(composite)


def smooth_health_indicator(
    hi_series: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to health indicator series.

    Reduces noise while preserving trend shape and important features
    like onset transitions. Uses polynomial fitting in a sliding window.

    Args:
        hi_series: Health indicator time series.
        window_length: Size of smoothing window (must be odd, >= polyorder + 2).
        polyorder: Order of polynomial for fitting (typically 2-4).

    Returns:
        Smoothed health indicator series (same length as input).

    Raises:
        ValueError: If window_length is invalid for the series length.
    """
    n_samples = len(hi_series)

    if n_samples < 3:
        # Too short to smooth meaningfully
        return hi_series.copy()

    # Adjust window_length if series is too short
    if window_length > n_samples:
        window_length = n_samples if n_samples % 2 == 1 else n_samples - 1

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length -= 1

    # Ensure polyorder < window_length
    if polyorder >= window_length:
        polyorder = window_length - 1

    # Minimum valid window
    if window_length < 3:
        return hi_series.copy()

    return savgol_filter(hi_series, window_length=window_length, polyorder=polyorder)


def get_all_bearing_ids(features_df: pd.DataFrame) -> list[str]:
    """Get list of all unique bearing IDs from features dataframe.

    Args:
        features_df: DataFrame with bearing_id column.

    Returns:
        Sorted list of unique bearing identifiers.
    """
    return sorted(features_df["bearing_id"].unique().tolist())


def load_all_bearings_health_series(
    features_df: pd.DataFrame,
    smooth: bool = False,
    smooth_window: int = 11,
) -> dict[str, BearingHealthSeries]:
    """Load health indicator series for all bearings in the dataset.

    Args:
        features_df: DataFrame with features for all bearings.
        smooth: If True, apply Savitzky-Golay smoothing to composite HI.
        smooth_window: Window size for smoothing.

    Returns:
        Dictionary mapping bearing_id to BearingHealthSeries.
    """
    bearing_ids = get_all_bearing_ids(features_df)
    result = {}

    for bearing_id in bearing_ids:
        health_series = load_bearing_health_series(bearing_id, features_df)

        if smooth:
            # Apply smoothing to composite indicator
            health_series.composite = smooth_health_indicator(
                health_series.composite,
                window_length=smooth_window,
            )

        result[bearing_id] = health_series

    return result
