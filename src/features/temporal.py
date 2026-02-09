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
"""Temporal feature enrichment for bearing degradation trajectories.

Computes lag features, rolling statistics, and rate-of-change features
from per-bearing time series. Designed for LightGBM tabular models that
need trajectory awareness beyond single-timestep snapshots.

All computations are per-bearing (grouped by bearing_id, sorted by file_idx)
to prevent cross-bearing data leakage. NaN values at the start of each
bearing's timeline are left as-is since LightGBM handles them natively.
"""

from __future__ import annotations

import pandas as pd

TEMPORAL_BASE_FEATURES = [
    "h_kurtosis",
    "v_kurtosis",
    "h_rms",
    "v_rms",
    "h_peak",
    "v_peak",
    "cross_correlation",
    "h_bpfo_band_power",
    "v_bpfo_band_power",
    "h_bpfi_band_power",
]

DEFAULT_LAGS = [1, 5, 10]
DEFAULT_ROLLING_WINDOWS = [5, 10]


def enrich_temporal_features(
    df: pd.DataFrame,
    base_features: list[str] | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Enrich a features DataFrame with temporal (lag, rolling, diff) features.

    For each bearing (grouped by bearing_id, sorted by file_idx), computes:
      - Lag features: f_lag_{k} for each lag k
      - Rolling mean: f_rmean_{w} for each window w
      - Rolling std: f_rstd_{w} for each window w
      - Rate of change: f_diff (first-order difference)

    Args:
        df: Features DataFrame with bearing_id, file_idx, and feature columns.
        base_features: Feature columns to compute temporal features for.
            Defaults to TEMPORAL_BASE_FEATURES.
        lags: Lag values. Defaults to [1, 5, 10].
        rolling_windows: Rolling window sizes. Defaults to [5, 10].

    Returns:
        Copy of df with new temporal columns appended.
    """
    if base_features is None:
        base_features = TEMPORAL_BASE_FEATURES
    if lags is None:
        lags = DEFAULT_LAGS
    if rolling_windows is None:
        rolling_windows = DEFAULT_ROLLING_WINDOWS

    missing = [f for f in base_features if f not in df.columns]
    if missing:
        raise ValueError(f"Base features not found in DataFrame: {missing}")

    result = df.copy()
    result = result.sort_values(["bearing_id", "file_idx"]).reset_index(drop=True)

    grouped = result.groupby("bearing_id", sort=False)

    for feat in base_features:
        col = result[feat]
        group_col = grouped[feat]

        for k in lags:
            result[f"{feat}_lag_{k}"] = group_col.shift(k)

        for w in rolling_windows:
            result[f"{feat}_rmean_{w}"] = group_col.transform(
                lambda x, _w=w: x.rolling(window=_w, min_periods=1).mean()
            )
            result[f"{feat}_rstd_{w}"] = group_col.transform(
                lambda x, _w=w: x.rolling(window=_w, min_periods=1).std()
            )

        result[f"{feat}_diff"] = group_col.diff()

    return result


def get_temporal_feature_names(
    base_features: list[str] | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> list[str]:
    """Return the list of temporal feature column names that would be created.

    Args:
        base_features: Base feature names. Defaults to TEMPORAL_BASE_FEATURES.
        lags: Lag values. Defaults to [1, 5, 10].
        rolling_windows: Rolling window sizes. Defaults to [5, 10].

    Returns:
        List of temporal feature column names.
    """
    if base_features is None:
        base_features = TEMPORAL_BASE_FEATURES
    if lags is None:
        lags = DEFAULT_LAGS
    if rolling_windows is None:
        rolling_windows = DEFAULT_ROLLING_WINDOWS

    names = []
    for feat in base_features:
        for k in lags:
            names.append(f"{feat}_lag_{k}")
        for w in rolling_windows:
            names.append(f"{feat}_rmean_{w}")
            names.append(f"{feat}_rstd_{w}")
        names.append(f"{feat}_diff")
    return names
