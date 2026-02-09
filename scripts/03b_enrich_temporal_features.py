#!/usr/bin/env python3
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
"""Temporal Feature Enrichment Script.

Reads extracted features (features_v2.csv) and adds temporal features
(lags, rolling statistics, rate of change) per bearing to produce
features_v3.csv for LightGBM models with trajectory awareness.

Usage:
    python scripts/03b_enrich_temporal_features.py
    python scripts/03b_enrich_temporal_features.py --input features_v2.csv --output features_v3.csv
    python scripts/03b_enrich_temporal_features.py --base-features h_kurtosis,v_kurtosis,h_rms
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.temporal import (
    TEMPORAL_BASE_FEATURES,
    enrich_temporal_features,
    get_temporal_feature_names,
)

DEFAULT_INPUT = "outputs/features/features_v2.csv"
DEFAULT_OUTPUT = "outputs/features/features_v3.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich bearing features with temporal (lag, rolling, diff) features."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Input features CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output features CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--base-features",
        type=str,
        default=None,
        help="Comma-separated list of base features (default: all 10 temporal base features)",
    )
    parser.add_argument(
        "--lags",
        type=str,
        default=None,
        help="Comma-separated lag values (default: 1,5,10)",
    )
    parser.add_argument(
        "--rolling-windows",
        type=str,
        default=None,
        help="Comma-separated rolling window sizes (default: 5,10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Parse optional parameters
    base_features = None
    if args.base_features:
        base_features = [f.strip() for f in args.base_features.split(",")]

    lags = None
    if args.lags:
        lags = [int(x.strip()) for x in args.lags.split(",")]

    rolling_windows = None
    if args.rolling_windows:
        rolling_windows = [int(x.strip()) for x in args.rolling_windows.split(",")]

    # Read input
    print(f"Reading features from {input_path}...")
    df = pd.read_csv(input_path)
    original_cols = len(df.columns)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {original_cols}")

    # Enrich
    features_used = base_features or TEMPORAL_BASE_FEATURES
    print(f"\nEnriching with temporal features for {len(features_used)} base features:")
    for f in features_used:
        print(f"  - {f}")

    df_enriched = enrich_temporal_features(
        df,
        base_features=base_features,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    new_cols = len(df_enriched.columns) - original_cols
    temporal_names = get_temporal_feature_names(
        base_features=base_features,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    # NaN summary
    nan_counts = df_enriched[temporal_names].isna().sum()
    total_nans = nan_counts.sum()

    print(f"\nResult:")
    print(f"  Original columns: {original_cols}")
    print(f"  New temporal columns: {new_cols}")
    print(f"  Total columns: {len(df_enriched.columns)}")
    print(f"  Total NaN values in temporal features: {total_nans:,}")
    print(f"  (NaNs are expected at the start of each bearing's timeline)")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
