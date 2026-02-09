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

"""Automated Onset Labeling Script (ONSET-10).

Applies ThresholdOnsetDetector and CUSUMOnsetDetector to each bearing's
health indicator series (kurtosis + RMS) to automatically detect degradation
onset. Compares results to manual labels and saves automated labels to CSV.

Strategy:
    1. Try ThresholdOnsetDetector on kurtosis (h, v avg) — best for impulsive defects
    2. Try ThresholdOnsetDetector on RMS (h, v avg) — catches progressive wear
    3. Apply CUSUMOnsetDetector on kurtosis as secondary detector
    4. Select the best onset per bearing using: kurtosis threshold first, then RMS
       threshold, then CUSUM as fallback

Usage:
    python scripts/08_generate_onset_labels.py
    python scripts/08_generate_onset_labels.py --features outputs/features/features_v2.csv
    python scripts/08_generate_onset_labels.py --tolerance 5

Output:
    outputs/onset/onset_labels_auto.csv
    Console comparison report: agreement rate and disagreement cases
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.onset.detectors import CUSUMOnsetDetector, OnsetResult, ThresholdOnsetDetector
from src.onset.health_indicators import load_bearing_health_series
from src.onset.labels import load_onset_labels

# Configuration
DEFAULT_FEATURES = "outputs/features/features_v2.csv"
DEFAULT_OUTPUT = "outputs/onset/onset_labels_auto.csv"
DEFAULT_TOLERANCE = 5  # samples tolerance for agreement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run_detector(
    series: np.ndarray,
    detector_type: str,
    healthy_fraction: float = 0.2,
) -> OnsetResult:
    """Run a single detector on a health indicator series.

    Args:
        series: Health indicator values ordered by file_idx.
        detector_type: 'threshold' or 'cusum'.
        healthy_fraction: Fraction of early samples for baseline.

    Returns:
        OnsetResult from the detector.
    """
    series = np.asarray(series, dtype=float)

    if detector_type == "threshold":
        # Match manual labeling criteria: >2σ for >5 consecutive samples
        detector = ThresholdOnsetDetector(threshold_sigma=2.0, min_consecutive=5)
    elif detector_type == "cusum":
        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
    else:
        raise ValueError(f"Unknown detector: {detector_type}")

    return detector.fit_detect(series, healthy_fraction=healthy_fraction)


def detect_all_indicators(
    bearing_id: str,
    features_df: pd.DataFrame,
) -> dict[str, tuple[int | None, float]]:
    """Run detectors on all health indicators for a bearing.

    Returns dict mapping indicator_detector -> (onset_idx, confidence).
    """
    health = load_bearing_health_series(bearing_id, features_df)

    kurtosis_avg = (health.kurtosis_h + health.kurtosis_v) / 2.0
    rms_avg = (health.rms_h + health.rms_v) / 2.0

    results = {}

    # ThresholdOnsetDetector on kurtosis (primary for impulsive defects)
    r = _run_detector(kurtosis_avg, "threshold")
    results["kurtosis_threshold"] = (r.onset_idx, r.confidence)

    # ThresholdOnsetDetector on RMS (primary for progressive wear)
    r = _run_detector(rms_avg, "threshold")
    results["rms_threshold"] = (r.onset_idx, r.confidence)

    # CUSUMOnsetDetector on kurtosis (secondary)
    r = _run_detector(kurtosis_avg, "cusum")
    results["kurtosis_cusum"] = (r.onset_idx, r.confidence)

    # CUSUMOnsetDetector on RMS (secondary)
    r = _run_detector(rms_avg, "cusum")
    results["rms_cusum"] = (r.onset_idx, r.confidence)

    return results, health.condition


def select_best_onset(
    indicator_results: dict[str, tuple[int | None, float]],
    n_samples: int,
    healthy_fraction: float = 0.2,
) -> tuple[int | None, str]:
    """Select the best onset index from multiple detector results.

    Priority: kurtosis_threshold > rms_threshold > kurtosis_cusum > rms_cusum.
    This matches the manual labeling methodology which prefers kurtosis for
    impulsive onset and falls back to RMS for progressive wear.

    Guard: If kurtosis detects onset within the healthy baseline period (first
    healthy_fraction of life), it's likely a false alarm from an unstable
    baseline. In that case, prefer RMS threshold detection.

    Returns:
        (onset_file_idx, detector_method) tuple.
    """
    healthy_cutoff = int(n_samples * healthy_fraction)

    # Check if kurtosis threshold detection is within the baseline period
    kt_idx, _ = indicator_results["kurtosis_threshold"]
    kurtosis_in_baseline = kt_idx is not None and kt_idx < healthy_cutoff

    if kurtosis_in_baseline:
        # Kurtosis triggered in baseline region — likely false alarm.
        # Prefer RMS threshold, then fall back to others.
        priority = ["rms_threshold", "kurtosis_cusum", "rms_cusum", "kurtosis_threshold"]
    else:
        priority = ["kurtosis_threshold", "rms_threshold", "kurtosis_cusum", "rms_cusum"]

    for method in priority:
        idx, conf = indicator_results[method]
        if idx is not None:
            return idx, method

    return None, "none"


def run_automated_labeling(
    features_path: str,
    output_path: str,
    tolerance: int = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Run automated onset labeling on all bearings.

    Args:
        features_path: Path to features_v2.csv.
        output_path: Path for output CSV.
        tolerance: Sample tolerance for agreement with manual labels.

    Returns:
        DataFrame with automated onset labels.
    """
    # Load features
    logger.info("Loading features from %s", features_path)
    features_df = pd.read_csv(features_path)
    logger.info(
        "Loaded %d rows, %d bearings",
        len(features_df),
        features_df["bearing_id"].nunique(),
    )

    # Load manual labels for comparison
    try:
        manual_labels = load_onset_labels()
        logger.info("Loaded manual labels for %d bearings", len(manual_labels))
    except (FileNotFoundError, KeyError) as e:
        logger.warning("Could not load manual labels: %s", e)
        manual_labels = {}

    # Process each bearing
    rows = []
    bearing_ids = sorted(features_df["bearing_id"].unique())

    for bearing_id in bearing_ids:
        indicator_results, condition = detect_all_indicators(bearing_id, features_df)

        # Number of samples for this bearing
        n_samples = len(features_df[features_df["bearing_id"] == bearing_id])

        # Select best onset
        best_idx, best_method = select_best_onset(indicator_results, n_samples)

        row = {
            "bearing_id": bearing_id,
            "condition": condition,
            "onset_file_idx": best_idx,
            "detector_method": best_method,
        }

        # Add individual detector results
        for method, (idx, conf) in indicator_results.items():
            row[f"{method}_idx"] = idx
            row[f"{method}_conf"] = round(conf, 4)

        rows.append(row)

        logger.info(
            "%s: selected=%s (method=%s) | kurt_thr=%s rms_thr=%s kurt_cus=%s rms_cus=%s",
            bearing_id,
            best_idx,
            best_method,
            indicator_results["kurtosis_threshold"][0],
            indicator_results["rms_threshold"][0],
            indicator_results["kurtosis_cusum"][0],
            indicator_results["rms_cusum"][0],
        )

    auto_df = pd.DataFrame(rows)

    # Add manual labels for comparison
    if manual_labels:
        auto_df["manual_onset_idx"] = auto_df["bearing_id"].map(
            lambda bid: manual_labels[bid].onset_file_idx if bid in manual_labels else None
        )
    else:
        auto_df["manual_onset_idx"] = None

    # Print comparison report
    print_comparison_report(auto_df, manual_labels, tolerance)

    # Save output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    auto_df.to_csv(output_path, index=False)
    logger.info("Saved automated labels to %s", output_path)

    return auto_df


def print_comparison_report(
    auto_df: pd.DataFrame,
    manual_labels: dict,
    tolerance: int,
) -> None:
    """Print comparison report between automated and manual labels."""
    print("\n" + "=" * 70)
    print("ONSET DETECTION COMPARISON REPORT")
    print("=" * 70)

    if not manual_labels:
        print("No manual labels available for comparison.")
        print("=" * 70)
        return

    # Compare selected onset with manual labels
    print("\n--- Selected Onset vs Manual ---")
    agree_count = 0
    disagree_cases = []
    total = 0

    for _, row in auto_df.iterrows():
        manual_idx = row.get("manual_onset_idx")
        auto_idx = row["onset_file_idx"]

        if manual_idx is None or pd.isna(manual_idx):
            continue

        total += 1
        manual_idx = int(manual_idx)

        if auto_idx is None or pd.isna(auto_idx):
            disagree_cases.append(
                f"  {row['bearing_id']}: auto=None, manual={manual_idx} [{row['detector_method']}]"
            )
            continue

        auto_idx = int(auto_idx)
        diff = abs(auto_idx - manual_idx)

        if diff <= tolerance:
            agree_count += 1
        else:
            disagree_cases.append(
                f"  {row['bearing_id']}: auto={auto_idx}, manual={manual_idx}, "
                f"diff={diff} [{row['detector_method']}]"
            )

    if total > 0:
        rate = agree_count / total * 100
        print(
            f"Agreement: {agree_count}/{total} "
            f"({rate:.1f}%) within {tolerance} samples"
        )

    if disagree_cases:
        print(f"Disagreements ({len(disagree_cases)}):")
        for case in disagree_cases:
            print(case)

    # Compare individual detectors
    detector_cols = [
        ("kurtosis_threshold", "Kurtosis Threshold"),
        ("rms_threshold", "RMS Threshold"),
        ("kurtosis_cusum", "Kurtosis CUSUM"),
        ("rms_cusum", "RMS CUSUM"),
    ]

    for col_prefix, label in detector_cols:
        col = f"{col_prefix}_idx"
        agree = 0
        n = 0
        for _, row in auto_df.iterrows():
            manual_idx = row.get("manual_onset_idx")
            auto_idx = row.get(col)
            if manual_idx is None or pd.isna(manual_idx):
                continue
            n += 1
            if auto_idx is not None and not pd.isna(auto_idx):
                if abs(int(auto_idx) - int(manual_idx)) <= tolerance:
                    agree += 1
        if n > 0:
            print(f"  {label}: {agree}/{n} ({agree / n * 100:.0f}%)")

    # Per-bearing summary table
    print(f"\n--- Per-Bearing Summary ---")
    print(
        f"{'Bearing':<15} {'Manual':>7} {'Selected':>9} {'Method':<20} {'Diff':>6}"
    )
    print("-" * 70)

    for _, row in auto_df.iterrows():
        manual = row.get("manual_onset_idx")
        selected = row["onset_file_idx"]
        method = row["detector_method"]

        m_str = str(int(manual)) if manual is not None and not pd.isna(manual) else "N/A"
        s_str = str(int(selected)) if selected is not None and not pd.isna(selected) else "None"

        diff_str = ""
        if manual is not None and not pd.isna(manual) and selected is not None and not pd.isna(selected):
            d = int(selected) - int(manual)
            diff_str = f"{d:+d}"

        print(f"{row['bearing_id']:<15} {m_str:>7} {s_str:>9} {method:<20} {diff_str:>6}")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate automated onset labels for all bearings"
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        help=f"Path to features CSV (default: {DEFAULT_FEATURES})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help=f"Agreement tolerance in samples (default: {DEFAULT_TOLERANCE})",
    )
    args = parser.parse_args()

    run_automated_labeling(args.features, args.output, args.tolerance)


if __name__ == "__main__":
    main()
