#!/usr/bin/env python3
"""Evaluate and aggregate DL model results for XJTU-SY Bearing Dataset.

Reads per-fold DL results from dl_model_results.csv, computes per-model
summary statistics, merges with existing model_comparison.csv, and
generates per-bearing breakdown CSVs.

Usage:
    python scripts/06_evaluate_dl_models.py
    python scripts/06_evaluate_dl_models.py --results-dir outputs/evaluation --output-dir outputs/evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and aggregate DL model training results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory containing dl_model_results.csv and predictions/ (default: outputs/evaluation)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory to write summary and comparison CSVs (default: outputs/evaluation)",
    )
    return parser.parse_args()


def aggregate_per_model_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std per model for each metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-fold results with columns: model_name, fold_id, rmse, mae, mape,
        phm08_score, phm08_score_normalized.

    Returns
    -------
    pd.DataFrame
        One row per model with columns: model_name, rmse_mean, rmse_std,
        mae_mean, mae_std, mape_mean, mape_std, phm08_mean, phm08_std,
        phm08_norm_mean, phm08_norm_std, n_folds.
    """
    metric_cols = {
        "rmse": ("rmse_mean", "rmse_std"),
        "mae": ("mae_mean", "mae_std"),
        "mape": ("mape_mean", "mape_std"),
        "phm08_score": ("phm08_mean", "phm08_std"),
        "phm08_score_normalized": ("phm08_norm_mean", "phm08_norm_std"),
    }

    rows = []
    for model_name, group in results_df.groupby("model_name"):
        row: dict = {"model_name": model_name}
        for col, (mean_name, std_name) in metric_cols.items():
            row[mean_name] = group[col].mean()
            row[std_name] = group[col].std()
        row["n_folds"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 80)
    print("DL Model Summary (mean ± std across folds)")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"\n  {row['model_name']}  ({int(row['n_folds'])} folds)")
        print(f"    RMSE:  {row['rmse_mean']:.2f} ± {row['rmse_std']:.2f}")
        print(f"    MAE:   {row['mae_mean']:.2f} ± {row['mae_std']:.2f}")
        print(f"    MAPE:  {row['mape_mean']:.2f} ± {row['mape_std']:.2f}")
        print(f"    PHM08: {row['phm08_mean']:.2f} ± {row['phm08_std']:.2f}")
    print("\n" + "=" * 80)


def main() -> None:
    """Main entry point for DL model evaluation."""
    args = parse_args()
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir:  {args.output_dir}")

    # --- EVAL-2: Aggregate per-fold results into per-model summary ---
    results_csv = args.results_dir / "dl_model_results.csv"
    if not results_csv.exists():
        print(f"\nERROR: {results_csv} not found. Run training first.")
        sys.exit(1)

    results_df = pd.read_csv(results_csv)
    print(f"\nLoaded {len(results_df)} fold results from {results_csv}")
    print(f"Models found: {sorted(results_df['model_name'].unique())}")

    summary_df = aggregate_per_model_summary(results_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "dl_model_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved per-model summary → {summary_csv}")

    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
