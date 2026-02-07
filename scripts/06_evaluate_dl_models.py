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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.metrics import per_bearing_metrics


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


def generate_per_bearing_breakdowns(
    results_df: pd.DataFrame,
    predictions_dir: Path,
    output_dir: Path,
) -> None:
    """Generate per-bearing metric breakdown CSVs for each DL model.

    For each model, loads all per-fold prediction CSVs, concatenates them
    (each fold has different bearings in the val set), and computes
    per-bearing metrics using ``per_bearing_metrics()``.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-fold results (used to determine which models were trained).
    predictions_dir : Path
        Directory containing ``{model_name}_fold{N}_predictions.csv`` files.
    output_dir : Path
        Directory to write ``{model_name}_per_bearing.csv`` files.
    """
    if not predictions_dir.exists():
        print(f"\nWARNING: {predictions_dir} not found — skipping EVAL-4 per-bearing breakdown.")
        return

    model_names = sorted(results_df["model_name"].unique())
    print(f"\n--- EVAL-4: Per-bearing breakdown for {len(model_names)} model(s) ---")

    for model_name in model_names:
        # Glob for all fold prediction CSVs for this model
        pred_csvs = sorted(predictions_dir.glob(f"{model_name}_fold*_predictions.csv"))
        if not pred_csvs:
            print(f"  {model_name}: no prediction CSVs found — skipping")
            continue

        # Concatenate all folds
        dfs = [pd.read_csv(p) for p in pred_csvs]
        combined = pd.concat(dfs, ignore_index=True)

        y_true = combined["y_true"].values
        y_pred = combined["y_pred"].values
        bearing_ids = combined["bearing_id"].values

        # Compute per-bearing metrics
        pb_df = per_bearing_metrics(y_true, y_pred, bearing_ids)

        # Save
        out_csv = output_dir / f"{model_name}_per_bearing.csv"
        pb_df.to_csv(out_csv, index=False)
        print(f"  {model_name}: {len(pred_csvs)} fold(s), {len(combined)} samples, "
              f"{len(pb_df)} bearings → {out_csv}")

    print("--- EVAL-4 complete ---")


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

    # --- EVAL-3: Merge DL results with existing model_comparison.csv ---
    merge_dl_into_model_comparison(summary_df, args.output_dir)

    # --- EVAL-4: Generate per-bearing breakdown CSVs for DL models ---
    predictions_dir = args.results_dir / "predictions"
    generate_per_bearing_breakdowns(results_df, predictions_dir, args.output_dir)


# ---------------------------------------------------------------------------
# Display name mapping: registry name → human-readable name for comparison CSV
# ---------------------------------------------------------------------------
_DISPLAY_NAMES: dict[str, str] = {
    "cnn1d_baseline": "1D CNN (CV)",
    "tcn_transformer_lstm": "TCN-LSTM (CV)",
    "tcn_transformer_transformer": "TCN-Transformer (CV)",
    "cnn2d_lstm": "2D CNN LSTM (CV)",
    "cnn2d_simple": "2D CNN Simple (CV)",
}

# Registry name → Type column value
_INPUT_TYPE_TO_DISPLAY: dict[str, str] = {
    "raw_signal": "DL - Raw Signal",
    "spectrogram": "DL - Spectrogram",
}


def _get_input_type(model_name: str) -> str:
    """Look up the input type for a model from the registry.

    Falls back to 'DL' if the registry is not importable.
    """
    try:
        from src.models.registry import get_model_info

        info = get_model_info(model_name)
        return _INPUT_TYPE_TO_DISPLAY.get(info.input_type, "DL")
    except (ImportError, KeyError):
        return "DL"


def merge_dl_into_model_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Append DL model rows to model_comparison.csv, skipping duplicates.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-model summary from ``aggregate_per_model_summary()``.
    output_dir : Path
        Directory containing ``model_comparison.csv``.
    """
    comparison_csv = output_dir / "model_comparison.csv"
    if not comparison_csv.exists():
        print(f"\nWARNING: {comparison_csv} not found — skipping EVAL-3 merge.")
        return

    comp_df = pd.read_csv(comparison_csv)
    existing_models = set(comp_df["Model"].values)

    new_rows = []
    for _, row in summary_df.iterrows():
        model_name: str = row["model_name"]
        display_name = _DISPLAY_NAMES.get(model_name, f"{model_name} (CV)")
        if display_name in existing_models:
            print(f"  Skipping {display_name} — already in model_comparison.csv")
            continue

        model_type = _get_input_type(model_name)
        new_rows.append(
            {
                "Model": display_name,
                "Type": model_type,
                "RMSE": row["rmse_mean"],
                "MAE": row["mae_mean"],
                "MAPE (%)": row["mape_mean"],
                "PHM08 Score": row["phm08_mean"],
                "PHM08 (norm)": row["phm08_norm_mean"],
            }
        )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        comp_df = pd.concat([comp_df, new_df], ignore_index=True)
        comp_df.to_csv(comparison_csv, index=False)
        print(f"\nAppended {len(new_rows)} DL model(s) to {comparison_csv}")
        for r in new_rows:
            print(f"  + {r['Model']} ({r['Type']}) — RMSE: {r['RMSE']:.2f}, MAE: {r['MAE']:.2f}")
    else:
        print("\nNo new DL models to add to model_comparison.csv")


if __name__ == "__main__":
    main()
