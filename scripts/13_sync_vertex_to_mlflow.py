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

"""Sync Vertex AI benchmark results into local MLflow.

Downloads training outputs (history CSVs, predictions, fold results) from GCS
and creates MLflow runs under the `bearing_rul_fulllife` experiment with:
- Per-epoch training curves (loss, val_loss, mae, val_mae)
- Final summary metrics (RMSE, MAE, PHM08)
- Predictions logged as artifacts
- Training config logged as artifact

Usage:
    # Download from GCS and sync to MLflow
    python scripts/13_sync_vertex_to_mlflow.py

    # Sync from local benchmark dir (already downloaded)
    python scripts/13_sync_vertex_to_mlflow.py --local

    # Sync specific models/protocols
    python scripts/13_sync_vertex_to_mlflow.py --models cnn2d_simple,dta_mlp --protocols lobo

    # Dry run — show what would be synced
    python scripts/13_sync_vertex_to_mlflow.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

# GCS paths (must match submit_vertex_jobs.py)
STAGING_BUCKET = "xjtu-bearing-failure-staging"
GCS_BENCHMARK_BASE = f"gs://{STAGING_BUCKET}/outputs/vertex_training/benchmark"

# Local paths
LOCAL_BENCHMARK_DIR = Path("outputs/benchmark")
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_EXPERIMENT_NAME = "bearing_rul_fulllife"

# Model × protocol matrix
BENCHMARK_MODELS = [
    "cnn1d_baseline", "cnn2d_simple", "dta_mlp",
    "tcn_transformer_lstm",
]
PROTOCOLS = ["lobo", "loco", "jin", "sun"]

CV_STRATEGY_MAP = {
    "lobo": "leave_one_bearing_out",
    "loco": "loco_per_bearing",
    "jin": "jin_fixed",
    "sun": "sun_fixed",
}


def download_from_gcs(model: str, protocol: str) -> Path:
    """Download benchmark results from GCS to local benchmark dir."""
    gcs_path = f"{GCS_BENCHMARK_BASE}/{model}_{protocol}/"
    local_path = LOCAL_BENCHMARK_DIR / f"{model}_{protocol}"
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {gcs_path} → {local_path}")
    result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_path, str(local_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    WARNING: gsutil failed: {result.stderr.strip()}")
        return local_path

    # Count downloaded files
    n_files = sum(1 for _ in local_path.rglob("*.csv"))
    print(f"    Downloaded {n_files} CSV files")
    return local_path


def find_fold_results(output_dir: Path) -> list[dict]:
    """Parse fold results CSV from a benchmark output directory.

    Prefers dl_model_results.csv (append-mode, survives parallel job
    overwrites) and falls back to *_fold_results.csv.
    """
    # Prefer the append-mode results file (reliable with parallel jobs)
    dl_results = output_dir / "dl_model_results.csv"
    if dl_results.exists():
        df = pd.read_csv(dl_results)
        if len(df) > 1:
            return df.to_dict("records")

    # Fallback to per-model fold results
    csvs = list(output_dir.glob("*_fold_results.csv"))
    if not csvs:
        return []
    df = pd.read_csv(csvs[0])
    return df.to_dict("records")


def find_history_csvs(output_dir: Path) -> dict[int, Path]:
    """Find per-fold history CSVs in the output directory."""
    history_dir = output_dir / "history"
    if not history_dir.exists():
        return {}
    result = {}
    for csv_path in history_dir.glob("*_fold*_history.csv"):
        # Extract fold_id from filename like "cnn2d_simple_fold3_history.csv"
        name = csv_path.stem  # e.g. "cnn2d_simple_fold3_history"
        parts = name.split("_fold")
        if len(parts) == 2:
            fold_str = parts[1].replace("_history", "")
            try:
                fold_id = int(fold_str)
                result[fold_id] = csv_path
            except ValueError:
                continue
    return result


def find_prediction_csvs(output_dir: Path) -> dict[int, Path]:
    """Find per-fold prediction CSVs in the output directory."""
    pred_dir = output_dir / "predictions"
    if not pred_dir.exists():
        return {}
    result = {}
    for csv_path in pred_dir.glob("*_fold*_predictions.csv"):
        name = csv_path.stem
        parts = name.split("_fold")
        if len(parts) == 2:
            fold_str = parts[1].replace("_predictions", "")
            try:
                fold_id = int(fold_str)
                result[fold_id] = csv_path
            except ValueError:
                continue
    return result


def sync_to_mlflow(
    model: str,
    protocol: str,
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """Import results from a single model × protocol into MLflow.

    Creates one MLflow run per fold with:
    - Training params (model, fold, cv_strategy, protocol)
    - Per-epoch metrics from history CSV (for training curves)
    - Final summary metrics (RMSE, MAE, PHM08)
    - Predictions CSV as artifact

    Returns number of runs created.
    """
    fold_results = find_fold_results(output_dir)
    if not fold_results:
        print(f"    No fold results found in {output_dir}")
        return 0

    history_csvs = find_history_csvs(output_dir)
    prediction_csvs = find_prediction_csvs(output_dir)

    runs_created = 0

    for fold in fold_results:
        fold_id = fold["fold_id"]
        model_name = fold["model_name"]
        val_bearings = fold.get("val_bearings", "")

        run_name = f"{model_name}_fold{fold_id}_{protocol}"

        if dry_run:
            has_history = fold_id in history_csvs
            has_preds = fold_id in prediction_csvs
            print(
                f"    [DRY RUN] {run_name}  "
                f"RMSE={fold['rmse']:.4f}  "
                f"history={'yes' if has_history else 'no'}  "
                f"preds={'yes' if has_preds else 'no'}"
            )
            runs_created += 1
            continue

        with mlflow.start_run(run_name=run_name) as run:
            # Log params
            mlflow.log_params({
                "model_name": model_name,
                "fold_id": fold_id,
                "val_bearings": val_bearings,
                "cv_strategy": CV_STRATEGY_MAP.get(protocol, protocol),
                "protocol": protocol,
                "benchmark_model": model,
                "eval_mode": "full_life",
                "source": "vertex_ai",
            })

            # Log per-epoch training curves from history CSV
            if fold_id in history_csvs:
                history_df = pd.read_csv(history_csvs[fold_id])
                for epoch_idx, row in history_df.iterrows():
                    step_metrics = {}
                    for col in history_df.columns:
                        if col == "epoch":
                            continue
                        step_metrics[col] = float(row[col])
                    mlflow.log_metrics(step_metrics, step=int(epoch_idx))

                # Log history CSV as artifact
                mlflow.log_artifact(str(history_csvs[fold_id]), artifact_path="history")

            # Log final summary metrics
            mlflow.log_metrics({
                "final_rmse": fold["rmse"],
                "final_mae": fold["mae"],
                "final_mape": fold.get("mape", 0.0),
                "final_phm08_score": fold.get("phm08_score", 0.0),
                "final_phm08_score_normalized": fold.get("phm08_score_normalized", 0.0),
            })

            # Log predictions CSV as artifact
            if fold_id in prediction_csvs:
                mlflow.log_artifact(str(prediction_csvs[fold_id]), artifact_path="predictions")

            # Log tags for UI filtering
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("protocol", protocol)
            mlflow.set_tag("benchmark_model", model)

        runs_created += 1

    return runs_created


def main():
    parser = argparse.ArgumentParser(
        description="Sync Vertex AI benchmark results into local MLflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"Comma-separated model names or 'all'. Available: {', '.join(BENCHMARK_MODELS)}",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        default="all",
        help=f"Comma-separated protocol names or 'all'. Available: {', '.join(PROTOCOLS)}",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Skip GCS download, sync from local benchmark dir only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without creating MLflow runs.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=MLFLOW_EXPERIMENT_NAME,
        help=f"MLflow experiment name (default: {MLFLOW_EXPERIMENT_NAME}).",
    )
    args = parser.parse_args()

    # Parse model/protocol lists
    models = BENCHMARK_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]
    protocols = PROTOCOLS if args.protocols == "all" else [p.strip() for p in args.protocols.split(",")]

    print("=" * 60)
    print("Sync Vertex AI Results → MLflow")
    print("=" * 60)
    print(f"  Models:     {', '.join(models)}")
    print(f"  Protocols:  {', '.join(protocols)}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Local only: {args.local}")
    print(f"  Dry run:    {args.dry_run}")
    print()

    # Set up MLflow
    if not args.dry_run:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(args.experiment)

    total_runs = 0

    for model in models:
        for protocol in protocols:
            output_dir = LOCAL_BENCHMARK_DIR / f"{model}_{protocol}"

            print(f"\n  {model} × {protocol}:")

            # Download from GCS if needed
            if not args.local:
                download_from_gcs(model, protocol)

            if not output_dir.exists():
                print(f"    Skipping (no output dir)")
                continue

            # Sync to MLflow
            n_runs = sync_to_mlflow(model, protocol, output_dir, dry_run=args.dry_run)
            total_runs += n_runs
            action = "would create" if args.dry_run else "created"
            print(f"    {action} {n_runs} MLflow runs")

    print(f"\n{'=' * 60}")
    action = "Would create" if args.dry_run else "Created"
    print(f"  {action} {total_runs} total MLflow runs")
    print(f"  View: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("=" * 60)


if __name__ == "__main__":
    main()
