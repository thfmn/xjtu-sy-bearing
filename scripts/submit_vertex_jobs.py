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

"""Vertex AI Training Job Submission CLI Tool.

Submits Vertex AI CustomJobs for training bearing RUL models with:
- Flexible fold selection (all, range, or specific folds)
- GPU or CPU-only machine configurations
- Cost estimation before submission
- Parallel or sequential job submission
- Results aggregation and local MLflow sync

Usage:
    # Dry run with cost estimate
    python scripts/submit_vertex_jobs.py --model cnn1d_baseline --estimate-cost --dry-run

    # Train all 15 folds in parallel with T4 GPU
    python scripts/submit_vertex_jobs.py --model cnn1d_baseline --folds all --yes

    # Train folds 1-4 and wait for aggregation
    python scripts/submit_vertex_jobs.py --model tcn_transformer_lstm --folds 1-4 --wait

    # CPU-only for testing
    python scripts/submit_vertex_jobs.py --model cnn1d_baseline --folds 0 --machine-type cpu-only
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry import list_models

# =============================================================================
# Constants
# =============================================================================

PROJECT_ID = "xjtu-bearing-failure"
REGION = "asia-southeast1"  # Singapore - Vertex AI supported region
BUCKET_NAME = "xjtu-bearing-failure-dev-data"  # Data bucket (asia-southeast3)
STAGING_BUCKET_NAME = "xjtu-bearing-failure-staging"  # Staging bucket (must be in REGION)
STAGING_BUCKET = f"gs://{STAGING_BUCKET_NAME}"

# Custom container in Artifact Registry
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/training/bearing-rul:latest"

# GCS paths (using /gcs/ fuse mount inside Vertex AI jobs)
# All paths must use staging bucket (same region as Vertex AI jobs)
GCS_DATA_ROOT = f"/gcs/{STAGING_BUCKET_NAME}/xjtu_data"
GCS_SPECTROGRAM_DIR = f"/gcs/{STAGING_BUCKET_NAME}/outputs/spectrograms/stft"
GCS_CWT_DIR = f"/gcs/{STAGING_BUCKET_NAME}/outputs/spectrograms/cwt"
GCS_FEATURES_CSV = f"/gcs/{STAGING_BUCKET_NAME}/outputs/features/features_v2.csv"
GCS_OUTPUT_BASE = f"/gcs/{STAGING_BUCKET_NAME}/outputs/vertex_training"

# CV strategy name → --cv-strategy value
CV_STRATEGY_MAP = {
    "lobo": "leave_one_bearing_out",
    "jin": "jin_fixed",
    "sun": "sun_fixed",
}

# Number of folds per CV strategy
FOLDS_PER_STRATEGY = {
    "lobo": 15,
    "jin": 1,
    "sun": 1,
}

# Benchmark model definitions: maps model key → (config_path, extra_args)
BENCHMARK_MODELS = {
    "cnn1d_baseline": ("configs/fulllife_cnn1d.yaml", ["--model", "cnn1d_baseline"]),
    "cnn2d_simple": ("configs/fulllife_cnn2d.yaml", ["--model", "cnn2d_simple"]),
    "dta_mlp": ("configs/benchmark_dta_mlp.yaml", ["--model", "dta_mlp"]),
    "tcn_transformer_lstm": ("configs/benchmark_tcn_transformer_lstm.yaml", ["--model", "tcn_transformer_lstm"]),
}

# Hourly costs (approximate, asia-southeast1)
HOURLY_COSTS_ON_DEMAND = {
    "gpu-t4": 0.54,  # n1-standard-4 + NVIDIA T4
    "gpu-t4-highmem": 0.64,  # n1-standard-8 + NVIDIA T4 (30 GB RAM)
    "cpu-only": 0.19,  # n1-standard-4
}

# Spot VM costs (~60-70% discount over on-demand)
HOURLY_COSTS_SPOT = {
    "gpu-t4": 0.18,  # n1-standard-4 + NVIDIA T4 (Spot)
    "gpu-t4-highmem": 0.22,  # n1-standard-8 + NVIDIA T4 (Spot)
    "cpu-only": 0.06,  # n1-standard-4 (Spot)
}

# Default: use Spot pricing (override with --no-spot)
HOURLY_COSTS = HOURLY_COSTS_SPOT

# Estimated training time per fold (hours) by model and machine type
TRAINING_TIME_ESTIMATES = {
    "cnn1d_baseline": {"gpu-t4": 0.5, "cpu-only": 2.0},
    "tcn_transformer_lstm": {"gpu-t4": 1.0, "cpu-only": 4.0},
    "tcn_transformer_transformer": {"gpu-t4": 1.2, "cpu-only": 5.0},
    "cnn2d_lstm": {"gpu-t4": 0.8, "cpu-only": 3.0},
    "cnn2d_simple": {"gpu-t4": 0.6, "cpu-only": 2.5},
}

# Default training time for models not in the lookup table
DEFAULT_TRAINING_TIME = {"gpu-t4": 1.0, "gpu-t4-highmem": 1.0, "cpu-only": 4.0}


# =============================================================================
# Machine Configurations
# =============================================================================


@dataclass(frozen=True)
class MachineConfig:
    """Configuration for Vertex AI worker pool."""

    name: str
    machine_type: str
    accelerator_type: str | None
    accelerator_count: int
    container_uri: str
    hourly_cost: float

    @property
    def has_gpu(self) -> bool:
        return self.accelerator_type is not None


def _build_machine_configs(spot: bool = True) -> dict[str, MachineConfig]:
    """Build machine configs with on-demand or spot pricing."""
    costs = HOURLY_COSTS_SPOT if spot else HOURLY_COSTS_ON_DEMAND
    return {
        "gpu-t4": MachineConfig(
            name="gpu-t4",
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            container_uri=CONTAINER_URI,
            hourly_cost=costs["gpu-t4"],
        ),
        "gpu-t4-highmem": MachineConfig(
            name="gpu-t4-highmem",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            container_uri=CONTAINER_URI,
            hourly_cost=costs["gpu-t4-highmem"],
        ),
        "cpu-only": MachineConfig(
            name="cpu-only",
            machine_type="n1-standard-4",
            accelerator_type=None,
            accelerator_count=0,
            container_uri=CONTAINER_URI,
            hourly_cost=costs["cpu-only"],
        ),
    }


# Default configs (spot pricing)
MACHINE_CONFIGS: dict[str, MachineConfig] = _build_machine_configs(spot=True)


# =============================================================================
# Fold Parsing
# =============================================================================


def parse_folds(folds_str: str, max_folds: int = 15) -> list[int]:
    """Parse fold selection string into list of fold IDs.

    Args:
        folds_str: Fold specification. Can be:
            - "all": Returns [0..max_folds-1]
            - "0-4": Returns range [0,1,2,3,4]
            - "1,3,5": Returns [1,3,5]
            - "7": Returns [7]
        max_folds: Maximum number of folds (default 15 for XJTU-SY).

    Returns:
        Sorted list of unique fold IDs.

    Raises:
        ValueError: If format is invalid or fold IDs are out of range.
    """
    folds_str = folds_str.strip().lower()

    if folds_str == "all":
        return list(range(max_folds))

    # Range format: "0-4"
    if "-" in folds_str and "," not in folds_str:
        parts = folds_str.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: '{folds_str}'. Use '0-4' format.")
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid range values: '{folds_str}'")
        if start > end:
            raise ValueError(f"Range start ({start}) > end ({end})")
        if start < 0 or end >= max_folds:
            raise ValueError(f"Fold IDs must be in 0-{max_folds-1}")
        return list(range(start, end + 1))

    # Comma-separated or single value: "1,3,5" or "7"
    try:
        fold_ids = [int(f.strip()) for f in folds_str.split(",")]
    except ValueError:
        raise ValueError(f"Invalid fold format: '{folds_str}'")

    for fid in fold_ids:
        if fid < 0 or fid >= max_folds:
            raise ValueError(f"Fold ID {fid} out of range (0-{max_folds-1})")

    return sorted(set(fold_ids))


# =============================================================================
# Cost Estimation
# =============================================================================


def estimate_cost(
    model_name: str,
    num_folds: int,
    machine_type: str,
    spot: bool = True,
) -> dict[str, float]:
    """Estimate training cost.

    Args:
        model_name: Name of the model to train.
        num_folds: Number of folds to train.
        machine_type: Machine configuration key.
        spot: If True, use Spot VM pricing.

    Returns:
        Dictionary with hours_per_fold, total_hours, hourly_rate, total_cost.
    """
    time_estimates = TRAINING_TIME_ESTIMATES.get(model_name, DEFAULT_TRAINING_TIME)
    hours_per_fold = time_estimates.get(machine_type, DEFAULT_TRAINING_TIME[machine_type])
    costs = HOURLY_COSTS_SPOT if spot else HOURLY_COSTS_ON_DEMAND
    hourly_rate = costs.get(machine_type, 0.50)

    total_hours = hours_per_fold * num_folds
    total_cost = total_hours * hourly_rate

    return {
        "hours_per_fold": hours_per_fold,
        "total_hours": total_hours,
        "hourly_rate": hourly_rate,
        "total_cost": total_cost,
    }


def print_cost_estimate(
    model_name: str,
    fold_ids: list[int],
    machine_type: str,
    spot: bool = True,
) -> None:
    """Print formatted cost estimate."""
    estimate = estimate_cost(model_name, len(fold_ids), machine_type, spot=spot)
    machine_configs = _build_machine_configs(spot=spot)
    machine_config = machine_configs[machine_type]

    print("\n" + "=" * 60)
    print("Cost Estimation")
    print("=" * 60)
    print(f"  Model:           {model_name}")
    print(f"  Machine type:    {machine_type}")
    print(f"    Instance:      {machine_config.machine_type}")
    if machine_config.has_gpu:
        print(f"    GPU:           {machine_config.accelerator_type}")
    print(f"  Folds:           {len(fold_ids)} ({fold_ids[0]}-{fold_ids[-1] if len(fold_ids) > 1 else fold_ids[0]})")
    print()
    print(f"  Hours per fold:  ~{estimate['hours_per_fold']:.1f}h")
    print(f"  Total hours:     ~{estimate['total_hours']:.1f}h")
    print(f"  Hourly rate:     ${estimate['hourly_rate']:.2f}/hr")
    print(f"  Estimated cost:  ${estimate['total_cost']:.2f}")
    print("=" * 60)
    print("\nNote: Costs are estimates. Actual costs depend on training duration.")


# =============================================================================
# Job Building
# =============================================================================


def build_job_config(
    model_name: str,
    fold_id: int,
    machine_config: MachineConfig,
    config_path: str | None = None,
    tracking_mode: str = "none",
    two_stage: bool = False,
    cv_strategy: str | None = None,
    output_dir_override: str | None = None,
) -> dict:
    """Build CustomJob configuration for a single fold.

    Args:
        model_name: Name of the model to train.
        fold_id: Fold ID to train.
        machine_config: Machine configuration.
        config_path: Optional path to training config YAML.
        tracking_mode: Experiment tracking mode.
        two_stage: Enable two-stage pipeline (onset detection + RUL).
        cv_strategy: CV strategy value (e.g. "leave_one_bearing_out", "jin_fixed").
        output_dir_override: Custom GCS output directory (overrides default).

    Returns:
        Dictionary with job_name and worker_pool_specs.
    """
    job_name = f"{model_name}-fold{fold_id}"
    output_dir = output_dir_override or f"{GCS_OUTPUT_BASE}/{model_name}"

    # Build training command args
    args = [
        "--model", model_name,
        "--folds", str(fold_id),
        "--data-root", GCS_DATA_ROOT,
        "--spectrogram-dir", GCS_SPECTROGRAM_DIR,
        "--cwt-dir", GCS_CWT_DIR,
        "--features-csv", GCS_FEATURES_CSV,
        "--output-dir", output_dir,
        "--tracking", tracking_mode,
    ]

    if config_path:
        args.extend(["--config", config_path])

    if cv_strategy:
        args.extend(["--cv-strategy", cv_strategy])

    if two_stage:
        args.append("--two-stage")

    # Build worker pool spec
    worker_pool_spec = {
        "machine_spec": {
            "machine_type": machine_config.machine_type,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": machine_config.container_uri,
            "command": ["python", "scripts/05_train_dl_models.py"],
            "args": args,
        },
    }

    # Add GPU if configured
    if machine_config.has_gpu:
        worker_pool_spec["machine_spec"]["accelerator_type"] = machine_config.accelerator_type
        worker_pool_spec["machine_spec"]["accelerator_count"] = machine_config.accelerator_count

    return {
        "job_name": job_name,
        "worker_pool_specs": [worker_pool_spec],
    }


def print_job_config(job_config: dict, fold_id: int) -> None:
    """Print formatted job configuration."""
    print(f"\n  Fold {fold_id}:")
    print(f"    Job name: {job_config['job_name']}")
    spec = job_config["worker_pool_specs"][0]
    print(f"    Machine:  {spec['machine_spec']['machine_type']}")
    if "accelerator_type" in spec["machine_spec"]:
        print(f"    GPU:      {spec['machine_spec']['accelerator_type']}")
    print(f"    Container: {spec['container_spec']['image_uri']}")
    print(f"    Command:  {' '.join(spec['container_spec']['command'] + spec['container_spec']['args'][:6])}...")


# =============================================================================
# Job Submission
# =============================================================================


def submit_jobs(
    job_configs: list[dict],
    parallel: bool = True,
    wait: bool = False,
    spot: bool = True,
    max_parallel: int | None = None,
) -> list:
    """Submit Vertex AI CustomJobs.

    Args:
        job_configs: List of job configurations from build_job_config().
        parallel: If True, submit all jobs simultaneously. If False, wait for each.
        wait: If True, wait for all jobs to complete.
        spot: If True, use Spot VMs for ~60-70% cost savings. Jobs are
            automatically restarted (up to 6 times) if preempted.
        max_parallel: Maximum concurrent jobs. When set, jobs are submitted in
            batches of this size, waiting for each batch to complete before
            starting the next. Implies wait=True between batches.

    Returns:
        List of submitted CustomJob objects.
    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types.custom_job import Scheduling

    scheduling_strategy = (
        Scheduling.Strategy.SPOT if spot else Scheduling.Strategy.ON_DEMAND
    )
    vm_type = "Spot" if spot else "On-Demand"

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
    )

    # Determine batch size
    if max_parallel and parallel:
        batch_size = max_parallel
    else:
        batch_size = len(job_configs)

    all_jobs = []
    total = len(job_configs)
    for batch_start in range(0, total, batch_size):
        batch_configs = job_configs[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        num_batches = (total + batch_size - 1) // batch_size

        if num_batches > 1:
            print(f"\n  --- Batch {batch_num}/{num_batches} ({len(batch_configs)} jobs) ---")

        batch_jobs = []
        for config in batch_configs:
            print(f"  Submitting ({vm_type}): {config['job_name']}")

            job = aiplatform.CustomJob(
                display_name=config["job_name"],
                worker_pool_specs=config["worker_pool_specs"],
                staging_bucket=STAGING_BUCKET,
            )

            if parallel:
                job.submit(scheduling_strategy=scheduling_strategy)
            else:
                job.run(sync=True, scheduling_strategy=scheduling_strategy)

            batch_jobs.append(job)

            # Print console URL
            console_url = (
                f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
                f"/{REGION}/{job.resource_name.split('/')[-1]}"
                f"?project={PROJECT_ID}"
            )
            print(f"    Console: {console_url}")

        all_jobs.extend(batch_jobs)

        # Wait for batch completion before submitting next batch
        need_batch_wait = (
            num_batches > 1
            and batch_start + batch_size < total
            and parallel
        )
        if need_batch_wait or (parallel and wait and batch_start + batch_size >= total):
            print(f"\n  Waiting for batch {batch_num} to complete...")
            for job in batch_jobs:
                job.wait()
                print(f"    {job.display_name}: {job.state.name}")

    return all_jobs


def check_job_status(jobs: list) -> dict:
    """Check status of submitted jobs.

    Args:
        jobs: List of CustomJob objects.

    Returns:
        Dictionary with status summary.
    """
    from google.cloud.aiplatform_v1 import JobState

    status_counts = {
        "SUCCEEDED": 0,
        "FAILED": 0,
        "RUNNING": 0,
        "PENDING": 0,
        "OTHER": 0,
    }

    for job in jobs:
        job.refresh()  # Refresh state from API
        state_name = job.state.name if hasattr(job.state, "name") else str(job.state)

        if state_name == "JOB_STATE_SUCCEEDED":
            status_counts["SUCCEEDED"] += 1
        elif state_name == "JOB_STATE_FAILED":
            status_counts["FAILED"] += 1
        elif state_name in ("JOB_STATE_RUNNING", "JOB_STATE_PENDING"):
            status_counts["RUNNING" if "RUNNING" in state_name else "PENDING"] += 1
        else:
            status_counts["OTHER"] += 1

    return status_counts


# =============================================================================
# Results Aggregation
# =============================================================================


def aggregate_results(
    model_name: str,
    fold_ids: list[int],
) -> pd.DataFrame | None:
    """Aggregate results from completed training jobs.

    Reads per-fold prediction CSVs from GCS and computes aggregate metrics.

    Args:
        model_name: Name of the model.
        fold_ids: List of fold IDs to aggregate.

    Returns:
        DataFrame with per-fold and aggregate metrics, or None if no results.
    """
    from google.cloud import storage

    from src.training.metrics import evaluate_predictions

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    results = []
    for fold_id in fold_ids:
        pred_path = f"outputs/vertex_training/{model_name}/predictions/{model_name}_fold{fold_id}_predictions.csv"
        blob = bucket.blob(pred_path)

        if not blob.exists():
            print(f"  Warning: Missing predictions for fold {fold_id}")
            continue

        # Download and parse predictions
        csv_content = blob.download_as_text()
        pred_df = pd.read_csv(pd.io.common.StringIO(csv_content))

        y_true = pred_df["y_true"].values
        y_pred = pred_df["y_pred"].values

        metrics = evaluate_predictions(y_true, y_pred)
        metrics["fold_id"] = fold_id
        metrics["model_name"] = model_name
        results.append(metrics)

    if not results:
        print("  No results found to aggregate.")
        return None

    results_df = pd.DataFrame(results)

    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print(f"Aggregate Results for {model_name}")
    print("=" * 60)
    print(f"\nPer-fold metrics ({len(results)} folds):")
    print(results_df[["fold_id", "rmse", "mae", "mape", "phm08_score"]].to_string(index=False))

    print(f"\nMean ± Std:")
    for metric in ["rmse", "mae", "mape", "phm08_score"]:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"  {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")

    print("=" * 60)

    return results_df


def sync_to_mlflow(
    results_df: pd.DataFrame,
    model_name: str,
) -> None:
    """Sync aggregated results to local MLflow.

    Creates a parent run with summary metrics and child runs for each fold.

    Args:
        results_df: DataFrame with per-fold metrics.
        model_name: Name of the model.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("bearing_rul_prediction")

        # Log aggregate metrics in a summary run
        with mlflow.start_run(run_name=f"{model_name}_vertex_aggregate"):
            # Log summary metrics
            for metric in ["rmse", "mae", "mape", "phm08_score"]:
                mlflow.log_metric(f"mean_{metric}", results_df[metric].mean())
                mlflow.log_metric(f"std_{metric}", results_df[metric].std())

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("num_folds", len(results_df))
            mlflow.log_param("source", "vertex_ai")

            # Log per-fold results as artifact
            results_csv = f"/tmp/{model_name}_vertex_results.csv"
            results_df.to_csv(results_csv, index=False)
            mlflow.log_artifact(results_csv, artifact_path="results")

        print(f"\n  MLflow: Logged aggregate results to experiment 'bearing_rul_prediction'")

    except ImportError:
        print("\n  MLflow not available. Skipping local sync.")
    except Exception as e:
        print(f"\n  MLflow sync failed: {e}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit Vertex AI training jobs for bearing RUL models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with cost estimate
  python scripts/submit_vertex_jobs.py --model cnn1d_baseline --estimate-cost --dry-run

  # Train all 15 folds in parallel with T4 GPU
  python scripts/submit_vertex_jobs.py --model cnn1d_baseline --folds all --yes

  # Train folds 1-4 and wait for aggregation
  python scripts/submit_vertex_jobs.py --model tcn_transformer_lstm --folds 1-4 --wait

  # CPU-only for testing
  python scripts/submit_vertex_jobs.py --model cnn1d_baseline --folds 0 --machine-type cpu-only
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., cnn1d_baseline, dta_mlp). Required unless --benchmark.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Fold selection: 'all', '0-4' (range), '1,3,5' (list), or single number. Default: all",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        choices=list(MACHINE_CONFIGS.keys()),
        default="gpu-t4",
        help="Machine configuration: 'gpu-t4' (n1-standard-4 + T4) or 'cpu-only'. Default: gpu-t4",
    )
    parser.add_argument(
        "--tracking",
        type=str,
        choices=["vertex", "mlflow", "both", "none"],
        default="vertex",
        help="Experiment tracking: 'vertex' (default), 'mlflow', 'both', or 'none'.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Submit all folds simultaneously (default behavior).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Wait for each fold to complete before submitting the next.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job configs without submitting.",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Show cost estimation.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for jobs to complete and aggregate results.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Custom training config YAML path.",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Enable two-stage pipeline (onset detection + RUL prediction).",
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        choices=list(CV_STRATEGY_MAP.values()),
        default=None,
        help="CV strategy: 'leave_one_bearing_out', 'jin_fixed', or 'sun_fixed'.",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Use on-demand VMs instead of Spot VMs (default: Spot for ~60-70%% savings).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Maximum number of jobs to run concurrently. Jobs are submitted in "
            "batches; each batch waits for completion before the next starts. "
            "Useful for respecting GPU quota limits (e.g. --max-parallel 15)."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark mode: submit jobs for all remaining model × protocol pairs.",
    )
    parser.add_argument(
        "--benchmark-models",
        type=str,
        default="all",
        help=(
            "Comma-separated model names for benchmark mode, or 'all'. "
            f"Available: {', '.join(sorted(BENCHMARK_MODELS))}"
        ),
    )
    parser.add_argument(
        "--benchmark-protocols",
        type=str,
        default="all",
        help="Comma-separated protocol names for benchmark mode. Available: lobo, jin, sun",
    )

    return parser.parse_args()


def validate_model(model_name: str) -> None:
    """Validate model name against registry."""
    registered = list_models()
    if model_name not in registered:
        available = ", ".join(registered) if registered else "(none)"
        print(f"ERROR: Model '{model_name}' not found in registry.")
        print(f"Available models: {available}")
        sys.exit(1)


def build_benchmark_jobs(
    models: list[str],
    protocols: list[str],
    machine_config: MachineConfig,
    tracking_mode: str = "none",
) -> list[tuple[str, str, int, dict]]:
    """Build all job configs for benchmark mode.

    Returns list of (model, protocol, fold_id, job_config) tuples.
    """
    all_jobs = []
    for model in models:
        config_path, _extra_args = BENCHMARK_MODELS[model]
        for protocol in protocols:
            cv_strategy = CV_STRATEGY_MAP[protocol]
            n_folds = FOLDS_PER_STRATEGY[protocol]
            output_dir = f"{GCS_OUTPUT_BASE}/benchmark/{model}_{protocol}"

            for fold_id in range(n_folds):
                job_config = build_job_config(
                    model_name=model,
                    fold_id=fold_id,
                    machine_config=machine_config,
                    config_path=config_path,
                    tracking_mode=tracking_mode,
                    cv_strategy=cv_strategy,
                    output_dir_override=output_dir,
                )
                # Override job name to include protocol
                job_config["job_name"] = f"bench-{model}-{protocol}-fold{fold_id}"
                all_jobs.append((model, protocol, fold_id, job_config))

    return all_jobs


def main() -> None:
    args = parse_args()

    use_spot = not args.no_spot
    vm_label = "Spot" if use_spot else "On-Demand"

    # Rebuild machine configs with correct pricing
    machine_configs = _build_machine_configs(spot=use_spot)

    print("=" * 60)
    print("Vertex AI Training Job Submission")
    print("=" * 60)
    print(f"  VM pricing:   {vm_label}")

    machine_config = machine_configs[args.machine_type]

    # ---- Benchmark mode ----
    if args.benchmark:
        # Parse model list
        if args.benchmark_models == "all":
            models = sorted(BENCHMARK_MODELS.keys())
        else:
            models = [m.strip() for m in args.benchmark_models.split(",")]
            for m in models:
                if m not in BENCHMARK_MODELS:
                    print(f"ERROR: Unknown model '{m}'. Available: {sorted(BENCHMARK_MODELS)}")
                    sys.exit(1)

        # Parse protocol list
        if args.benchmark_protocols == "all":
            protocols = list(CV_STRATEGY_MAP.keys())
        else:
            protocols = [p.strip() for p in args.benchmark_protocols.split(",")]
            for p in protocols:
                if p not in CV_STRATEGY_MAP:
                    print(f"ERROR: Unknown protocol '{p}'. Available: {sorted(CV_STRATEGY_MAP)}")
                    sys.exit(1)

        # Build all job configs
        all_jobs = build_benchmark_jobs(models, protocols, machine_config, args.tracking)
        total_jobs = len(all_jobs)

        # Estimate cost
        total_hours = sum(
            TRAINING_TIME_ESTIMATES.get(model, DEFAULT_TRAINING_TIME).get(
                args.machine_type, DEFAULT_TRAINING_TIME[args.machine_type]
            )
            for model, _, _, _ in all_jobs
        )
        total_cost = total_hours * machine_config.hourly_cost

        print(f"\n  Benchmark Mode")
        print(f"  Models:    {', '.join(models)}")
        print(f"  Protocols: {', '.join(protocols)}")
        print(f"  Machine:   {args.machine_type} ({machine_config.machine_type})")
        if machine_config.has_gpu:
            print(f"  GPU:       {machine_config.accelerator_type}")
        print(f"  Total jobs: {total_jobs}")
        print(f"  Est. cost:  ${total_cost:.2f} ({total_hours:.1f} GPU-hours)")
        print()

        # Print job plan
        for model, protocol, fold_id, job_config in all_jobs:
            folds_total = FOLDS_PER_STRATEGY[protocol]
            print(f"  {job_config['job_name']:45s}  ({model} × {protocol}, fold {fold_id}/{folds_total})")

        if args.dry_run:
            print(f"\n{'=' * 60}")
            print("Dry run complete. No jobs submitted.")
            print("=" * 60)
            return

        # Confirmation
        if not args.yes:
            response = input(f"\n  Submit {total_jobs} jobs (est. ${total_cost:.2f})? [y/N]: ").strip().lower()
            if response != "y":
                print("  Aborted.")
                return

        # Submit all jobs in parallel
        print(f"\n{'=' * 60}")
        print("Submitting Benchmark Jobs")
        print("=" * 60)

        job_configs = [jc for _, _, _, jc in all_jobs]
        jobs = submit_jobs(
            job_configs=job_configs,
            parallel=not args.sequential,
            wait=args.wait,
            spot=use_spot,
            max_parallel=args.max_parallel,
        )

        print(f"\n  Submitted {len(jobs)} benchmark jobs.")

        if args.wait:
            status = check_job_status(jobs)
            print(f"\n  Final Status:")
            print(f"    Succeeded: {status['SUCCEEDED']}")
            print(f"    Failed:    {status['FAILED']}")

        print(f"\n{'=' * 60}")
        print(f"  Output base: gs://{STAGING_BUCKET_NAME}/outputs/vertex_training/benchmark/")
        print(f"  Sync results: python scripts/13_sync_vertex_to_mlflow.py")
        print("=" * 60)
        return

    # ---- Single-model mode (original behavior) ----
    if not args.model:
        print("ERROR: --model is required (or use --benchmark for batch mode).")
        sys.exit(1)
    validate_model(args.model)
    print(f"  Model:        {args.model}")

    # Parse folds
    try:
        fold_ids = parse_folds(args.folds)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"  Folds:        {len(fold_ids)} ({fold_ids[0]}-{fold_ids[-1] if len(fold_ids) > 1 else fold_ids[0]})")

    print(f"  Machine:      {args.machine_type}")
    print(f"    Instance:   {machine_config.machine_type}")
    if machine_config.has_gpu:
        print(f"    GPU:        {machine_config.accelerator_type}")
    print(f"  Container:    {machine_config.container_uri}")
    print(f"  Tracking:     {args.tracking}")
    print(f"  Two-stage:    {args.two_stage}")
    print(f"  Parallel:     {not args.sequential}")

    # Cost estimation
    if args.estimate_cost:
        print_cost_estimate(args.model, fold_ids, args.machine_type, spot=use_spot)

    # Build job configs
    job_configs = []
    for fold_id in fold_ids:
        config = build_job_config(
            model_name=args.model,
            fold_id=fold_id,
            machine_config=machine_config,
            config_path=args.config,
            tracking_mode=args.tracking,
            two_stage=args.two_stage,
            cv_strategy=args.cv_strategy,
        )
        job_configs.append(config)

    # Dry run: print configs and exit
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Job Configurations")
        print("=" * 60)
        for i, (fold_id, config) in enumerate(zip(fold_ids, job_configs)):
            print_job_config(config, fold_id)
        print("\n" + "=" * 60)
        print("Dry run complete. No jobs submitted.")
        print("=" * 60)
        return

    # Confirmation prompt
    if not args.yes:
        estimate = estimate_cost(args.model, len(fold_ids), args.machine_type, spot=use_spot)
        print(f"\n  Estimated cost: ${estimate['total_cost']:.2f}")
        response = input("\n  Submit jobs? [y/N]: ").strip().lower()
        if response != "y":
            print("  Aborted.")
            return

    # Submit jobs
    print("\n" + "=" * 60)
    print("Submitting Jobs")
    print("=" * 60)

    parallel = not args.sequential
    jobs = submit_jobs(
        job_configs=job_configs,
        parallel=parallel,
        wait=args.wait,
        spot=use_spot,
        max_parallel=args.max_parallel,
    )

    print(f"\n  Submitted {len(jobs)} jobs.")

    # Wait and aggregate if requested
    if args.wait:
        print("\n" + "=" * 60)
        print("Aggregating Results")
        print("=" * 60)

        # Check final status
        status = check_job_status(jobs)
        print(f"\n  Job Status:")
        print(f"    Succeeded: {status['SUCCEEDED']}")
        print(f"    Failed:    {status['FAILED']}")
        print(f"    Running:   {status['RUNNING']}")
        print(f"    Pending:   {status['PENDING']}")

        if status["SUCCEEDED"] > 0:
            results_df = aggregate_results(args.model, fold_ids)
            if results_df is not None:
                sync_to_mlflow(results_df, args.model)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
