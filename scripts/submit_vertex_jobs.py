#!/usr/bin/env python3
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
GCS_FEATURES_CSV = f"/gcs/{STAGING_BUCKET_NAME}/outputs/features/features_v2.csv"
GCS_OUTPUT_BASE = f"/gcs/{STAGING_BUCKET_NAME}/outputs/vertex_training"

# Hourly costs (approximate, asia-southeast1)
HOURLY_COSTS = {
    "gpu-t4": 0.54,  # n1-standard-4 + NVIDIA T4
    "cpu-only": 0.19,  # n1-standard-4
}

# Estimated training time per fold (hours) by model and machine type
TRAINING_TIME_ESTIMATES = {
    "cnn1d_baseline": {"gpu-t4": 0.5, "cpu-only": 2.0},
    "tcn_transformer_lstm": {"gpu-t4": 1.0, "cpu-only": 4.0},
    "tcn_transformer_transformer": {"gpu-t4": 1.2, "cpu-only": 5.0},
    "cnn2d_lstm": {"gpu-t4": 0.8, "cpu-only": 3.0},
    "cnn2d_simple": {"gpu-t4": 0.6, "cpu-only": 2.5},
}

# Default training time for models not in the lookup table
DEFAULT_TRAINING_TIME = {"gpu-t4": 1.0, "cpu-only": 4.0}


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


MACHINE_CONFIGS: dict[str, MachineConfig] = {
    "gpu-t4": MachineConfig(
        name="gpu-t4",
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        container_uri=CONTAINER_URI,
        hourly_cost=HOURLY_COSTS["gpu-t4"],
    ),
    "cpu-only": MachineConfig(
        name="cpu-only",
        machine_type="n1-standard-4",
        accelerator_type=None,
        accelerator_count=0,
        container_uri=CONTAINER_URI,
        hourly_cost=HOURLY_COSTS["cpu-only"],
    ),
}


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
) -> dict[str, float]:
    """Estimate training cost.

    Args:
        model_name: Name of the model to train.
        num_folds: Number of folds to train.
        machine_type: Machine configuration key.

    Returns:
        Dictionary with hours_per_fold, total_hours, hourly_rate, total_cost.
    """
    time_estimates = TRAINING_TIME_ESTIMATES.get(model_name, DEFAULT_TRAINING_TIME)
    hours_per_fold = time_estimates.get(machine_type, DEFAULT_TRAINING_TIME[machine_type])
    hourly_rate = HOURLY_COSTS.get(machine_type, 0.50)

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
) -> None:
    """Print formatted cost estimate."""
    estimate = estimate_cost(model_name, len(fold_ids), machine_type)
    machine_config = MACHINE_CONFIGS[machine_type]

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
    tracking_mode: str = "vertex",
) -> dict:
    """Build CustomJob configuration for a single fold.

    Args:
        model_name: Name of the model to train.
        fold_id: Fold ID to train.
        machine_config: Machine configuration.
        config_path: Optional path to training config YAML.
        tracking_mode: Experiment tracking mode.

    Returns:
        Dictionary with job_name and worker_pool_specs.
    """
    job_name = f"{model_name}-fold{fold_id}"
    output_dir = f"{GCS_OUTPUT_BASE}/{model_name}"

    # Build training command args
    args = [
        "--model", model_name,
        "--folds", str(fold_id),
        "--data-root", GCS_DATA_ROOT,
        "--spectrogram-dir", GCS_SPECTROGRAM_DIR,
        "--features-csv", GCS_FEATURES_CSV,
        "--output-dir", output_dir,
        "--tracking", tracking_mode,
    ]

    if config_path:
        args.extend(["--config", config_path])

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
) -> list:
    """Submit Vertex AI CustomJobs.

    Args:
        job_configs: List of job configurations from build_job_config().
        parallel: If True, submit all jobs simultaneously. If False, wait for each.
        wait: If True, wait for all jobs to complete.

    Returns:
        List of submitted CustomJob objects.
    """
    from google.cloud import aiplatform

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
    )

    jobs = []
    for config in job_configs:
        print(f"  Submitting: {config['job_name']}")

        job = aiplatform.CustomJob(
            display_name=config["job_name"],
            worker_pool_specs=config["worker_pool_specs"],
            staging_bucket=STAGING_BUCKET,
        )

        if parallel:
            # Submit without blocking
            job.submit()
        else:
            # Run synchronously (blocks until complete)
            job.run(sync=True)

        jobs.append(job)

        # Print console URL
        console_url = (
            f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
            f"/{REGION}/{job.resource_name.split('/')[-1]}"
            f"?project={PROJECT_ID}"
        )
        print(f"    Console: {console_url}")

    if parallel and wait:
        print("\n  Waiting for all jobs to complete...")
        for job in jobs:
            job.wait()
            print(f"    {job.display_name}: {job.state.name}")

    return jobs


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
        required=True,
        help="Model name (e.g., cnn1d_baseline, tcn_transformer_lstm).",
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

    return parser.parse_args()


def validate_model(model_name: str) -> None:
    """Validate model name against registry."""
    registered = list_models()
    if model_name not in registered:
        available = ", ".join(registered) if registered else "(none)"
        print(f"ERROR: Model '{model_name}' not found in registry.")
        print(f"Available models: {available}")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Vertex AI Training Job Submission")
    print("=" * 60)

    # Validate model
    validate_model(args.model)
    print(f"  Model:        {args.model}")

    # Parse folds
    try:
        fold_ids = parse_folds(args.folds)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"  Folds:        {len(fold_ids)} ({fold_ids[0]}-{fold_ids[-1] if len(fold_ids) > 1 else fold_ids[0]})")

    # Machine config
    machine_config = MACHINE_CONFIGS[args.machine_type]
    print(f"  Machine:      {args.machine_type}")
    print(f"    Instance:   {machine_config.machine_type}")
    if machine_config.has_gpu:
        print(f"    GPU:        {machine_config.accelerator_type}")
    print(f"  Container:    {machine_config.container_uri}")
    print(f"  Tracking:     {args.tracking}")
    print(f"  Parallel:     {not args.sequential}")

    # Cost estimation
    if args.estimate_cost:
        print_cost_estimate(args.model, fold_ids, args.machine_type)

    # Build job configs
    job_configs = []
    for fold_id in fold_ids:
        config = build_job_config(
            model_name=args.model,
            fold_id=fold_id,
            machine_config=machine_config,
            config_path=args.config,
            tracking_mode=args.tracking,
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
        estimate = estimate_cost(args.model, len(fold_ids), args.machine_type)
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
