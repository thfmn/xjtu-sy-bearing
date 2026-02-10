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

"""Benchmark runner: orchestrates all model × protocol training combinations.

Runs each (model, protocol) pair by invoking the appropriate training script
with the correct config file and --cv-strategy flag. Results are collected
into a unified CSV at outputs/benchmark/results_matrix.csv.

Usage:
    # Dry run — show plan without training
    python scripts/12_run_benchmark.py --dry-run

    # Run all baselines under all protocols
    python scripts/12_run_benchmark.py --models feature_lstm,lgbm,cnn1d_baseline

    # Run a single model under a single protocol
    python scripts/12_run_benchmark.py --models dta_mlp --protocols jin

    # Resume (skip already-completed pairs)
    python scripts/12_run_benchmark.py --resume
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BENCHMARK_DIR = Path("outputs/benchmark")

# Protocol name → --cv-strategy value
PROTOCOL_MAP = {
    "lobo": "leave_one_bearing_out",
    "jin": "jin_fixed",
    "sun": "sun_fixed",
}


@dataclass
class ModelSpec:
    """Specification for a benchmark model."""

    name: str
    script: str  # Training script to invoke
    config: str  # YAML config file path
    extra_args: list[str]  # Any additional CLI arguments


# Model definitions: maps model key → ModelSpec
MODEL_SPECS: dict[str, ModelSpec] = {
    "cnn1d_baseline": ModelSpec(
        name="cnn1d_baseline",
        script="scripts/05_train_dl_models.py",
        config="configs/fulllife_cnn1d.yaml",
        extra_args=["--model", "cnn1d_baseline"],
    ),
    "feature_lstm": ModelSpec(
        name="feature_lstm",
        script="scripts/11_train_feature_lstm.py",
        config="configs/fulllife_feature_lstm.yaml",
        extra_args=[],
    ),
    "lgbm": ModelSpec(
        name="lgbm",
        script="scripts/10_train_lgbm.py",
        config="configs/twostage_lgbm_fulllife.yaml",
        extra_args=[],
    ),
    "dta_mlp": ModelSpec(
        name="dta_mlp",
        script="scripts/05_train_dl_models.py",
        config="configs/benchmark_dta_mlp.yaml",
        extra_args=["--model", "dta_mlp"],
    ),
    "cnn2d_simple": ModelSpec(
        name="cnn2d_simple",
        script="scripts/05_train_dl_models.py",
        config="configs/fulllife_cnn2d.yaml",
        extra_args=["--model", "cnn2d_simple"],
    ),
    "tcn_transformer_lstm": ModelSpec(
        name="tcn_transformer_lstm",
        script="scripts/05_train_dl_models.py",
        config="configs/benchmark_tcn_transformer_lstm.yaml",
        extra_args=["--model", "tcn_transformer_lstm"],
    ),
}


def get_output_dir(model_name: str, protocol: str) -> Path:
    """Get the output directory for a model-protocol pair."""
    return BENCHMARK_DIR / f"{model_name}_{protocol}"


def is_completed(model_name: str, protocol: str) -> bool:
    """Check if a model-protocol pair has already been trained.

    Looks for any *_fold_results.csv in the output directory (model name in
    the filename comes from the YAML config, which may differ from our key).
    """
    output_dir = get_output_dir(model_name, protocol)
    return any(output_dir.glob("*_fold_results.csv"))


def build_command(spec: ModelSpec, protocol: str) -> list[str]:
    """Build the subprocess command for a model-protocol pair."""
    output_dir = get_output_dir(spec.name, protocol)
    cv_strategy = PROTOCOL_MAP[protocol]

    cmd = [
        sys.executable,
        spec.script,
        "--config", spec.config,
        "--cv-strategy", cv_strategy,
        "--output-dir", str(output_dir),
    ]
    # Only 05_train_dl_models.py supports --tracking
    if "05_train_dl_models" in spec.script:
        cmd.extend(["--tracking", "mlflow"])
    cmd.extend(spec.extra_args)
    return cmd


def run_pair(spec: ModelSpec, protocol: str, dry_run: bool = False) -> dict:
    """Run a single model-protocol training pair.

    Returns a result dict with model, protocol, status, duration, and error info.
    """
    cmd = build_command(spec, protocol)
    output_dir = get_output_dir(spec.name, protocol)

    result = {
        "model": spec.name,
        "protocol": protocol,
        "cv_strategy": PROTOCOL_MAP[protocol],
        "output_dir": str(output_dir),
        "command": " ".join(cmd),
        "status": "pending",
        "duration_minutes": 0.0,
        "error": "",
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    print(f"\n{'=' * 70}")
    print(f"  Running: {spec.name} × {protocol}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output:  {output_dir}")
    print(f"{'=' * 70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            text=True,
        )
        elapsed = (time.time() - start) / 60
        result["status"] = "success"
        result["duration_minutes"] = round(elapsed, 1)
        print(f"\n  Completed {spec.name} × {protocol} in {elapsed:.1f} min")
    except subprocess.CalledProcessError as e:
        elapsed = (time.time() - start) / 60
        result["status"] = "failed"
        result["duration_minutes"] = round(elapsed, 1)
        result["error"] = str(e)
        print(f"\n  FAILED {spec.name} × {protocol} after {elapsed:.1f} min: {e}")
    except KeyboardInterrupt:
        elapsed = (time.time() - start) / 60
        result["status"] = "interrupted"
        result["duration_minutes"] = round(elapsed, 1)
        print(f"\n  Interrupted {spec.name} × {protocol} after {elapsed:.1f} min")
        raise

    return result


def collect_results_matrix(models: list[str], protocols: list[str]) -> None:
    """Scan completed benchmark outputs and compile results_matrix.csv.

    Reads each model-protocol's fold_results.csv and combines them into a
    single wide-format results matrix.
    """
    import pandas as pd

    all_rows = []
    for model in models:
        for protocol in protocols:
            output_dir = get_output_dir(model, protocol)
            # Model name in CSV may differ from our key (comes from YAML config)
            csvs = list(output_dir.glob("*_fold_results.csv"))
            if not csvs:
                continue
            df = pd.read_csv(csvs[0])
            df["benchmark_model"] = model
            df["protocol"] = protocol
            df["cv_strategy"] = PROTOCOL_MAP[protocol]
            all_rows.append(df)

    if not all_rows:
        print("\nNo completed results to collect.")
        return

    matrix = pd.concat(all_rows, ignore_index=True)
    matrix_path = BENCHMARK_DIR / "results_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    print(f"\n  Results matrix: {matrix_path} ({len(matrix)} rows)")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("  Benchmark Results Summary")
    print(f"{'=' * 70}")
    summary = matrix.groupby(["model_name", "protocol"]).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        n_folds=("fold_id", "count"),
    ).round(4)
    print(summary.to_string())
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark training across models and protocols.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/12_run_benchmark.py --dry-run
    python scripts/12_run_benchmark.py --models feature_lstm,lgbm
    python scripts/12_run_benchmark.py --models dta_mlp --protocols jin
    python scripts/12_run_benchmark.py --resume
        """,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Comma-separated model names or 'all'. "
            f"Available: {', '.join(sorted(MODEL_SPECS))}"
        ),
    )
    parser.add_argument(
        "--protocols",
        type=str,
        default="all",
        help="Comma-separated protocol names or 'all'. Available: lobo, jin, sun",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without actually training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip model-protocol pairs that already have results.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect existing results into results_matrix.csv (no training).",
    )
    args = parser.parse_args()

    # Parse model list
    if args.models == "all":
        models = sorted(MODEL_SPECS.keys())
    else:
        models = [m.strip() for m in args.models.split(",")]
        for m in models:
            if m not in MODEL_SPECS:
                print(f"ERROR: Unknown model '{m}'. Available: {sorted(MODEL_SPECS)}")
                sys.exit(1)

    # Parse protocol list
    if args.protocols == "all":
        protocols = list(PROTOCOL_MAP.keys())
    else:
        protocols = [p.strip() for p in args.protocols.split(",")]
        for p in protocols:
            if p not in PROTOCOL_MAP:
                print(f"ERROR: Unknown protocol '{p}'. Available: {sorted(PROTOCOL_MAP)}")
                sys.exit(1)

    # Collect-only mode
    if args.collect_only:
        collect_results_matrix(models, protocols)
        return

    # Build run plan
    pairs = []
    for model in models:
        for protocol in protocols:
            skip = args.resume and is_completed(model, protocol)
            pairs.append((model, protocol, skip))

    # Print plan
    print(f"\n{'=' * 70}")
    print("  Benchmark Plan")
    print(f"{'=' * 70}")
    print(f"  Models:    {', '.join(models)}")
    print(f"  Protocols: {', '.join(protocols)}")
    print(f"  Total:     {len(pairs)} combinations")
    if args.resume:
        skipped = sum(1 for _, _, s in pairs if s)
        print(f"  Skipping:  {skipped} (already completed)")
    print()

    for model, protocol, skip in pairs:
        status = "SKIP (completed)" if skip else ("DRY RUN" if args.dry_run else "RUN")
        cmd = " ".join(build_command(MODEL_SPECS[model], protocol))
        print(f"  [{status:>16s}] {model:20s} × {protocol:4s}")
        if args.dry_run:
            print(f"                     {cmd}")

    print(f"{'=' * 70}\n")

    if args.dry_run:
        return

    # Run all pairs
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    run_log = []
    total_start = time.time()

    for model, protocol, skip in pairs:
        if skip:
            print(f"  Skipping {model} × {protocol} (already completed)")
            continue

        try:
            result = run_pair(MODEL_SPECS[model], protocol, dry_run=False)
            run_log.append(result)
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user. Saving partial results...")
            break

    total_elapsed = (time.time() - total_start) / 60

    # Save run log
    if run_log:
        log_path = BENCHMARK_DIR / "benchmark_run_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_log[0].keys())
            writer.writeheader()
            writer.writerows(run_log)
        print(f"\n  Run log: {log_path}")

    # Collect results matrix
    collect_results_matrix(models, protocols)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  Total benchmark time: {total_elapsed:.1f} minutes")
    succeeded = sum(1 for r in run_log if r["status"] == "success")
    failed = sum(1 for r in run_log if r["status"] == "failed")
    print(f"  Succeeded: {succeeded}, Failed: {failed}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
