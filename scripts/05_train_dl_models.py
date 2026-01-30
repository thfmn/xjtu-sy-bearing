#!/usr/bin/env python3
"""Unified Deep Learning Model Training Script for XJTU-SY Bearing Dataset.

Trains any registered DL model with leave-one-bearing-out cross-validation
and MLflow experiment tracking.

Usage:
    python scripts/05_train_dl_models.py --model cnn1d_baseline --config configs/cnn1d_baseline.yaml
    python scripts/05_train_dl_models.py --model all
    python scripts/05_train_dl_models.py --model cnn1d_baseline --folds 0,1,2
    python scripts/05_train_dl_models.py --model cnn1d_baseline --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.cv import leave_one_bearing_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DL models for bearing RUL prediction with CV and MLflow tracking.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g. cnn1d_baseline), comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML. If omitted, looks for configs/{model_name}.yaml, "
        "then falls back to configs/training_default.yaml.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated fold IDs to train (e.g. 0,1,2). Default: all 15 folds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build models and print summaries without training.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="assets/Data/XJTU-SY_Bearing_Datasets",
        help="Root directory of raw bearing CSV data.",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=str,
        default="outputs/spectrograms/stft",
        help="Directory containing .npy spectrogram files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory for saving results, predictions, and history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("DL Model Training Script")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  Config:          {args.config or '(auto-resolve)'}")
    print(f"  Folds:           {args.folds or 'all'}")
    print(f"  Dry run:         {args.dry_run}")
    print(f"  Data root:       {args.data_root}")
    print(f"  Spectrogram dir: {args.spectrogram_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print("=" * 60)

    # --- TRAIN-2: Load metadata and generate CV folds ---
    features_csv = Path("outputs/features/features_v2.csv")
    print(f"\nLoading metadata from {features_csv} ...")
    metadata_df = pd.read_csv(features_csv)
    print(f"  Loaded {len(metadata_df)} rows, {len(metadata_df.columns)} columns")

    cv_split = leave_one_bearing_out(metadata_df)
    print(f"  Generated {len(cv_split)} CV folds ({cv_split.strategy})")

    # Filter folds if --folds specified
    if args.folds is not None:
        fold_ids = [int(f.strip()) for f in args.folds.split(",")]
        folds = [fold for fold in cv_split if fold.fold_id in fold_ids]
        if not folds:
            print(f"ERROR: No folds matched IDs {fold_ids}. Available: 0-{len(cv_split) - 1}")
            sys.exit(1)
    else:
        folds = list(cv_split)

    # Print fold summary
    print(f"\n  Folds to train: {len(folds)}")
    print(f"  {'Fold':>5s}  {'Train':>6s}  {'Val':>5s}  {'Val Bearing(s)'}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*30}")
    for fold in folds:
        val_bearings_str = ", ".join(fold.val_bearings)
        print(f"  {fold.fold_id:>5d}  {len(fold.train_indices):>6d}  {len(fold.val_indices):>5d}  {val_bearings_str}")
    print()


if __name__ == "__main__":
    main()
