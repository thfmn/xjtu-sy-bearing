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
"""LightGBM Two-Stage RUL Training Script.

Trains a LightGBM model on 65 extracted features with leave-one-bearing-out
cross-validation and MLflow experiment tracking. Uses two-stage onset filtering
so only post-onset samples are used for training and evaluation.

Usage:
    python scripts/10_train_lgbm.py --config configs/twostage_lgbm.yaml
    python scripts/10_train_lgbm.py --config configs/twostage_lgbm.yaml --folds 0,1,2
    python scripts/10_train_lgbm.py --config configs/twostage_lgbm.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.rul_labels import compute_twostage_rul, generate_rul_for_bearing
from src.models.baselines.lgbm_baseline import (
    LGBMConfig,
    LightGBMBaseline,
    get_feature_columns,
)
from src.onset.labels import load_onset_labels
from src.training.cv import generate_cv_folds
from src.training.metrics import evaluate_predictions
from src.utils.tracking import ExperimentTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM RUL model with two-stage onset filtering and LOBO CV.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/twostage_lgbm.yaml).",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default="outputs/features/features_v2.csv",
        help="Path to features CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory for saving results and predictions.",
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
        help="Load data and print summary without training.",
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="leave_one_bearing_out",
        choices=["leave_one_bearing_out", "jin_fixed", "li_fixed"],
        help="Cross-validation strategy. Default: leave_one_bearing_out (15-fold LOBO). "
        "jin_fixed: Jin et al. 2025 protocol (2 train, 13 test). "
        "li_fixed: Li et al. 2024 protocol (4 train, 6 test, Conds 1-2 only).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_twostage_features(
    features_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Load onset labels, compute two-stage RUL, and filter pre-onset samples.

    Args:
        features_df: Full features DataFrame with bearing_id, file_idx, rul columns.
        config: Parsed YAML config dict with onset and training sections.

    Returns:
        Filtered DataFrame with only post-onset samples and two-stage RUL labels.
    """
    onset_cfg = config.get("onset", {})
    training_cfg = config.get("training", {})
    max_rul = training_cfg.get("max_rul", 125)
    filter_pre_onset = training_cfg.get("filter_pre_onset", True)

    labels_path = onset_cfg.get("labels_path", "configs/onset_labels.yaml")
    print(f"  Loading onset labels from {labels_path} ...")
    onset_labels = load_onset_labels(labels_path)
    print(f"  Loaded onset labels for {len(onset_labels)} bearings")

    df = features_df.copy()
    df["rul_original"] = df["rul"].copy()
    df["is_post_onset"] = 0

    for bearing_id in df["bearing_id"].unique():
        if bearing_id not in onset_labels:
            print(f"  WARNING: {bearing_id} not in onset labels, skipping")
            continue

        mask = df["bearing_id"] == bearing_id
        onset_idx = onset_labels[bearing_id].onset_file_idx
        num_files = mask.sum()

        twostage_rul = compute_twostage_rul(num_files, onset_idx, max_rul=max_rul)
        df.loc[mask, "rul"] = twostage_rul
        df.loc[mask, "is_post_onset"] = (df.loc[mask, "file_idx"] >= onset_idx).astype(int)

    n_pre = (df["is_post_onset"] == 0).sum()
    n_post = (df["is_post_onset"] == 1).sum()
    print(f"  Onset split: {n_pre} pre-onset, {n_post} post-onset samples")

    if filter_pre_onset:
        original_count = len(df)
        df = df[df["is_post_onset"] == 1].reset_index(drop=True)
        print(f"  Filtered pre-onset: {original_count} -> {len(df)} samples")

    return df


def prepare_full_life_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize RUL to [0, 1] per bearing for full-life evaluation.

    All samples are included (no onset filtering). RUL is normalized per bearing
    so that the first file = 1.0 and the last file = 0.0. This produces metrics
    directly comparable to published papers (e.g., Jin et al. 2025).

    Args:
        features_df: Full features DataFrame with bearing_id column.

    Returns:
        DataFrame with RUL normalized to [0, 1] per bearing.
    """
    df = features_df.copy()
    for bearing_id in df["bearing_id"].unique():
        mask = df["bearing_id"] == bearing_id
        num_files = mask.sum()
        df.loc[mask, "rul"] = generate_rul_for_bearing(
            num_files, strategy="linear", normalize=True
        )
    return df


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = config.get("model_name", "lgbm_baseline")
    model_version = config.get("model_version", "v1")
    seed = config.get("seed", 42)

    eval_mode_label = config.get("training", {}).get("eval_mode", "post_onset")
    print("=" * 60)
    print(f"LightGBM RUL Training ({eval_mode_label})")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Model:        {model_name} {model_version}")
    print(f"  Features CSV: {args.features_csv}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  CV strategy:  {args.cv_strategy}")
    print(f"  Folds:        {args.folds or 'all'}")
    print(f"  Dry run:      {args.dry_run}")
    print("=" * 60)

    # Load features
    features_csv = Path(args.features_csv)
    print(f"\nLoading features from {features_csv} ...")
    features_df = pd.read_csv(features_csv)
    print(f"  Loaded {len(features_df)} rows, {len(features_df.columns)} columns")

    # Data preparation: branch on eval_mode
    eval_mode = config.get("training", {}).get("eval_mode", "post_onset")
    if eval_mode == "full_life":
        print("\nPreparing full-life data (normalized RUL [0, 1]) ...")
        features_df = prepare_full_life_features(features_df)
        rul_range = features_df["rul"]
        print(f"  RUL range: [{rul_range.min():.4f}, {rul_range.max():.4f}]")
    else:
        print("\nPreparing two-stage data ...")
        features_df = prepare_twostage_features(features_df, config)

    # Identify feature columns
    feature_cols = get_feature_columns(features_df)
    print(f"  Using {len(feature_cols)} features")

    # Generate CV folds
    cv_strategy = args.cv_strategy
    cv_split = generate_cv_folds(features_df, strategy=cv_strategy)
    print(f"  Generated {len(cv_split)} CV folds ({cv_split.strategy})")

    # Filter folds if specified
    if args.folds is not None:
        fold_ids = [int(f.strip()) for f in args.folds.split(",")]
        folds = [fold for fold in cv_split if fold.fold_id in fold_ids]
        if not folds:
            print(f"ERROR: No folds matched IDs {fold_ids}.")
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

    # Build LightGBM config from YAML
    lgbm_params = config.get("lgbm", {})
    lgbm_config = LGBMConfig(
        objective=lgbm_params.get("objective", "regression"),
        metric=lgbm_params.get("metric", "rmse"),
        num_leaves=lgbm_params.get("num_leaves", 31),
        learning_rate=lgbm_params.get("learning_rate", 0.05),
        n_estimators=lgbm_params.get("n_estimators", 500),
        max_depth=lgbm_params.get("max_depth", -1),
        min_child_samples=lgbm_params.get("min_child_samples", 20),
        subsample=lgbm_params.get("subsample", 0.8),
        colsample_bytree=lgbm_params.get("colsample_bytree", 0.8),
        reg_alpha=lgbm_params.get("reg_alpha", 0.1),
        reg_lambda=lgbm_params.get("reg_lambda", 0.1),
        random_state=lgbm_params.get("random_state", seed),
        n_jobs=lgbm_params.get("n_jobs", -1),
        verbose=lgbm_params.get("verbose", -1),
        early_stopping_rounds=lgbm_params.get("early_stopping_rounds", 50),
    )

    # Dry run
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN — Summary")
        print(f"{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(features_df)}")
        print(f"  Folds: {len(folds)}")
        print(f"  LightGBM params:")
        for k, v in lgbm_config.to_lgb_params().items():
            print(f"    {k}: {v}")
        print(f"\n{'=' * 60}")
        print("Dry run complete. No training performed.")
        print(f"{'=' * 60}")
        return

    # Set up experiment tracking
    tracking_cfg = config.get("tracking", {})
    experiment_name = tracking_cfg.get("experiment_name", "bearing_rul_twostage")
    tracking_uri = tracking_cfg.get("tracking_uri", "mlruns")

    tracker = ExperimentTracker(
        backend="mlflow",
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
    print(f"\n  MLflow tracking: experiment={experiment_name}")

    # Output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    all_results: list[dict] = []
    all_importance_dfs: list[pd.DataFrame] = []

    for fold_idx, fold in enumerate(folds):
        fold_id = fold.fold_id
        print(f"\n{'─' * 60}")
        print(f"  [{fold_idx + 1}/{len(folds)}] Training {model_name} — fold {fold_id}")
        print(f"  Val bearing(s): {', '.join(fold.val_bearings)}")
        print(f"  Train: {len(fold.train_indices)} samples, Val: {len(fold.val_indices)} samples")
        print(f"{'─' * 60}")

        # Split data
        train_df = features_df.iloc[fold.train_indices]
        val_df = features_df.iloc[fold.val_indices]

        X_train = train_df[feature_cols].values
        y_train = train_df["rul"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["rul"].values

        # Train model
        model = LightGBMBaseline(lgbm_config)
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

        # Predict
        y_pred = model.predict(X_val)

        # Evaluate
        metrics = evaluate_predictions(y_val, y_pred)

        # Feature importance
        importance_df = model.get_feature_importance("gain")
        importance_df["fold_id"] = fold_id
        all_importance_dfs.append(importance_df)

        # Log to MLflow
        run_name = f"{model_name}_{model_version}_fold{fold_id}"
        training_params = {
            "model_name": model_name,
            "model_version": model_version,
            "fold_id": fold_id,
            "val_bearings": ", ".join(fold.val_bearings),
            "train_samples": len(fold.train_indices),
            "val_samples": len(fold.val_indices),
            "eval_mode": eval_mode,
            "n_features": len(feature_cols),
            "n_estimators": lgbm_config.n_estimators,
            "learning_rate": lgbm_config.learning_rate,
            "num_leaves": lgbm_config.num_leaves,
        }
        final_metrics = {
            "final_rmse": metrics["rmse"],
            "final_mae": metrics["mae"],
            "final_mape": metrics["mape"],
            "final_phm08_score": metrics["phm08_score"],
            "final_phm08_score_normalized": metrics["phm08_score_normalized"],
        }
        if model.best_iteration is not None:
            final_metrics["best_iteration"] = float(model.best_iteration)

        with tracker.start_run(run_name=run_name) as run:
            run.log_params(training_params)
            run.log_metrics(final_metrics)

        print(f"\n  Fold {fold_id} results:")
        print(f"    RMSE:  {metrics['rmse']:.4f}")
        print(f"    MAE:   {metrics['mae']:.4f}")
        print(f"    PHM08: {metrics['phm08_score']:.4f}")
        if model.best_iteration is not None:
            print(f"    Best iteration: {model.best_iteration}")

        # Save per-fold predictions
        pred_df = pd.DataFrame({
            "bearing_id": val_df["bearing_id"].values,
            "condition": val_df["condition"].values,
            "file_idx": val_df["file_idx"].values,
            "y_true": y_val,
            "y_pred": y_pred,
        })
        pred_csv = predictions_dir / f"{model_name}_fold{fold_id}_predictions.csv"
        pred_df.to_csv(pred_csv, index=False)
        print(f"  Saved predictions -> {pred_csv}")

        all_results.append({
            "model_name": model_name,
            "fold_id": fold_id,
            "val_bearings": ", ".join(fold.val_bearings),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "phm08_score": metrics["phm08_score"],
            "phm08_score_normalized": metrics["phm08_score_normalized"],
        })

    # Save fold results CSV
    results_df = pd.DataFrame(all_results)
    results_csv = output_dir / f"{model_name}_fold_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n  Saved fold results -> {results_csv}")

    # Append to combined dl_model_results.csv
    combined_csv = output_dir / "dl_model_results.csv"
    write_header = not combined_csv.exists()
    results_df.to_csv(combined_csv, mode="a", header=write_header, index=False)
    print(f"  Appended {len(results_df)} fold results -> {combined_csv}")

    # Save aggregated feature importance
    if all_importance_dfs:
        all_importance = pd.concat(all_importance_dfs, ignore_index=True)
        agg_importance = (
            all_importance.groupby("feature")["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean", ascending=False)
            .reset_index(drop=True)
        )
        agg_importance.columns = ["feature", "importance_mean", "importance_std"]
        importance_csv = output_dir / f"{model_name}_feature_importance.csv"
        agg_importance.to_csv(importance_csv, index=False)
        print(f"  Saved feature importance -> {importance_csv}")

        print(f"\n  Top 10 features:")
        for _, row in agg_importance.head(10).iterrows():
            print(f"    {row['feature']:<30s}  {row['importance_mean']:.1f} +/- {row['importance_std']:.1f}")

    # Print aggregate metrics
    rmses = [r["rmse"] for r in all_results]
    maes = [r["mae"] for r in all_results]
    print(f"\n{'─' * 60}")
    print(f"  Aggregate for {model_name} ({len(all_results)} folds):")
    print(f"    RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"    MAE:  {np.mean(maes):.4f} +/- {np.std(maes):.4f}")
    print(f"{'─' * 60}")

    print(f"\n{'=' * 60}")
    print("Training complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
