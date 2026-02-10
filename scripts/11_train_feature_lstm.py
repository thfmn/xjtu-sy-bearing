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
"""Feature Sequence LSTM Two-Stage RUL Training Script.

Trains a BiLSTM model on sliding windows of 65 extracted features with
leave-one-bearing-out cross-validation and MLflow experiment tracking.

The key difference from DL script 05 is that this model needs sliding windows
across consecutive files with per-bearing normalization, which is a
fundamentally different data flow from individual file loading.

Usage:
    python scripts/11_train_feature_lstm.py --config configs/twostage_feature_lstm.yaml
    python scripts/11_train_feature_lstm.py --config configs/twostage_feature_lstm.yaml --folds 0,1,2
    python scripts/11_train_feature_lstm.py --config configs/twostage_feature_lstm.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_windows import (
    build_feature_window_tf_dataset,
    create_rul_feature_windows,
    split_feature_windows_by_bearing,
)
from src.models.baselines.feature_lstm import (
    FeatureLSTMConfig,
    build_feature_lstm_model,
)
from src.models.baselines.lgbm_baseline import get_feature_columns
from src.onset.labels import load_onset_labels
from src.training.cv import generate_cv_folds
from src.training.metrics import evaluate_predictions
from src.utils.tracking import ExperimentTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Feature LSTM RUL model with two-stage onset filtering and LOBO CV.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/twostage_feature_lstm.yaml).",
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
        help="Build model and print summary without training.",
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="leave_one_bearing_out",
        choices=["leave_one_bearing_out", "loco_per_bearing", "jin_fixed", "sun_fixed"],
        help="Cross-validation strategy. Default: leave_one_bearing_out (15-fold LOBO). "
        "loco_per_bearing: LOCO-LOBO hybrid (15 folds, cross-condition training). "
        "jin_fixed: Jin et al. 2025 protocol (2 train, 13 test). "
        "sun_fixed: Sun et al. 2024 protocol (4 train, 6 test, Conds 1-2 only).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class _EpochMetricsCallback(keras.callbacks.Callback):
    """Logs per-epoch metrics to an active ExperimentTracker run."""

    def __init__(self, tracker: ExperimentTracker):
        super().__init__()
        self._tracker = tracker

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs:
            self._tracker.log_metrics(
                {k: float(v) for k, v in logs.items()},
                step=epoch,
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = config.get("model_name", "feature_lstm")
    model_version = config.get("model_version", "v1")
    seed = config.get("seed", 9)

    # Feature LSTM architecture config
    lstm_cfg = config.get("feature_lstm", {})
    lstm_config = FeatureLSTMConfig(
        window_size=lstm_cfg.get("window_size", 10),
        n_features=lstm_cfg.get("n_features", 65),
        lstm_units=lstm_cfg.get("lstm_units", 16),
        bidirectional=lstm_cfg.get("bidirectional", True),
        dropout_rate=lstm_cfg.get("dropout_rate", 0.2),
        dense_units=lstm_cfg.get("dense_units", 16),
    )

    # Training config
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 100)

    # Optimizer config
    opt_cfg = config.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adamw")
    learning_rate = opt_cfg.get("learning_rate", 0.001)
    weight_decay = opt_cfg.get("weight_decay", 0.0001)

    # Loss config
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "huber")
    loss_delta = loss_cfg.get("delta", 10.0)

    # Callback config
    cb_cfg = config.get("callbacks", {})
    checkpoint_dir = Path(cb_cfg.get("checkpoint_dir", "outputs/models/checkpoints"))
    early_stop_patience = cb_cfg.get("early_stop_patience", 7)
    lr_reduce_patience = cb_cfg.get("lr_reduce_patience", 5)
    lr_reduce_factor = cb_cfg.get("lr_reduce_factor", 0.5)
    lr_min = cb_cfg.get("lr_min", 1e-6)

    # Onset/training config
    onset_cfg = config.get("onset", {})
    training_cfg = config.get("training", {})
    max_rul = training_cfg.get("max_rul", 125)
    filter_pre_onset = training_cfg.get("filter_pre_onset", True)
    eval_mode = training_cfg.get("eval_mode", "post_onset")

    print("=" * 60)
    print(f"Feature LSTM RUL Training ({eval_mode})")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Model:        {model_name} {model_version}")
    print(f"  Architecture: BiLSTM({lstm_config.lstm_units}) -> Dense({lstm_config.dense_units})")
    print(f"  Window size:  {lstm_config.window_size}")
    print(f"  N features:   {lstm_config.n_features}")
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

    # Get feature columns
    feature_cols = get_feature_columns(features_df)
    n_features = len(feature_cols)
    print(f"  Using {n_features} features")

    # Update config if feature count differs
    if n_features != lstm_config.n_features:
        print(f"  NOTE: Updating n_features from {lstm_config.n_features} to {n_features}")
        lstm_config.n_features = n_features

    # Load onset labels (not needed in full_life mode)
    if eval_mode == "full_life":
        print("\n  Full-life mode: skipping onset labels, normalized [0, 1] RUL")
        onset_labels = None
    else:
        labels_path = onset_cfg.get("labels_path", "configs/onset_labels.yaml")
        print(f"\n  Loading onset labels from {labels_path} ...")
        onset_labels = load_onset_labels(labels_path)
        print(f"  Loaded onset labels for {len(onset_labels)} bearings")

    # Create all feature windows
    print(f"\n  Creating feature windows (window_size={lstm_config.window_size}, eval_mode={eval_mode}) ...")
    all_windows = create_rul_feature_windows(
        features_df=features_df,
        feature_cols=feature_cols,
        onset_labels=onset_labels,
        window_size=lstm_config.window_size,
        max_rul=max_rul,
        filter_pre_onset=filter_pre_onset if eval_mode != "full_life" else False,
        eval_mode=eval_mode,
    )
    print(f"  Created {len(all_windows.rul_labels)} windows")
    print(f"  Window shape: {all_windows.windows.shape}")

    unique_bearings = sorted(set(all_windows.bearing_ids))
    print(f"  Bearings with windows: {len(unique_bearings)}")

    # Generate CV folds. Bearing IDs from folds are used to split windows.
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

    # Dry run
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN — Building model and printing summary")
        print(f"{'=' * 60}")
        model = build_feature_lstm_model(lstm_config)
        model.summary()
        print(f"\n  Input shape:  {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total params: {model.count_params():,}")
        print(f"  Windows:      {len(all_windows.rul_labels)}")
        print(f"\n{'=' * 60}")
        print("Dry run complete. No training performed.")
        print(f"{'=' * 60}")
        return

    # Set up experiment tracking
    experiment_name = cb_cfg.get("mlflow_experiment_name", "bearing_rul_twostage")
    tracking_uri = cb_cfg.get("mlflow_tracking_uri", "mlruns")

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
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    all_results: list[dict] = []

    for fold_idx, fold in enumerate(folds):
        fold_id = fold.fold_id
        val_bearings = fold.val_bearings

        # Check if val bearings have any windows
        val_in_windows = [b for b in val_bearings if b in set(all_windows.bearing_ids)]
        if not val_in_windows:
            print(f"\n  Fold {fold_id}: Val bearing(s) {val_bearings} have no windows, skipping")
            continue

        print(f"\n{'─' * 60}")
        print(f"  [{fold_idx + 1}/{len(folds)}] Training {model_name} — fold {fold_id}")
        print(f"  Val bearing(s): {', '.join(val_bearings)}")

        # Split windows by bearing
        split = split_feature_windows_by_bearing(all_windows, val_bearings)
        n_train = len(split.train.rul_labels)
        n_val = len(split.val.rul_labels)
        print(f"  Train: {n_train} windows, Val: {n_val} windows")
        print(f"{'─' * 60}")

        if n_train == 0 or n_val == 0:
            print(f"  Skipping fold {fold_id}: insufficient data")
            continue

        # Build tf.data datasets
        train_ds = build_feature_window_tf_dataset(
            split.train, batch_size=batch_size, shuffle=True,
        )
        val_ds = build_feature_window_tf_dataset(
            split.val, batch_size=batch_size, shuffle=False,
        )

        # Build fresh model
        model = build_feature_lstm_model(lstm_config)

        # Compile
        if opt_name == "adamw":
            optimizer = keras.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay,
            )
        elif opt_name == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

        if loss_name == "huber":
            loss_fn = keras.losses.Huber(delta=loss_delta)
        elif loss_name == "mse":
            loss_fn = keras.losses.MeanSquaredError()
        else:
            loss_fn = keras.losses.MeanAbsoluteError()

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae"])

        # Callbacks
        checkpoint_path = str(checkpoint_dir / f"{model_name}_fold{fold_id}.keras")
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=0,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_reduce_factor,
                patience=lr_reduce_patience,
                min_lr=lr_min,
                verbose=1,
            ),
        ]

        # MLflow run
        run_name = f"{model_name}_{model_version}_fold{fold_id}"
        training_params = {
            "model_name": model_name,
            "model_version": model_version,
            "fold_id": fold_id,
            "val_bearings": ", ".join(val_bearings),
            "train_windows": n_train,
            "val_windows": n_val,
            "window_size": lstm_config.window_size,
            "n_features": lstm_config.n_features,
            "lstm_units": lstm_config.lstm_units,
            "bidirectional": lstm_config.bidirectional,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "optimizer": opt_name,
            "loss": loss_name,
            "eval_mode": eval_mode,
            "two_stage": eval_mode != "full_life",
        }

        with tracker.start_run(run_name=run_name) as run:
            run.log_params(training_params)

            # Add epoch metrics callback
            cbs = callbacks + [_EpochMetricsCallback(tracker)]

            # Train
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=cbs,
                verbose=2,
            )

            # Predict on val set
            y_true_list = []
            y_pred_list = []
            for x_batch, y_batch in val_ds:
                preds = model.predict(x_batch, verbose=0)
                y_true_list.append(y_batch.numpy())
                y_pred_list.append(preds.ravel())

            y_true = np.concatenate(y_true_list)
            y_pred = np.concatenate(y_pred_list)
            # Clip predictions: [0, 1] for full-life normalized, [0, inf) otherwise
            y_pred_max = 1.0 if eval_mode == "full_life" else None
            y_pred = np.clip(y_pred, 0, y_pred_max)

            # Evaluate
            metrics = evaluate_predictions(y_true, y_pred)

            final_metrics = {
                "final_rmse": metrics["rmse"],
                "final_mae": metrics["mae"],
                "final_mape": metrics["mape"],
                "final_phm08_score": metrics["phm08_score"],
                "final_phm08_score_normalized": metrics["phm08_score_normalized"],
            }
            run.log_metrics(final_metrics)

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df["epoch"] = range(len(history_df))
        history_csv = history_dir / f"{model_name}_fold{fold_id}_history.csv"
        history_df.to_csv(history_csv, index=False)

        print(f"\n  Fold {fold_id} results:")
        print(f"    RMSE:  {metrics['rmse']:.4f}")
        print(f"    MAE:   {metrics['mae']:.4f}")
        print(f"    PHM08: {metrics['phm08_score']:.4f}")

        # Save per-fold predictions
        # For windowed models, bearing_id comes from the window result
        val_bearing_arr = np.array(split.val.bearing_ids)
        pred_df = pd.DataFrame({
            "bearing_id": val_bearing_arr,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        pred_csv = predictions_dir / f"{model_name}_fold{fold_id}_predictions.csv"
        pred_df.to_csv(pred_csv, index=False)
        print(f"  Saved predictions -> {pred_csv}")

        all_results.append({
            "model_name": model_name,
            "fold_id": fold_id,
            "val_bearings": ", ".join(val_bearings),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "phm08_score": metrics["phm08_score"],
            "phm08_score_normalized": metrics["phm08_score_normalized"],
        })

        # Save intermediate results
        results_df = pd.DataFrame(all_results)
        results_csv = output_dir / f"{model_name}_fold_results.csv"
        results_df.to_csv(results_csv, index=False)

    # Final results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = output_dir / f"{model_name}_fold_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\n  Saved fold results -> {results_csv}")

        # Append to combined results
        combined_csv = output_dir / "dl_model_results.csv"
        write_header = not combined_csv.exists()
        results_df.to_csv(combined_csv, mode="a", header=write_header, index=False)
        print(f"  Appended {len(results_df)} fold results -> {combined_csv}")

        # Aggregate metrics
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
