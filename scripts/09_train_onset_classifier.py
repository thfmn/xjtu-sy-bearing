#!/usr/bin/env python3
"""Train onset classifier with leave-one-bearing-out cross-validation.

Trains the LSTM-based onset classifier (Stage 1 of two-stage RUL pipeline)
using leave-one-bearing-out CV. Each fold holds out one bearing for validation
and trains on the remaining 14.

Usage:
    python scripts/09_train_onset_classifier.py
    python scripts/09_train_onset_classifier.py --folds 0,1,2
    python scripts/09_train_onset_classifier.py --epochs 30 --dry-run
    python scripts/09_train_onset_classifier.py --tracking none
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.onset.dataset import (
    build_onset_tf_dataset,
    compute_class_weights,
    create_onset_dataset,
    split_by_bearing,
)
from src.onset.labels import load_onset_labels
from src.onset.models import build_onset_classifier, compile_onset_classifier

DEFAULT_FEATURES_CSV = "outputs/features/features_v2.csv"
DEFAULT_OUTPUT_DIR = "outputs/models"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WINDOW_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train onset classifier with leave-one-bearing-out CV and MLflow tracking.",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=DEFAULT_FEATURES_CSV,
        help=f"Path to features CSV file. Default: {DEFAULT_FEATURES_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saving models and results. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs per fold. Default: {DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate for Adam optimizer. Default: {DEFAULT_LEARNING_RATE}",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Sliding window size. Default: {DEFAULT_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated fold indices to train (e.g. 0,1,2). Default: all 15 folds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model and print summary without training.",
    )
    parser.add_argument(
        "--tracking",
        type=str,
        choices=["mlflow", "none"],
        default="mlflow",
        help="Experiment tracking backend. Default: mlflow.",
    )
    return parser.parse_args()


def compute_fold_metrics(
    model,
    val_ds,
    val_labels: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics on validation set.

    Returns dict with: accuracy, precision, recall, f1, auc_roc.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Collect predictions
    y_prob_list: list[np.ndarray] = []
    for x_batch, _ in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_prob_list.append(preds.ravel())
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(val_labels, y_pred)),
        "precision": float(precision_score(val_labels, y_pred, zero_division=0.0)),
        "recall": float(recall_score(val_labels, y_pred, zero_division=0.0)),
        "f1": float(f1_score(val_labels, y_pred, zero_division=0.0)),
    }

    # AUC-ROC requires both classes present in validation set
    if len(np.unique(val_labels)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(val_labels, y_prob))
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def train_single_fold(
    fold_idx: int,
    val_bearing_id: str,
    dataset_result,
    args: argparse.Namespace,
    tracker=None,
) -> dict:
    """Train onset classifier for one CV fold.

    Returns dict with fold_idx, val_bearing, metrics, and training time.
    """
    from tensorflow import keras

    print(f"\n{'─' * 60}")
    print(f"  Fold {fold_idx}: val_bearing = {val_bearing_id}")
    print(f"{'─' * 60}")

    # Split dataset
    split = split_by_bearing(dataset_result, val_bearing_ids=[val_bearing_id])

    train_labels = split.train.labels
    val_labels = split.val.labels

    n_train = len(train_labels)
    n_val = len(val_labels)
    print(f"  Train: {n_train} windows ({np.sum(train_labels == 0)} healthy, {np.sum(train_labels == 1)} degraded)")
    print(f"  Val:   {n_val} windows ({np.sum(val_labels == 0)} healthy, {np.sum(val_labels == 1)} degraded)")

    if n_val == 0:
        print(f"  SKIP: No validation samples for {val_bearing_id}")
        return {
            "fold_idx": fold_idx,
            "val_bearing": val_bearing_id,
            "metrics": None,
            "train_time_s": 0.0,
            "skipped": True,
        }

    # Build tf.data pipelines
    train_ds = build_onset_tf_dataset(split.train, batch_size=args.batch_size, shuffle=True)
    val_ds = build_onset_tf_dataset(split.val, batch_size=args.batch_size, shuffle=False)

    # Build fresh model each fold
    model = build_onset_classifier()
    compile_onset_classifier(model, learning_rate=args.learning_rate)

    # Class weights for imbalanced training data
    class_weights = compute_class_weights(train_labels)

    # Callbacks: early stopping on val loss
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    # Train
    t0 = time.monotonic()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.monotonic() - t0

    # Compute metrics
    metrics = compute_fold_metrics(model, val_ds, val_labels)
    n_epochs_run = len(history.history["loss"])

    print(f"  Epochs: {n_epochs_run}/{args.epochs} (early stopping)")
    print(f"  Time:   {train_time:.1f}s")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    # MLflow logging
    if tracker is not None:
        run_name = f"onset_fold{fold_idx}_{val_bearing_id}"
        with tracker.start_run(run_name=run_name) as run:
            run.log_params({
                "fold_idx": fold_idx,
                "val_bearing": val_bearing_id,
                "epochs": args.epochs,
                "epochs_run": n_epochs_run,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "window_size": args.window_size,
                "train_samples": n_train,
                "val_samples": n_val,
                "class_weight_0": class_weights[0],
                "class_weight_1": class_weights[1],
            })
            run.log_metrics({
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc_roc": metrics["auc_roc"],
                "train_time_s": train_time,
            })

    return {
        "fold_idx": fold_idx,
        "val_bearing": val_bearing_id,
        "metrics": metrics,
        "train_time_s": train_time,
        "skipped": False,
        "model": model,
        "n_epochs_run": n_epochs_run,
    }


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Onset Classifier Training (LOBO CV)")
    print("=" * 60)
    print(f"  Features CSV:  {args.features_csv}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Window size:   {args.window_size}")
    print(f"  Folds:         {args.folds or 'all'}")
    print(f"  Dry run:       {args.dry_run}")
    print(f"  Tracking:      {args.tracking}")
    print("=" * 60)

    # Load data
    features_csv = Path(args.features_csv)
    if not features_csv.exists():
        print(f"ERROR: Features CSV not found: {features_csv}")
        sys.exit(1)

    print(f"\nLoading features from {features_csv} ...")
    features_df = pd.read_csv(features_csv)
    print(f"  Loaded {len(features_df)} rows")

    print("Loading onset labels ...")
    onset_labels = load_onset_labels()
    print(f"  Loaded {len(onset_labels)} bearing labels")

    # Create windowed dataset from all bearings
    print(f"\nCreating onset dataset (window_size={args.window_size}) ...")
    dataset_result = create_onset_dataset(
        features_df, onset_labels, window_size=args.window_size
    )
    n_total = len(dataset_result.labels)
    n_healthy = int(np.sum(dataset_result.labels == 0))
    n_degraded = int(np.sum(dataset_result.labels == 1))
    print(f"  Total windows: {n_total} ({n_healthy} healthy, {n_degraded} degraded)")

    # Determine bearing list and fold indices
    all_bearing_ids = sorted(set(dataset_result.bearing_ids))
    print(f"  Bearings: {len(all_bearing_ids)}")

    if args.folds is not None:
        fold_indices = [int(f.strip()) for f in args.folds.split(",")]
        if any(i < 0 or i >= len(all_bearing_ids) for i in fold_indices):
            print(f"ERROR: Fold indices must be 0-{len(all_bearing_ids) - 1}. Got: {fold_indices}")
            sys.exit(1)
    else:
        fold_indices = list(range(len(all_bearing_ids)))

    print(f"\n  Folds to train: {len(fold_indices)}")
    print(f"  {'Fold':>5s}  {'Val Bearing':<15s}")
    print(f"  {'─' * 5}  {'─' * 15}")
    for i in fold_indices:
        print(f"  {i:>5d}  {all_bearing_ids[i]:<15s}")

    # Dry run
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN — Building model and printing summary")
        print(f"{'=' * 60}")
        model = build_onset_classifier()
        model.summary()
        print(f"\n  Input shape:  {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total params: {model.count_params():,}")
        print(f"\n{'=' * 60}")
        print("Dry run complete. No training performed.")
        print(f"{'=' * 60}")
        return

    # Set up MLflow tracking
    tracker = None
    if args.tracking == "mlflow":
        from src.utils.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            backend="mlflow",
            experiment_name="onset-classifier",
            tracking_uri="mlruns",
        )
        print("\n  MLflow tracking: enabled (experiment=onset-classifier)")
    else:
        print("\n  MLflow tracking: disabled")

    # Train all folds
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    best_f1 = -1.0
    best_model = None
    total_start = time.monotonic()

    for fold_idx in fold_indices:
        val_bearing_id = all_bearing_ids[fold_idx]
        result = train_single_fold(
            fold_idx=fold_idx,
            val_bearing_id=val_bearing_id,
            dataset_result=dataset_result,
            args=args,
            tracker=tracker,
        )
        all_results.append(result)

        # Track best model by F1
        if not result["skipped"] and result["metrics"]["f1"] > best_f1:
            best_f1 = result["metrics"]["f1"]
            best_model = result["model"]

    total_time = time.monotonic() - total_start

    # Aggregate results
    completed = [r for r in all_results if not r["skipped"]]

    if not completed:
        print("\nERROR: No folds completed successfully.")
        sys.exit(1)

    f1_scores = [r["metrics"]["f1"] for r in completed]
    auc_scores = [r["metrics"]["auc_roc"] for r in completed if not np.isnan(r["metrics"]["auc_roc"])]
    acc_scores = [r["metrics"]["accuracy"] for r in completed]
    prec_scores = [r["metrics"]["precision"] for r in completed]
    rec_scores = [r["metrics"]["recall"] for r in completed]

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE RESULTS ({len(completed)} folds)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {np.mean(acc_scores):.4f} +/- {np.std(acc_scores):.4f}")
    print(f"  Precision: {np.mean(prec_scores):.4f} +/- {np.std(prec_scores):.4f}")
    print(f"  Recall:    {np.mean(rec_scores):.4f} +/- {np.std(rec_scores):.4f}")
    print(f"  F1:        {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    if auc_scores:
        print(f"  AUC-ROC:   {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 60}")

    # Save best model
    best_model_path = output_dir / "onset_classifier.keras"
    if best_model is not None:
        best_model.save(best_model_path)
        print(f"\n  Best model saved (F1={best_f1:.4f}) -> {best_model_path}")

    # Save per-fold results CSV
    rows = []
    for r in completed:
        rows.append({
            "fold_idx": r["fold_idx"],
            "val_bearing": r["val_bearing"],
            "accuracy": r["metrics"]["accuracy"],
            "precision": r["metrics"]["precision"],
            "recall": r["metrics"]["recall"],
            "f1": r["metrics"]["f1"],
            "auc_roc": r["metrics"]["auc_roc"],
            "train_time_s": r["train_time_s"],
            "epochs_run": r["n_epochs_run"],
        })
    results_df = pd.DataFrame(rows)
    results_csv = output_dir / "onset_classifier_cv_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"  CV results saved -> {results_csv}")

    # Log aggregate to MLflow
    if tracker is not None:
        with tracker.start_run(run_name="onset_aggregate") as run:
            run.log_params({
                "n_folds": len(completed),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "window_size": args.window_size,
            })
            run.log_metrics({
                "mean_accuracy": float(np.mean(acc_scores)),
                "mean_precision": float(np.mean(prec_scores)),
                "mean_recall": float(np.mean(rec_scores)),
                "mean_f1": float(np.mean(f1_scores)),
                "std_f1": float(np.std(f1_scores)),
                "mean_auc_roc": float(np.mean(auc_scores)) if auc_scores else float("nan"),
                "total_time_s": total_time,
            })
            if best_model_path.exists():
                run.log_artifact(str(best_model_path), artifact_path="model")
            run.log_artifact(str(results_csv), artifact_path="results")
        print("  Aggregate metrics logged to MLflow")

    print(f"\n{'=' * 60}")
    print("Training complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
