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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import build_dataset_for_model
from src.models.registry import build_model, get_model_info, list_models
from src.training.config import MLflowCallback, TrainingConfig, build_callbacks, compile_model
from src.training.cv import CVFold, leave_one_bearing_out
from src.training.metrics import evaluate_predictions
from src.utils.tracking import ExperimentTracker

CONFIGS_DIR = Path("configs")
DEFAULT_CONFIG = CONFIGS_DIR / "training_default.yaml"


def resolve_model_names(model_arg: str) -> list[str]:
    """Resolve --model argument to a list of validated model names.

    Accepts a single name, comma-separated names, or 'all'.
    Raises SystemExit if any name is not in the registry.
    """
    registered = list_models()
    if model_arg == "all":
        if not registered:
            print("ERROR: No models registered in the registry.")
            sys.exit(1)
        return registered

    names = [n.strip() for n in model_arg.split(",")]
    for name in names:
        if name not in registered:
            available = ", ".join(registered) or "(none)"
            print(f"ERROR: Model '{name}' is not registered. Available: {available}")
            sys.exit(1)
    return names


def resolve_config_path(model_name: str, explicit_config: str | None) -> Path:
    """Resolve which config YAML to use for a model.

    Priority: explicit --config > configs/{model_name}.yaml > configs/training_default.yaml.
    Raises SystemExit if the resolved path does not exist.
    """
    if explicit_config is not None:
        path = Path(explicit_config)
        if not path.exists():
            print(f"ERROR: Config file not found: {path}")
            sys.exit(1)
        return path

    model_config = CONFIGS_DIR / f"{model_name}.yaml"
    if model_config.exists():
        return model_config

    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG

    print(f"ERROR: No config found for '{model_name}'. Tried: {model_config}, {DEFAULT_CONFIG}")
    sys.exit(1)


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


def train_single_fold(
    model_name: str,
    fold: CVFold,
    metadata_df: pd.DataFrame,
    training_config: TrainingConfig,
    data_root: Path,
    spectrogram_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
) -> dict:
    """Train a model on one CV fold. Returns dict with metrics and predictions.

    Builds a fresh model each fold to avoid weight leakage, compiles it,
    constructs train/val tf.data.Datasets, trains with callbacks, then
    evaluates on the validation set. Each fold is tracked as an MLflow run
    via ExperimentTracker.

    Args:
        model_name: Registered model name (e.g. "cnn1d_baseline").
        fold: CVFold with train_indices, val_indices, fold_id.
        metadata_df: Full metadata DataFrame (features_v2.csv).
        training_config: TrainingConfig loaded from YAML.
        data_root: Root directory of raw bearing CSV data.
        spectrogram_dir: Directory containing .npy spectrogram files.
        output_dir: Directory for saving checkpoints and results.
        config_path: Path to the YAML config file (for artifact logging).

    Returns:
        Dict with keys: fold_id, model_name, metrics, y_true, y_pred, history.
    """
    fold_id = fold.fold_id
    print(f"\n{'─' * 60}")
    print(f"  Training {model_name} — fold {fold_id}")
    print(f"  Val bearing(s): {', '.join(fold.val_bearings)}")
    print(f"  Train: {len(fold.train_indices)} samples, Val: {len(fold.val_indices)} samples")
    print(f"{'─' * 60}")

    # 1. Build fresh model
    model = build_model(model_name)

    # 2. Compile
    compile_model(model, training_config)

    # 3. Build train/val datasets
    train_ds = build_dataset_for_model(
        model_name=model_name,
        metadata_df=metadata_df,
        indices=fold.train_indices,
        batch_size=training_config.batch_size,
        shuffle=True,
        data_root=str(data_root),
        spectrogram_dir=str(spectrogram_dir),
    )
    val_ds = build_dataset_for_model(
        model_name=model_name,
        metadata_df=metadata_df,
        indices=fold.val_indices,
        batch_size=training_config.batch_size,
        shuffle=False,
        data_root=str(data_root),
        spectrogram_dir=str(spectrogram_dir),
    )

    # 4. Build callbacks — checkpoint path includes model name and fold ID
    #    Remove MLflowCallback to avoid nested runs (we use ExperimentTracker instead)
    checkpoint_name = f"{model_name}_fold{fold_id}"
    callbacks = build_callbacks(training_config, model_name=checkpoint_name)
    callbacks = [cb for cb in callbacks if not isinstance(cb, MLflowCallback)]

    # 5. Set up ExperimentTracker for MLflow run context
    cb_config = training_config.callbacks
    tracker = ExperimentTracker(
        backend="mlflow",
        experiment_name=cb_config.mlflow_experiment_name,
        tracking_uri=cb_config.mlflow_tracking_uri,
    )

    run_name = f"{model_name}_fold{fold_id}"
    with tracker.start_run(run_name=run_name) as run:
        # Log training params
        run.log_params({
            "model_name": model_name,
            "fold_id": fold_id,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.optimizer.learning_rate,
            "epochs": training_config.epochs,
            "optimizer": training_config.optimizer.name,
            "loss": training_config.loss.name,
            "val_bearings": ", ".join(fold.val_bearings),
            "train_samples": len(fold.train_indices),
            "val_samples": len(fold.val_indices),
        })

        # 6. Train
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=training_config.epochs,
            callbacks=callbacks,
            verbose=training_config.verbose,
        )

        # 7. Collect predictions on val set (iterate all val batches)
        y_true_list = []
        y_pred_list = []
        for x_batch, y_batch in val_ds:
            preds = model.predict(x_batch, verbose=0)
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(preds.squeeze())

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)

        # 8. Compute metrics
        metrics = evaluate_predictions(y_true, y_pred)

        # Log final eval metrics to MLflow
        run.log_metrics({
            "final_rmse": metrics["rmse"],
            "final_mae": metrics["mae"],
            "final_mape": metrics["mape"],
            "final_phm08_score": metrics["phm08_score"],
            "final_phm08_score_normalized": metrics["phm08_score_normalized"],
        })

        # Log per-epoch metrics to MLflow
        for epoch_idx in range(len(history.history.get("loss", []))):
            epoch_metrics = {}
            for key in history.history:
                epoch_metrics[key] = history.history[key][epoch_idx]
            run.log_metrics(epoch_metrics, step=epoch_idx)

    # Save training history to CSV
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = range(len(history_df))
    history_csv = history_dir / f"{model_name}_fold{fold_id}_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"  Saved training history → {history_csv}")

    print(f"\n  Fold {fold_id} results:")
    print(f"    RMSE:  {metrics['rmse']:.4f}")
    print(f"    MAE:   {metrics['mae']:.4f}")
    print(f"    PHM08: {metrics['phm08_score']:.4f}")

    # 9. Save per-fold predictions to disk
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    val_meta = metadata_df.iloc[fold.val_indices].reset_index(drop=True)
    pred_df = pd.DataFrame(
        {
            "bearing_id": val_meta["bearing_id"].values,
            "condition": val_meta["condition"].values,
            "file_idx": val_meta["file_idx"].values,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    pred_csv = predictions_dir / f"{model_name}_fold{fold_id}_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"  Saved predictions → {pred_csv}")

    # 10. Return result dict
    return {
        "fold_id": fold_id,
        "model_name": model_name,
        "val_bearings": ", ".join(fold.val_bearings),
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "history": history.history,
    }


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

    # --- TRAIN-3: Resolve models and configs ---
    model_names = resolve_model_names(args.model)
    model_configs: dict[str, Path] = {}
    for name in model_names:
        model_configs[name] = resolve_config_path(name, args.config)

    print(f"Models to train: {len(model_names)}")
    print(f"  {'Model':<35s}  {'Input Type':<15s}  Config")
    print(f"  {'─'*35}  {'─'*15}  {'─'*40}")
    for name in model_names:
        info = get_model_info(name)
        print(f"  {name:<35s}  {info.input_type:<15s}  {model_configs[name]}")
    print()

    # --- TRAIN-4: Dry-run mode ---
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — Building models and printing summaries")
        print("=" * 60)
        for name in model_names:
            print(f"\n{'─' * 60}")
            print(f"Model: {name}")
            print(f"{'─' * 60}")
            model = build_model(name)
            model.summary()
            print(f"\n  Input shape:  {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            print(f"  Total params: {model.count_params():,}")
        print(f"\n{'=' * 60}")
        print("Dry run complete. No training performed.")
        print(f"{'=' * 60}")
        return

    # --- TRAIN-7: Main training loop over models and folds ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    spectrogram_dir = Path(args.spectrogram_dir)

    all_results: list[dict] = []

    for model_name in model_names:
        config_path = model_configs[model_name]
        training_config = TrainingConfig.from_yaml(config_path)
        print(f"\n{'=' * 60}")
        print(f"  MODEL: {model_name}")
        print(f"  Config: {config_path}")
        print(f"  Folds: {len(folds)}")
        print(f"{'=' * 60}")

        model_results: list[dict] = []

        for fold_idx, fold in enumerate(folds):
            print(f"\n  [{fold_idx + 1}/{len(folds)}] Training {model_name} fold {fold.fold_id}")

            result = train_single_fold(
                model_name=model_name,
                fold=fold,
                metadata_df=metadata_df,
                training_config=training_config,
                data_root=data_root,
                spectrogram_dir=spectrogram_dir,
                output_dir=output_dir,
                config_path=config_path,
            )
            model_results.append(result)
            all_results.append(result)

            # Save intermediate per-fold results CSV (crash recovery)
            _save_fold_results_csv(model_results, model_name, output_dir)

        # Print aggregate metrics for this model
        _print_aggregate_metrics(model_results, model_name)

        # Append this model's fold results to the combined dl_model_results.csv
        _append_dl_model_results(model_results, output_dir)

    # Save final combined results CSV across all models
    if all_results:
        _save_fold_results_csv(all_results, "all_models_combined", output_dir)

    print(f"\n{'=' * 60}")
    print("Training complete.")
    print(f"{'=' * 60}")


def _build_results_df(results: list[dict]) -> pd.DataFrame:
    """Convert a list of fold result dicts to a DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "model_name": r["model_name"],
                "fold_id": r["fold_id"],
                "val_bearings": r.get("val_bearings", ""),
                "rmse": r["metrics"]["rmse"],
                "mae": r["metrics"]["mae"],
                "mape": r["metrics"]["mape"],
                "phm08_score": r["metrics"]["phm08_score"],
                "phm08_score_normalized": r["metrics"]["phm08_score_normalized"],
            }
        )
    return pd.DataFrame(rows)


def _save_fold_results_csv(
    results: list[dict], name: str, output_dir: Path
) -> None:
    """Save a list of fold result dicts as a CSV file (overwrite mode)."""
    df = _build_results_df(results)
    csv_path = output_dir / f"{name}_fold_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved intermediate results → {csv_path}")


def _append_dl_model_results(results: list[dict], output_dir: Path) -> None:
    """Append fold results to dl_model_results.csv.

    Writes header only if the file does not already exist. This allows
    running different models separately and accumulating results in one file.
    """
    df = _build_results_df(results)
    csv_path = output_dir / "dl_model_results.csv"
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    print(f"  Appended {len(df)} fold results → {csv_path}")


def _print_aggregate_metrics(results: list[dict], model_name: str) -> None:
    """Print mean ± std of RMSE and MAE across folds for a model."""
    rmses = [r["metrics"]["rmse"] for r in results]
    maes = [r["metrics"]["mae"] for r in results]
    print(f"\n{'─' * 60}")
    print(f"  Aggregate for {model_name} ({len(results)} folds):")
    print(f"    RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"    MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
