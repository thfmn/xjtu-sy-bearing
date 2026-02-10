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

"""Unified Deep Learning Model Training Script for XJTU-SY Bearing Dataset.

Trains any registered DL model with leave-one-bearing-out cross-validation
and MLflow experiment tracking.

Supports two-stage mode where RUL model trains only on post-onset samples
with onset-relative RUL labels.

Usage:
    python scripts/05_train_dl_models.py --model cnn1d_baseline --config configs/cnn1d_baseline.yaml
    python scripts/05_train_dl_models.py --model all
    python scripts/05_train_dl_models.py --model cnn1d_baseline --folds 0,1,2
    python scripts/05_train_dl_models.py --model cnn1d_baseline --dry-run
    python scripts/05_train_dl_models.py --model cnn1d_baseline --config configs/twostage_pipeline.yaml --two-stage
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

from src.data.dataset_builders import build_dataset_for_model
from src.data.rul_labels import compute_twostage_rul, generate_rul_for_bearing
from src.models.registry import build_model, get_model_info, list_models
from src.onset.labels import load_onset_labels
from src.training.config import MLflowCallback, TrainingConfig, build_callbacks, compile_model
from src.training.cv import CVFold, generate_cv_folds
from src.training.metrics import evaluate_predictions
from src.utils.tracking import ExperimentTracker

CONFIGS_DIR = Path("configs")
DEFAULT_CONFIG = CONFIGS_DIR / "training_default.yaml"


class _EpochMetricsCallback(keras.callbacks.Callback):
    """Keras callback that logs per-epoch metrics to an active ExperimentTracker run.

    Unlike KerasTrackingCallback in src/utils/tracking.py, this callback does NOT
    manage run lifecycle — it assumes a run is already active on the tracker and
    simply logs epoch-end metrics with the correct step number. This avoids
    nested/duplicate runs when used inside an ExperimentTracker context manager.
    """

    def __init__(self, tracker: ExperimentTracker):
        super().__init__()
        self._tracker = tracker

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs:
            self._tracker.log_metrics(
                {k: float(v) for k, v in logs.items()},
                step=epoch,
            )


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
        "--cwt-dir",
        type=str,
        default="outputs/spectrograms/cwt",
        help="Directory containing .npy CWT scaleogram files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory for saving results, predictions, and history.",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default="outputs/features/features_v2.csv",
        help="Path to features CSV file. Use GCS fuse path for Vertex AI jobs.",
    )
    parser.add_argument(
        "--tracking",
        type=str,
        choices=["mlflow", "vertex", "both", "none"],
        default="mlflow",
        help="Experiment tracking backend(s). 'mlflow' (default), 'vertex', 'both', or 'none' (no tracking).",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Enable two-stage training mode: load onset labels, filter pre-onset "
        "samples (if config has training.filter_pre_onset=true), and use "
        "onset-relative RUL labels. Requires onset section in config YAML.",
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


def prepare_twostage_data(
    metadata_df: pd.DataFrame,
    training_config: TrainingConfig,
) -> pd.DataFrame:
    """Prepare metadata for two-stage training: recompute RUL and optionally filter pre-onset samples.

    Loads onset labels from the config, computes two-stage RUL labels
    (constant max_rul pre-onset, linear decay post-onset), and optionally
    removes pre-onset samples from the dataset.

    Args:
        metadata_df: Full metadata DataFrame (features_v2.csv) with at least
            columns: bearing_id, file_idx, rul.
        training_config: TrainingConfig with onset and training sections.

    Returns:
        Modified copy of metadata_df with:
        - 'rul' column replaced by two-stage RUL
        - 'rul_original' column preserving the original piecewise-linear RUL
        - Pre-onset rows removed if filter_pre_onset=true
        - 'is_post_onset' column (1 if post-onset, 0 otherwise)
    """
    onset_config = training_config.get_onset_config()
    if onset_config is None:
        raise ValueError(
            "Two-stage mode requires an 'onset' section in the config YAML. "
            "See configs/twostage_pipeline.yaml for an example."
        )

    twostage_config = training_config.get_twostage_training_config()
    max_rul = twostage_config.max_rul

    # Load onset labels
    labels_path = onset_config.labels_path
    print(f"  Loading onset labels from {labels_path} ...")
    onset_labels = load_onset_labels(labels_path)
    print(f"  Loaded onset labels for {len(onset_labels)} bearings")

    # Work on a copy
    df = metadata_df.copy()

    # Preserve original RUL
    df["rul_original"] = df["rul"].copy()

    # Compute two-stage RUL and add is_post_onset flag per bearing
    df["is_post_onset"] = 0
    bearings_processed = 0
    bearings_missing = []

    for bearing_id in df["bearing_id"].unique():
        if bearing_id not in onset_labels:
            bearings_missing.append(bearing_id)
            continue

        mask = df["bearing_id"] == bearing_id
        onset_idx = onset_labels[bearing_id].onset_file_idx
        num_files = mask.sum()

        # Compute two-stage RUL for this bearing
        twostage_rul = compute_twostage_rul(num_files, onset_idx, max_rul=max_rul)
        df.loc[mask, "rul"] = twostage_rul

        # Mark post-onset samples
        df.loc[mask, "is_post_onset"] = (df.loc[mask, "file_idx"] >= onset_idx).astype(int)
        bearings_processed += 1

    if bearings_missing:
        print(f"  WARNING: {len(bearings_missing)} bearings not in onset labels: {bearings_missing}")

    # Report onset split
    n_pre = (df["is_post_onset"] == 0).sum()
    n_post = (df["is_post_onset"] == 1).sum()
    print(f"  Onset split: {n_pre} pre-onset, {n_post} post-onset samples")

    # Filter pre-onset samples if configured
    if twostage_config.filter_pre_onset:
        original_count = len(df)
        df = df[df["is_post_onset"] == 1].reset_index(drop=True)
        print(f"  Filtered pre-onset: {original_count} -> {len(df)} samples ({original_count - len(df)} removed)")
    else:
        print("  filter_pre_onset=false: keeping all samples (with two-stage RUL labels)")

    return df


def train_single_fold(
    model_name: str,
    fold: CVFold,
    metadata_df: pd.DataFrame,
    training_config: TrainingConfig,
    data_root: Path,
    spectrogram_dir: Path,
    cwt_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    tracking_mode: str = "mlflow",
    two_stage: bool = False,
    eval_mode: str = "default",
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
        spectrogram_dir: Directory containing .npy STFT spectrogram files.
        cwt_dir: Directory containing .npy CWT scaleogram files.
        output_dir: Directory for saving checkpoints and results.
        config_path: Path to the YAML config file (for artifact logging).
        tracking_mode: Experiment tracking backend(s): "mlflow", "vertex", "both", or "none".
        two_stage: Whether two-stage mode is active (for logging).

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

    # 3. Build train/val datasets (augmentation on training only)
    train_ds = build_dataset_for_model(
        model_name=model_name,
        metadata_df=metadata_df,
        indices=fold.train_indices,
        batch_size=training_config.batch_size,
        shuffle=True,
        data_root=str(data_root),
        spectrogram_dir=str(spectrogram_dir),
        cwt_dir=str(cwt_dir),
        augment=True,
    )
    val_ds = build_dataset_for_model(
        model_name=model_name,
        metadata_df=metadata_df,
        indices=fold.val_indices,
        batch_size=training_config.batch_size,
        shuffle=False,
        data_root=str(data_root),
        spectrogram_dir=str(spectrogram_dir),
        cwt_dir=str(cwt_dir),
        augment=False,
    )

    # 4. Build callbacks — checkpoint path includes model name and fold ID
    #    Remove MLflowCallback to avoid nested runs (we use ExperimentTracker instead)
    checkpoint_name = f"{model_name}_fold{fold_id}"
    callbacks = build_callbacks(training_config, model_name=checkpoint_name)
    callbacks = [cb for cb in callbacks if not isinstance(cb, MLflowCallback)]

    # 5. Set up experiment tracking based on tracking_mode
    use_mlflow = tracking_mode in ("mlflow", "both")
    use_vertex = tracking_mode in ("vertex", "both")

    cb_config = training_config.callbacks
    tracker = None
    if use_mlflow:
        tracker = ExperimentTracker(
            backend="mlflow",
            experiment_name=cb_config.mlflow_experiment_name,
            tracking_uri=cb_config.mlflow_tracking_uri,
        )
        print(f"  MLflow tracking: enabled (experiment={cb_config.mlflow_experiment_name})")
    else:
        print("  MLflow tracking: disabled")

    # 5b. Set up optional Vertex AI dual-logging
    vertex_tracker = None
    if use_vertex:
        vertex_config = training_config.get_extra("vertex", {})
        try:
            vertex_tracker = ExperimentTracker(
                backend="vertex",
                experiment_name=vertex_config.get("experiment_name", "bearing-rul-dl"),
                project_id=vertex_config.get("project_id"),
                location=vertex_config.get("location", "asia-southeast3"),
            )
            print(f"  Vertex AI Experiments: logging to {vertex_config.get('experiment_name', 'bearing-rul-dl')}")
        except Exception as e:
            print(f"  Vertex AI Experiments: failed to initialize ({e}), continuing without Vertex")
            vertex_tracker = None
    else:
        print("  Vertex AI Experiments: disabled")

    model_version = training_config.get_extra("model_version", "v0")
    run_name = f"{model_name}_{model_version}_fold{fold_id}"
    training_params = {
        "model_name": model_name,
        "model_version": model_version,
        "fold_id": fold_id,
        "batch_size": training_config.batch_size,
        "learning_rate": training_config.optimizer.learning_rate,
        "weight_decay": training_config.optimizer.weight_decay,
        "epochs": training_config.epochs,
        "optimizer": training_config.optimizer.name,
        "loss": training_config.loss.name,
        "val_bearings": ", ".join(fold.val_bearings),
        "train_samples": len(fold.train_indices),
        "val_samples": len(fold.val_indices),
        "two_stage": two_stage,
        "eval_mode": eval_mode,
    }
    if two_stage:
        onset_config = training_config.get_onset_config()
        twostage_config = training_config.get_twostage_training_config()
        training_params.update({
            "onset_method": onset_config.method if onset_config else "none",
            "onset_labels_path": onset_config.labels_path if onset_config else "",
            "filter_pre_onset": twostage_config.filter_pre_onset,
            "max_rul": twostage_config.max_rul,
        })

    # --- Training and evaluation (with optional MLflow run context) ---
    def _run_training(run=None):
        """Execute training, prediction, and metric computation.

        If ``run`` is provided, logs params, metrics, and artifacts to it.
        Returns (metrics, final_metrics, y_true, y_pred, history).
        """
        if run is not None:
            run.log_params(training_params)
            # Set MLflow tags for UI filtering (tags are separate from params)
            if tracker is not None and tracker.backend_name == "mlflow":
                import mlflow
                mlflow.set_tag("model_version", model_version)
                mlflow.set_tag("model_name", model_name)
            if config_path is not None and config_path.exists():
                run.log_artifact(str(config_path), artifact_path="config")

        # 6. Train — with real-time per-epoch metric logging via callback
        cbs = list(callbacks)
        if tracker is not None:
            cbs.append(_EpochMetricsCallback(tracker))
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=training_config.epochs,
            callbacks=cbs,
            verbose=training_config.verbose,
        )

        # 7. Collect predictions on val set (iterate all val batches)
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

        # 8. Compute metrics
        metrics = evaluate_predictions(y_true, y_pred)

        final_metrics = {
            "final_rmse": metrics["rmse"],
            "final_mae": metrics["mae"],
            "final_mape": metrics["mape"],
            "final_phm08_score": metrics["phm08_score"],
            "final_phm08_score_normalized": metrics["phm08_score_normalized"],
        }

        if run is not None:
            run.log_metrics(final_metrics)
            best_checkpoint = (
                Path(training_config.callbacks.checkpoint_dir)
                / f"{model_name}_fold{fold_id}.keras"
            )
            if best_checkpoint.exists():
                run.log_artifact(str(best_checkpoint), artifact_path="model")
                print(f"  Logged model artifact → {best_checkpoint}")

                # Register in MLflow Model Registry for lineage tracking
                if tracker is not None and tracker.backend_name == "mlflow":
                    try:
                        import mlflow
                        registry_name = f"bearing-rul-{model_name}"
                        run_id = mlflow.active_run().info.run_id
                        model_uri = f"runs:/{run_id}/model"
                        rv = mlflow.register_model(model_uri, registry_name)
                        print(f"  Registered model → {registry_name} version {rv.version}")
                    except Exception as e:
                        print(f"  Model registry skipped ({e})")

            # Save onset config alongside RUL model in two-stage mode
            if two_stage:
                onset_cfg = training_config.get_onset_config()
                if onset_cfg is not None:
                    import json as _json
                    from dataclasses import asdict as _asdict
                    onset_cfg_path = (
                        Path(training_config.callbacks.checkpoint_dir)
                        / f"{model_name}_fold{fold_id}_onset_config.json"
                    )
                    with open(onset_cfg_path, "w") as _f:
                        _json.dump(_asdict(onset_cfg), _f, indent=2)
                    run.log_artifact(str(onset_cfg_path), artifact_path="onset")
                    print(f"  Saved onset config → {onset_cfg_path}")

        return metrics, final_metrics, y_true, y_pred, history

    if tracker is not None:
        with tracker.start_run(run_name=run_name) as run:
            metrics, final_metrics, y_true, y_pred, history = _run_training(run)
    else:
        metrics, final_metrics, y_true, y_pred, history = _run_training()

    # 5c. Mirror logging to Vertex AI (outside MLflow context, wrapped in try/except)
    if vertex_tracker is not None:
        try:
            with vertex_tracker.start_run(run_name=f"{model_name}-fold{fold_id}") as vtx_run:
                vtx_run.log_params(training_params)
                vtx_run.log_metrics(final_metrics)
            print(f"  Vertex AI: logged run {model_name}-fold{fold_id}")
        except Exception as e:
            print(f"  Vertex AI: failed to log run ({e}), continuing")

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


def _resolve_tracking_mode(cli_tracking: str, config: TrainingConfig) -> str:
    """Resolve effective tracking mode from CLI flag and YAML vertex.enabled.

    Logic:
    - If the user explicitly passes ``--tracking vertex`` or ``--tracking both``,
      that takes priority (they know what they want).
    - If ``--tracking`` is left at the default (``mlflow``) **and** the YAML
      config has ``vertex.enabled: true``, upgrade to ``both`` so that Vertex
      logging activates automatically from the config alone.
    - ``--tracking none`` always wins (disables everything).

    Args:
        cli_tracking: The value of ``--tracking`` from argparse.
        config: TrainingConfig loaded from the YAML file.

    Returns:
        One of ``"mlflow"``, ``"vertex"``, ``"both"``, or ``"none"``.
    """
    if cli_tracking in ("vertex", "both", "none"):
        return cli_tracking

    # cli_tracking == "mlflow" (the default) — check YAML override
    vertex_config = config.get_extra("vertex", {})
    if isinstance(vertex_config, dict) and vertex_config.get("enabled", False):
        return "both"

    return cli_tracking


def main() -> None:
    args = parse_args()

    two_stage = args.two_stage

    print("=" * 60)
    print("DL Model Training Script")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  Config:          {args.config or '(auto-resolve)'}")
    print(f"  Folds:           {args.folds or 'all'}")
    print(f"  Dry run:         {args.dry_run}")
    print(f"  Two-stage:       {two_stage}")
    print(f"  CV strategy:     {args.cv_strategy}")
    print(f"  Eval mode:       (resolved after config load)")
    print(f"  Data root:       {args.data_root}")
    print(f"  Spectrogram dir: {args.spectrogram_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Features CSV:    {args.features_csv}")
    print(f"  Tracking:        {args.tracking}")
    print("=" * 60)

    # --- TRAIN-2: Load metadata and generate CV folds ---
    features_csv = Path(args.features_csv)
    print(f"\nLoading metadata from {features_csv} ...")
    metadata_df = pd.read_csv(features_csv)
    print(f"  Loaded {len(metadata_df)} rows, {len(metadata_df.columns)} columns")

    # --- Determine eval_mode from config ---
    model_names_for_config = resolve_model_names(args.model)
    first_config_path = resolve_config_path(model_names_for_config[0], args.config)
    raw_config = yaml.safe_load(open(first_config_path))
    config_eval_mode = raw_config.get("training", {}).get("eval_mode", None)
    eval_mode = "post_onset" if two_stage else "default"
    if config_eval_mode == "full_life":
        eval_mode = "full_life"

    # --- Full-life mode: normalize RUL to [0, 1] per bearing ---
    if eval_mode == "full_life":
        print(f"\n  Full-life mode enabled (config: {first_config_path})")
        for bearing_id in metadata_df["bearing_id"].unique():
            mask = metadata_df["bearing_id"] == bearing_id
            num_files = mask.sum()
            metadata_df.loc[mask, "rul"] = generate_rul_for_bearing(
                num_files, strategy="linear", normalize=True
            )
        print(f"  Full-life mode: RUL normalized to [{metadata_df['rul'].min():.4f}, {metadata_df['rul'].max():.4f}]")
        print(f"  Total samples: {len(metadata_df)}")

    # --- Two-stage: prepare data before CV split (filtered indices must align) ---
    elif two_stage:
        # Need a config to read onset/training sections — use the first model's config
        # (two-stage params are shared, not per-model)
        twostage_config_path = first_config_path
        twostage_training_config = TrainingConfig.from_yaml(twostage_config_path)
        print(f"\n  Two-stage mode enabled (config: {twostage_config_path})")
        metadata_df = prepare_twostage_data(metadata_df, twostage_training_config)

    cv_strategy = args.cv_strategy
    cv_split = generate_cv_folds(metadata_df, strategy=cv_strategy)
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
    cwt_dir = Path(args.cwt_dir)

    all_results: list[dict] = []

    for model_name in model_names:
        config_path = model_configs[model_name]
        training_config = TrainingConfig.from_yaml(config_path)

        # Resolve tracking mode: YAML vertex.enabled can upgrade the CLI default,
        # but an explicit CLI flag always wins.
        tracking_mode = _resolve_tracking_mode(args.tracking, training_config)

        print(f"\n{'=' * 60}")
        print(f"  MODEL: {model_name}")
        print(f"  Config: {config_path}")
        print(f"  Tracking: {tracking_mode}")
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
                cwt_dir=cwt_dir,
                output_dir=output_dir,
                config_path=config_path,
                tracking_mode=tracking_mode,
                two_stage=two_stage,
                eval_mode=eval_mode,
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
