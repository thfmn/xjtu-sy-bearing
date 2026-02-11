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
"""Data loading and path constants for the XJTU-SY dashboard."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from src.onset.labels import load_onset_labels

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
FEATURES_CSV = OUTPUTS / "features" / "features_v2.csv"
FEATURE_IMPORTANCE_CSV = OUTPUTS / "evaluation" / "lgbm_feature_importance.csv"
PER_BEARING_CSV = OUTPUTS / "evaluation" / "lgbm_per_bearing.csv"
MODELS_DIR = OUTPUTS / "models"
AUDIO_DIR = OUTPUTS / "audio"
BENCHMARK_DIR = OUTPUTS / "benchmark"
ONSET_AUTO_CSV = OUTPUTS / "onset" / "onset_labels_auto.csv"
ONSET_CV_CSV = OUTPUTS / "models" / "onset_classifier_cv_results.csv"
ONSET_LABELS_YAML = BASE_DIR / "configs" / "onset_labels.yaml"

# ---------------------------------------------------------------------------
# Canonical 6-model registry (all normalized [0,1] RUL, 15-fold LOBO)
# ---------------------------------------------------------------------------
BENCHMARK_MODELS: dict[str, dict] = {
    "feature_lstm_fulllife": {
        "display_name": "Feature LSTM",
        "fold_results": BENCHMARK_DIR / "feature_lstm_lobo" / "feature_lstm_fulllife_fold_results.csv",
        "predictions_dir": BENCHMARK_DIR / "feature_lstm_lobo" / "predictions",
    },
    "lgbm_fulllife": {
        "display_name": "LightGBM",
        "fold_results": BENCHMARK_DIR / "lgbm_lobo" / "lgbm_fulllife_fold_results.csv",
        "predictions_dir": BENCHMARK_DIR / "lgbm_lobo" / "predictions",
    },
    "cnn1d_baseline": {
        "display_name": "1D CNN",
        "fold_results": OUTPUTS / "evaluation" / "cnn1d_baseline_fold_results.csv",
        "predictions_dir": OUTPUTS / "evaluation" / "predictions",
    },
    "cnn2d_simple": {
        "display_name": "CNN2D",
        "fold_results": BENCHMARK_DIR / "cnn2d_simple_lobo" / "dl_model_results.csv",
        "predictions_dir": BENCHMARK_DIR / "cnn2d_simple_lobo" / "predictions",
    },
    "dta_mlp": {
        "display_name": "DTA-MLP",
        "fold_results": BENCHMARK_DIR / "dta_mlp_lobo" / "dl_model_results.csv",
        "predictions_dir": BENCHMARK_DIR / "dta_mlp_lobo" / "predictions",
    },
    "tcn_transformer_lstm": {
        "display_name": "TCN-Transformer",
        "fold_results": BENCHMARK_DIR / "tcn_transformer_lstm_lobo" / "dl_model_results.csv",
        "predictions_dir": BENCHMARK_DIR / "tcn_transformer_lstm_lobo" / "predictions",
    },
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    key: cfg["display_name"] for key, cfg in BENCHMARK_MODELS.items()
}

# Reverse mapping: display name → internal model key
DISPLAY_NAME_TO_KEY: dict[str, str] = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}

# Metadata columns excluded when determining feature columns
_METADATA_COLS = {
    "condition", "bearing_id", "filename", "file_idx",
    "total_files", "rul", "rul_original", "rul_twostage", "is_post_onset",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature column names from DataFrame, excluding metadata columns."""
    return [col for col in df.columns if col not in _METADATA_COLS]


def load_data() -> dict:
    """Load all pre-computed data at startup. Called once."""
    required_files = {
        FEATURES_CSV: "scripts/03_extract_features.py",
        FEATURE_IMPORTANCE_CSV: "notebooks/30_evaluation.ipynb",
        PER_BEARING_CSV: "notebooks/30_evaluation.ipynb",
    }
    missing = {
        path: script for path, script in required_files.items() if not path.exists()
    }
    if missing:
        msg_lines = [
            "Required data files are missing. Run the pipeline scripts to generate them:",
        ]
        for path, script in missing.items():
            msg_lines.append(f"  - {path.relative_to(BASE_DIR)}  (generate with: {script})")
        msg_lines.append(
            "See the 'Reproducing Results' section in README.md for the full pipeline."
        )
        raise FileNotFoundError("\n".join(msg_lines))

    data: dict = {}

    # 1. Load features CSV (main dataset for EDA)
    data["features_df"] = pd.read_csv(FEATURES_CSV)
    data["feature_cols"] = get_feature_columns(data["features_df"])

    # 2. Load LightGBM-specific evaluation artifacts (feature importance, SHAP)
    data["feature_importance"] = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    data["per_bearing"] = pd.read_csv(PER_BEARING_CSV)

    # 3. Build model_comparison from the 6 benchmark fold result files
    comparison_rows = []
    for model_key, cfg in BENCHMARK_MODELS.items():
        csv_path = cfg["fold_results"]
        if not csv_path.exists():
            logger.warning("Fold results not found: %s", csv_path)
            continue
        fold_df = pd.read_csv(csv_path)
        comparison_rows.append({
            "Model": cfg["display_name"],
            "RMSE": fold_df["rmse"].mean(),
            "MAE": fold_df["mae"].mean(),
        })
    data["model_comparison"] = pd.DataFrame(comparison_rows)
    logger.info("Model comparison: %d models loaded", len(comparison_rows))

    # 4. Load predictions from the 6 benchmark prediction directories
    data["predictions"] = {}  # {model_key: {bearing_id: {"y_true": [...], "y_pred": [...]}}}
    for model_key, cfg in BENCHMARK_MODELS.items():
        pred_dir = cfg["predictions_dir"]
        if not pred_dir.exists():
            logger.warning("Predictions dir not found: %s", pred_dir)
            continue
        model_preds = {}
        for csv_path in sorted(pred_dir.glob(f"{model_key}_fold*_predictions.csv")):
            pred_df = pd.read_csv(csv_path)
            for bearing_id, group in pred_df.groupby("bearing_id"):
                model_preds[bearing_id] = {
                    "y_true": group["y_true"].values,
                    "y_pred": group["y_pred"].values,
                }
        if model_preds:
            data["predictions"][model_key] = model_preds
            logger.info("  %s: %d bearing predictions", cfg["display_name"], len(model_preds))

    # 5. Build per-bearing metrics from predictions (all models)
    data["dl_per_bearing"] = {}
    for model_key, bearings in data["predictions"].items():
        rows = []
        for bearing_id, preds in sorted(bearings.items()):
            y_true = np.array(preds["y_true"])
            y_pred = np.array(preds["y_pred"])
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rows.append({
                "bearing_id": bearing_id,
                "n_samples": len(y_true),
                "rmse": rmse,
                "mae": mae,
            })
        if rows:
            data["dl_per_bearing"][model_key] = pd.DataFrame(rows)

    # 6. Load DL training history (if available)
    data["dl_history"] = {}
    history_dirs: list[Path] = []
    history_dir = OUTPUTS / "evaluation" / "history"
    if history_dir.exists():
        history_dirs.append(history_dir)
    for cfg in BENCHMARK_MODELS.values():
        bench_history = cfg["fold_results"].parent / "history"
        if bench_history.exists() and bench_history not in history_dirs:
            history_dirs.append(bench_history)
    for h_dir in history_dirs:
        for csv_path in sorted(h_dir.glob("*_history.csv")):
            parts = csv_path.stem.rsplit("_fold", 1)
            if len(parts) == 2:
                model_name = parts[0]
                if model_name not in data["dl_history"]:
                    data["dl_history"][model_name] = []
                data["dl_history"][model_name].append(pd.read_csv(csv_path))
    if data["dl_history"]:
        total_folds = sum(len(v) for v in data["dl_history"].values())
        logger.info("DL history: %d models, %d fold histories loaded", len(data["dl_history"]), total_folds)

    # 7. Load onset detection data
    data["onset_labels_curated"] = {}
    try:
        data["onset_labels_curated"] = load_onset_labels(ONSET_LABELS_YAML)
        logger.info("Onset labels: %d bearings loaded from YAML", len(data["onset_labels_curated"]))
    except Exception as e:
        logger.warning("Could not load onset labels YAML: %s", e)

    data["onset_labels_auto"] = None
    if ONSET_AUTO_CSV.exists():
        data["onset_labels_auto"] = pd.read_csv(ONSET_AUTO_CSV)
        logger.info("Onset auto labels: %d rows from %s", len(data["onset_labels_auto"]), ONSET_AUTO_CSV.name)

    data["onset_cv_results"] = None
    if ONSET_CV_CSV.exists():
        data["onset_cv_results"] = pd.read_csv(ONSET_CV_CSV)
        logger.info("Onset classifier CV: %d folds from %s", len(data["onset_cv_results"]), ONSET_CV_CSV.name)

    return data
