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

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baselines.lgbm_baseline import train_with_cv, get_feature_columns
from src.onset.labels import load_onset_labels

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
FEATURES_CSV = OUTPUTS / "features" / "features_v2.csv"
MODEL_COMPARISON_CSV = OUTPUTS / "evaluation" / "model_comparison.csv"
FEATURE_IMPORTANCE_CSV = OUTPUTS / "evaluation" / "lgbm_feature_importance.csv"
PER_BEARING_CSV = OUTPUTS / "evaluation" / "lgbm_per_bearing.csv"
MODELS_DIR = OUTPUTS / "models"
AUDIO_DIR = OUTPUTS / "audio"
PREDICTIONS_DIR = OUTPUTS / "evaluation" / "predictions"
DL_SUMMARY_CSV = OUTPUTS / "evaluation" / "dl_model_summary.csv"
ONSET_AUTO_CSV = OUTPUTS / "onset" / "onset_labels_auto.csv"
ONSET_CV_CSV = OUTPUTS / "models" / "onset_classifier_cv_results.csv"
ONSET_LABELS_YAML = BASE_DIR / "configs" / "onset_labels.yaml"

# ---------------------------------------------------------------------------
# Display name mapping for DL models
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "LightGBM": "LightGBM (CV)",
    "cnn1d_baseline": "1D CNN (Fold 0)",
    "tcn_transformer_lstm": "TCN-LSTM (Fold 0)",
    "tcn_transformer_transformer": "TCN-Transformer (Fold 0)",
    "pattern2_simple": "CNN2D Simple (Fold 0)",
    "cnn2d_lstm": "2D CNN LSTM (Fold 0)",
    "cnn2d_simple": "2D CNN Simple (Fold 0)",
}


def load_data() -> dict:
    """Load all pre-computed data at startup. Called once."""
    data: dict = {}

    # 1. Load features CSV (main dataset)
    data["features_df"] = pd.read_csv(FEATURES_CSV)
    data["feature_cols"] = get_feature_columns(data["features_df"])

    # 2. Load evaluation CSVs
    data["model_comparison"] = pd.read_csv(MODEL_COMPARISON_CSV)
    data["feature_importance"] = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    data["per_bearing"] = pd.read_csv(PER_BEARING_CSV)

    # 2b. Dynamically merge DL fold results into model_comparison
    eval_dir = OUTPUTS / "evaluation"
    dl_fold_rows = []
    for csv_path in sorted(eval_dir.glob("*_fold_results.csv")):
        if csv_path.name.startswith("all_models"):
            continue
        fold_df = pd.read_csv(csv_path)
        if fold_df.empty:
            continue
        model_name = fold_df["model_name"].iloc[0]
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        n_folds = len(fold_df)
        dl_fold_rows.append({
            "Model": display,
            "Type": f"DL - Fold 0 only" if n_folds == 1 else f"DL - {n_folds}-fold CV",
            "RMSE": fold_df["rmse"].mean(),
            "MAE": fold_df["mae"].mean(),
            "MAPE (%)": fold_df["mape"].mean(),
            "PHM08 Score": fold_df["phm08_score"].mean(),
            "PHM08 (norm)": fold_df["phm08_score_normalized"].mean(),
        })
    if dl_fold_rows:
        data["model_comparison"] = pd.concat(
            [data["model_comparison"], pd.DataFrame(dl_fold_rows)],
            ignore_index=True,
        )
        print(f"  Merged {len(dl_fold_rows)} DL model(s) into model comparison")

    # 3. Retrain LightGBM with CV for interactive predictions
    data["predictions"] = {}
    data["has_model"] = False
    try:
        cv_results, _ = train_with_cv(
            data["features_df"], data["feature_cols"], verbose=False
        )
        for result in cv_results:
            bearing = result.val_bearing[0]  # one bearing per fold
            data["predictions"][bearing] = {
                "y_true": result.y_true,
                "y_pred": result.y_pred,
            }
        data["has_model"] = True
    except Exception as e:
        print(f"Warning: LightGBM retraining failed: {e}")

    # 4. Load DL model predictions (if available)
    data["dl_predictions"] = {}  # {model_name: {bearing_id: {"y_true": [...], "y_pred": [...]}}}
    if PREDICTIONS_DIR.exists():
        for csv_path in sorted(PREDICTIONS_DIR.glob("*_predictions.csv")):
            # filename pattern: {model_name}_fold{N}_predictions.csv
            parts = csv_path.stem.rsplit("_fold", 1)
            if len(parts) == 2:
                model_name = parts[0]
                pred_df = pd.read_csv(csv_path)
                if model_name not in data["dl_predictions"]:
                    data["dl_predictions"][model_name] = {}
                for bearing_id, group in pred_df.groupby("bearing_id"):
                    data["dl_predictions"][model_name][bearing_id] = {
                        "y_true": group["y_true"].values,
                        "y_pred": group["y_pred"].values,
                    }

    # 5. Build DL per-bearing metrics from prediction CSVs (in-memory)
    data["dl_per_bearing"] = {}  # {model_name: DataFrame}
    for model_name, bearings in data["dl_predictions"].items():
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
            data["dl_per_bearing"][model_name] = pd.DataFrame(rows)

    # Also load any pre-existing per-bearing CSVs (excluding lgbm)
    for csv_path in sorted(eval_dir.glob("*_per_bearing.csv")):
        model_name = csv_path.stem.replace("_per_bearing", "")
        if model_name != "lgbm" and model_name not in data["dl_per_bearing"]:
            data["dl_per_bearing"][model_name] = pd.read_csv(csv_path)

    # 6. Load DL model summary (if available)
    if DL_SUMMARY_CSV.exists():
        data["dl_summary"] = pd.read_csv(DL_SUMMARY_CSV)
        print(f"DL summary: {len(data['dl_summary'])} models loaded from {DL_SUMMARY_CSV.name}")
    else:
        data["dl_summary"] = None

    # 7. Load DL training history (if available)
    data["dl_history"] = {}  # {model_name: [fold0_df, fold1_df, ...]}
    history_dir = OUTPUTS / "evaluation" / "history"
    if history_dir.exists():
        for csv_path in sorted(history_dir.glob("*_history.csv")):
            parts = csv_path.stem.rsplit("_fold", 1)
            if len(parts) == 2:
                model_name = parts[0]
                if model_name not in data["dl_history"]:
                    data["dl_history"][model_name] = []
                data["dl_history"][model_name].append(pd.read_csv(csv_path))
    if data["dl_history"]:
        total_folds = sum(len(v) for v in data["dl_history"].values())
        print(f"DL history: {len(data['dl_history'])} models, {total_folds} fold histories loaded")

    # 8. Load onset detection data
    data["onset_labels_manual"] = {}
    try:
        data["onset_labels_manual"] = load_onset_labels(ONSET_LABELS_YAML)
        print(f"  Onset labels: {len(data['onset_labels_manual'])} bearings loaded from YAML")
    except Exception as e:
        print(f"Warning: Could not load onset labels YAML: {e}")

    data["onset_labels_auto"] = None
    if ONSET_AUTO_CSV.exists():
        data["onset_labels_auto"] = pd.read_csv(ONSET_AUTO_CSV)
        print(f"  Onset auto labels: {len(data['onset_labels_auto'])} rows from {ONSET_AUTO_CSV.name}")

    data["onset_cv_results"] = None
    if ONSET_CV_CSV.exists():
        data["onset_cv_results"] = pd.read_csv(ONSET_CV_CSV)
        print(f"  Onset classifier CV: {len(data['onset_cv_results'])} folds from {ONSET_CV_CSV.name}")

    return data
