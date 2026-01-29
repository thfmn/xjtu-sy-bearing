"""XJTU-SY Bearing RUL Prediction Dashboard.

Interactive Gradio dashboard for showcasing the bearing RUL prediction pipeline.
Displays pre-computed results and retrains a fast LightGBM model at startup
for interactive per-bearing predictions.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from src.data.loader import CONDITIONS, BEARINGS_PER_CONDITION
from src.models.baselines.lgbm_baseline import train_with_cv, get_feature_columns

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
OUTPUTS = BASE_DIR / "outputs"
FEATURES_CSV = OUTPUTS / "features" / "features_v2.csv"
MODEL_COMPARISON_CSV = OUTPUTS / "evaluation" / "model_comparison.csv"
FEATURE_IMPORTANCE_CSV = OUTPUTS / "evaluation" / "lgbm_feature_importance.csv"
PER_BEARING_CSV = OUTPUTS / "evaluation" / "lgbm_per_bearing.csv"
MODELS_DIR = OUTPUTS / "models"
AUDIO_DIR = OUTPUTS / "audio"

# ---------------------------------------------------------------------------
# Global data store (populated once at startup)
# ---------------------------------------------------------------------------
DATA: dict = {}


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

    return data


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    """Build the Gradio Blocks application with 4 tabs."""
    with gr.Blocks(title="XJTU-SY Bearing RUL Dashboard") as app:
        gr.Markdown("# XJTU-SY Bearing RUL Prediction Dashboard")
        gr.Markdown(
            "Remaining Useful Life prediction for rolling element bearings. "
            "15 bearings, 3 operating conditions, 9,216 vibration recordings."
        )
        with gr.Tabs():
            with gr.Tab("EDA"):
                gr.Markdown("*Coming in READY-7*")
            with gr.Tab("Model Results"):
                gr.Markdown("*Coming in READY-8*")
            with gr.Tab("Predictions"):
                gr.Markdown("*Coming in READY-9*")
            with gr.Tab("Audio Analysis"):
                gr.Markdown("*Coming in READY-10*")
    return app


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    DATA = load_data()
    print(f"  Features: {len(DATA['features_df'])} rows, {len(DATA['feature_cols'])} feature columns")
    print(f"  Model trained: {DATA['has_model']}")
    if DATA["has_model"]:
        print(f"  Predictions for {len(DATA['predictions'])} bearings")

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
