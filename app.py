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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# Helper functions
# ---------------------------------------------------------------------------

def plot_degradation_trend(condition: str, bearing_id: str) -> go.Figure:
    """Plot RMS and kurtosis over file index for a single bearing (dual Y-axis)."""
    df = DATA["features_df"]
    mask = (df["condition"] == condition) & (df["bearing_id"] == bearing_id)
    bearing_df = df.loc[mask].sort_values("file_idx")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=bearing_df["file_idx"],
            y=bearing_df["h_rms"],
            name="RMS (horizontal)",
            line=dict(color="#1f77b4"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=bearing_df["file_idx"],
            y=bearing_df["h_kurtosis"],
            name="Kurtosis (horizontal)",
            line=dict(color="#d62728"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Degradation Trend — {bearing_id} ({condition})",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="File Index (≈ minutes)")
    fig.update_yaxes(title_text="RMS", secondary_y=False)
    fig.update_yaxes(title_text="Kurtosis", secondary_y=True)

    return fig


def compute_dataset_overview() -> pd.DataFrame:
    """Summary statistics table: files and bearing lifetimes per condition."""
    df = DATA["features_df"]
    rows = []
    for condition in sorted(df["condition"].unique()):
        cond_df = df[df["condition"] == condition]
        lifetimes = cond_df.groupby("bearing_id")["total_files"].first()
        rows.append(
            {
                "Condition": condition,
                "Bearings": cond_df["bearing_id"].nunique(),
                "Total Files": len(cond_df),
                "Min Lifetime": int(lifetimes.min()),
                "Max Lifetime": int(lifetimes.max()),
                "Mean Lifetime": round(lifetimes.mean(), 1),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_importance(top_n: int = 20) -> go.Figure:
    """Interactive horizontal bar chart of feature importance with error bars."""
    df = DATA["feature_importance"].sort_values("importance_mean", ascending=False).head(int(top_n))
    # Reverse so highest importance is at top of horizontal bar chart
    df = df.iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=df["importance_mean"],
            y=df["feature"],
            orientation="h",
            error_x=dict(type="data", array=df["importance_std"].values),
            marker_color="#1f77b4",
        )
    )
    fig.update_layout(
        title=f"Top {int(top_n)} Feature Importance (LightGBM, gain-based)",
        xaxis_title="Mean Importance",
        yaxis_title="Feature",
        height=max(400, int(top_n) * 25),
        margin=dict(l=200),
    )
    return fig


def plot_feature_distribution(feature_name: str) -> go.Figure:
    """Box plot of a selected feature across the 3 operating conditions."""
    df = DATA["features_df"]
    fig = px.box(
        df,
        x="condition",
        y=feature_name,
        color="condition",
        title=f"Distribution of {feature_name} by Condition",
    )
    fig.update_layout(
        xaxis_title="Operating Condition",
        yaxis_title=feature_name,
        showlegend=False,
    )
    return fig


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
                gr.Markdown("### Exploratory Data Analysis")

                # --- Dataset overview table ---
                overview_table = gr.Dataframe(
                    value=compute_dataset_overview(),
                    label="Dataset Overview",
                    interactive=False,
                )

                # --- Cascading dropdowns: condition → bearing ---
                condition_choices = list(CONDITIONS.keys())
                default_condition = condition_choices[0]
                default_bearings = BEARINGS_PER_CONDITION[default_condition]

                with gr.Row():
                    eda_condition_dd = gr.Dropdown(
                        choices=condition_choices,
                        value=default_condition,
                        label="Operating Condition",
                    )
                    eda_bearing_dd = gr.Dropdown(
                        choices=default_bearings,
                        value=default_bearings[0],
                        label="Bearing",
                    )

                def _update_bearing_choices(condition: str):
                    bearings = BEARINGS_PER_CONDITION.get(condition, [])
                    return gr.Dropdown(choices=bearings, value=bearings[0] if bearings else None)

                eda_condition_dd.change(
                    fn=_update_bearing_choices,
                    inputs=eda_condition_dd,
                    outputs=eda_bearing_dd,
                )

                # --- Degradation trend plot ---
                trend_plot = gr.Plot(
                    value=plot_degradation_trend(default_condition, default_bearings[0]),
                    label="Degradation Trend",
                )

                eda_bearing_dd.change(
                    fn=plot_degradation_trend,
                    inputs=[eda_condition_dd, eda_bearing_dd],
                    outputs=trend_plot,
                )

                # --- Feature distribution ---
                gr.Markdown("### Feature Distribution by Condition")
                feature_dd = gr.Dropdown(
                    choices=DATA.get("feature_cols", []),
                    value="h_rms",
                    label="Select Feature",
                )
                dist_plot = gr.Plot(
                    value=plot_feature_distribution("h_rms"),
                    label="Feature Distribution",
                )
                feature_dd.change(
                    fn=plot_feature_distribution,
                    inputs=feature_dd,
                    outputs=dist_plot,
                )
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
