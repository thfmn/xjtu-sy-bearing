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
PREDICTIONS_DIR = OUTPUTS / "evaluation" / "predictions"
DL_SUMMARY_CSV = OUTPUTS / "evaluation" / "dl_model_summary.csv"

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

    # 5. Load DL per-bearing metrics (if available)
    data["dl_per_bearing"] = {}  # {model_name: DataFrame}
    eval_dir = OUTPUTS / "evaluation"
    for csv_path in sorted(eval_dir.glob("*_per_bearing.csv")):
        model_name = csv_path.stem.replace("_per_bearing", "")
        if model_name != "lgbm":  # Skip the existing LightGBM file
            data["dl_per_bearing"][model_name] = pd.read_csv(csv_path)

    # 6. Load DL model summary (if available)
    if DL_SUMMARY_CSV.exists():
        data["dl_summary"] = pd.read_csv(DL_SUMMARY_CSV)
        print(f"DL summary: {len(data['dl_summary'])} models loaded from {DL_SUMMARY_CSV.name}")
    else:
        data["dl_summary"] = None

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


def plot_rul_curve(bearing_id: str, model_name: str = "LightGBM") -> go.Figure:
    """Line chart: predicted vs actual RUL over file index for one bearing."""
    if model_name == "LightGBM":
        preds = DATA.get("predictions", {}).get(bearing_id)
    else:
        preds = DATA.get("dl_predictions", {}).get(model_name, {}).get(bearing_id)

    if preds is None:
        fig = go.Figure()
        fig.update_layout(title=f"No predictions available for {bearing_id} ({model_name})")
        return fig

    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])
    file_idx = np.arange(len(y_true))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=file_idx,
            y=y_true,
            name="Ground Truth",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=file_idx,
            y=y_pred,
            name="Predicted",
            line=dict(color="#d62728", width=2),
        )
    )
    fig.update_layout(
        title=f"RUL Prediction — {bearing_id} ({model_name})",
        xaxis_title="File Index (≈ minutes)",
        yaxis_title="Remaining Useful Life",
        hovermode="x unified",
    )
    return fig


def plot_scatter_all_predictions(model_name: str = "LightGBM") -> go.Figure:
    """Scatter: all predicted vs actual RUL from all 15 CV folds, colored by bearing."""
    if model_name == "LightGBM":
        predictions = DATA.get("predictions", {})
    else:
        predictions = DATA.get("dl_predictions", {}).get(model_name, {})

    if not predictions:
        fig = go.Figure()
        fig.update_layout(title=f"No predictions available for {model_name}")
        return fig

    fig = go.Figure()

    for bearing_id, preds in sorted(predictions.items()):
        y_true = np.array(preds["y_true"])
        y_pred = np.array(preds["y_pred"])
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name=bearing_id,
                marker=dict(size=4, opacity=0.6),
            )
        )

    # Diagonal reference line (perfect prediction)
    all_true = np.concatenate([np.array(p["y_true"]) for p in predictions.values()])
    max_val = float(np.max(all_true)) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Perfect",
            line=dict(color="black", dash="dash", width=1),
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"Predicted vs Actual RUL — {model_name} (all bearings)",
        xaxis_title="Actual RUL",
        yaxis_title="Predicted RUL",
        height=550,
    )
    return fig


def get_audio_path(condition: str, bearing_id: str, stage_key: str) -> str | None:
    """Return path to WAV file, or None if not found.

    Args:
        condition: Operating condition, e.g. "35Hz12kN".
        bearing_id: Bearing identifier, e.g. "Bearing1_1".
        stage_key: Lifecycle stage key, e.g. "healthy_0pct".
    """
    path = AUDIO_DIR / condition / bearing_id / f"{bearing_id}_{stage_key}_h.wav"
    return str(path) if path.exists() else None


def plot_waveform(wav_path: str | None) -> go.Figure | None:
    """Plot audio waveform from WAV file. Returns None if path is None."""
    if wav_path is None:
        return None

    import scipy.io.wavfile as wavfile

    sample_rate, data = wavfile.read(wav_path)

    # Convert int16 to float for display
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0

    # Use only first channel if stereo
    if data.ndim == 2:
        data = data[:, 0]

    # Subsample to ~5000 points for rendering speed
    n_samples = len(data)
    if n_samples > 5000:
        step = n_samples // 5000
        data = data[::step]
        n_samples = len(data)

    time_ms = np.linspace(0, len(data) * 1000.0 / sample_rate, n_samples, endpoint=False)

    fig = go.Figure(
        go.Scatter(x=time_ms, y=data, mode="lines", line=dict(width=0.5))
    )
    fig.update_layout(
        xaxis_title="Time (ms)",
        yaxis_title="Amplitude",
        height=250,
        margin=dict(l=50, r=20, t=30, b=40),
    )
    return fig


AUDIO_STAGES = {
    "Healthy (0%)": "healthy_0pct",
    "Degrading (50%)": "degrading_50pct",
    "Failed (100%)": "failed_100pct",
}


def update_audio_comparison(condition: str, bearing_id: str):
    """Return 6 outputs: (audio_path, waveform_plot) x 3 stages."""
    results = []
    for _label, stage_key in AUDIO_STAGES.items():
        audio_path = get_audio_path(condition, bearing_id, stage_key)
        waveform = plot_waveform(audio_path)
        results.append(audio_path)
        results.append(waveform)
    return results


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
                gr.Markdown("### Model Comparison")
                # --- Model comparison table ---
                comparison_df = DATA["model_comparison"][
                    ["Model", "Type", "RMSE", "MAE", "PHM08 (norm)"]
                ].round(2)
                gr.Dataframe(
                    value=comparison_df,
                    label="Model Comparison (all models)",
                    interactive=False,
                )

                # --- Feature importance chart with slider ---
                gr.Markdown("### Feature Importance (LightGBM)")
                fi_slider = gr.Slider(
                    minimum=5,
                    maximum=65,
                    value=20,
                    step=1,
                    label="Top N Features",
                )
                fi_plot = gr.Plot(
                    value=plot_feature_importance(20),
                    label="Feature Importance",
                )
                fi_slider.change(
                    fn=plot_feature_importance,
                    inputs=fi_slider,
                    outputs=fi_plot,
                )

                # --- Per-bearing performance table ---
                gr.Markdown("### Per-Bearing Performance (LightGBM CV)")
                per_bearing_df = DATA["per_bearing"][
                    ["bearing_id", "n_samples", "rmse", "mae"]
                ].round(2)
                gr.Dataframe(
                    value=per_bearing_df,
                    label="Per-Bearing Metrics",
                    interactive=False,
                )

                # --- SHAP plot (pre-generated image) ---
                gr.Markdown("### SHAP Feature Importance")
                shap_path = MODELS_DIR / "lgbm_shap_bar.png"
                if shap_path.exists():
                    gr.Image(
                        value=str(shap_path),
                        label="SHAP Bar Plot",
                    )
                else:
                    gr.Markdown("*SHAP image not found.*")
            with gr.Tab("Predictions"):
                gr.Markdown("### RUL Predictions")
                if not DATA.get("has_model", False) and not DATA.get("dl_predictions"):
                    gr.Markdown(
                        "⚠️ **No model predictions available.** "
                        "Displaying pre-generated prediction plots instead."
                    )
                    pred_png = MODELS_DIR / "lgbm_predictions.png"
                    scatter_png = MODELS_DIR / "lgbm_scatter.png"
                    if pred_png.exists():
                        gr.Image(value=str(pred_png), label="LightGBM Predictions")
                    if scatter_png.exists():
                        gr.Image(value=str(scatter_png), label="Predicted vs Actual")
                else:
                    # --- Build model choices list ---
                    model_choices = []
                    if DATA.get("has_model", False):
                        model_choices.append("LightGBM")
                    if DATA.get("dl_predictions"):
                        model_choices += sorted(DATA["dl_predictions"].keys())
                    default_model = model_choices[0] if model_choices else None

                    def _get_bearings_for_model(model_name: str) -> list[str]:
                        """Return sorted bearing list for a given model."""
                        if model_name == "LightGBM":
                            return sorted(DATA.get("predictions", {}).keys())
                        return sorted(DATA.get("dl_predictions", {}).get(model_name, {}).keys())

                    default_bearings_pred = _get_bearings_for_model(default_model) if default_model else []

                    # --- Model selector dropdown ---
                    pred_model_dd = gr.Dropdown(
                        choices=model_choices,
                        value=default_model,
                        label="Model",
                    )

                    # --- Bearing selector dropdown ---
                    pred_bearing_dd = gr.Dropdown(
                        choices=default_bearings_pred,
                        value=default_bearings_pred[0] if default_bearings_pred else None,
                        label="Select Bearing",
                    )

                    def _update_bearing_choices_for_model(model_name: str):
                        """Update bearing dropdown when model changes."""
                        bearings = _get_bearings_for_model(model_name)
                        return gr.Dropdown(
                            choices=bearings,
                            value=bearings[0] if bearings else None,
                        )

                    pred_model_dd.change(
                        fn=_update_bearing_choices_for_model,
                        inputs=pred_model_dd,
                        outputs=pred_bearing_dd,
                    )

                    # --- RUL prediction curve ---
                    rul_curve_plot = gr.Plot(
                        value=plot_rul_curve(
                            default_bearings_pred[0], default_model
                        ) if default_bearings_pred and default_model else None,
                        label="RUL Prediction Curve",
                    )

                    pred_model_dd.change(
                        fn=plot_rul_curve,
                        inputs=[pred_bearing_dd, pred_model_dd],
                        outputs=rul_curve_plot,
                    )
                    pred_bearing_dd.change(
                        fn=plot_rul_curve,
                        inputs=[pred_bearing_dd, pred_model_dd],
                        outputs=rul_curve_plot,
                    )

                    # --- Predicted vs Actual scatter (reactive to model selection) ---
                    gr.Markdown("### Predicted vs Actual RUL (all bearings)")
                    scatter_plot = gr.Plot(
                        value=plot_scatter_all_predictions(default_model),
                        label="Predicted vs Actual RUL",
                    )

                    pred_model_dd.change(
                        fn=plot_scatter_all_predictions,
                        inputs=pred_model_dd,
                        outputs=scatter_plot,
                    )

                    # --- Uncertainty visualizations (pre-generated PNGs) ---
                    gr.Markdown("### Uncertainty Quantification")
                    uncertainty_intervals_path = MODELS_DIR / "uncertainty_prediction_intervals.png"
                    uncertainty_vs_rul_path = MODELS_DIR / "uncertainty_vs_rul.png"
                    with gr.Row():
                        if uncertainty_intervals_path.exists():
                            gr.Image(
                                value=str(uncertainty_intervals_path),
                                label="Prediction Intervals (68% & 95% CI)",
                            )
                        else:
                            gr.Markdown("*Prediction intervals image not found.*")
                        if uncertainty_vs_rul_path.exists():
                            gr.Image(
                                value=str(uncertainty_vs_rul_path),
                                label="Uncertainty vs RUL",
                            )
                        else:
                            gr.Markdown("*Uncertainty vs RUL image not found.*")

                    # --- Per-bearing error table ---
                    gr.Markdown("### Per-Bearing Error Breakdown (LightGBM CV)")
                    pred_per_bearing_df = DATA["per_bearing"][
                        ["bearing_id", "n_samples", "rmse", "mae"]
                    ].round(2)
                    gr.Dataframe(
                        value=pred_per_bearing_df,
                        label="Per-Bearing Metrics",
                        interactive=False,
                    )
            with gr.Tab("Audio Analysis"):
                gr.Markdown("### Audio Analysis — Bearing Lifecycle Sonification")
                gr.Markdown(
                    "Listen to vibration signals converted to audio. "
                    "Compare healthy, degrading, and failed states."
                )

                # --- Cascading dropdowns (separate from EDA tab) ---
                audio_condition_choices = list(CONDITIONS.keys())
                audio_default_cond = audio_condition_choices[0]
                audio_default_bearings = BEARINGS_PER_CONDITION[audio_default_cond]

                with gr.Row():
                    audio_condition_dd = gr.Dropdown(
                        choices=audio_condition_choices,
                        value=audio_default_cond,
                        label="Operating Condition",
                    )
                    audio_bearing_dd = gr.Dropdown(
                        choices=audio_default_bearings,
                        value=audio_default_bearings[0],
                        label="Bearing",
                    )

                def _update_audio_bearing_choices(condition: str):
                    bearings = BEARINGS_PER_CONDITION.get(condition, [])
                    return gr.Dropdown(
                        choices=bearings,
                        value=bearings[0] if bearings else None,
                    )

                audio_condition_dd.change(
                    fn=_update_audio_bearing_choices,
                    inputs=audio_condition_dd,
                    outputs=audio_bearing_dd,
                )

                # --- Three-column comparison layout ---
                stage_labels = list(AUDIO_STAGES.keys())
                audio_components = []  # collect (audio, plot) pairs

                with gr.Row():
                    for label in stage_labels:
                        with gr.Column():
                            gr.Markdown(f"**{label}**")
                            audio_comp = gr.Audio(
                                type="filepath",
                                label=label,
                                interactive=False,
                            )
                            waveform_comp = gr.Plot(label=f"Waveform — {label}")
                            audio_components.append(audio_comp)
                            audio_components.append(waveform_comp)

                # Initial values
                _init_results = update_audio_comparison(
                    audio_default_cond, audio_default_bearings[0]
                )
                for i, comp in enumerate(audio_components):
                    comp.value = _init_results[i]

                # Wire callback: bearing change updates all 6 components
                audio_bearing_dd.change(
                    fn=update_audio_comparison,
                    inputs=[audio_condition_dd, audio_bearing_dd],
                    outputs=audio_components,
                )
                # Also update when condition changes (after bearing dropdown updates)
                audio_condition_dd.change(
                    fn=update_audio_comparison,
                    inputs=[audio_condition_dd, audio_bearing_dd],
                    outputs=audio_components,
                )
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
    dl_preds = DATA.get("dl_predictions", {})
    if dl_preds:
        total_bearings = sum(len(v) for v in dl_preds.values())
        print(f"  DL predictions: {len(dl_preds)} models, {total_bearings} bearings total")

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
