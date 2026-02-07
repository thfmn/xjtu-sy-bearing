"""XJTU-SY Bearing RUL Prediction Dashboard.

Interactive Gradio dashboard for showcasing the bearing RUL prediction pipeline.
Displays pre-computed results and retrains a fast LightGBM model at startup
for interactive per-bearing predictions.
"""

from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.data import (
    AUDIO_DIR,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    load_data,
)
from src.data.loader import CONDITIONS, BEARINGS_PER_CONDITION
from src.onset.health_indicators import load_bearing_health_series

# ---------------------------------------------------------------------------
# Global data store (populated once at startup)
# ---------------------------------------------------------------------------
DATA: dict = {}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def plot_degradation_trend(condition: str, bearing_id: str) -> go.Figure:
    """Plot RMS and kurtosis over file index for a single bearing (dual Y-axis).

    Includes onset marker and healthy/degraded region shading when onset labels
    are available.
    """
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

    # Add onset marker if available
    onset_labels = DATA.get("onset_labels_manual", {})
    if bearing_id in onset_labels:
        onset_idx = onset_labels[bearing_id].onset_file_idx
        max_idx = int(bearing_df["file_idx"].max())

        # Green region (healthy) and red region (degraded)
        fig.add_vrect(
            x0=0, x1=onset_idx,
            fillcolor="green", opacity=0.05, line_width=0,
        )
        fig.add_vrect(
            x0=onset_idx, x1=max_idx,
            fillcolor="red", opacity=0.05, line_width=0,
        )

        # Vertical onset line
        fig.add_vline(
            x=onset_idx, line_dash="dash", line_color="darkgreen", line_width=2,
            annotation_text=f"Onset ({onset_idx})",
            annotation_position="top left",
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


def get_per_bearing_table(model_name: str = "LightGBM") -> pd.DataFrame:
    """Return per-bearing metrics table for the selected model."""
    if model_name == "LightGBM":
        df = DATA.get("per_bearing")
    else:
        df = DATA.get("dl_per_bearing", {}).get(model_name)

    if df is None or df.empty:
        return pd.DataFrame({"Info": ["No per-bearing data available for this model"]})

    cols = [c for c in ["bearing_id", "n_samples", "rmse", "mae"] if c in df.columns]
    return df[cols].round(2)


def plot_model_comparison_bars() -> go.Figure:
    """Grouped bar chart comparing RMSE and MAE across all models."""
    df = DATA["model_comparison"].copy()
    df = df.sort_values("RMSE", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="RMSE", x=df["Model"], y=df["RMSE"],
        marker_color="#1f77b4",
    ))
    fig.add_trace(go.Bar(
        name="MAE", x=df["Model"], y=df["MAE"],
        marker_color="#ff7f0e",
    ))
    fig.update_layout(
        title="Model Comparison — RMSE and MAE (lower is better)",
        barmode="group", yaxis_title="Error",
        height=450,
    )
    return fig


def plot_per_bearing_comparison() -> go.Figure:
    """Grouped bar chart: per-bearing RMSE for each model."""
    fig = go.Figure()

    # LightGBM
    lgbm_df = DATA.get("per_bearing")
    if lgbm_df is not None:
        lgbm_sorted = lgbm_df.sort_values("bearing_id")
        fig.add_trace(go.Bar(
            name="LightGBM", x=lgbm_sorted["bearing_id"], y=lgbm_sorted["rmse"],
        ))

    # DL models
    for model_name, df in sorted(DATA.get("dl_per_bearing", {}).items()):
        df_sorted = df.sort_values("bearing_id")
        fig.add_trace(go.Bar(
            name=model_name, x=df_sorted["bearing_id"], y=df_sorted["rmse"],
        ))

    fig.update_layout(
        title="Per-Bearing RMSE Comparison Across Models",
        barmode="group", xaxis_title="Bearing",
        yaxis_title="RMSE", height=500,
    )
    return fig


def plot_training_curves(model_name: str) -> go.Figure:
    """Plot training and validation loss over epochs for a DL model.

    Shows all folds as semi-transparent lines.
    """
    histories = DATA.get("dl_history", {}).get(model_name, [])
    if not histories:
        fig = go.Figure()
        fig.update_layout(title=f"No training history for {model_name}")
        return fig

    fig = go.Figure()
    for i, hist_df in enumerate(histories):
        fig.add_trace(go.Scatter(
            x=hist_df["epoch"], y=hist_df["loss"],
            name=f"Fold {i} train", opacity=0.3,
            line=dict(color="#1f77b4"), showlegend=(i == 0),
        ))
        fig.add_trace(go.Scatter(
            x=hist_df["epoch"], y=hist_df["val_loss"],
            name=f"Fold {i} val", opacity=0.3,
            line=dict(color="#d62728"), showlegend=(i == 0),
        ))

    fig.update_layout(
        title=f"Training Curves — {model_name} (all folds)",
        xaxis_title="Epoch", yaxis_title="Loss (Huber)",
        height=450,
    )
    return fig


def compute_model_architecture_table() -> pd.DataFrame:
    """Summary of all model architectures: name, type, input, params."""
    rows = [
        {"Model": "LightGBM", "Family": "Gradient Boosting", "Input": "65 features", "Input Shape": "65"},
        {"Model": "RMS Threshold", "Family": "Statistical", "Input": "RMS signal", "Input Shape": "N/A"},
        {"Model": "Kurtosis Trending", "Family": "Statistical", "Input": "Kurtosis", "Input Shape": "N/A"},
        {"Model": "Health Indicator Fusion", "Family": "Statistical", "Input": "RMS + Kurtosis", "Input Shape": "N/A"},
    ]
    try:
        from src.models.registry import list_models, get_model_info
        for name in list_models():
            try:
                info = get_model_info(name)
                rows.append({
                    "Model": _MODEL_DISPLAY_NAMES.get(name, name),
                    "Family": "Deep Learning",
                    "Input": info.input_type.replace("_", " ").title(),
                    "Input Shape": str(info.default_input_shape),
                })
            except Exception:
                pass
    except ImportError:
        pass
    return pd.DataFrame(rows)


_MODEL_DISPLAY_NAMES = MODEL_DISPLAY_NAMES


def get_model_metrics_summary(model_name: str) -> str:
    """Return markdown string with model's aggregate metrics."""
    comp_df = DATA.get("model_comparison")
    if comp_df is None or comp_df.empty:
        return f"**{model_name}** — No aggregate metrics available"

    display_name = _MODEL_DISPLAY_NAMES.get(model_name, model_name)
    row = comp_df[comp_df["Model"] == display_name]
    if row.empty:
        return f"**{model_name}** — No aggregate metrics available"
    row = row.iloc[0]
    return (
        f"**{row['Model']}** — "
        f"RMSE: {row['RMSE']:.2f} | "
        f"MAE: {row['MAE']:.2f} | "
        f"Type: {row['Type']}"
    )


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


def plot_prediction_intervals(model_name: str = "LightGBM") -> go.Figure:
    """Prediction intervals from CV residuals: predicted ± empirical CI, sorted by true RUL."""
    if model_name == "LightGBM":
        predictions = DATA.get("predictions", {})
    else:
        predictions = DATA.get("dl_predictions", {}).get(model_name, {})

    if not predictions:
        fig = go.Figure()
        fig.update_layout(title=f"No predictions available for {model_name}")
        return fig

    # Pool all predictions
    all_true, all_pred = [], []
    for preds in predictions.values():
        all_true.extend(preds["y_true"])
        all_pred.extend(preds["y_pred"])
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Sort by true RUL
    sort_idx = np.argsort(all_true)
    true_sorted = all_true[sort_idx]
    pred_sorted = all_pred[sort_idx]

    # Sliding window empirical CI (window = 5% of samples, min 20)
    n = len(true_sorted)
    win = max(20, n // 20)
    pred_mean = np.convolve(pred_sorted, np.ones(win) / win, mode="same")
    pred_std = np.array([
        pred_sorted[max(0, i - win // 2):min(n, i + win // 2)].std()
        for i in range(n)
    ])

    x = np.arange(n)
    fig = go.Figure()
    # 95% CI
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([pred_mean + 1.96 * pred_std, (pred_mean - 1.96 * pred_std)[::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.1)", line=dict(width=0),
        name="95% CI (2σ)",
    ))
    # 68% CI
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([pred_mean + pred_std, (pred_mean - pred_std)[::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.25)", line=dict(width=0),
        name="68% CI (1σ)",
    ))
    fig.add_trace(go.Scatter(x=x, y=true_sorted, name="Ground Truth", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=x, y=pred_mean, name="Predicted (smoothed)", line=dict(color="#636EFA", width=2)))

    fig.update_layout(
        title=f"Prediction Intervals — {model_name} (empirical from CV residuals)",
        xaxis_title="Sample Index (sorted by true RUL)",
        yaxis_title="RUL", height=450, hovermode="x unified",
    )
    return fig


def plot_residuals_vs_rul(model_name: str = "LightGBM") -> go.Figure:
    """Scatter: residual (pred - true) vs true RUL, colored by bearing."""
    if model_name == "LightGBM":
        predictions = DATA.get("predictions", {})
    else:
        predictions = DATA.get("dl_predictions", {}).get(model_name, {})

    if not predictions:
        fig = go.Figure()
        fig.update_layout(title=f"No predictions available for {model_name}")
        return fig

    fig = go.Figure()
    all_true, all_resid = [], []
    for bearing_id, preds in sorted(predictions.items()):
        y_true = np.array(preds["y_true"])
        y_pred = np.array(preds["y_pred"])
        residuals = y_pred - y_true
        all_true.extend(y_true)
        all_resid.extend(residuals)
        fig.add_trace(go.Scatter(
            x=y_true, y=residuals, mode="markers", name=bearing_id,
            marker=dict(size=4, opacity=0.5),
        ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

    # Trend line
    all_true = np.array(all_true)
    all_resid = np.array(all_resid)
    if len(all_true) > 2:
        z = np.polyfit(all_true, all_resid, 1)
        x_line = np.linspace(all_true.min(), all_true.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=np.polyval(z, x_line),
            mode="lines", name=f"Trend (slope={z[0]:.3f})",
            line=dict(color="red", dash="dash", width=2),
        ))

    fig.update_layout(
        title=f"Residuals vs True RUL — {model_name}",
        xaxis_title="True RUL", yaxis_title="Residual (Predicted − Actual)",
        height=450,
    )
    return fig


def build_onset_overview_table() -> pd.DataFrame:
    """Build overview table merging manual + auto onset labels for all 15 bearings."""
    manual = DATA.get("onset_labels_manual", {})
    auto_df = DATA.get("onset_labels_auto")

    rows = []
    for bearing_id, entry in sorted(manual.items()):
        row = {
            "Bearing": bearing_id,
            "Condition": entry.condition,
            "Manual Onset": entry.onset_file_idx,
            "Confidence": entry.confidence,
            "Method": entry.detection_method,
        }
        # Merge auto onset
        if auto_df is not None:
            auto_row = auto_df[auto_df["bearing_id"] == bearing_id]
            if not auto_row.empty:
                row["Auto Onset"] = int(auto_row["onset_file_idx"].iloc[0])
                row["Auto Method"] = auto_row["detector_method"].iloc[0]
            else:
                row["Auto Onset"] = None
                row["Auto Method"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def plot_health_indicators(condition: str, bearing_id: str) -> go.Figure:
    """Dual-axis Plotly: kurtosis + RMS with onset markers and region shading."""
    try:
        hs = load_bearing_health_series(bearing_id, DATA["features_df"])
    except ValueError:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {bearing_id}")
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Average kurtosis across channels
    kurtosis_avg = (hs.kurtosis_h + hs.kurtosis_v) / 2
    rms_avg = (hs.rms_h + hs.rms_v) / 2

    fig.add_trace(
        go.Scatter(
            x=hs.file_indices, y=kurtosis_avg,
            name="Kurtosis (H+V avg)", line=dict(color="#d62728"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hs.file_indices, y=rms_avg,
            name="RMS (H+V avg)", line=dict(color="#1f77b4"),
        ),
        secondary_y=True,
    )

    # Onset markers
    manual = DATA.get("onset_labels_manual", {})
    auto_df = DATA.get("onset_labels_auto")
    max_idx = int(hs.file_indices[-1]) if len(hs.file_indices) > 0 else 100

    if bearing_id in manual:
        m_idx = manual[bearing_id].onset_file_idx
        fig.add_vrect(x0=0, x1=m_idx, fillcolor="green", opacity=0.05, line_width=0)
        fig.add_vrect(x0=m_idx, x1=max_idx, fillcolor="red", opacity=0.05, line_width=0)
        fig.add_vline(
            x=m_idx, line_dash="dash", line_color="blue", line_width=2,
            annotation_text=f"Manual ({m_idx})", annotation_position="top left",
        )

    if auto_df is not None:
        auto_row = auto_df[auto_df["bearing_id"] == bearing_id]
        if not auto_row.empty:
            a_idx = int(auto_row["onset_file_idx"].iloc[0])
            fig.add_vline(
                x=a_idx, line_dash="dashdot", line_color="red", line_width=2,
                annotation_text=f"Auto ({a_idx})", annotation_position="top right",
            )

    fig.update_layout(
        title=f"Health Indicators — {bearing_id} ({condition})",
        hovermode="x unified", height=500,
    )
    fig.update_xaxes(title_text="File Index")
    fig.update_yaxes(title_text="Kurtosis (avg)", secondary_y=False)
    fig.update_yaxes(title_text="RMS (avg)", secondary_y=True)
    return fig


def build_onset_classifier_table() -> tuple[pd.DataFrame, str]:
    """Return onset classifier CV results table and summary markdown."""
    cv_df = DATA.get("onset_cv_results")
    if cv_df is None or cv_df.empty:
        return pd.DataFrame({"Info": ["No onset classifier results available"]}), ""

    display_df = cv_df[["val_bearing", "f1", "auc_roc", "precision", "recall", "accuracy"]].copy()
    display_df.columns = ["Bearing", "F1", "AUC-ROC", "Precision", "Recall", "Accuracy"]
    display_df = display_df.round(4)

    mean_f1 = cv_df["f1"].mean()
    std_f1 = cv_df["f1"].std()
    mean_auc = cv_df["auc_roc"].dropna().mean()
    summary = (
        f"**LSTM Onset Classifier (15-fold LOBO CV):** "
        f"Mean F1 = {mean_f1:.3f} +/- {std_f1:.3f} | "
        f"Mean AUC-ROC = {mean_auc:.3f}"
    )
    return display_df, summary


def build_detector_comparison_table() -> pd.DataFrame:
    """Table comparing each detector's onset index per bearing."""
    auto_df = DATA.get("onset_labels_auto")
    if auto_df is None:
        return pd.DataFrame({"Info": ["No auto-detector results available"]})

    cols = ["bearing_id", "manual_onset_idx", "kurtosis_threshold_idx",
            "rms_threshold_idx", "kurtosis_cusum_idx", "rms_cusum_idx",
            "onset_file_idx", "detector_method"]
    available = [c for c in cols if c in auto_df.columns]
    df = auto_df[available].copy()

    # Rename for readability
    rename_map = {
        "bearing_id": "Bearing",
        "manual_onset_idx": "Manual",
        "onset_file_idx": "Auto (selected)",
        "detector_method": "Selected Method",
        "kurtosis_threshold_idx": "Kurt. Threshold",
        "rms_threshold_idx": "RMS Threshold",
        "kurtosis_cusum_idx": "Kurt. CUSUM",
        "rms_cusum_idx": "RMS CUSUM",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


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
    """Build the Gradio Blocks application with 5 tabs."""
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
                gr.Markdown(
                    "> *Note: DL models evaluated on Fold 0 only (Bearing1_1, 123 samples). "
                    "Full 15-fold CV pending. LightGBM uses full 15-fold leave-one-bearing-out CV.*"
                )
                # --- Model comparison table ---
                comparison_df = DATA["model_comparison"][
                    ["Model", "Type", "RMSE", "MAE", "PHM08 (norm)"]
                ].round(2)
                gr.Dataframe(
                    value=comparison_df,
                    label="Model Comparison (all models)",
                    interactive=False,
                )

                # --- Model comparison bar chart ---
                gr.Plot(
                    value=plot_model_comparison_bars(),
                    label="Model Comparison Chart",
                )

                # --- Model architecture summary table ---
                gr.Markdown("### Model Architectures")
                gr.Dataframe(
                    value=compute_model_architecture_table(),
                    label="Model Architecture Overview",
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

                # --- Training convergence curves ---
                gr.Markdown("### Training Convergence")
                dl_history_models = sorted(DATA.get("dl_history", {}).keys())
                if dl_history_models:
                    convergence_model_dd = gr.Dropdown(
                        choices=dl_history_models,
                        value=dl_history_models[0],
                        label="DL Model",
                    )
                    convergence_plot = gr.Plot(
                        value=plot_training_curves(dl_history_models[0]),
                        label="Training Curves",
                    )
                    convergence_model_dd.change(
                        fn=plot_training_curves,
                        inputs=convergence_model_dd,
                        outputs=convergence_plot,
                    )
                else:
                    gr.Markdown("*No training history available. Train DL models first.*")

                # --- Per-bearing comparison across models ---
                if DATA.get("dl_per_bearing"):
                    gr.Markdown("### Per-Bearing Model Comparison")
                    gr.Plot(
                        value=plot_per_bearing_comparison(),
                        label="Per-Bearing RMSE Comparison",
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

                    # --- Model metrics summary (reactive) ---
                    model_metrics_md = gr.Markdown(
                        value=get_model_metrics_summary(default_model) if default_model else "",
                    )

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

                    pred_model_dd.change(
                        fn=get_model_metrics_summary,
                        inputs=pred_model_dd,
                        outputs=model_metrics_md,
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

                    # --- Residual analysis (replaces broken uncertainty PNGs) ---
                    gr.Markdown("### Residual Analysis")
                    intervals_plot = gr.Plot(
                        value=plot_prediction_intervals(default_model),
                        label="Prediction Intervals (empirical from CV)",
                    )
                    residuals_plot = gr.Plot(
                        value=plot_residuals_vs_rul(default_model),
                        label="Residuals vs True RUL",
                    )
                    pred_model_dd.change(
                        fn=plot_prediction_intervals,
                        inputs=pred_model_dd,
                        outputs=intervals_plot,
                    )
                    pred_model_dd.change(
                        fn=plot_residuals_vs_rul,
                        inputs=pred_model_dd,
                        outputs=residuals_plot,
                    )

                    # --- Per-bearing error table (reactive to model selection) ---
                    gr.Markdown("### Per-Bearing Error Breakdown")
                    per_bearing_table = gr.Dataframe(
                        value=get_per_bearing_table(default_model) if default_model else pd.DataFrame(),
                        label="Per-Bearing Metrics",
                        interactive=False,
                    )

                    pred_model_dd.change(
                        fn=get_per_bearing_table,
                        inputs=pred_model_dd,
                        outputs=per_bearing_table,
                    )
            with gr.Tab("Onset Detection"):
                gr.Markdown("### Degradation Onset Detection")
                gr.Markdown(
                    "Two-stage onset detection pipeline: 5 statistical detectors + "
                    "LSTM classifier. Identifies the transition from healthy to degraded "
                    "operation for each bearing."
                )

                # --- Onset overview table ---
                gr.Markdown("#### Onset Labels Overview")
                onset_overview = build_onset_overview_table()
                if not onset_overview.empty:
                    gr.Dataframe(
                        value=onset_overview,
                        label="Manual + Auto Onset Labels (15 bearings)",
                        interactive=False,
                    )
                else:
                    gr.Markdown("*No onset labels available.*")

                # --- Interactive health indicator plot ---
                gr.Markdown("#### Health Indicator Explorer")
                onset_condition_choices = list(CONDITIONS.keys())
                onset_default_cond = onset_condition_choices[0]
                onset_default_bearings = BEARINGS_PER_CONDITION[onset_default_cond]

                with gr.Row():
                    onset_condition_dd = gr.Dropdown(
                        choices=onset_condition_choices,
                        value=onset_default_cond,
                        label="Operating Condition",
                    )
                    onset_bearing_dd = gr.Dropdown(
                        choices=onset_default_bearings,
                        value=onset_default_bearings[0],
                        label="Bearing",
                    )

                def _update_onset_bearing_choices(condition: str):
                    bearings = BEARINGS_PER_CONDITION.get(condition, [])
                    return gr.Dropdown(choices=bearings, value=bearings[0] if bearings else None)

                onset_condition_dd.change(
                    fn=_update_onset_bearing_choices,
                    inputs=onset_condition_dd,
                    outputs=onset_bearing_dd,
                )

                hi_plot = gr.Plot(
                    value=plot_health_indicators(onset_default_cond, onset_default_bearings[0]),
                    label="Health Indicators with Onset Markers",
                )

                onset_bearing_dd.change(
                    fn=plot_health_indicators,
                    inputs=[onset_condition_dd, onset_bearing_dd],
                    outputs=hi_plot,
                )
                onset_condition_dd.change(
                    fn=plot_health_indicators,
                    inputs=[onset_condition_dd, onset_bearing_dd],
                    outputs=hi_plot,
                )

                # --- Onset classifier performance ---
                gr.Markdown("#### LSTM Onset Classifier Performance")
                classifier_df, classifier_summary = build_onset_classifier_table()
                if classifier_summary:
                    gr.Markdown(classifier_summary)
                gr.Dataframe(
                    value=classifier_df,
                    label="15-fold Leave-One-Bearing-Out CV Results",
                    interactive=False,
                )

                # --- Detector comparison ---
                gr.Markdown("#### Detector Comparison")
                gr.Markdown(
                    "Onset file index detected by each algorithm. "
                    "'Auto (selected)' is the best detector chosen per bearing."
                )
                gr.Dataframe(
                    value=build_detector_comparison_table(),
                    label="Per-Bearing Detector Onset Indices",
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
