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
"""Plotting and visualization functions for the XJTU-SY dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.data import AUDIO_DIR, MODEL_DISPLAY_NAMES, MODELS_DIR
from src.onset.health_indicators import load_bearing_health_series

# ---------------------------------------------------------------------------
# Module-level data store — set by app.py at startup via init_plots()
# ---------------------------------------------------------------------------
DATA: dict = {}

_MODEL_DISPLAY_NAMES = MODEL_DISPLAY_NAMES

AUDIO_STAGES = {
    "Healthy (0%)": "healthy_0pct",
    "Degrading (50%)": "degrading_50pct",
    "Failed (100%)": "failed_100pct",
}


def init_plots(data: dict) -> None:
    """Bind the global data store. Called once at startup from app.py."""
    global DATA
    DATA = data


# ---------------------------------------------------------------------------
# EDA plots
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
    onset_labels = DATA.get("onset_labels_curated", {})
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
# Model results plots
# ---------------------------------------------------------------------------

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


def plot_model_comparison_bars() -> go.Figure:
    """Horizontal bar chart comparing RMSE and MAE across all models."""
    df = DATA["model_comparison"].copy()
    df = df.sort_values("MAE", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="RMSE", y=df["Model"], x=df["RMSE"],
        marker_color="#1f77b4", orientation="h",
    ))
    fig.add_trace(go.Bar(
        name="MAE", y=df["Model"], x=df["MAE"],
        marker_color="#ff7f0e", orientation="h",
    ))
    fig.update_layout(
        title="Model Comparison — RMSE and MAE (lower is better)",
        barmode="group", xaxis_title="Error",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(df) * 60),
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


# ---------------------------------------------------------------------------
# Prediction plots
# ---------------------------------------------------------------------------

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


def plot_prediction_intervals(model_name: str = "LightGBM") -> go.Figure:
    """Prediction intervals from CV residuals: predicted +/- empirical CI, sorted by true RUL."""
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


# ---------------------------------------------------------------------------
# Onset detection plots
# ---------------------------------------------------------------------------

def build_onset_overview_table() -> pd.DataFrame:
    """Build overview table merging curated + auto onset labels for all 15 bearings."""
    manual = DATA.get("onset_labels_curated", {})
    auto_df = DATA.get("onset_labels_auto")

    rows = []
    for bearing_id, entry in sorted(manual.items()):
        row = {
            "Bearing": bearing_id,
            "Condition": entry.condition,
            "Curated Onset": entry.onset_file_idx,
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
    manual = DATA.get("onset_labels_curated", {})
    auto_df = DATA.get("onset_labels_auto")
    max_idx = int(hs.file_indices[-1]) if len(hs.file_indices) > 0 else 100

    if bearing_id in manual:
        m_idx = manual[bearing_id].onset_file_idx
        fig.add_vrect(x0=0, x1=m_idx, fillcolor="green", opacity=0.05, line_width=0)
        fig.add_vrect(x0=m_idx, x1=max_idx, fillcolor="red", opacity=0.05, line_width=0)
        fig.add_vline(
            x=m_idx, line_dash="dash", line_color="blue", line_width=2,
            annotation_text=f"Curated ({m_idx})", annotation_position="top left",
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
        "curated_onset_idx": "Curated",
        "onset_file_idx": "Auto (selected)",
        "detector_method": "Selected Method",
        "kurtosis_threshold_idx": "Kurt. Threshold",
        "rms_threshold_idx": "RMS Threshold",
        "kurtosis_cusum_idx": "Kurt. CUSUM",
        "rms_cusum_idx": "RMS CUSUM",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


# ---------------------------------------------------------------------------
# Audio plots
# ---------------------------------------------------------------------------

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


def update_audio_comparison(condition: str, bearing_id: str):
    """Return 6 outputs: (audio_path, waveform_plot) x 3 stages."""
    results = []
    for _label, stage_key in AUDIO_STAGES.items():
        audio_path = get_audio_path(condition, bearing_id, stage_key)
        waveform = plot_waveform(audio_path)
        results.append(audio_path)
        results.append(waveform)
    return results
