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

"""XJTU-SY Bearing RUL Prediction Dashboard.

Interactive Gradio dashboard for showcasing the bearing RUL prediction pipeline.
Displays pre-computed benchmark results from 6 models evaluated on normalized
[0, 1] RUL with 15-fold leave-one-bearing-out cross-validation.
"""

from __future__ import annotations

import pandas as pd
import gradio as gr

from app.data import (
    MODELS_DIR,
    MODEL_DISPLAY_NAMES,
    DISPLAY_NAME_TO_KEY,
    load_data,
)
from app.plots import (
    AUDIO_STAGES,
    init_plots,
    plot_degradation_trend,
    compute_dataset_overview,
    plot_feature_distribution,
    plot_feature_importance,
    plot_model_comparison_bars,
    compute_model_architecture_table,
    plot_training_curves,
    plot_per_bearing_comparison,
    plot_rul_curve,
    plot_scatter_all_predictions,
    get_per_bearing_table,
    get_model_metrics_summary,
    plot_prediction_intervals,
    plot_residuals_vs_rul,
    build_onset_overview_table,
    plot_health_indicators,
    build_onset_classifier_table,
    build_detector_comparison_table,
    update_audio_comparison,
)
from src.data.loader import CONDITIONS, BEARINGS_PER_CONDITION

# ---------------------------------------------------------------------------
# Global data store (populated once at startup)
# ---------------------------------------------------------------------------
DATA: dict = {}

# Display name list for model dropdowns (ordered by performance)
_DISPLAY_NAMES = list(MODEL_DISPLAY_NAMES.values())


def _update_bearing_dd(condition: str):
    """Update bearing dropdown choices when operating condition changes."""
    bearings = BEARINGS_PER_CONDITION.get(condition, [])
    return gr.Dropdown(choices=bearings, value=bearings[0] if bearings else None)


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
                overview_table = gr.Dataframe(
                    value=compute_dataset_overview(),
                    label="Dataset Overview",
                    interactive=False,
                )
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

                eda_condition_dd.change(
                    fn=_update_bearing_dd,
                    inputs=eda_condition_dd,
                    outputs=eda_bearing_dd,
                )
                trend_plot = gr.Plot(
                    value=plot_degradation_trend(default_condition, default_bearings[0]),
                    label="Degradation Trend",
                )

                eda_bearing_dd.change(
                    fn=plot_degradation_trend,
                    inputs=[eda_condition_dd, eda_bearing_dd],
                    outputs=trend_plot,
                )
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
                    "> *All models evaluated with 15-fold leave-one-bearing-out CV "
                    "on normalized [0, 1] RUL scale.*"
                )
                comparison_df = DATA["model_comparison"][
                    ["Model", "RMSE", "MAE"]
                ].round(3)
                gr.Dataframe(
                    value=comparison_df,
                    label="Model Comparison (6 benchmark models)",
                    interactive=False,
                )

                gr.Plot(
                    value=plot_model_comparison_bars(),
                    label="Model Comparison Chart",
                )

                gr.Markdown("### Model Architectures")
                gr.Dataframe(
                    value=compute_model_architecture_table(),
                    label="Model Architecture Overview",
                    interactive=False,
                )

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

                gr.Markdown("### Per-Bearing Model Comparison")
                if DATA.get("dl_per_bearing"):
                    gr.Plot(
                        value=plot_per_bearing_comparison(),
                        label="Per-Bearing RMSE Comparison",
                    )
                else:
                    gr.Markdown("*No per-bearing data available.*")

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

                # Explanation callout
                gr.Markdown(
                    "> **Reading the prediction curves:** The RMSE shown "
                    "(e.g., Feature LSTM = 0.160) is the mean error across all "
                    "15 bearings and all timesteps on a 0-to-1 normalized scale. "
                    "Individual per-bearing curves vary in quality — some track "
                    "nearly perfectly (Bearing1\_3: 0.090 RMSE), while harder "
                    "cases (Bearing3\_1: 2,538 files, gradual degradation) show "
                    "more deviation. This is expected and consistent across all "
                    "models. Use the model dropdown to compare models on the "
                    "same bearing."
                )

                predictions = DATA.get("predictions", {})
                if not predictions:
                    gr.Markdown(
                        "No model predictions available. "
                        "Run the benchmark scripts first."
                    )
                else:
                    default_model = _DISPLAY_NAMES[0]
                    default_model_key = DISPLAY_NAME_TO_KEY[default_model]

                    def _bearings_for_model(display_name: str) -> list[str]:
                        key = DISPLAY_NAME_TO_KEY.get(display_name, display_name)
                        return sorted(predictions.get(key, {}).keys())

                    def _update_pred_bearing_dd(display_name: str):
                        bearings = _bearings_for_model(display_name)
                        return gr.Dropdown(choices=bearings, value=bearings[0] if bearings else None)

                    default_bearings_pred = _bearings_for_model(default_model)

                    model_metrics_md = gr.Markdown(
                        value=get_model_metrics_summary(default_model),
                    )
                    pred_model_dd = gr.Dropdown(
                        choices=_DISPLAY_NAMES, value=default_model, label="Model",
                    )
                    pred_bearing_dd = gr.Dropdown(
                        choices=default_bearings_pred,
                        value=default_bearings_pred[0] if default_bearings_pred else None,
                        label="Select Bearing",
                    )

                    pred_model_dd.change(
                        fn=_update_pred_bearing_dd,
                        inputs=pred_model_dd,
                        outputs=pred_bearing_dd,
                    )

                    pred_model_dd.change(
                        fn=get_model_metrics_summary,
                        inputs=pred_model_dd,
                        outputs=model_metrics_md,
                    )

                    rul_curve_plot = gr.Plot(
                        value=plot_rul_curve(
                            default_bearings_pred[0], default_model
                        ) if default_bearings_pred else None,
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

                    gr.Markdown("### Per-Bearing Error Breakdown")
                    per_bearing_table = gr.Dataframe(
                        value=get_per_bearing_table(default_model),
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

                gr.Markdown("#### Onset Labels Overview")
                onset_overview = build_onset_overview_table()
                if not onset_overview.empty:
                    gr.Dataframe(
                        value=onset_overview,
                        label="Curated + Auto Onset Labels (15 bearings)",
                        interactive=False,
                    )
                else:
                    gr.Markdown("*No onset labels available.*")

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

                onset_condition_dd.change(
                    fn=_update_bearing_dd,
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

                gr.Markdown("#### LSTM Onset Classifier Performance")
                classifier_df, classifier_summary = build_onset_classifier_table()
                if classifier_summary:
                    gr.Markdown(classifier_summary)
                gr.Dataframe(
                    value=classifier_df,
                    label="15-fold Leave-One-Bearing-Out CV Results",
                    interactive=False,
                )

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

                audio_condition_dd.change(
                    fn=_update_bearing_dd,
                    inputs=audio_condition_dd,
                    outputs=audio_bearing_dd,
                )

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
    init_plots(DATA)
    print(f"  Features: {len(DATA['features_df'])} rows, {len(DATA['feature_cols'])} feature columns")
    print(f"  Models: {len(DATA['model_comparison'])} benchmark models")
    preds = DATA.get("predictions", {})
    if preds:
        total_bearings = sum(len(v) for v in preds.values())
        print(f"  Predictions: {len(preds)} models, {total_bearings} bearings total")

    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
