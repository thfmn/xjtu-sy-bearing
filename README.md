# XJTU-SY Bearing RUL Prediction

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-459%20passed-brightgreen)
![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)

An MLOps pipeline that predicts the **Remaining Useful Life (RUL)** of rolling element bearings from vibration sensor data. Built on the XJTU-SY benchmark dataset with GCP infrastructure for data processing, model training, and experiment tracking.

## The Problem

Bearings are among the most critical — and most failure-prone — components in rotating machinery. A single unexpected bearing failure in a wind turbine, industrial pump, or rail axle can halt production for days and cost tens of thousands of dollars.

Traditional maintenance strategies are either **reactive** (fix it when it breaks — expensive downtime) or **time-based** (replace on a schedule — wastes perfectly good parts). **Predictive maintenance** uses sensor data to estimate how much useful life remains, enabling repairs at exactly the right time.

This project takes raw vibration signals from accelerometers mounted on bearings and predicts how many minutes of operation remain before failure — the **Remaining Useful Life**.

## Architecture Overview

```
Raw Vibration CSVs (25.6 kHz, 2-channel)
    │
    ├─→ Feature Extraction (65 features)  ─→  LightGBM  ─→  RUL Prediction
    │
    ├─→ 1D Signal Windowing (32768×2)     ─→  1D CNN / TCN-Transformer  ─→  RUL
    │
    ├─→ STFT Spectrograms (128×128×2)     ─→  2D CNN / CNN-LSTM  ─→  RUL
    │
    └─→ Health Indicators (kurtosis, RMS)  ─→  Onset Detection  ─→  Two-Stage RUL
```

Three parallel input representations feed different model families — from gradient-boosted trees on hand-crafted features to deep learning on raw signals and spectrograms. A two-stage onset detection pipeline identifies the transition from healthy to degraded operation.

## Dataset

**XJTU-SY Bearing Dataset** — a widely-used benchmark for bearing prognostics research.

| Property | Value |
|---|---|
| Bearings | 15 (run-to-failure) |
| Operating conditions | 3 (35 Hz/12 kN, 37.5 Hz/11 kN, 40 Hz/10 kN) |
| Sampling rate | 25.6 kHz |
| Channels | 2 (horizontal + vertical vibration) |
| Total files | ~9,216 |

Each CSV file contains 32,768 samples (1.28 s recording) captured at regular intervals throughout a bearing's lifetime.

## Models

Five architectures are registered in the model registry (`src/models/registry.py`):

| Model | Input Type | Input Shape | Architecture |
|---|---|---|---|
| **LightGBM** | 65 features | tabular | Gradient-boosted trees (baseline) |
| **1D CNN** | Raw signal | 32768 × 2 | Conv1D → BatchNorm → GlobalAvgPool → Dense |
| **TCN-LSTM** | Raw signal | 32768 × 2 | Temporal conv → LSTM sequence head |
| **CNN2D Simple** | Spectrogram | 128 × 128 × 2 | 2D CNN with progressive downsampling |
| **CNN2D LSTM** | Spectrogram | 128 × 128 × 2 | 2D CNN encoder → LSTM sequence head |

## Results

### RUL Prediction

| Model | RMSE | MAE | Evaluation | Status |
|---|---|---|---|---|
| LightGBM | 22.43 | 14.84 | 15-fold LOBO CV | Complete |
| TCN-LSTM | 13.38 | 10.96 | Fold 0 | Complete |
| CNN2D Simple | 14.39 | 12.15 | Fold 0 | Complete |
| 1D CNN | 17.52 | 14.22 | Fold 0 | Complete |

> **Note:** Deep learning models show fold-0 results only (Bearing1_1, 123 samples). Full 15-fold leave-one-bearing-out evaluation is pending. LightGBM uses complete 15-fold CV across all bearings.

### Onset Detection

Two-stage pipeline that first identifies when degradation begins, then predicts RUL only in the degraded region.

| Component | Method | Performance |
|---|---|---|
| Statistical detectors | Kurtosis threshold, RMS threshold, Kurtosis CUSUM, RMS CUSUM | 15/15 bearings labeled |
| LSTM classifier | 8-feature z-score input (5,793 params) | F1 = 0.844 ± 0.243 (15-fold LOBO CV) |
| Manual labels | Expert-verified onset indices | 11 high, 2 medium, 2 low confidence |

## Interactive Dashboard

A Gradio dashboard (`app.py`) provides interactive visualization of all results:

```bash
uv run python app.py
# Opens at http://localhost:7860
```

**Tabs:**
- **EDA** — Degradation trends with onset markers, feature distributions
- **Model Results** — Cross-model comparison (LightGBM + DL), feature importance, training curves
- **Predictions** — Per-bearing RUL curves, residual analysis
- **Onset Detection** — Health indicator explorer, classifier performance, detector comparison
- **Audio Analysis** — Vibration signals sonified to audio (healthy → degrading → failed)

## Quick Start

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Run tests (459 tests)
uv run pytest tests/

# Launch the interactive dashboard
uv run python app.py

# Train a model (e.g., 1D CNN baseline, fold 0)
python scripts/05_train_dl_models.py --model cnn1d_baseline --folds 0

# Train with a specific config
python scripts/05_train_dl_models.py --model cnn2d_simple --config configs/cnn2d.yaml

# Evaluate trained models
python scripts/06_evaluate_dl_models.py

# Launch experiment tracking UI
bash scripts/mlflow_server.sh
```

## Project Structure

```
├── app.py                    # Gradio interactive dashboard
├── configs/                  # YAML training configurations
│   ├── cnn1d_baseline.yaml
│   ├── cnn2d.yaml
│   ├── onset_labels.yaml     # Manual onset labels (15 bearings)
│   ├── tcn_transformer.yaml
│   └── training_default.yaml
├── notebooks/                # EDA, training, and evaluation notebooks
│   ├── 01-03_eda_*.ipynb     #   Exploratory data analysis
│   ├── 20-24_model_*.ipynb   #   Model development
│   ├── 30_evaluation.ipynb   #   Cross-model comparison
│   └── 40-41_onset_*.ipynb   #   Onset detection evaluation
├── scripts/                  # Pipeline scripts (numbered by stage)
│   ├── 01_upload_to_gcs_with_hive_partitioning.py
│   ├── 02_preprocessing.py
│   ├── 03_extract_features.py
│   ├── 04_generate_spectrograms.py
│   ├── 05_train_dl_models.py
│   └── 06_evaluate_dl_models.py
├── src/
│   ├── data/                 # Data loading, windowing, augmentation, RUL labels
│   ├── features/             # 65-feature extraction (time + frequency domain)
│   ├── models/               # Model registry and architectures
│   │   ├── baselines/        #   LightGBM, 1D CNN
│   │   ├── pattern1/         #   TCN-Transformer variants
│   │   └── cnn2d/            #   Spectrogram-based (2D CNN, CNN-LSTM)
│   ├── onset/                # Degradation onset detection pipeline
│   │   ├── detectors.py      #   4 statistical detectors + ensemble
│   │   ├── health_indicators.py  # Kurtosis/RMS health indicator computation
│   │   ├── labels.py         #   Onset label loading and management
│   │   ├── models.py         #   LSTM onset classifier
│   │   └── pipeline.py       #   End-to-end detection pipeline
│   ├── training/             # Config, cross-validation, metrics
│   └── utils/                # Experiment tracking, helpers
├── tests/                    # pytest suite (459 tests)
└── pyproject.toml
```

## Experiment Tracking

Dual-backend setup for local development and cloud reproducibility:

- **MLflow** (local) — `bash scripts/mlflow_server.sh` launches the UI at `localhost:5000`
- **Vertex AI Experiments** (cloud) — automatic logging when running on GCP with `--tracking vertex`

## Tech Stack

- **ML Frameworks:** TensorFlow/Keras, LightGBM
- **Signal Processing:** SciPy, PyWavelets
- **Data:** Pandas, NumPy, BigQuery
- **Infrastructure:** GCS, Vertex AI, MLflow
- **Package Management:** uv
- **Visualization:** Plotly, Gradio (interactive dashboard)
