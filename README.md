# XJTU-SY Bearing RUL Prediction

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-89%20passed-brightgreen)
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
    └─→ STFT Spectrograms (128×128×2)     ─→  2D CNN / CNN-LSTM  ─→  RUL
```

Three parallel input representations feed different model families — from gradient-boosted trees on hand-crafted features to deep learning on raw signals and spectrograms.

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
| **TCN-Transformer** | Raw signal | 32768 × 2 | Temporal conv + multi-head attention |
| **Pattern2 Simple** | Spectrogram | 128 × 128 × 2 | 2D CNN with progressive downsampling |
| **Pattern2 LSTM** | Spectrogram | 128 × 128 × 2 | 2D CNN encoder → LSTM sequence head |

## Results

| Model | RMSE | MAE | Evaluation | Status |
|---|---|---|---|---|
| LightGBM | 22.43 | 14.84 | 15-fold CV | ✅ Complete |
| 1D CNN | 17.52 | 14.22 | Fold 0 | ✅ Complete |
| Pattern2 2D CNN | 14.39 | 12.15 | Fold 0 | ✅ Complete |
| TCN-Transformer | — | — | — | 🔄 Training |

> **Note:** Deep learning models show fold-0 results only; full 15-fold leave-one-bearing-out evaluation is pending. RMSE/MAE are in percentage of total lifetime.

## Quick Start

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Run tests
uv run pytest tests/

# Train a model (e.g., 1D CNN baseline, fold 0)
python scripts/05_train_dl_models.py --model cnn1d_baseline --folds 0

# Train with a specific config
python scripts/05_train_dl_models.py --model pattern2_simple --config configs/pattern2_cnn2d.yaml

# Evaluate trained models
python scripts/06_evaluate_dl_models.py

# Launch experiment tracking UI
bash scripts/mlflow_server.sh
```

## Project Structure

```
├── configs/                  # YAML training configurations
│   ├── cnn1d_baseline.yaml
│   ├── pattern2_cnn2d.yaml
│   ├── tcn_transformer.yaml
│   └── training_default.yaml
├── notebooks/                # EDA, training, and evaluation notebooks
│   ├── 01-03_eda_*.ipynb     #   Exploratory data analysis
│   ├── 20-24_model_*.ipynb   #   Model development
│   └── 30_evaluation.ipynb   #   Cross-model comparison
├── scripts/                  # Pipeline scripts (numbered by stage)
│   ├── 01_upload_to_gcs_with_hive_partitioning.py
│   ├── 02_preprocessing.py
│   ├── 03_extract_features.py
│   ├── 04_generate_spectrograms.py
│   ├── 05_train_dl_models.py
│   └── 06_evaluate_dl_models.py
├── src/
│   ├── data/                 # Data loading, windowing, RUL labels
│   ├── features/             # 65-feature extraction (time + frequency domain)
│   ├── models/               # Model registry and architectures
│   │   ├── baselines/        #   LightGBM, 1D CNN
│   │   ├── pattern1/         #   TCN-Transformer variants
│   │   └── pattern2/         #   Spectrogram-based (2D CNN, CNN-LSTM)
│   ├── training/             # Config, cross-validation, metrics
│   └── utils/                # Experiment tracking, helpers
├── tests/                    # pytest suite (89 tests)
└── pyproject.toml
```

## Experiment Tracking

Dual-backend setup for local development and cloud reproducibility:

- **MLflow** (local) — `bash scripts/mlflow_server.sh` launches the UI at `localhost:5000`
- **Vertex AI Experiments** (cloud) — automatic logging when running on GCP with `--tracking vertex`

## Tech Stack

- **ML Frameworks:** TensorFlow/Keras, PyTorch, LightGBM
- **Signal Processing:** SciPy, PyWavelets
- **Data:** Pandas, NumPy, BigQuery
- **Infrastructure:** GCS, Vertex AI, MLflow
- **Package Management:** uv
- **Visualization:** Seaborn, Plotly, Gradio (demo UI)
