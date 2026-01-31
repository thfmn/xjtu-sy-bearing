# Experiment Tracking

This project uses a dual-backend experiment tracking system that logs training runs to both **MLflow** (local) and **Google Cloud Vertex AI Experiments** (cloud). The backends share a unified API via `ExperimentTracker` in `src/utils/tracking.py`.

## Architecture

```
Training Script (scripts/05_train_dl_models.py)
        |
        v
  ExperimentTracker
   /            \
  v              v
MLflowBackend   VertexBackend
  |               |
  v               v
mlruns/         Vertex AI Experiments
(local)         (GCP Console)
```

Each training fold produces one run per backend. The `ExperimentTracker` context manager handles run lifecycle (start, log, end) and graceful failure isolation — a Vertex AI error will not interrupt MLflow logging.

### What gets logged per run

| Data               | Example                              |
|--------------------|--------------------------------------|
| Parameters         | model_name, fold_id, batch_size, lr  |
| Per-epoch metrics  | loss, val_loss, mae, val_mae         |
| Final eval metrics | final_rmse, final_mae, phm08_score   |
| Artifacts          | Best `.keras` checkpoint, config YAML|

## MLflow (Local Development)

MLflow is the default backend. Runs are stored in the `mlruns/` directory at the project root.

### Start the MLflow UI

```bash
bash scripts/mlflow_server.sh          # http://localhost:5000
bash scripts/mlflow_server.sh 8080     # custom port
```

### Train with MLflow tracking (default)

```bash
python scripts/05_train_dl_models.py --model cnn1d_baseline --config configs/cnn1d_baseline.yaml
python scripts/05_train_dl_models.py --model cnn1d_baseline --tracking mlflow   # explicit
```

### Compare runs programmatically

```python
from src.utils.tracking import ExperimentTracker, compare_runs, plot_metric_comparison

tracker = ExperimentTracker(backend="mlflow", experiment_name="bearing_rul_dl", tracking_uri="mlruns")
runs = tracker.list_runs()
comparison = compare_runs(runs, metrics=["final_rmse", "final_mae"])
fig = plot_metric_comparison(runs, metric_name="final_rmse")
```

See `notebooks/04_experiment_comparison.ipynb` for a full walkthrough.

## Vertex AI Experiments (Cloud)

Vertex AI Experiments is the secondary backend for production runs on GCP.

### Prerequisites

1. GCP project with Vertex AI API enabled
2. Authenticated via `gcloud auth application-default login`
3. Set `vertex.enabled: true` in the model config YAML

### Config (per-model YAML)

```yaml
vertex:
  enabled: true
  experiment_name: bearing-rul-dl
  project_id: xjtu-bearing-failure
  location: asia-southeast3
```

### Train with both backends

```bash
python scripts/05_train_dl_models.py --model cnn1d_baseline --tracking both
```

### Train with Vertex AI only

```bash
python scripts/05_train_dl_models.py --model cnn1d_baseline --tracking vertex
```

### Disable all tracking (debug)

```bash
python scripts/05_train_dl_models.py --model cnn1d_baseline --tracking none
```

## CLI `--tracking` flag

| Value    | MLflow | Vertex AI | Use case                |
|----------|--------|-----------|-------------------------|
| `mlflow` | Yes    | No        | Local dev (default)     |
| `vertex` | No     | Yes       | Cloud-only              |
| `both`   | Yes    | Yes       | Production dual-logging |
| `none`   | No     | No        | Fast debug runs         |

## Cost

| Backend            | Storage         | Compute  | Monthly cost |
|--------------------|-----------------|----------|--------------|
| MLflow (local)     | Local disk      | None     | $0           |
| Vertex AI Metadata | GCP managed     | None     | ~$0*         |
| Vertex AI Artifacts| GCS bucket      | None     | ~$0.02/GB    |

*Vertex AI Experiments metadata storage is free for typical experiment volumes. GCS artifact storage is the only marginal cost.
