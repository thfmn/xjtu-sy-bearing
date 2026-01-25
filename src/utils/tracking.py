"""Unified experiment tracking module for MLflow and Vertex Experiments.

This module provides a unified interface for experiment tracking that works
with both MLflow (local/development) and Google Cloud Vertex AI Experiments
(production). It supports logging parameters, metrics, artifacts, and model
registry integration.

Usage:
    # Local development with MLflow
    tracker = ExperimentTracker(backend="mlflow", experiment_name="my_experiment")

    # Production with Vertex AI
    tracker = ExperimentTracker(
        backend="vertex",
        experiment_name="my_experiment",
        project_id="my-gcp-project",
        location="us-central1"
    )

    # Context manager usage
    with tracker.start_run(run_name="run_001") as run:
        run.log_params({"learning_rate": 0.001, "batch_size": 32})
        run.log_metrics({"loss": 0.5, "mae": 10.0}, step=0)
        run.log_artifact("outputs/model.keras")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from contextlib import contextmanager
import json
import os


@dataclass
class RunInfo:
    """Information about an experiment run."""

    run_id: str
    run_name: str | None
    experiment_name: str
    status: str
    start_time: str | None = None
    end_time: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


class TrackingBackend(ABC):
    """Abstract base class for experiment tracking backends."""

    @abstractmethod
    def set_experiment(self, experiment_name: str) -> None:
        """Set or create an experiment."""
        pass

    @abstractmethod
    def start_run(self, run_name: str | None = None) -> "RunContext":
        """Start a new run and return a context for logging."""
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for the current run."""
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None
    ) -> None:
        """Log metrics for the current run."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file or directory as an artifact."""
        pass

    @abstractmethod
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None
    ) -> None:
        """Log a model artifact and optionally register it."""
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        pass

    @abstractmethod
    def get_run(self, run_id: str) -> RunInfo:
        """Get information about a specific run."""
        pass

    @abstractmethod
    def list_runs(
        self,
        experiment_name: str | None = None,
        filter_string: str | None = None,
        max_results: int = 100
    ) -> list[RunInfo]:
        """List runs, optionally filtered."""
        pass


class RunContext:
    """Context manager for a single experiment run."""

    def __init__(self, backend: TrackingBackend, run_name: str | None = None):
        self.backend = backend
        self.run_name = run_name
        self._run_id: str | None = None

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.backend.end_run(status=status)
        return False

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""
        self.backend.log_params(params)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.backend.log_params({key: value})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        self.backend.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        self.backend.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact."""
        self.backend.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None
    ) -> None:
        """Log a model."""
        self.backend.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )


class MLflowBackend(TrackingBackend):
    """MLflow tracking backend for local/development use."""

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        registry_uri: str | None = None
    ):
        """Initialize MLflow backend.

        Args:
            tracking_uri: MLflow tracking server URI or local path.
            registry_uri: Optional model registry URI.
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self._mlflow = None
        self._active_run = None
        self._experiment_name: str | None = None

    def _setup(self):
        """Lazy import and setup MLflow."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                if self.registry_uri:
                    mlflow.set_registry_uri(self.registry_uri)
            except ImportError:
                raise ImportError(
                    "mlflow is required for MLflowBackend. "
                    "Install it with: uv add mlflow"
                )

    def set_experiment(self, experiment_name: str) -> None:
        """Set or create an MLflow experiment."""
        self._setup()
        self._mlflow.set_experiment(experiment_name)
        self._experiment_name = experiment_name

    def start_run(self, run_name: str | None = None) -> RunContext:
        """Start a new MLflow run."""
        self._setup()
        self._active_run = self._mlflow.start_run(run_name=run_name)
        return RunContext(self, run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        self._setup()
        self._mlflow.log_params(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None
    ) -> None:
        """Log metrics to MLflow."""
        self._setup()
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact to MLflow."""
        self._setup()
        self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None
    ) -> None:
        """Log a Keras model to MLflow."""
        self._setup()
        # Use mlflow.keras for Keras models
        self._mlflow.keras.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        self._setup()
        self._mlflow.end_run(status=status)
        self._active_run = None

    def get_run(self, run_id: str) -> RunInfo:
        """Get information about a specific MLflow run."""
        self._setup()
        run = self._mlflow.get_run(run_id)
        return RunInfo(
            run_id=run.info.run_id,
            run_name=run.info.run_name,
            experiment_name=self._experiment_name or "",
            status=run.info.status,
            start_time=str(run.info.start_time) if run.info.start_time else None,
            end_time=str(run.info.end_time) if run.info.end_time else None,
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
            artifacts=[],  # Would need to query artifact store
        )

    def list_runs(
        self,
        experiment_name: str | None = None,
        filter_string: str | None = None,
        max_results: int = 100
    ) -> list[RunInfo]:
        """List MLflow runs."""
        self._setup()

        exp_name = experiment_name or self._experiment_name
        if exp_name is None:
            raise ValueError("No experiment name specified")

        experiment = self._mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            return []

        runs = self._mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string or "",
            max_results=max_results
        )

        result = []
        for _, row in runs.iterrows():
            run_info = RunInfo(
                run_id=row["run_id"],
                run_name=row.get("tags.mlflow.runName", ""),
                experiment_name=exp_name,
                status=row["status"],
                start_time=str(row["start_time"]) if row["start_time"] else None,
                end_time=str(row["end_time"]) if row["end_time"] else None,
                metrics={k.replace("metrics.", ""): v for k, v in row.items()
                        if k.startswith("metrics.")},
                params={k.replace("params.", ""): v for k, v in row.items()
                       if k.startswith("params.")},
            )
            result.append(run_info)

        return result

    def get_tracking_uri(self) -> str:
        """Get the MLflow tracking URI."""
        self._setup()
        return self._mlflow.get_tracking_uri()

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: dict[str, str] | None = None
    ) -> Any:
        """Register a model in the MLflow model registry."""
        self._setup()
        result = self._mlflow.register_model(model_uri, name)
        if tags:
            client = self._mlflow.MlflowClient()
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=name,
                    version=result.version,
                    key=key,
                    value=value
                )
        return result


class VertexBackend(TrackingBackend):
    """Google Cloud Vertex AI Experiments backend for production use."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "us-central1",
        staging_bucket: str | None = None,
        credentials: Any | None = None
    ):
        """Initialize Vertex Experiments backend.

        Args:
            project_id: GCP project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
            location: GCP region for Vertex AI.
            staging_bucket: GCS bucket for staging artifacts.
            credentials: Optional GCP credentials.
        """
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.staging_bucket = staging_bucket
        self.credentials = credentials
        self._aiplatform = None
        self._experiment = None
        self._experiment_name: str | None = None
        self._active_run = None

    def _setup(self):
        """Lazy import and setup Vertex AI."""
        if self._aiplatform is None:
            try:
                from google.cloud import aiplatform
                self._aiplatform = aiplatform
                aiplatform.init(
                    project=self.project_id,
                    location=self.location,
                    staging_bucket=self.staging_bucket,
                    credentials=self.credentials
                )
            except ImportError:
                raise ImportError(
                    "google-cloud-aiplatform is required for VertexBackend. "
                    "Install it with: uv add google-cloud-aiplatform"
                )

    def set_experiment(self, experiment_name: str) -> None:
        """Set or create a Vertex AI Experiment."""
        self._setup()
        self._experiment_name = experiment_name

        # Create experiment if it doesn't exist
        try:
            self._experiment = self._aiplatform.Experiment.get_or_create(
                experiment_name=experiment_name,
                project=self.project_id,
                location=self.location
            )
        except Exception:
            # Experiment.get_or_create may not be available in all versions
            # Fall back to trying to get, then create
            try:
                self._experiment = self._aiplatform.Experiment(
                    experiment_name=experiment_name,
                    project=self.project_id,
                    location=self.location
                )
            except Exception:
                self._experiment = self._aiplatform.Experiment.create(
                    experiment_name=experiment_name,
                    project=self.project_id,
                    location=self.location
                )

    def start_run(self, run_name: str | None = None) -> RunContext:
        """Start a new Vertex AI Experiment run."""
        self._setup()
        if self._experiment is None:
            raise ValueError("No experiment set. Call set_experiment() first.")

        self._active_run = self._aiplatform.ExperimentRun.create(
            run_name=run_name or f"run-{self._generate_run_id()}",
            experiment=self._experiment_name,
            project=self.project_id,
            location=self.location
        )
        return RunContext(self, run_name=run_name)

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        import datetime
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to Vertex AI."""
        self._setup()
        if self._active_run is None:
            raise ValueError("No active run. Call start_run() first.")
        self._active_run.log_params(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None
    ) -> None:
        """Log metrics to Vertex AI."""
        self._setup()
        if self._active_run is None:
            raise ValueError("No active run. Call start_run() first.")

        # Vertex AI doesn't support step in the same way as MLflow
        # We include it as a metric if provided
        if step is not None:
            metrics = {**metrics, "_step": float(step)}

        self._active_run.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact to GCS via Vertex AI."""
        self._setup()
        if self._active_run is None:
            raise ValueError("No active run. Call start_run() first.")

        # Vertex AI handles artifacts differently
        # We need to upload to GCS and log the URI
        from google.cloud import storage

        if self.staging_bucket is None:
            raise ValueError("staging_bucket required for artifact logging")

        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact not found: {local_path}")

        # Upload to GCS
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.staging_bucket.replace("gs://", ""))

        gcs_path = artifact_path or local_path.name
        blob = bucket.blob(f"experiments/{self._experiment_name}/{gcs_path}")

        if local_path.is_file():
            blob.upload_from_filename(str(local_path))
        else:
            # Upload directory recursively
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(local_path)
                    file_blob = bucket.blob(
                        f"experiments/{self._experiment_name}/{gcs_path}/{rel_path}"
                    )
                    file_blob.upload_from_filename(str(file_path))

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None
    ) -> None:
        """Log a Keras model to Vertex AI."""
        self._setup()

        # Save model locally first
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / f"{artifact_path}.keras"
            model.save(model_path)

            # Upload to GCS
            self.log_artifact(str(model_path), artifact_path=artifact_path)

            # Register in Vertex Model Registry if name provided
            if registered_model_name:
                self._register_model(
                    model_path=f"gs://{self.staging_bucket}/experiments/{self._experiment_name}/{artifact_path}",
                    model_name=registered_model_name
                )

    def _register_model(self, model_path: str, model_name: str) -> Any:
        """Register a model in Vertex Model Registry."""
        self._setup()
        return self._aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=model_path,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
        )

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current Vertex AI run."""
        self._setup()
        if self._active_run is not None:
            self._active_run.end_run(status.lower())
            self._active_run = None

    def get_run(self, run_id: str) -> RunInfo:
        """Get information about a specific Vertex AI run."""
        self._setup()
        run = self._aiplatform.ExperimentRun(
            run_name=run_id,
            experiment=self._experiment_name,
            project=self.project_id,
            location=self.location
        )

        return RunInfo(
            run_id=run_id,
            run_name=run_id,
            experiment_name=self._experiment_name or "",
            status=run.state.name if hasattr(run, "state") else "UNKNOWN",
            metrics=dict(run.get_metrics()) if hasattr(run, "get_metrics") else {},
            params=dict(run.get_params()) if hasattr(run, "get_params") else {},
        )

    def list_runs(
        self,
        experiment_name: str | None = None,
        filter_string: str | None = None,
        max_results: int = 100
    ) -> list[RunInfo]:
        """List Vertex AI Experiment runs."""
        self._setup()

        exp_name = experiment_name or self._experiment_name
        if exp_name is None:
            raise ValueError("No experiment name specified")

        experiment = self._aiplatform.Experiment(
            experiment_name=exp_name,
            project=self.project_id,
            location=self.location
        )

        runs = experiment.get_data_frame()
        if runs is None or runs.empty:
            return []

        result = []
        for _, row in runs.head(max_results).iterrows():
            run_info = RunInfo(
                run_id=row.get("run_name", ""),
                run_name=row.get("run_name", ""),
                experiment_name=exp_name,
                status=row.get("state", "UNKNOWN"),
                metrics={k: v for k, v in row.items() if k.startswith("metric.")},
                params={k: v for k, v in row.items() if k.startswith("param.")},
            )
            result.append(run_info)

        return result


class ExperimentTracker:
    """Unified experiment tracker supporting multiple backends.

    This class provides a high-level interface for experiment tracking
    that works with both MLflow and Vertex AI Experiments.

    Example:
        >>> tracker = ExperimentTracker(backend="mlflow")
        >>> tracker.set_experiment("my_experiment")
        >>> with tracker.start_run("run_001") as run:
        ...     run.log_params({"lr": 0.001})
        ...     run.log_metrics({"loss": 0.5})
    """

    def __init__(
        self,
        backend: Literal["mlflow", "vertex"] = "mlflow",
        experiment_name: str | None = None,
        **backend_kwargs
    ):
        """Initialize experiment tracker.

        Args:
            backend: Tracking backend to use ("mlflow" or "vertex").
            experiment_name: Optional experiment name to set immediately.
            **backend_kwargs: Additional arguments passed to the backend.
        """
        self.backend_name = backend

        if backend == "mlflow":
            self._backend: TrackingBackend = MLflowBackend(**backend_kwargs)
        elif backend == "vertex":
            self._backend = VertexBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if experiment_name:
            self.set_experiment(experiment_name)

    def set_experiment(self, experiment_name: str) -> None:
        """Set the current experiment."""
        self._backend.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str | None = None):
        """Start a new run as a context manager.

        Example:
            >>> with tracker.start_run("my_run") as run:
            ...     run.log_metrics({"loss": 0.5})
        """
        run_context = self._backend.start_run(run_name=run_name)
        try:
            yield run_context
        finally:
            pass  # RunContext.__exit__ handles end_run

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the current run."""
        self._backend.log_params(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None
    ) -> None:
        """Log metrics to the current run."""
        self._backend.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact to the current run."""
        self._backend.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None
    ) -> None:
        """Log a model to the current run."""
        self._backend.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        self._backend.end_run(status=status)

    def get_run(self, run_id: str) -> RunInfo:
        """Get information about a specific run."""
        return self._backend.get_run(run_id)

    def list_runs(
        self,
        experiment_name: str | None = None,
        filter_string: str | None = None,
        max_results: int = 100
    ) -> list[RunInfo]:
        """List runs in the experiment."""
        return self._backend.list_runs(
            experiment_name=experiment_name,
            filter_string=filter_string,
            max_results=max_results
        )


# ============================================================================
# Experiment Comparison Utilities
# ============================================================================

def compare_runs(
    runs: list[RunInfo],
    metrics: list[str] | None = None,
    params: list[str] | None = None
) -> dict[str, Any]:
    """Compare multiple runs by their metrics and parameters.

    Args:
        runs: List of RunInfo objects to compare.
        metrics: Optional list of metric names to include.
        params: Optional list of parameter names to include.

    Returns:
        Dictionary with comparison data including:
        - 'summary': DataFrame-like dict with all runs
        - 'best_run': Run with best primary metric
        - 'metric_stats': Statistics for each metric
    """
    import pandas as pd

    if not runs:
        return {"summary": {}, "best_run": None, "metric_stats": {}}

    # Build comparison DataFrame
    data = []
    for run in runs:
        row = {
            "run_id": run.run_id,
            "run_name": run.run_name,
            "status": run.status,
        }

        # Add metrics
        for key, value in run.metrics.items():
            if metrics is None or key in metrics:
                row[f"metric_{key}"] = value

        # Add params
        for key, value in run.params.items():
            if params is None or key in params:
                row[f"param_{key}"] = value

        data.append(row)

    df = pd.DataFrame(data)

    # Calculate metric statistics
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    metric_stats = {}
    for col in metric_cols:
        metric_name = col.replace("metric_", "")
        numeric_values = pd.to_numeric(df[col], errors="coerce")
        metric_stats[metric_name] = {
            "mean": numeric_values.mean(),
            "std": numeric_values.std(),
            "min": numeric_values.min(),
            "max": numeric_values.max(),
            "best_run_id": df.loc[numeric_values.idxmin(), "run_id"]
            if not numeric_values.isna().all() else None
        }

    # Find best run (by first metric, assuming lower is better)
    best_run = None
    if metric_cols:
        first_metric = metric_cols[0]
        numeric_values = pd.to_numeric(df[first_metric], errors="coerce")
        if not numeric_values.isna().all():
            best_idx = numeric_values.idxmin()
            best_run = runs[best_idx]

    return {
        "summary": df.to_dict(orient="records"),
        "best_run": best_run,
        "metric_stats": metric_stats,
        "dataframe": df  # For further analysis
    }


def plot_metric_comparison(
    runs: list[RunInfo],
    metric_name: str,
    figsize: tuple[int, int] = (10, 6)
) -> Any:
    """Plot a comparison of a specific metric across runs.

    Args:
        runs: List of RunInfo objects.
        metric_name: Name of the metric to plot.
        figsize: Figure size tuple.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    run_names = []
    metric_values = []

    for run in runs:
        if metric_name in run.metrics:
            run_names.append(run.run_name or run.run_id[:8])
            metric_values.append(run.metrics[metric_name])

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(run_names, metric_values)

    # Highlight best (lowest) value
    if metric_values:
        best_idx = metric_values.index(min(metric_values))
        bars[best_idx].set_color("green")

    ax.set_xlabel("Run")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Comparison of {metric_name} across runs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def export_comparison_report(
    runs: list[RunInfo],
    output_path: str | Path,
    format: Literal["json", "csv", "html"] = "html"
) -> None:
    """Export a comparison report of runs to a file.

    Args:
        runs: List of RunInfo objects.
        output_path: Path to save the report.
        format: Output format.
    """
    import pandas as pd

    comparison = compare_runs(runs)
    df = comparison["dataframe"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "html":
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .best {{ background-color: #90EE90; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Experiment Comparison Report</h1>
    <h2>Runs Summary</h2>
    {df.to_html(classes='dataframe', index=False)}
    <h2>Metric Statistics</h2>
    <pre>{json.dumps(comparison['metric_stats'], indent=2)}</pre>
</body>
</html>
"""
        with open(output_path, "w") as f:
            f.write(html)


# ============================================================================
# Keras Callback Integration
# ============================================================================

class UnifiedTrackingCallback:
    """Factory for creating Keras callbacks that use the unified tracking API.

    This replaces the MLflowCallback in config.py with a more flexible version
    that works with any tracking backend.
    """

    @staticmethod
    def create(
        tracker: ExperimentTracker,
        run_name: str | None = None,
        log_models: bool = True
    ) -> "KerasTrackingCallback":
        """Create a Keras callback for the given tracker.

        Args:
            tracker: ExperimentTracker instance.
            run_name: Name for the experiment run.
            log_models: Whether to log the model at training end.

        Returns:
            Keras callback instance.
        """
        return KerasTrackingCallback(
            tracker=tracker,
            run_name=run_name,
            log_models=log_models
        )


# Need to import keras here for the callback
from tensorflow import keras


class KerasTrackingCallback(keras.callbacks.Callback):
    """Keras callback for unified experiment tracking.

    Works with any ExperimentTracker backend (MLflow or Vertex).
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        run_name: str | None = None,
        log_models: bool = True
    ):
        super().__init__()
        self.tracker = tracker
        self.run_name = run_name
        self.log_models = log_models
        self._run_context: RunContext | None = None

    def on_train_begin(self, logs=None):
        """Start tracking run at training start."""
        self._run_context = self.tracker._backend.start_run(run_name=self.run_name)

        # Log model architecture info
        if hasattr(self.model, "count_params"):
            self.tracker.log_params({
                "total_params": self.model.count_params(),
                "model_name": self.model.name if hasattr(self.model, "name") else "model"
            })

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch."""
        if logs:
            self.tracker.log_metrics(logs, step=epoch)

    def on_train_end(self, logs=None):
        """End tracking run and optionally log model."""
        if self.log_models and self.model is not None:
            self.tracker.log_model(self.model, artifact_path="model")

        self.tracker.end_run()
        self._run_context = None


# ============================================================================
# Convenience Functions
# ============================================================================

def create_mlflow_tracker(
    experiment_name: str,
    tracking_uri: str = "mlruns"
) -> ExperimentTracker:
    """Create an MLflow-backed experiment tracker.

    Args:
        experiment_name: Name of the experiment.
        tracking_uri: MLflow tracking URI.

    Returns:
        Configured ExperimentTracker.
    """
    return ExperimentTracker(
        backend="mlflow",
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )


def create_vertex_tracker(
    experiment_name: str,
    project_id: str | None = None,
    location: str = "us-central1",
    staging_bucket: str | None = None
) -> ExperimentTracker:
    """Create a Vertex AI Experiments-backed tracker.

    Args:
        experiment_name: Name of the experiment.
        project_id: GCP project ID.
        location: GCP region.
        staging_bucket: GCS bucket for artifacts.

    Returns:
        Configured ExperimentTracker.
    """
    return ExperimentTracker(
        backend="vertex",
        experiment_name=experiment_name,
        project_id=project_id,
        location=location,
        staging_bucket=staging_bucket
    )
