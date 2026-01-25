"""Training configuration module for bearing RUL prediction.

This module provides configuration dataclasses and Keras callbacks
for model training, including experiment tracking via MLflow.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal
import json
import yaml

import tensorflow as tf
from tensorflow import keras


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: Literal["adam", "adamw", "sgd"] = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # For AdamW
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    momentum: float = 0.9  # For SGD

    def build(self) -> keras.optimizers.Optimizer:
        """Build Keras optimizer from config."""
        if self.name == "adamw":
            return keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
            )
        elif self.name == "adam":
            return keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
            )
        elif self.name == "sgd":
            return keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.name}")


@dataclass
class LossConfig:
    """Loss function configuration."""

    name: Literal["huber", "mse", "mae"] = "huber"
    delta: float = 10.0  # For Huber loss

    def build(self) -> keras.losses.Loss:
        """Build Keras loss from config."""
        if self.name == "huber":
            return keras.losses.Huber(delta=self.delta)
        elif self.name == "mse":
            return keras.losses.MeanSquaredError()
        elif self.name == "mae":
            return keras.losses.MeanAbsoluteError()
        else:
            raise ValueError(f"Unknown loss: {self.name}")


@dataclass
class CallbackConfig:
    """Callback configuration."""

    # ModelCheckpoint
    checkpoint_dir: str = "outputs/models/checkpoints"
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"

    # ReduceLROnPlateau
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 10
    lr_min: float = 1e-6

    # EarlyStopping
    early_stop_patience: int = 20
    restore_best_weights: bool = True

    # MLflow
    mlflow_tracking: bool = True
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "bearing_rul_prediction"


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Basic training params
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    shuffle: bool = True
    verbose: int = 1

    # Random seed for reproducibility
    seed: int = 42

    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)

    def __post_init__(self):
        """Convert dicts to dataclasses if loaded from YAML/JSON."""
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)
        if isinstance(self.callbacks, dict):
            self.callbacks = CallbackConfig(**self.callbacks)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        import numpy as np
        import random

        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)


class MLflowCallback(keras.callbacks.Callback):
    """Custom Keras callback for MLflow experiment tracking.

    Logs metrics, parameters, and artifacts to MLflow.
    """

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "bearing_rul_prediction",
        run_name: str | None = None,
        log_models: bool = True,
    ):
        """Initialize MLflow callback.

        Args:
            tracking_uri: MLflow tracking server URI or local path.
            experiment_name: Name of the MLflow experiment.
            run_name: Optional name for this run.
            log_models: Whether to log the model artifact.
        """
        super().__init__()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_models = log_models
        self._mlflow = None
        self._run = None

    def _setup_mlflow(self):
        """Lazy import and setup MLflow."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
            except ImportError:
                raise ImportError(
                    "mlflow is required for MLflowCallback. "
                    "Install it with: pip install mlflow"
                )

    def on_train_begin(self, logs=None):
        """Start MLflow run at training start."""
        self._setup_mlflow()
        self._run = self._mlflow.start_run(run_name=self.run_name)

        # Log model architecture summary
        if hasattr(self.model, "count_params"):
            self._mlflow.log_param("total_params", self.model.count_params())

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch."""
        if logs is None:
            return

        for key, value in logs.items():
            self._mlflow.log_metric(key, value, step=epoch)

    def on_train_end(self, logs=None):
        """End MLflow run and optionally log model."""
        if self.log_models and self.model is not None:
            # Log the model as an artifact
            self._mlflow.keras.log_model(self.model, "model")

        self._mlflow.end_run()


def build_callbacks(
    config: TrainingConfig,
    model_name: str = "model",
) -> list[keras.callbacks.Callback]:
    """Build list of Keras callbacks from configuration.

    Args:
        config: Training configuration.
        model_name: Name for model checkpoint files.

    Returns:
        List of configured Keras callbacks.
    """
    cb_config = config.callbacks
    callbacks = []

    # ModelCheckpoint - save best model
    checkpoint_path = Path(cb_config.checkpoint_dir) / f"{model_name}.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=cb_config.monitor,
            mode=cb_config.mode,
            save_best_only=cb_config.save_best_only,
            verbose=1,
        )
    )

    # ReduceLROnPlateau - reduce learning rate when validation loss plateaus
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=cb_config.monitor,
            factor=cb_config.lr_reduce_factor,
            patience=cb_config.lr_reduce_patience,
            min_lr=cb_config.lr_min,
            verbose=1,
        )
    )

    # EarlyStopping - stop training when validation loss stops improving
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=cb_config.monitor,
            patience=cb_config.early_stop_patience,
            restore_best_weights=cb_config.restore_best_weights,
            verbose=1,
        )
    )

    # MLflow tracking (optional)
    if cb_config.mlflow_tracking:
        callbacks.append(
            MLflowCallback(
                tracking_uri=cb_config.mlflow_tracking_uri,
                experiment_name=cb_config.mlflow_experiment_name,
                run_name=model_name,
            )
        )

    return callbacks


def compile_model(
    model: keras.Model,
    config: TrainingConfig,
    metrics: list[str] | None = None,
) -> keras.Model:
    """Compile a Keras model with the given configuration.

    Args:
        model: Keras model to compile.
        config: Training configuration.
        metrics: Optional list of metric names. Defaults to ["mae"].

    Returns:
        Compiled Keras model.
    """
    if metrics is None:
        metrics = ["mae"]

    model.compile(
        optimizer=config.optimizer.build(),
        loss=config.loss.build(),
        metrics=metrics,
    )

    return model


# Default configuration instance
DEFAULT_CONFIG = TrainingConfig()
