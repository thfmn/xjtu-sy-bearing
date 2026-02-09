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

"""Simple 1D CNN Baseline for RUL prediction.

This module provides a lightweight 1D CNN architecture for bearing RUL
prediction using raw vibration signals. Based on the pattern from
00_scratchpad.ipynb, adapted for regression.

Architecture:
    Input -> [Conv1D -> BatchNorm -> ReLU] x 3 -> GlobalAvgPool -> Dense -> RUL

Input shape: (batch_size, 32768, 2) - raw signals with H/V channels
Output shape: (batch_size, 1) - RUL prediction
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Constants matching the data loader
INPUT_LENGTH = 32768  # Samples per file
NUM_CHANNELS = 2  # Horizontal + Vertical


@dataclass
class CNN1DConfig:
    """Configuration for 1D CNN baseline model.

    Attributes:
        num_conv_layers: Number of convolutional blocks (default: 3).
        filters: Number of filters per conv layer (default: 64).
        kernel_size: Kernel size for Conv1D layers (default: 3).
        activation: Activation function (default: "relu").
        use_batch_norm: Whether to use batch normalization (default: True).
        dropout_rate: Dropout rate after global pooling (default: 0.0).
        dense_units: Units in optional dense layer before output (default: None).
    """

    num_conv_layers: int = 3
    filters: int = 64
    kernel_size: int = 3
    activation: Literal["relu", "gelu", "swish"] = "relu"
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    dense_units: int | None = None  # Optional dense layer before output

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


def build_cnn1d_model(
    config: CNN1DConfig | None = None,
    input_shape: tuple[int, int] = (INPUT_LENGTH, NUM_CHANNELS),
    name: str = "cnn1d_rul",
) -> keras.Model:
    """Build a simple 1D CNN model for RUL prediction.

    Architecture follows the pattern from 00_scratchpad.ipynb:
    - 3 convolutional blocks with BatchNorm and ReLU
    - GlobalAveragePooling1D for temporal aggregation
    - Single Dense output for RUL regression

    Args:
        config: Model configuration. Uses defaults if None.
        input_shape: Shape of input signals (length, channels).
        name: Model name.

    Returns:
        Uncompiled Keras model.
    """
    if config is None:
        config = CNN1DConfig()

    # Input layer
    inputs = layers.Input(shape=input_shape, name="input_signal")
    x = inputs

    # Convolutional blocks
    for i in range(config.num_conv_layers):
        x = layers.Conv1D(
            filters=config.filters,
            kernel_size=config.kernel_size,
            padding="same",
            name=f"conv1d_{i + 1}",
        )(x)

        if config.use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)

        if config.activation == "relu":
            x = layers.ReLU(name=f"relu_{i + 1}")(x)
        elif config.activation == "gelu":
            x = layers.Activation("gelu", name=f"gelu_{i + 1}")(x)
        elif config.activation == "swish":
            x = layers.Activation("swish", name=f"swish_{i + 1}")(x)

    # Global pooling to aggregate temporal information
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Optional dropout
    if config.dropout_rate > 0:
        x = layers.Dropout(config.dropout_rate, name="dropout")(x)

    # Optional dense layer before output
    if config.dense_units is not None:
        x = layers.Dense(config.dense_units, activation="relu", name="dense")(x)

    # RUL output - linear activation; clipping to non-negative is done post-prediction
    outputs = layers.Dense(1, name="rul_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def get_model_summary(model: keras.Model) -> dict[str, Any]:
    """Get model summary statistics.

    Args:
        model: Keras model.

    Returns:
        Dictionary with model statistics.
    """
    total_params = model.count_params()
    trainable_params = sum(
        int(np.prod(w.shape)) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    return {
        "name": model.name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "num_layers": len(model.layers),
    }


def print_model_summary(model: keras.Model) -> None:
    """Print formatted model summary.

    Args:
        model: Keras model.
    """
    summary = get_model_summary(model)
    print(f"\n{'=' * 60}")
    print(f"Model: {summary['name']}")
    print(f"{'=' * 60}")
    print(f"Input shape:  {summary['input_shape']}")
    print(f"Output shape: {summary['output_shape']}")
    print(f"{'=' * 60}")
    print(f"Total params:         {summary['total_params']:,}")
    print(f"Trainable params:     {summary['trainable_params']:,}")
    print(f"Non-trainable params: {summary['non_trainable_params']:,}")
    print(f"Number of layers:     {summary['num_layers']}")
    print(f"{'=' * 60}\n")


class CNN1DBaseline:
    """Simple 1D CNN baseline for RUL prediction.

    This class wraps the functional model with training utilities,
    making it easier to use with the existing training infrastructure.

    Example:
        >>> from src.models.baselines import CNN1DBaseline
        >>> from src.training import TrainingConfig, compile_model
        >>>
        >>> baseline = CNN1DBaseline()
        >>> model = baseline.model
        >>> config = TrainingConfig()
        >>> compile_model(model, config)
        >>>
        >>> # Train with raw signals
        >>> model.fit(x_train, y_train, ...)
    """

    def __init__(
        self,
        config: CNN1DConfig | None = None,
        input_shape: tuple[int, int] = (INPUT_LENGTH, NUM_CHANNELS),
    ):
        """Initialize CNN1D baseline.

        Args:
            config: Model configuration.
            input_shape: Shape of input signals.
        """
        self.config = config or CNN1DConfig()
        self.input_shape = input_shape
        self._model: keras.Model | None = None

    @property
    def model(self) -> keras.Model:
        """Get or create the Keras model."""
        if self._model is None:
            self._model = build_cnn1d_model(
                config=self.config,
                input_shape=self.input_shape,
            )
        return self._model

    def summary(self) -> dict[str, Any]:
        """Get model summary statistics."""
        return get_model_summary(self.model)

    def print_summary(self) -> None:
        """Print formatted model summary."""
        print_model_summary(self.model)

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model (.keras format).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "CNN1DBaseline":
        """Load model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            CNN1DBaseline instance with loaded model.
        """
        path = Path(path)
        model = keras.models.load_model(str(path))

        # Reconstruct baseline wrapper
        baseline = cls.__new__(cls)
        baseline.config = CNN1DConfig()  # Default config (actual params in model)
        baseline.input_shape = model.input_shape[1:]
        baseline._model = model
        return baseline


def create_default_cnn1d() -> keras.Model:
    """Create the default 1D CNN model as specified in PRD.

    This creates the exact model specified in MODEL-3:
    - 3 Conv1D layers with 64 filters each
    - BatchNormalization after each conv
    - GlobalAveragePooling1D
    - Single Dense output for RUL
    - Total params under 1M

    Returns:
        Compiled Keras model ready for training.
    """
    config = CNN1DConfig(
        num_conv_layers=3,
        filters=64,
        kernel_size=3,
        activation="relu",
        use_batch_norm=True,
        dropout_rate=0.0,
        dense_units=None,
    )
    return build_cnn1d_model(config)
