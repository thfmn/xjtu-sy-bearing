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
"""Feature Sequence LSTM for RUL prediction.

A lightweight LSTM model that operates on sliding windows of extracted features
to capture degradation trajectories over time. Unlike single-snapshot models,
this approach can learn temporal patterns in feature evolution.

Architecture:
    Input(window_size, n_features) -> BiLSTM(units) -> Dropout -> Dense(units, relu) -> Dense(1)

The model is intentionally small (~11K params) to avoid overfitting on the
small per-condition training sets in XJTU-SY.
"""

from dataclasses import dataclass, asdict
from typing import Any

from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class FeatureLSTMConfig:
    """Configuration for Feature Sequence LSTM model.

    Attributes:
        window_size: Number of consecutive samples per window.
        n_features: Number of input features per timestep.
        lstm_units: Number of LSTM units.
        bidirectional: Whether to use bidirectional LSTM.
        dropout_rate: Dropout rate after LSTM.
        dense_units: Units in the hidden dense layer.
    """

    window_size: int = 10
    n_features: int = 65
    lstm_units: int = 16
    bidirectional: bool = True
    dropout_rate: float = 0.2
    dense_units: int = 16

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


def build_feature_lstm_model(
    config: FeatureLSTMConfig | None = None,
    name: str = "feature_lstm_rul",
) -> keras.Model:
    """Build a Feature Sequence LSTM model for RUL prediction.

    Args:
        config: Model configuration. Uses defaults if None.
        name: Model name.

    Returns:
        Uncompiled Keras model.
    """
    if config is None:
        config = FeatureLSTMConfig()

    inputs = layers.Input(
        shape=(config.window_size, config.n_features),
        name="feature_window",
    )

    # LSTM layer (optionally bidirectional)
    lstm = layers.LSTM(config.lstm_units, name="lstm")
    if config.bidirectional:
        x = layers.Bidirectional(lstm, name="bilstm")(inputs)
    else:
        x = lstm(inputs)

    # Dropout for regularization
    if config.dropout_rate > 0:
        x = layers.Dropout(config.dropout_rate, name="dropout")(x)

    # Dense hidden layer
    x = layers.Dense(config.dense_units, activation="relu", name="dense")(x)

    # RUL output - linear activation; clipping to non-negative is done post-prediction
    outputs = layers.Dense(1, name="rul_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def create_default_feature_lstm() -> keras.Model:
    """Create the default Feature LSTM model.

    Uses default config: window_size=10, n_features=65, BiLSTM(16),
    Dropout(0.2), Dense(16), Dense(1). Approximately 11K parameters.

    Returns:
        Uncompiled Keras model.
    """
    config = FeatureLSTMConfig(
        window_size=10,
        n_features=65,
        lstm_units=16,
        bidirectional=True,
        dropout_rate=0.2,
        dense_units=16,
    )
    return build_feature_lstm_model(config)
