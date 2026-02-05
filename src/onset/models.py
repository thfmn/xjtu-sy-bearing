"""Onset classifier model for two-stage RUL prediction (Stage 1).

Lightweight LSTM-based binary classifier that predicts whether a bearing
is in a healthy (0) or degraded (1) state, based on a sliding window of
health indicator features (h_kurtosis, v_kurtosis, h_rms, v_rms).

Architecture: Input -> LSTM(32) -> Dropout(0.3) -> Dense(16, relu) -> Dropout(0.3) -> Dense(1, sigmoid)
Total parameters: ~5,300 (well under 10K target for fast inference).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.onset.dataset import N_FEATURES


@dataclass
class OnsetClassifierConfig:
    """Configuration for the onset classifier model.

    Attributes:
        window_size: Number of timesteps in each input window.
        n_features: Number of health indicator features per timestep.
        lstm_units: Number of units in the LSTM layer.
        dense_units: Number of units in the hidden Dense layer.
        dropout_rate: Dropout rate applied after LSTM and Dense layers.
        l2_factor: L2 regularization factor for Dense kernel weights.
    """

    window_size: int = 10
    n_features: int = N_FEATURES  # 8 (4 signed z-scores + 4 absolute z-scores)
    lstm_units: int = 32
    dense_units: int = 16
    dropout_rate: float = 0.3
    l2_factor: float = 1e-3


def build_onset_classifier(
    config: OnsetClassifierConfig | None = None,
    name: str = "onset_classifier",
):
    """Build a lightweight LSTM onset classifier using the Keras Functional API.

    Args:
        config: Model configuration. Uses defaults if None.
        name: Name for the Keras model.

    Returns:
        Uncompiled keras.Model with:
            - Input shape: (None, window_size, n_features)
            - Output shape: (None, 1) with sigmoid activation
    """
    import keras
    from keras import layers, regularizers

    if config is None:
        config = OnsetClassifierConfig()

    inputs = keras.Input(
        shape=(config.window_size, config.n_features),
        name="input",
    )

    # LSTM layer processes the temporal window
    x = layers.LSTM(
        config.lstm_units,
        name="lstm",
    )(inputs)

    x = layers.Dropout(config.dropout_rate, name="dropout_1")(x)

    # Hidden dense layer with L2 regularization
    x = layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=regularizers.L2(config.l2_factor),
        name="dense_hidden",
    )(x)

    x = layers.Dropout(config.dropout_rate, name="dropout_2")(x)

    # Output layer: single sigmoid for binary classification
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="onset_output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_onset_classifier(
    model,
    learning_rate: float = 1e-3,
):
    """Compile the onset classifier with binary crossentropy loss.

    Configures the model for binary classification with:
    - Binary crossentropy loss (sigmoid output, from_logits=False)
    - Adam optimizer with configurable learning rate
    - Metrics: accuracy, AUC-ROC, precision, recall

    Class weights for imbalanced data should be passed to model.fit()
    via the class_weight parameter (use compute_class_weights() from
    src.onset.dataset).

    Args:
        model: Uncompiled Keras model from build_onset_classifier().
        learning_rate: Learning rate for Adam optimizer. Default 1e-3.

    Returns:
        Compiled Keras model (same object, modified in-place).
    """
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def predict_proba(model, x, **kwargs):
    """Return class probabilities from the onset classifier.

    Equivalent to scikit-learn's ``predict_proba()`` interface.  The model
    uses a sigmoid output, so ``model.predict(x)`` already returns P(degraded).
    This helper returns a two-column array ``[P(healthy), P(degraded)]``
    matching the sklearn convention.

    Args:
        model: Compiled onset classifier (sigmoid output).
        x: Input array with shape ``(n_samples, window_size, n_features)``.
        **kwargs: Extra keyword arguments forwarded to ``model.predict()``.

    Returns:
        numpy.ndarray of shape ``(n_samples, 2)`` with columns
        ``[P(class=0), P(class=1)]``.
    """
    import numpy as np

    p1 = model.predict(x, **kwargs)  # (n_samples, 1)
    p1 = np.asarray(p1).reshape(-1, 1)
    return np.hstack([1.0 - p1, p1])


def create_onset_classifier(
    input_dim: int = N_FEATURES,
    window_size: int = 10,
):
    """Factory function matching the PRD interface.

    Convenience wrapper around build_onset_classifier() that accepts
    input_dim and window_size directly.

    Args:
        input_dim: Number of features per timestep. Default 4.
        window_size: Number of timesteps in each window. Default 10.

    Returns:
        Uncompiled keras.Model.
    """
    config = OnsetClassifierConfig(
        window_size=window_size,
        n_features=input_dim,
    )
    return build_onset_classifier(config=config)
