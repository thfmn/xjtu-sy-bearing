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

"""Per-sensor stem for initial feature extraction from raw 1D signals.

The stem applies separate convolutional processing to each sensor channel
before fusion, allowing the model to learn channel-specific features.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class StemConfig:
    """Configuration for per-sensor stem.

    Attributes:
        filters: Number of output filters for both conv layers.
        kernel_size_1: Kernel size for first (wider) convolution.
        kernel_size_2: Kernel size for second (narrower) convolution.
        use_batch_norm: Whether to apply batch normalization.
        dropout_rate: Dropout rate after activation (0 to disable).
    """
    filters: int = 64
    kernel_size_1: int = 7
    kernel_size_2: int = 3
    use_batch_norm: bool = True
    dropout_rate: float = 0.0


def build_sensor_stem(
    input_shape: tuple[int, int],
    config: Optional[StemConfig] = None,
    name: str = "sensor_stem"
) -> keras.Model:
    """Build a per-sensor stem for one channel.

    Architecture: Conv1D(k=7) -> GELU -> Conv1D(k=3) -> GELU
    Optionally includes BatchNorm and Dropout.

    Args:
        input_shape: Shape of single-channel input (time_steps, 1).
        config: Stem configuration. Uses defaults if None.
        name: Name prefix for the model.

    Returns:
        Keras model for single-channel feature extraction.
    """
    if config is None:
        config = StemConfig()

    inputs = keras.Input(shape=input_shape, name=f"{name}_input")

    # First conv block: wider kernel for initial feature extraction
    x = layers.Conv1D(
        filters=config.filters,
        kernel_size=config.kernel_size_1,
        padding="same",
        name=f"{name}_conv1"
    )(inputs)

    if config.use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)

    x = layers.Activation("gelu", name=f"{name}_gelu1")(x)

    if config.dropout_rate > 0:
        x = layers.Dropout(config.dropout_rate, name=f"{name}_dropout1")(x)

    # Second conv block: narrower kernel for refinement
    x = layers.Conv1D(
        filters=config.filters,
        kernel_size=config.kernel_size_2,
        padding="same",
        name=f"{name}_conv2"
    )(x)

    if config.use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    x = layers.Activation("gelu", name=f"{name}_gelu2")(x)

    if config.dropout_rate > 0:
        x = layers.Dropout(config.dropout_rate, name=f"{name}_dropout2")(x)

    return keras.Model(inputs=inputs, outputs=x, name=name)


class DualChannelStem(keras.layers.Layer):
    """Process both horizontal and vertical channels with shared or separate stems.

    This layer splits the dual-channel input and processes each channel
    through its own stem network, then returns both processed features.

    Attributes:
        config: Stem configuration for both channels.
        share_weights: If True, use same weights for both channels.
    """

    def __init__(
        self,
        config: Optional[StemConfig] = None,
        share_weights: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else StemConfig()
        self.share_weights = share_weights

        # Will be built in build()
        self.h_stem = None
        self.v_stem = None

    def build(self, input_shape):
        """Build the stem layers.

        Args:
            input_shape: Expected (batch, time_steps, 2) for dual-channel.
        """
        time_steps = input_shape[1]
        single_channel_shape = (time_steps, 1)

        # Build horizontal stem
        self.h_stem = build_sensor_stem(
            input_shape=single_channel_shape,
            config=self.config,
            name="h_stem"
        )

        if self.share_weights:
            # Use same stem for both channels
            self.v_stem = self.h_stem
        else:
            # Separate stem for vertical channel
            self.v_stem = build_sensor_stem(
                input_shape=single_channel_shape,
                config=self.config,
                name="v_stem"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Process dual-channel input through stems.

        Args:
            inputs: Tensor of shape (batch, time_steps, 2).
            training: Training mode flag.

        Returns:
            Tuple of (h_features, v_features), each (batch, time_steps, filters).
        """
        # Split channels: [..., 0:1] keeps dim, [..., 0] would squeeze
        h_input = inputs[..., 0:1]  # (batch, time_steps, 1)
        v_input = inputs[..., 1:2]  # (batch, time_steps, 1)

        # Process through stems
        h_features = self.h_stem(h_input, training=training)
        v_features = self.v_stem(v_input, training=training)

        return h_features, v_features

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "filters": self.config.filters,
                "kernel_size_1": self.config.kernel_size_1,
                "kernel_size_2": self.config.kernel_size_2,
                "use_batch_norm": self.config.use_batch_norm,
                "dropout_rate": self.config.dropout_rate,
            },
            "share_weights": self.share_weights,
        })
        return config

    @classmethod
    def from_config(cls, config):
        stem_config = StemConfig(**config.pop("config"))
        return cls(config=stem_config, **config)


def create_default_stem() -> DualChannelStem:
    """Create default dual-channel stem with PRD-specified settings.

    Returns:
        DualChannelStem with default configuration.
    """
    config = StemConfig(
        filters=64,
        kernel_size_1=7,
        kernel_size_2=3,
        use_batch_norm=True,
        dropout_rate=0.0,
    )
    return DualChannelStem(config=config, share_weights=False)
