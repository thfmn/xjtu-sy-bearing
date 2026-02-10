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

"""2D CNN backbone for spectrogram feature extraction.

This module implements the convolutional backbone that processes spectrograms
(either STFT or CWT) and extracts spatial features for temporal aggregation.

Architecture follows standard CNN design with 4 convolutional blocks:
- Conv2D -> BatchNorm -> Activation -> MaxPool (repeated 4 times)
- GlobalAveragePooling2D for spatial aggregation to embedding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import keras
import tensorflow as tf
from keras import layers


@dataclass
class CNN2DBackboneConfig:
    """Configuration for 2D CNN backbone.

    Attributes:
        filters: List of filter counts for each conv block.
        kernel_sizes: List of kernel sizes for each conv block.
        pool_sizes: List of pool sizes for each conv block.
        activation: Activation function name.
        use_batch_norm: Whether to use batch normalization.
        dropout_rate: Dropout rate after each block (0 to disable).
        use_global_pool: Whether to apply global pooling at the end.
        global_pool_type: Type of global pooling ('avg' or 'max').
    """

    filters: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3, 3])
    pool_sizes: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    activation: str = "relu"
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    use_global_pool: bool = True
    global_pool_type: Literal["avg", "max"] = "avg"

    def __post_init__(self):
        """Validate configuration."""
        n_blocks = len(self.filters)
        if len(self.kernel_sizes) != n_blocks:
            raise ValueError(
                f"kernel_sizes length ({len(self.kernel_sizes)}) "
                f"must match filters length ({n_blocks})"
            )
        if len(self.pool_sizes) != n_blocks:
            raise ValueError(
                f"pool_sizes length ({len(self.pool_sizes)}) "
                f"must match filters length ({n_blocks})"
            )


class ConvBlock2D(keras.layers.Layer):
    """Single 2D convolutional block.

    Architecture: Conv2D -> BatchNorm (optional) -> Activation -> MaxPool
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.conv = None
        self.bn = None
        self.act = None
        self.pool = None
        self.dropout = None

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            name=f"{self.name}_conv"
        )

        if self.use_batch_norm:
            self.bn = layers.BatchNormalization(name=f"{self.name}_bn")

        self.act = layers.Activation(self.activation, name=f"{self.name}_act")

        if self.pool_size > 1:
            self.pool = layers.MaxPooling2D(
                pool_size=self.pool_size,
                padding="same",
                name=f"{self.name}_pool"
            )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(
                self.dropout_rate,
                name=f"{self.name}_dropout"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Apply convolutional block.

        Args:
            inputs: Input tensor, shape (batch, height, width, channels).
            training: Training mode flag.

        Returns:
            Output tensor after conv, bn, activation, pooling.
        """
        x = self.conv(inputs)

        if self.bn is not None:
            x = self.bn(x, training=training)

        x = self.act(x)

        if self.pool is not None:
            x = self.pool(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "activation": self.activation,
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
        })
        return config


class CNN2DBackbone(keras.layers.Layer):
    """2D CNN backbone for spectrogram processing.

    Extracts spatial features from spectrograms using a stack of
    convolutional blocks followed by global pooling.

    Input: (batch, height, width, channels) - e.g., (batch, 128, 128, 2)
    Output: (batch, embedding_dim) - pooled feature vector
    """

    def __init__(
        self,
        config: CNN2DBackboneConfig | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else CNN2DBackboneConfig()
        self.conv_blocks = []
        self.global_pool = None

    def build(self, input_shape):
        # Create convolutional blocks
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(
                self.config.filters,
                self.config.kernel_sizes,
                self.config.pool_sizes
            )
        ):
            block = ConvBlock2D(
                filters=filters,
                kernel_size=kernel_size,
                pool_size=pool_size,
                activation=self.config.activation,
                use_batch_norm=self.config.use_batch_norm,
                dropout_rate=self.config.dropout_rate,
                name=f"{self.name}_block_{i}"
            )
            self.conv_blocks.append(block)

        # Global pooling
        if self.config.use_global_pool:
            if self.config.global_pool_type == "avg":
                self.global_pool = layers.GlobalAveragePooling2D(
                    name=f"{self.name}_gap"
                )
            else:
                self.global_pool = layers.GlobalMaxPooling2D(
                    name=f"{self.name}_gmp"
                )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Extract features from spectrogram.

        Args:
            inputs: Spectrogram tensor, shape (batch, height, width, channels).
            training: Training mode flag.

        Returns:
            Feature tensor, shape (batch, embedding_dim) if global pool,
            else (batch, h', w', filters[-1]).
        """
        x = inputs

        for block in self.conv_blocks:
            x = block(x, training=training)

        if self.global_pool is not None:
            x = self.global_pool(x)

        return x

    def get_output_dim(self) -> int:
        """Get the output embedding dimension.

        Returns:
            Output dimension (last filter count if global pool is used).
        """
        if self.config.use_global_pool:
            return self.config.filters[-1]
        else:
            raise ValueError(
                "Output dimension is spatial without global pooling. "
                "Call with actual input shape to determine output shape."
            )

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "filters": self.config.filters,
                "kernel_sizes": self.config.kernel_sizes,
                "pool_sizes": self.config.pool_sizes,
                "activation": self.config.activation,
                "use_batch_norm": self.config.use_batch_norm,
                "dropout_rate": self.config.dropout_rate,
                "use_global_pool": self.config.use_global_pool,
                "global_pool_type": self.config.global_pool_type,
            }
        })
        return config


class DualChannelCNN2DBackbone(keras.layers.Layer):
    """Dual-channel 2D CNN backbone with optional weight sharing.

    Processes spectrograms from both horizontal and vertical channels
    separately and returns per-channel embeddings for fusion.

    Can use shared weights (Siamese-like) or separate weights per channel.
    """

    def __init__(
        self,
        config: CNN2DBackboneConfig | None = None,
        share_weights: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else CNN2DBackboneConfig()
        self.share_weights = share_weights
        self.h_backbone = None
        self.v_backbone = None

    def build(self, input_shape):
        # Create backbone(s)
        self.h_backbone = CNN2DBackbone(
            config=self.config,
            name=f"{self.name}_h_backbone"
        )

        if self.share_weights:
            # Use same backbone for both channels
            self.v_backbone = self.h_backbone
        else:
            # Separate backbone for vertical channel
            self.v_backbone = CNN2DBackbone(
                config=self.config,
                name=f"{self.name}_v_backbone"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Process dual-channel spectrogram.

        Args:
            inputs: Spectrogram tensor, shape (batch, height, width, 2).
                Channel 0 = horizontal, Channel 1 = vertical.
            training: Training mode flag.

        Returns:
            Tuple of (h_embedding, v_embedding), each shape (batch, embedding_dim).
        """
        # Split channels
        h_spec = inputs[:, :, :, 0:1]  # (batch, h, w, 1)
        v_spec = inputs[:, :, :, 1:2]  # (batch, h, w, 1)

        # Process through backbones
        h_embedding = self.h_backbone(h_spec, training=training)
        v_embedding = self.v_backbone(v_spec, training=training)

        return h_embedding, v_embedding

    def get_output_dim(self) -> int:
        """Get the output embedding dimension per channel."""
        return self.config.filters[-1]

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "filters": self.config.filters,
                "kernel_sizes": self.config.kernel_sizes,
                "pool_sizes": self.config.pool_sizes,
                "activation": self.config.activation,
                "use_batch_norm": self.config.use_batch_norm,
                "dropout_rate": self.config.dropout_rate,
                "use_global_pool": self.config.use_global_pool,
                "global_pool_type": self.config.global_pool_type,
            },
            "share_weights": self.share_weights,
        })
        return config


def create_default_backbone(
    num_blocks: int = 4,
    base_filters: int = 32,
    share_weights: bool = True,
) -> DualChannelCNN2DBackbone:
    """Create default 2D CNN backbone with PRD-specified settings.

    Args:
        num_blocks: Number of convolutional blocks.
        base_filters: Number of filters in first block (doubles each block).
        share_weights: Whether to share weights between channels.

    Returns:
        Configured DualChannelCNN2DBackbone.
    """
    filters = [base_filters * (2 ** i) for i in range(num_blocks)]

    config = CNN2DBackboneConfig(
        filters=filters,
        kernel_sizes=[3] * num_blocks,
        pool_sizes=[2] * num_blocks,
        activation="relu",
        use_batch_norm=True,
        dropout_rate=0.0,
        use_global_pool=True,
        global_pool_type="avg",
    )

    return DualChannelCNN2DBackbone(
        config=config,
        share_weights=share_weights,
    )
