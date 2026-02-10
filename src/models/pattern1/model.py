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

"""TCN-Transformer Pattern 1 model assembly for RUL prediction.

This module assembles the full model architecture:
    Input (32768, 2) -> Stem -> TCN -> Cross-Attention -> Aggregator -> RUL Head

Supports both LSTM and Transformer aggregators for temporal summarization.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import tensorflow as tf
import keras
from keras import layers

from .stem import DualChannelStem, StemConfig
from .tcn import DualChannelTCN, TCNConfig
from .attention import BidirectionalCrossAttention, AttentionConfig, ChannelFusion
from .aggregator import (
    LSTMAggregator, LSTMAggregatorConfig,
    TransformerAggregator, TransformerAggregatorConfig,
)


@dataclass
class TCNTransformerConfig:
    """Configuration for the full TCN-Transformer model.

    Attributes:
        input_length: Length of input signal (e.g., 32768).
        num_channels: Number of input channels (2 for h/v).
        stem_config: Configuration for per-sensor stem.
        tcn_config: Configuration for TCN encoder.
        attention_config: Configuration for cross-attention.
        aggregator_type: Type of temporal aggregator ('lstm' or 'transformer').
        lstm_config: Configuration for LSTM aggregator (if used).
        transformer_config: Configuration for Transformer aggregator (if used).
        fusion_mode: How to fuse dual-channel features ('concat', 'add', 'avg', 'weighted').
        hidden_dim: Hidden dimension for RUL head MLP.
        dropout_rate: Dropout rate for RUL head.
        use_downsampling: Whether to downsample before aggregator (for memory).
        downsample_factor: Factor by which to downsample temporal dimension.
    """
    input_length: int = 32768
    num_channels: int = 2
    stem_config: StemConfig = field(default_factory=StemConfig)
    tcn_config: TCNConfig = field(default_factory=TCNConfig)
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    aggregator_type: Literal["lstm", "transformer"] = "lstm"
    lstm_config: LSTMAggregatorConfig = field(default_factory=LSTMAggregatorConfig)
    transformer_config: TransformerAggregatorConfig = field(
        default_factory=TransformerAggregatorConfig
    )
    fusion_mode: str = "concat"
    hidden_dim: int = 64
    dropout_rate: float = 0.1
    use_downsampling: bool = True
    downsample_factor: int = 16


class RULHead(keras.layers.Layer):
    """RUL prediction head with monotonic output constraint.

    Produces non-negative RUL predictions using ReLU activation.
    Optionally outputs uncertainty via Gaussian parametrization.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        output_uncertainty: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.output_uncertainty = output_uncertainty

        self.dense1 = None
        self.dropout = None
        self.dense_mean = None
        self.dense_var = None

    def build(self, input_shape):
        self.dense1 = layers.Dense(
            self.hidden_dim,
            activation="gelu",
            name=f"{self.name}_hidden"
        )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(
                self.dropout_rate,
                name=f"{self.name}_dropout"
            )

        # Output: mean RUL (non-negative via ReLU)
        self.dense_mean = layers.Dense(
            1,
            activation="relu",  # Ensures non-negative RUL
            name=f"{self.name}_mean"
        )

        # Optional: output variance for uncertainty quantification
        if self.output_uncertainty:
            self.dense_var = layers.Dense(
                1,
                activation="softplus",  # Ensures positive variance
                name=f"{self.name}_var"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Predict RUL from aggregated features.

        Args:
            inputs: Aggregated feature tensor.
            training: Training mode flag.

        Returns:
            RUL prediction (batch, 1), or (mean, variance) if uncertainty enabled.
        """
        x = self.dense1(inputs)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        mean = self.dense_mean(x)

        if self.output_uncertainty and self.dense_var is not None:
            var = self.dense_var(x)
            return mean, var

        return mean

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "output_uncertainty": self.output_uncertainty,
        })
        return config


class TemporalDownsampler(keras.layers.Layer):
    """Downsample temporal dimension to reduce memory for aggregator."""

    def __init__(self, factor: int = 16, mode: str = "avg", **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.mode = mode
        self.pool = None

    def build(self, input_shape):
        if self.mode == "avg":
            self.pool = layers.AveragePooling1D(
                pool_size=self.factor,
                strides=self.factor,
                padding="valid",
                name=f"{self.name}_avg_pool"
            )
        elif self.mode == "max":
            self.pool = layers.MaxPooling1D(
                pool_size=self.factor,
                strides=self.factor,
                padding="valid",
                name=f"{self.name}_max_pool"
            )
        elif self.mode == "strided_conv":
            # Strided convolution for learnable downsampling
            self.pool = layers.Conv1D(
                filters=input_shape[-1],
                kernel_size=self.factor,
                strides=self.factor,
                padding="valid",
                name=f"{self.name}_strided_conv"
            )
        super().build(input_shape)

    def call(self, inputs):
        return self.pool(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "factor": self.factor,
            "mode": self.mode,
        })
        return config


def build_tcn_transformer_model(
    config: Optional[TCNTransformerConfig] = None,
    name: str = "tcn_transformer"
) -> keras.Model:
    """Build the full TCN-Transformer model for RUL prediction.

    Architecture:
        Input (batch, 32768, 2)
            ↓
        DualChannelStem → (h_stem_out, v_stem_out)
            ↓
        DualChannelTCN → (h_tcn_out, v_tcn_out)
            ↓
        BidirectionalCrossAttention → (h_attn_out, v_attn_out)
            ↓
        ChannelFusion → fused_features
            ↓
        [Optional Downsampling]
            ↓
        Aggregator (LSTM or Transformer) → aggregated
            ↓
        RULHead → RUL prediction

    Args:
        config: Model configuration.
        name: Model name.

    Returns:
        Compiled Keras model.
    """
    if config is None:
        config = TCNTransformerConfig()

    # Input layer
    inputs = keras.Input(
        shape=(config.input_length, config.num_channels),
        name="input"
    )

    # Per-sensor stem processing
    stem = DualChannelStem(
        config=config.stem_config,
        share_weights=False,
        name="dual_stem"
    )
    h_stem, v_stem = stem(inputs)

    # TCN encoding
    tcn = DualChannelTCN(
        config=config.tcn_config,
        share_weights=False,
        name="dual_tcn"
    )
    h_tcn, v_tcn = tcn((h_stem, v_stem))

    # Downsample BEFORE cross-attention to avoid O(n^2) memory explosion
    # Cross-attention on 32768x32768 would require ~17GB for attention scores alone
    if config.use_downsampling and config.downsample_factor > 1:
        h_downsampler = TemporalDownsampler(
            factor=config.downsample_factor,
            mode="avg",
            name="h_temporal_downsample"
        )
        v_downsampler = TemporalDownsampler(
            factor=config.downsample_factor,
            mode="avg",
            name="v_temporal_downsample"
        )
        h_tcn = h_downsampler(h_tcn)
        v_tcn = v_downsampler(v_tcn)

    # Cross-attention fusion (now on downsampled sequences, e.g., 2048 instead of 32768)
    cross_attention = BidirectionalCrossAttention(
        config=config.attention_config,
        name="cross_attention"
    )
    h_attn, v_attn = cross_attention((h_tcn, v_tcn))

    # Channel fusion
    fusion = ChannelFusion(
        fusion_mode=config.fusion_mode,
        output_dim=None,  # Keep as is
        name="channel_fusion"
    )
    fused = fusion((h_attn, v_attn))

    # Temporal aggregation
    if config.aggregator_type == "lstm":
        aggregator = LSTMAggregator(
            config=config.lstm_config,
            name="lstm_aggregator"
        )
    else:
        aggregator = TransformerAggregator(
            config=config.transformer_config,
            name="transformer_aggregator"
        )
    aggregated = aggregator(fused)

    # RUL prediction head
    rul_head = RULHead(
        hidden_dim=config.hidden_dim,
        dropout_rate=config.dropout_rate,
        output_uncertainty=False,
        name="rul_head"
    )
    output = rul_head(aggregated)

    model = keras.Model(inputs=inputs, outputs=output, name=name)
    return model


def create_tcn_transformer_lstm(
    input_length: int = 32768,
    filters: int = 32,
    dilations: list[int] | None = None,
    lstm_units: int = 32,
) -> keras.Model:
    """Create TCN-Transformer model with LSTM aggregator.

    This is the v1 aggregator mentioned in PRD.
    v2: reduced capacity (64→32 filters/units, 6→4 dilations) to combat
    overfitting on the ~9K sample dataset.

    Args:
        input_length: Input sequence length.
        filters: Number of filters for stem and TCN.
        dilations: TCN dilation rates.
        lstm_units: Number of LSTM units.

    Returns:
        Configured Keras model.
    """
    if dilations is None:
        dilations = [1, 2, 4, 8]

    config = TCNTransformerConfig(
        input_length=input_length,
        stem_config=StemConfig(filters=filters),
        tcn_config=TCNConfig(filters=filters, dilations=dilations, dropout_rate=0.3),
        attention_config=AttentionConfig(num_heads=2, key_dim=filters),
        aggregator_type="lstm",
        lstm_config=LSTMAggregatorConfig(
            units=lstm_units,
            bidirectional=True,
            pooling="last"
        ),
        fusion_mode="concat",
        hidden_dim=32,
        dropout_rate=0.3,
        use_downsampling=True,
        downsample_factor=16,
    )

    return build_tcn_transformer_model(config, name="tcn_transformer_lstm")


def create_tcn_transformer_transformer(
    input_length: int = 32768,
    filters: int = 64,
    dilations: list[int] | None = None,
    num_transformer_layers: int = 2,
    num_heads: int = 4,
) -> keras.Model:
    """Create TCN-Transformer model with Transformer aggregator.

    This is the v2 aggregator mentioned in PRD (upgrade).

    Args:
        input_length: Input sequence length.
        filters: Number of filters for stem and TCN.
        dilations: TCN dilation rates.
        num_transformer_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.

    Returns:
        Configured Keras model.
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]

    config = TCNTransformerConfig(
        input_length=input_length,
        stem_config=StemConfig(filters=filters),
        tcn_config=TCNConfig(filters=filters, dilations=dilations),
        attention_config=AttentionConfig(num_heads=num_heads, key_dim=filters),
        aggregator_type="transformer",
        transformer_config=TransformerAggregatorConfig(
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            key_dim=filters,
            ff_dim=filters * 2,
            use_cls_token=True,
            pooling="cls",
        ),
        fusion_mode="concat",
        use_downsampling=True,
        downsample_factor=16,
    )

    return build_tcn_transformer_model(config, name="tcn_transformer_transformer")


def get_model_summary(model: keras.Model) -> dict:
    """Get summary statistics for a model.

    Args:
        model: Keras model.

    Returns:
        Dictionary with model statistics.
    """
    trainable = sum(
        tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables
    )
    non_trainable = sum(
        tf.reduce_prod(v.shape).numpy() for v in model.non_trainable_variables
    )

    return {
        "name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "num_layers": len(model.layers),
    }


def print_model_summary(model: keras.Model) -> None:
    """Print formatted model summary."""
    stats = get_model_summary(model)
    print(f"Model: {stats['name']}")
    print(f"  Input shape:  {stats['input_shape']}")
    print(f"  Output shape: {stats['output_shape']}")
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Trainable:    {stats['trainable_params']:,}")
    print(f"  Non-trainable: {stats['non_trainable_params']:,}")
    print(f"  Layers:       {stats['num_layers']}")
