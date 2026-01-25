"""2D CNN + Temporal Pattern 2 model assembly for RUL prediction.

This module assembles the full Pattern 2 architecture:
    Input (spectrograms) -> CNN Backbone -> Late Fusion -> Temporal Aggregator -> RUL Head

The model can operate in two modes:
1. Pre-computed spectrograms: Input shape (batch, height, width, 2)
2. Raw signals with on-the-fly STFT: Input shape (batch, samples, 2)

Architecture supports both LSTM and Transformer temporal aggregators,
with optional uncertainty quantification via Gaussian output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .backbone import CNN2DBackboneConfig, CNN2DBackbone, DualChannelCNN2DBackbone
from .aggregator import (
    SequenceAggregatorConfig,
    Pattern2Aggregator,
    LSTMAggregatorConfig,
    TransformerAggregatorConfig,
)


@dataclass
class Pattern2Config:
    """Configuration for the full Pattern 2 model.

    Attributes:
        input_mode: Input type ('spectrogram' for pre-computed, 'raw' for on-the-fly).
        spectrogram_height: Height of input spectrograms.
        spectrogram_width: Width of input spectrograms.
        backbone_config: Configuration for 2D CNN backbone.
        share_backbone_weights: Whether to share CNN weights between channels.
        fusion_mode: How to fuse channel embeddings ('concat', 'add', 'weighted').
        aggregator_type: Type of temporal aggregator ('lstm', 'transformer', 'mean').
        lstm_config: Configuration for LSTM aggregator.
        transformer_config: Configuration for Transformer aggregator.
        rul_hidden_dim: Hidden dimension for RUL head MLP.
        rul_dropout_rate: Dropout rate for RUL head.
        output_uncertainty: Whether to output uncertainty (mean + variance).
    """

    input_mode: Literal["spectrogram", "raw"] = "spectrogram"
    spectrogram_height: int = 128
    spectrogram_width: int = 128
    backbone_config: CNN2DBackboneConfig = field(default_factory=CNN2DBackboneConfig)
    share_backbone_weights: bool = True
    fusion_mode: Literal["concat", "add", "weighted"] = "concat"
    aggregator_type: Literal["lstm", "transformer", "mean", "none"] = "lstm"
    lstm_config: LSTMAggregatorConfig = field(default_factory=LSTMAggregatorConfig)
    transformer_config: TransformerAggregatorConfig = field(
        default_factory=TransformerAggregatorConfig
    )
    rul_hidden_dim: int = 64
    rul_dropout_rate: float = 0.1
    output_uncertainty: bool = False


class LateFusion(keras.layers.Layer):
    """Late fusion module for combining dual-channel embeddings.

    Supports multiple fusion strategies:
    - concat: Concatenate embeddings (doubles feature dimension)
    - add: Element-wise addition
    - weighted: Learnable weighted combination
    """

    def __init__(
        self,
        fusion_mode: str = "concat",
        output_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fusion_mode = fusion_mode
        self.output_dim = output_dim
        self.projection = None
        self.h_weight = None
        self.v_weight = None

    def build(self, input_shape):
        # Input is tuple of (h_embedding, v_embedding)
        h_shape, v_shape = input_shape
        h_dim = h_shape[-1]
        v_dim = v_shape[-1]

        if self.fusion_mode == "concat":
            fused_dim = h_dim + v_dim
        else:
            fused_dim = h_dim

        # Optional output projection
        if self.output_dim is not None and self.output_dim != fused_dim:
            self.projection = layers.Dense(
                self.output_dim,
                name=f"{self.name}_proj"
            )

        # Learnable weights for weighted fusion
        if self.fusion_mode == "weighted":
            self.h_weight = self.add_weight(
                name="h_weight",
                shape=(1,),
                initializer="ones",
                trainable=True
            )
            self.v_weight = self.add_weight(
                name="v_weight",
                shape=(1,),
                initializer="ones",
                trainable=True
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Fuse dual-channel embeddings.

        Args:
            inputs: Tuple of (h_embedding, v_embedding).
            training: Training mode flag.

        Returns:
            Fused embedding tensor.
        """
        h_embedding, v_embedding = inputs

        if self.fusion_mode == "concat":
            fused = keras.ops.concatenate([h_embedding, v_embedding], axis=-1)
        elif self.fusion_mode == "add":
            fused = h_embedding + v_embedding
        elif self.fusion_mode == "weighted":
            # Softmax-normalized weights
            weights = keras.ops.softmax(
                keras.ops.stack([self.h_weight, self.v_weight], axis=0)
            )
            fused = weights[0] * h_embedding + weights[1] * v_embedding
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        if self.projection is not None:
            fused = self.projection(fused)

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            "fusion_mode": self.fusion_mode,
            "output_dim": self.output_dim,
        })
        return config


class RULHead(keras.layers.Layer):
    """RUL prediction head with optional uncertainty output.

    Produces non-negative RUL predictions using ReLU activation.
    Optionally outputs mean and variance for uncertainty quantification.
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

        # Mean output (non-negative via ReLU)
        self.dense_mean = layers.Dense(
            1,
            activation="relu",
            name=f"{self.name}_mean"
        )

        # Optional variance output (positive via softplus)
        if self.output_uncertainty:
            self.dense_var = layers.Dense(
                1,
                activation="softplus",
                name=f"{self.name}_var"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Predict RUL from fused features.

        Args:
            inputs: Fused feature tensor.
            training: Training mode flag.

        Returns:
            RUL prediction (batch, 1), or dict with 'mean' and 'var' if uncertainty.
        """
        x = self.dense1(inputs)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        mean = self.dense_mean(x)

        if self.output_uncertainty and self.dense_var is not None:
            var = self.dense_var(x)
            return {"mean": mean, "var": var}

        return mean

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "output_uncertainty": self.output_uncertainty,
        })
        return config


def build_pattern2_model(
    config: Optional[Pattern2Config] = None,
    name: str = "pattern2_cnn_temporal"
) -> keras.Model:
    """Build the full Pattern 2 model for RUL prediction.

    Architecture:
        Input (batch, height, width, 2) [spectrograms]
            ↓
        DualChannelCNN2DBackbone → (h_embedding, v_embedding)
            ↓
        LateFusion → fused_embedding
            ↓
        [Optional: Expand to sequence for temporal aggregation]
            ↓
        Pattern2Aggregator → aggregated
            ↓
        RULHead → RUL prediction (and optionally variance)

    Args:
        config: Model configuration.
        name: Model name.

    Returns:
        Compiled Keras model.
    """
    if config is None:
        config = Pattern2Config()

    # Input layer
    inputs = keras.Input(
        shape=(config.spectrogram_height, config.spectrogram_width, 2),
        name="spectrogram_input"
    )

    # 2D CNN backbone for feature extraction
    backbone = DualChannelCNN2DBackbone(
        config=config.backbone_config,
        share_weights=config.share_backbone_weights,
        name="dual_backbone"
    )
    h_embedding, v_embedding = backbone(inputs)

    # Late fusion
    fusion = LateFusion(
        fusion_mode=config.fusion_mode,
        name="late_fusion"
    )
    fused = fusion((h_embedding, v_embedding))

    # For temporal aggregation, we need a sequence dimension
    # In single-spectrogram mode, expand dims to (batch, 1, features)
    if config.aggregator_type != "none":
        # Use Keras Reshape layer instead of tf.expand_dims
        fused_seq = layers.Reshape((1, -1), name="expand_to_sequence")(fused)

        # Temporal aggregation
        agg_config = SequenceAggregatorConfig(
            aggregator_type=config.aggregator_type,
            lstm_config=config.lstm_config,
            transformer_config=config.transformer_config,
        )
        aggregator = Pattern2Aggregator(
            config=agg_config,
            name="temporal_aggregator"
        )
        aggregated = aggregator(fused_seq)
    else:
        aggregated = fused

    # RUL prediction head
    rul_head = RULHead(
        hidden_dim=config.rul_hidden_dim,
        dropout_rate=config.rul_dropout_rate,
        output_uncertainty=config.output_uncertainty,
        name="rul_head"
    )
    output = rul_head(aggregated)

    # Handle uncertainty output
    if config.output_uncertainty:
        model = keras.Model(
            inputs=inputs,
            outputs=[output["mean"], output["var"]],
            name=name
        )
    else:
        model = keras.Model(inputs=inputs, outputs=output, name=name)

    return model


def create_pattern2_lstm(
    spectrogram_shape: tuple[int, int] = (128, 128),
    num_conv_blocks: int = 4,
    base_filters: int = 32,
    lstm_units: int = 64,
    share_weights: bool = True,
) -> keras.Model:
    """Create Pattern 2 model with LSTM aggregator.

    This is the v1 aggregator mentioned in PRD.

    Args:
        spectrogram_shape: Input spectrogram (height, width).
        num_conv_blocks: Number of convolutional blocks.
        base_filters: Base number of filters (doubles each block).
        lstm_units: Number of LSTM units.
        share_weights: Whether to share CNN weights between channels.

    Returns:
        Configured Keras model.
    """
    filters = [base_filters * (2 ** i) for i in range(num_conv_blocks)]

    config = Pattern2Config(
        spectrogram_height=spectrogram_shape[0],
        spectrogram_width=spectrogram_shape[1],
        backbone_config=CNN2DBackboneConfig(
            filters=filters,
            kernel_sizes=[3] * num_conv_blocks,
            pool_sizes=[2] * num_conv_blocks,
        ),
        share_backbone_weights=share_weights,
        fusion_mode="concat",
        aggregator_type="lstm",
        lstm_config=LSTMAggregatorConfig(
            units=lstm_units,
            bidirectional=True,
            pooling="last"
        ),
    )

    return build_pattern2_model(config, name="pattern2_lstm")


def create_pattern2_transformer(
    spectrogram_shape: tuple[int, int] = (128, 128),
    num_conv_blocks: int = 4,
    base_filters: int = 32,
    num_transformer_layers: int = 2,
    num_heads: int = 4,
    share_weights: bool = True,
) -> keras.Model:
    """Create Pattern 2 model with Transformer aggregator.

    This is the v2 aggregator mentioned in PRD.

    Args:
        spectrogram_shape: Input spectrogram (height, width).
        num_conv_blocks: Number of convolutional blocks.
        base_filters: Base number of filters (doubles each block).
        num_transformer_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        share_weights: Whether to share CNN weights between channels.

    Returns:
        Configured Keras model.
    """
    filters = [base_filters * (2 ** i) for i in range(num_conv_blocks)]

    config = Pattern2Config(
        spectrogram_height=spectrogram_shape[0],
        spectrogram_width=spectrogram_shape[1],
        backbone_config=CNN2DBackboneConfig(
            filters=filters,
            kernel_sizes=[3] * num_conv_blocks,
            pool_sizes=[2] * num_conv_blocks,
        ),
        share_backbone_weights=share_weights,
        fusion_mode="concat",
        aggregator_type="transformer",
        transformer_config=TransformerAggregatorConfig(
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            key_dim=64,
            use_cls_token=True,
            pooling="cls"
        ),
    )

    return build_pattern2_model(config, name="pattern2_transformer")


def create_pattern2_with_uncertainty(
    spectrogram_shape: tuple[int, int] = (128, 128),
    num_conv_blocks: int = 4,
    base_filters: int = 32,
    aggregator_type: str = "lstm",
) -> keras.Model:
    """Create Pattern 2 model with uncertainty quantification.

    Outputs both mean RUL prediction and variance estimate.

    Args:
        spectrogram_shape: Input spectrogram (height, width).
        num_conv_blocks: Number of convolutional blocks.
        base_filters: Base number of filters.
        aggregator_type: Type of aggregator ('lstm' or 'transformer').

    Returns:
        Configured Keras model with two outputs (mean, variance).
    """
    filters = [base_filters * (2 ** i) for i in range(num_conv_blocks)]

    config = Pattern2Config(
        spectrogram_height=spectrogram_shape[0],
        spectrogram_width=spectrogram_shape[1],
        backbone_config=CNN2DBackboneConfig(
            filters=filters,
            kernel_sizes=[3] * num_conv_blocks,
            pool_sizes=[2] * num_conv_blocks,
        ),
        share_backbone_weights=True,
        fusion_mode="concat",
        aggregator_type=aggregator_type,
        output_uncertainty=True,
    )

    return build_pattern2_model(config, name="pattern2_uncertainty")


def create_simple_pattern2(
    spectrogram_shape: tuple[int, int] = (128, 128),
    num_conv_blocks: int = 4,
    base_filters: int = 32,
) -> keras.Model:
    """Create simple Pattern 2 model without temporal aggregation.

    For single-spectrogram mode where no sequence aggregation is needed.

    Args:
        spectrogram_shape: Input spectrogram (height, width).
        num_conv_blocks: Number of convolutional blocks.
        base_filters: Base number of filters.

    Returns:
        Configured Keras model.
    """
    filters = [base_filters * (2 ** i) for i in range(num_conv_blocks)]

    config = Pattern2Config(
        spectrogram_height=spectrogram_shape[0],
        spectrogram_width=spectrogram_shape[1],
        backbone_config=CNN2DBackboneConfig(
            filters=filters,
            kernel_sizes=[3] * num_conv_blocks,
            pool_sizes=[2] * num_conv_blocks,
        ),
        share_backbone_weights=True,
        fusion_mode="concat",
        aggregator_type="none",  # No temporal aggregation
    )

    return build_pattern2_model(config, name="pattern2_simple")


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
