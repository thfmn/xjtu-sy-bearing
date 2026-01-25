"""Temporal aggregation for 2D CNN + Temporal Pattern 2.

This module provides temporal aggregation layers for processing sequences
of spectrogram embeddings. In the Pattern 2 architecture, the 2D CNN
extracts per-spectrogram embeddings, and the temporal aggregator combines
them into a single representation for RUL prediction.

Two modes are supported:
1. Single-spectrogram mode: Process one spectrogram per sample (no temporal sequence)
2. Sequence mode: Process multiple spectrograms per bearing (temporal context)

The aggregators from pattern1 can be reused since they operate on
(batch, time, features) tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Re-export aggregators from pattern1 for consistency
from src.models.pattern1.aggregator import (
    LSTMAggregator,
    LSTMAggregatorConfig,
    TransformerAggregator,
    TransformerAggregatorConfig,
    PositionalEncoding,
    TransformerEncoderBlock,
)


@dataclass
class SequenceAggregatorConfig:
    """Configuration for sequence-level aggregation in Pattern 2.

    In Pattern 2, we can operate on sequences of spectrograms from
    a single bearing (multiple time windows), requiring temporal
    aggregation to produce a single RUL prediction.

    Attributes:
        aggregator_type: Type of aggregator ('lstm', 'transformer', 'mean', 'last').
        lstm_config: Configuration for LSTM aggregator.
        transformer_config: Configuration for Transformer aggregator.
        sequence_length: Expected sequence length (for positional encoding).
    """

    aggregator_type: Literal["lstm", "transformer", "mean", "last"] = "lstm"
    lstm_config: LSTMAggregatorConfig | None = None
    transformer_config: TransformerAggregatorConfig | None = None
    sequence_length: int = 1

    def __post_init__(self):
        if self.lstm_config is None:
            self.lstm_config = LSTMAggregatorConfig(
                units=64,
                bidirectional=True,
                pooling="last"
            )
        if self.transformer_config is None:
            self.transformer_config = TransformerAggregatorConfig(
                num_layers=2,
                num_heads=4,
                key_dim=64,
                use_cls_token=True,
                pooling="cls"
            )


class SimplePoolingAggregator(keras.layers.Layer):
    """Simple pooling-based aggregator for sequence reduction.

    For cases where full temporal modeling is not needed, this provides
    simple mean or last-timestep pooling.
    """

    def __init__(
        self,
        mode: Literal["mean", "max", "last", "first"] = "mean",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.pool_layer = None

    def build(self, input_shape):
        if self.mode == "mean":
            self.pool_layer = layers.GlobalAveragePooling1D(name=f"{self.name}_gap")
        elif self.mode == "max":
            self.pool_layer = layers.GlobalMaxPooling1D(name=f"{self.name}_gmp")
        # 'last' and 'first' don't need special layers
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Aggregate sequence to single vector.

        Args:
            inputs: Sequence tensor, shape (batch, time, features).
            training: Training mode flag.

        Returns:
            Aggregated tensor, shape (batch, features).
        """
        if self.mode == "mean":
            return self.pool_layer(inputs)
        elif self.mode == "max":
            return self.pool_layer(inputs)
        elif self.mode == "last":
            return inputs[:, -1, :]
        elif self.mode == "first":
            return inputs[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config


class Pattern2Aggregator(keras.layers.Layer):
    """Unified temporal aggregator for Pattern 2 architecture.

    Wraps different aggregation strategies (LSTM, Transformer, simple pooling)
    behind a unified interface.
    """

    def __init__(
        self,
        config: SequenceAggregatorConfig | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else SequenceAggregatorConfig()
        self.aggregator = None

    def build(self, input_shape):
        if self.config.aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(
                config=self.config.lstm_config,
                name=f"{self.name}_lstm"
            )
        elif self.config.aggregator_type == "transformer":
            self.aggregator = TransformerAggregator(
                config=self.config.transformer_config,
                name=f"{self.name}_transformer"
            )
        elif self.config.aggregator_type in ["mean", "last", "first", "max"]:
            self.aggregator = SimplePoolingAggregator(
                mode=self.config.aggregator_type,
                name=f"{self.name}_pool"
            )
        else:
            raise ValueError(f"Unknown aggregator type: {self.config.aggregator_type}")

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Aggregate temporal sequence.

        Args:
            inputs: Sequence tensor, shape (batch, time, features).
                For single-spectrogram mode, time=1 and aggregation is trivial.
            training: Training mode flag.

        Returns:
            Aggregated tensor, shape (batch, output_dim).
        """
        return self.aggregator(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "aggregator_type": self.config.aggregator_type,
                "sequence_length": self.config.sequence_length,
            }
        })
        return config


def create_lstm_aggregator(
    units: int = 64,
    bidirectional: bool = True,
    num_layers: int = 1,
) -> Pattern2Aggregator:
    """Create Pattern 2 aggregator with LSTM.

    Args:
        units: Number of LSTM units.
        bidirectional: Whether to use bidirectional LSTM.
        num_layers: Number of stacked LSTM layers.

    Returns:
        Configured Pattern2Aggregator.
    """
    lstm_config = LSTMAggregatorConfig(
        units=units,
        num_layers=num_layers,
        bidirectional=bidirectional,
        pooling="last",
    )
    config = SequenceAggregatorConfig(
        aggregator_type="lstm",
        lstm_config=lstm_config,
    )
    return Pattern2Aggregator(config=config)


def create_transformer_aggregator(
    num_layers: int = 2,
    num_heads: int = 4,
    key_dim: int = 64,
) -> Pattern2Aggregator:
    """Create Pattern 2 aggregator with Transformer.

    Args:
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        key_dim: Dimension of attention keys.

    Returns:
        Configured Pattern2Aggregator.
    """
    transformer_config = TransformerAggregatorConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        key_dim=key_dim,
        use_cls_token=True,
        pooling="cls",
    )
    config = SequenceAggregatorConfig(
        aggregator_type="transformer",
        transformer_config=transformer_config,
    )
    return Pattern2Aggregator(config=config)


def create_simple_aggregator(
    mode: Literal["mean", "last", "first", "max"] = "mean"
) -> Pattern2Aggregator:
    """Create Pattern 2 aggregator with simple pooling.

    Args:
        mode: Pooling mode ('mean', 'last', 'first', 'max').

    Returns:
        Configured Pattern2Aggregator.
    """
    config = SequenceAggregatorConfig(
        aggregator_type=mode,
    )
    return Pattern2Aggregator(config=config)
