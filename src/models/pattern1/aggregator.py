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

"""Temporal aggregation modules for sequence-to-single-value reduction.

Provides two aggregation strategies:
1. Bidirectional LSTM/GRU for sequential aggregation
2. Transformer encoder for attention-based aggregation

Both output a fixed-size representation from variable-length sequences.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import tensorflow as tf
import keras
from keras import layers


@dataclass
class LSTMAggregatorConfig:
    """Configuration for LSTM/GRU temporal aggregator.

    Attributes:
        units: Number of units in RNN layer.
        num_layers: Number of stacked RNN layers.
        cell_type: Type of RNN cell ('lstm' or 'gru').
        bidirectional: Whether to use bidirectional RNN.
        dropout_rate: Dropout rate between layers.
        recurrent_dropout: Dropout for recurrent connections.
        return_sequences: If True, return all timesteps; if False, return last.
        pooling: Pooling mode when return_sequences is True.
    """
    units: int = 64
    num_layers: int = 1
    cell_type: Literal["lstm", "gru"] = "lstm"
    bidirectional: bool = True
    dropout_rate: float = 0.1
    recurrent_dropout: float = 0.0
    return_sequences: bool = False
    pooling: Literal["last", "mean", "max", "attention"] = "last"


class LSTMAggregator(keras.layers.Layer):
    """LSTM/GRU-based temporal aggregation.

    Processes temporal sequences and produces fixed-size output through
    recurrent processing with optional bidirectionality.
    """

    def __init__(
        self,
        config: Optional[LSTMAggregatorConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else LSTMAggregatorConfig()
        self.rnn_layers = []
        self.pooling_layer = None
        self.attention_dense = None

    def build(self, input_shape):
        # Create RNN layers
        for i in range(self.config.num_layers):
            # Last layer returns sequences based on pooling strategy
            return_seq = (
                self.config.return_sequences or
                i < self.config.num_layers - 1 or
                self.config.pooling in ["mean", "max", "attention"]
            )

            if self.config.cell_type == "lstm":
                rnn = layers.LSTM(
                    units=self.config.units,
                    return_sequences=return_seq,
                    dropout=self.config.dropout_rate if i < self.config.num_layers - 1 else 0,
                    recurrent_dropout=self.config.recurrent_dropout,
                    name=f"{self.name}_lstm_{i}"
                )
            else:
                rnn = layers.GRU(
                    units=self.config.units,
                    return_sequences=return_seq,
                    dropout=self.config.dropout_rate if i < self.config.num_layers - 1 else 0,
                    recurrent_dropout=self.config.recurrent_dropout,
                    name=f"{self.name}_gru_{i}"
                )

            if self.config.bidirectional:
                rnn = layers.Bidirectional(
                    rnn,
                    name=f"{self.name}_bidir_{i}"
                )

            self.rnn_layers.append(rnn)

        # Pooling for sequence aggregation
        if self.config.pooling == "mean":
            self.pooling_layer = layers.GlobalAveragePooling1D(
                name=f"{self.name}_gap"
            )
        elif self.config.pooling == "max":
            self.pooling_layer = layers.GlobalMaxPooling1D(
                name=f"{self.name}_gmp"
            )
        elif self.config.pooling == "attention":
            # Attention pooling: learn which timesteps to attend to
            self.attention_dense = layers.Dense(1, name=f"{self.name}_attn_dense")

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Apply temporal aggregation.

        Args:
            inputs: Tensor of shape (batch, time_steps, features).
            training: Training mode flag.

        Returns:
            Aggregated tensor of shape (batch, output_dim).
        """
        x = inputs

        for rnn in self.rnn_layers:
            x = rnn(x, training=training)

        # Apply pooling if needed
        if self.config.pooling == "last":
            # RNN already returns last state if return_sequences=False
            pass
        elif self.config.pooling in ["mean", "max"]:
            x = self.pooling_layer(x)
        elif self.config.pooling == "attention":
            # Attention pooling
            # x shape: (batch, time, features)
            scores = self.attention_dense(x)  # (batch, time, 1)
            weights = tf.nn.softmax(scores, axis=1)
            x = tf.reduce_sum(x * weights, axis=1)  # (batch, features)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "units": self.config.units,
                "num_layers": self.config.num_layers,
                "cell_type": self.config.cell_type,
                "bidirectional": self.config.bidirectional,
                "dropout_rate": self.config.dropout_rate,
                "recurrent_dropout": self.config.recurrent_dropout,
                "return_sequences": self.config.return_sequences,
                "pooling": self.config.pooling,
            }
        })
        return config


@dataclass
class TransformerAggregatorConfig:
    """Configuration for Transformer temporal aggregator.

    Attributes:
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        key_dim: Dimension of attention keys per head.
        ff_dim: Feed-forward network hidden dimension.
        dropout_rate: Dropout rate.
        use_cls_token: If True, prepend CLS token and use it for output.
        pooling: Pooling mode ('cls', 'mean', 'max', 'first', 'last').
        max_sequence_length: Maximum sequence length for positional encoding.
    """
    num_layers: int = 2
    num_heads: int = 4
    key_dim: int = 64
    ff_dim: int = 128
    dropout_rate: float = 0.1
    use_cls_token: bool = True
    pooling: Literal["cls", "mean", "max", "first", "last"] = "cls"
    max_sequence_length: int = 32768


class PositionalEncoding(keras.layers.Layer):
    """Learnable positional encoding for sequences."""

    def __init__(self, max_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.pos_embedding = None

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(self.max_length, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_embedding[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerEncoderBlock(keras.layers.Layer):
    """Single Transformer encoder block with self-attention and FFN."""

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attention = None
        self.ffn = None
        self.ln1 = None
        self.ln2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        embed_dim = input_shape[-1]

        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name=f"{self.name}_mha"
        )

        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(embed_dim),
        ], name=f"{self.name}_ffn")

        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln1")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln2")
        self.dropout1 = layers.Dropout(self.dropout_rate, name=f"{self.name}_drop1")
        self.dropout2 = layers.Dropout(self.dropout_rate, name=f"{self.name}_drop2")

        super().build(input_shape)

    def call(self, inputs, training=None):
        # Self-attention with residual and layer norm
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.ln1(inputs + attn_output)

        # Feed-forward with residual and layer norm
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.ln2(x + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class TransformerAggregator(keras.layers.Layer):
    """Transformer encoder-based temporal aggregation.

    Uses self-attention to aggregate temporal information into a
    fixed-size representation.
    """

    def __init__(
        self,
        config: Optional[TransformerAggregatorConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else TransformerAggregatorConfig()
        self.pos_encoding = None
        self.encoder_blocks = []
        self.cls_token = None
        self.pooling_layer = None

    def build(self, input_shape):
        embed_dim = input_shape[-1]

        # CLS token for pooling
        if self.config.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, embed_dim),
                initializer="glorot_uniform",
                trainable=True
            )

        # Positional encoding
        # For long sequences, use subsampling to keep positional encoding manageable
        max_len = min(self.config.max_sequence_length, 4096)  # Cap for memory
        self.pos_encoding = PositionalEncoding(
            max_length=max_len,
            embed_dim=embed_dim,
            name=f"{self.name}_pos_enc"
        )

        # Transformer encoder blocks
        for i in range(self.config.num_layers):
            block = TransformerEncoderBlock(
                num_heads=self.config.num_heads,
                key_dim=self.config.key_dim,
                ff_dim=self.config.ff_dim,
                dropout_rate=self.config.dropout_rate,
                name=f"{self.name}_encoder_{i}"
            )
            self.encoder_blocks.append(block)

        # Pooling
        if self.config.pooling == "mean":
            self.pooling_layer = layers.GlobalAveragePooling1D(
                name=f"{self.name}_gap"
            )
        elif self.config.pooling == "max":
            self.pooling_layer = layers.GlobalMaxPooling1D(
                name=f"{self.name}_gmp"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Apply Transformer aggregation.

        Args:
            inputs: Tensor of shape (batch, time_steps, features).
            training: Training mode flag.

        Returns:
            Aggregated tensor of shape (batch, output_dim).
        """
        batch_size = tf.shape(inputs)[0]
        _seq_len = tf.shape(inputs)[1]

        x = inputs

        # Add CLS token if used
        if self.config.use_cls_token and self.cls_token is not None:
            cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
            x = tf.concat([cls_tokens, x], axis=1)

        # Add positional encoding
        # After downsampling, sequence length should be manageable (e.g., 32768/16 = 2048)
        # Positional encoding handles up to max_sequence_length
        x = self.pos_encoding(x)

        # Apply Transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x, training=training)

        # Pool to single vector
        if self.config.pooling == "cls" and self.config.use_cls_token:
            # Use CLS token output (first position)
            x = x[:, 0, :]
        elif self.config.pooling == "first":
            x = x[:, 0, :]
        elif self.config.pooling == "last":
            x = x[:, -1, :]
        elif self.config.pooling in ["mean", "max"]:
            x = self.pooling_layer(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "key_dim": self.config.key_dim,
                "ff_dim": self.config.ff_dim,
                "dropout_rate": self.config.dropout_rate,
                "use_cls_token": self.config.use_cls_token,
                "pooling": self.config.pooling,
                "max_sequence_length": self.config.max_sequence_length,
            }
        })
        return config


def create_lstm_aggregator(
    units: int = 64,
    bidirectional: bool = True,
    pooling: Literal["last", "mean", "max", "attention"] = "last"
) -> LSTMAggregator:
    """Create LSTM aggregator with specified configuration.

    Args:
        units: Number of LSTM units.
        bidirectional: Whether to use bidirectional LSTM.
        pooling: Pooling mode.

    Returns:
        Configured LSTMAggregator.
    """
    config = LSTMAggregatorConfig(
        units=units,
        num_layers=1,
        cell_type="lstm",
        bidirectional=bidirectional,
        dropout_rate=0.1,
        pooling=pooling,
    )
    return LSTMAggregator(config=config)


def create_transformer_aggregator(
    num_layers: int = 2,
    num_heads: int = 4,
    pooling: Literal["cls", "mean", "max", "first", "last"] = "cls"
) -> TransformerAggregator:
    """Create Transformer aggregator with specified configuration.

    Args:
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        pooling: Pooling mode.

    Returns:
        Configured TransformerAggregator.
    """
    config = TransformerAggregatorConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        key_dim=64,
        ff_dim=128,
        dropout_rate=0.1,
        use_cls_token=(pooling == "cls"),
        pooling=pooling,
    )
    return TransformerAggregator(config=config)
