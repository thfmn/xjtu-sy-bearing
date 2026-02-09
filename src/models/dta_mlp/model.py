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

"""DTA-MLP model implementation.

Reproduces the architecture from Jin et al. (2025):
    CWT Scaleogram → CNN Frontend → Patch Embedding → Transformer (DTA) → CT-MLP → RUL

Implementation assumptions (paper does not specify):
    - CNN frontend: 4 Conv2D blocks with filter doubling (32/64/128/256),
      3×3 kernels, BatchNorm, GELU activation, 2×2 MaxPool
    - DTA: Standard multi-head attention + learned temporal decay bias
      (inspired by ALiBi but with trainable slope parameters)
    - CT-MLP: MLP-Mixer style channel+temporal mixing with GELU and skip connections
    - Patch embedding: Flatten spatial dims from CNN → project to model_dim
    - Total target: ~5.9M parameters (as reported in paper)
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class DTAMLPConfig:
    """Configuration for the DTA-MLP model.

    Attributes:
        input_height: Height of CWT scaleogram input.
        input_width: Width of CWT scaleogram input.
        input_channels: Number of input channels (2 for dual-channel).
        cnn_filters: Filter counts for each CNN block.
        cnn_kernel_size: Kernel size for Conv2D layers.
        model_dim: Transformer/MLP model dimension.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward dimension in Transformer blocks.
        num_transformer_layers: Number of Transformer encoder layers.
        num_ct_mlp_layers: Number of CT-MLP layers.
        dropout_rate: Dropout rate throughout.
        temporal_bias: Whether to use dynamic temporal attention bias.
    """

    input_height: int = 64
    input_width: int = 128
    input_channels: int = 2
    cnn_filters: tuple[int, ...] = (32, 64, 128, 256)
    cnn_kernel_size: int = 3
    model_dim: int = 256
    num_heads: int = 8
    ff_dim: int = 1024
    num_transformer_layers: int = 4
    num_ct_mlp_layers: int = 4
    dropout_rate: float = 0.1
    temporal_bias: bool = True


class CWTFeatureExtractor(layers.Layer):
    """CNN frontend that processes CWT scaleograms into feature sequences.

    Architecture: 4 Conv2D blocks → reshape to sequence of spatial patches.
    Each block: Conv2D → BatchNorm → GELU → Conv2D → BatchNorm → GELU → MaxPool2D

    After the CNN, the spatial feature map is reshaped into a sequence for
    the Transformer. Each spatial position becomes a "token."
    """

    def __init__(self, filters: tuple[int, ...] = (32, 64, 128, 256),
                 kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.conv_blocks = []
        for f in filters:
            block = [
                layers.Conv2D(f, kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("gelu"),
                layers.Conv2D(f, kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("gelu"),
                layers.MaxPool2D(2, 2),
            ]
            self.conv_blocks.append(block)

    def call(self, x, training=None):
        for block in self.conv_blocks:
            for layer in block:
                x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)
        # x shape: (batch, H/16, W/16, last_filters)
        # Reshape to sequence: (batch, num_patches, features)
        batch_size = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        x = tf.reshape(x, [batch_size, h * w, c])
        return x


class DynamicTemporalAttention(layers.Layer):
    """Multi-head attention with dynamic temporal bias.

    Implements standard scaled dot-product attention with an additive temporal
    bias that gives recent positions higher importance. The bias is parameterized
    as: bias[i,j] = -slope * |i - j| where slope is a learned parameter per head.

    This is inspired by ALiBi (Press et al., 2022) but with trainable slopes
    rather than fixed geometric ones.
    """

    def __init__(self, model_dim: int, num_heads: int, dropout_rate: float = 0.1,
                 use_temporal_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.use_temporal_bias = use_temporal_bias

        self.wq = layers.Dense(model_dim)
        self.wk = layers.Dense(model_dim)
        self.wv = layers.Dense(model_dim)
        self.wo = layers.Dense(model_dim)
        self.attn_dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        if self.use_temporal_bias:
            # Learned slopes for temporal decay, one per head
            # Initialize with small positive values so bias decays with distance
            self.temporal_slopes = self.add_weight(
                name="temporal_slopes",
                shape=(self.num_heads,),
                initializer=keras.initializers.Constant(0.1),
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project Q, K, V
        q = self.wq(x)  # (batch, seq, model_dim)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to (batch, heads, seq, head_dim)
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Scaled dot-product attention
        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attn_logits = tf.matmul(q, k, transpose_b=True) / scale  # (batch, heads, seq, seq)

        # Add temporal bias
        if self.use_temporal_bias:
            # distance matrix: |i - j| for all positions
            positions = tf.cast(tf.range(seq_len), tf.float32)
            dist = tf.abs(positions[:, None] - positions[None, :])  # (seq, seq)
            # slopes are softplus'd to ensure positivity
            slopes = tf.nn.softplus(self.temporal_slopes)  # (heads,)
            # bias = -slope * distance, broadcast over batch
            bias = -slopes[:, None, None] * dist[None, :, :]  # (heads, seq, seq)
            attn_logits = attn_logits + bias[None, :, :, :]  # (batch, heads, seq, seq)

        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Apply attention to values
        output = tf.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)
        output = tf.transpose(output, [0, 2, 1, 3])  # (batch, seq, heads, head_dim)
        output = tf.reshape(output, [batch_size, seq_len, self.model_dim])

        return self.wo(output)


class TransformerEncoderBlock(layers.Layer):
    """Single Transformer encoder block with DTA attention.

    Pre-norm architecture: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    """

    def __init__(self, model_dim: int, num_heads: int, ff_dim: int,
                 dropout_rate: float = 0.1, use_temporal_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization()
        self.attn = DynamicTemporalAttention(
            model_dim, num_heads, dropout_rate, use_temporal_bias,
        )
        self.drop1 = layers.Dropout(dropout_rate)

        self.ln2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(model_dim),
            layers.Dropout(dropout_rate),
        ])

    def call(self, x, training=None):
        # Self-attention with residual
        residual = x
        x = self.ln1(x)
        x = self.attn(x, training=training)
        x = self.drop1(x, training=training)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x, training=training)
        x = x + residual

        return x


class CTMixedMLP(layers.Layer):
    """Channel-Temporal Mixed MLP layer (MLP-Mixer style).

    Alternates between:
    1. Channel mixing: MLP across feature dimensions at each position
    2. Temporal mixing: MLP across temporal positions for each feature

    Both pathways use skip connections and GELU activation.
    """

    def __init__(self, model_dim: int, seq_len: int, expansion_factor: int = 4,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim

        # Channel mixing: operates on the feature dimension
        self.ln_channel = layers.LayerNormalization()
        self.channel_mlp = keras.Sequential([
            layers.Dense(model_dim * expansion_factor, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(model_dim),
            layers.Dropout(dropout_rate),
        ])

        # Temporal mixing: operates on the sequence dimension
        self.ln_temporal = layers.LayerNormalization()
        # For temporal mixing, we transpose and apply an MLP across positions
        self.temporal_dense1 = layers.Dense(seq_len * 2, activation="gelu")
        self.temporal_drop = layers.Dropout(dropout_rate)
        self.temporal_dense2 = layers.Dense(seq_len)
        self.temporal_drop2 = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        # Channel mixing with skip connection
        residual = x
        x = self.ln_channel(x)
        x = self.channel_mlp(x, training=training)
        x = x + residual

        # Temporal mixing with skip connection
        residual = x
        x = self.ln_temporal(x)
        # Transpose to (batch, model_dim, seq_len), apply MLP on seq dimension
        x = tf.transpose(x, [0, 2, 1])
        x = self.temporal_dense1(x)
        x = self.temporal_drop(x, training=training)
        x = self.temporal_dense2(x)
        x = self.temporal_drop2(x, training=training)
        x = tf.transpose(x, [0, 2, 1])  # back to (batch, seq, model_dim)
        x = x + residual

        return x


def build_dta_mlp(config: DTAMLPConfig | None = None) -> keras.Model:
    """Build the full DTA-MLP model.

    Architecture:
        Input (H, W, 2) → CNN Frontend → Patch Embed → Transformer (DTA) × N
        → CT-MLP × N → Global Average Pooling → Dense → RUL

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Compiled Keras model with input shape (None, H, W, 2) and output (None, 1).
    """
    if config is None:
        config = DTAMLPConfig()

    # Input
    inputs = keras.Input(
        shape=(config.input_height, config.input_width, config.input_channels),
        name="cwt_input",
    )

    # CNN Frontend: extract features → sequence
    x = CWTFeatureExtractor(
        filters=config.cnn_filters,
        kernel_size=config.cnn_kernel_size,
        name="cwt_frontend",
    )(inputs)
    # x shape: (batch, num_patches, cnn_last_filters)

    # Project to model_dim if CNN output channels != model_dim
    cnn_out_dim = config.cnn_filters[-1]
    if cnn_out_dim != config.model_dim:
        x = layers.Dense(config.model_dim, name="patch_projection")(x)

    # Compute seq_len from input dims: each CNN block has one MaxPool2D(2,2)
    pool_factor = 2 ** len(config.cnn_filters)
    seq_len = (config.input_height // pool_factor) * (config.input_width // pool_factor)

    # Learnable positional embedding
    pos_embedding = layers.Embedding(seq_len, config.model_dim, name="pos_embedding")
    positions = tf.range(seq_len)
    x = x + pos_embedding(positions)
    x = layers.Dropout(config.dropout_rate)(x)

    # Transformer encoder with DTA
    for i in range(config.num_transformer_layers):
        x = TransformerEncoderBlock(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
            use_temporal_bias=config.temporal_bias,
            name=f"transformer_block_{i}",
        )(x)

    # CT-MLP layers
    for i in range(config.num_ct_mlp_layers):
        x = CTMixedMLP(
            model_dim=config.model_dim,
            seq_len=seq_len,
            dropout_rate=config.dropout_rate,
            name=f"ct_mlp_{i}",
        )(x)

    # Final layer norm
    x = layers.LayerNormalization(name="final_ln")(x)

    # Global average pooling over sequence dimension
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # RUL head
    x = layers.Dense(128, activation="gelu", name="rul_dense1")(x)
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(64, activation="gelu", name="rul_dense2")(x)
    x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="rul_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dta_mlp")
    return model


def create_default_dta_mlp() -> keras.Model:
    """Create DTA-MLP with default configuration for XJTU-SY CWT scaleograms.

    Input shape: (64, 128, 2) matching pre-generated CWT scaleograms.
    Target: ~5.9M parameters (as reported in Jin et al. 2025).
    """
    return build_dta_mlp(DTAMLPConfig())
