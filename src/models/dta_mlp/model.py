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

Loosely inspired by the architecture from Jin et al. (2025):
    CWT Scaleogram → CNN Frontend → Patch Embedding → Transformer (DTA) → CT-MLP → RUL

The paper describes a CWT → CNN → Dynamic Attention → CT-MLP pipeline. Our
implementation follows the high-level architecture but interprets underspecified
details with the following deviations:

    - CNN frontend: 3 single-conv blocks (Conv2D → BN → ReLU → MaxPool),
      filters (32, 64, 128). Paper shows 3-5 conv layers with ReLU.
    - Attention rectification: soft-threshold gating on attention weights
      (learn a per-head threshold, smoothly zero out weights below it).
      The paper describes "attenuating less important features by weakening
      their attention or masking through zeroing out" but does not specify
      the exact mechanism.
    - CT-MLP interleaved inside each transformer block (Attention → FFN →
      CT-MLP repeated ×N), matching Figure 3 of the paper.
    - Linear output layer (no sigmoid), consistent with paper Fig. 10/14
      where predictions go negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import tensorflow as tf
from keras import layers


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
        num_layers: Number of Transformer + CT-MLP blocks (interleaved).
        dropout_rate: Dropout rate throughout.
        attention_rectification: Whether to use attention weight rectification.
    """

    input_height: int = 64
    input_width: int = 128
    input_channels: int = 2
    cnn_filters: tuple[int, ...] = (32, 64, 128)
    cnn_kernel_size: int = 3
    model_dim: int = 256
    num_heads: int = 8
    ff_dim: int = 1024
    num_layers: int = 4
    dropout_rate: float = 0.1
    attention_rectification: bool = True


class CWTFeatureExtractor(layers.Layer):
    """CNN frontend that processes CWT scaleograms into feature sequences.

    Architecture: Single-conv blocks → reshape to sequence of spatial patches.
    Each block: Conv2D → BatchNorm → ReLU → MaxPool2D(2,2)

    After the CNN, the spatial feature map is reshaped into a sequence for
    the Transformer. Each spatial position becomes a "token."
    """

    def __init__(self, filters: tuple[int, ...] = (32, 64, 128),
                 kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.conv_blocks = []
        for f in filters:
            block = [
                layers.Conv2D(f, kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPool2D(2, 2),
            ]
            self.conv_blocks.append(block)

    def call(self, x, training=None):
        for block in self.conv_blocks:
            for layer in block:
                x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)
        # x shape: (batch, H/2^n, W/2^n, last_filters)
        # Reshape to sequence: (batch, num_patches, features)
        batch_size = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        x = tf.reshape(x, [batch_size, h * w, c])
        return x


class DynamicTemporalAttention(layers.Layer):
    """Multi-head attention with dynamic attention rectification.

    Implements standard scaled dot-product attention with optional
    soft-threshold gating that attenuates low-importance attention weights.

    After softmax, a learned per-head threshold is applied: attention weights
    below the threshold are smoothly gated toward zero via a sigmoid gate.
    This implements the paper's concept of "attenuating less important features
    by weakening their attention or masking through zeroing out."
    """

    def __init__(self, model_dim: int, num_heads: int, dropout_rate: float = 0.1,
                 use_attention_rectification: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.use_attention_rectification = use_attention_rectification

        self.wq = layers.Dense(model_dim)
        self.wk = layers.Dense(model_dim)
        self.wv = layers.Dense(model_dim)
        self.wo = layers.Dense(model_dim)
        self.attn_dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        if self.use_attention_rectification:
            # Learned threshold for attention rectification, one per head.
            # Initialized near 0 so sigmoid(threshold) ≈ 0.5, gating starts moderate.
            self.learned_threshold = self.add_weight(
                name="learned_threshold",
                shape=(self.num_heads,),
                initializer=keras.initializers.Zeros(),
                trainable=True,
            )
            # Sharpness of the gating sigmoid. Higher = sharper cutoff.
            self.gate_sharpness = self.add_weight(
                name="gate_sharpness",
                shape=(self.num_heads,),
                initializer=keras.initializers.Constant(10.0),
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

        attn_weights = tf.nn.softmax(attn_logits, axis=-1)

        # Attention rectification: soft-threshold gating
        if self.use_attention_rectification:
            # threshold in (0, 1) via sigmoid
            threshold = tf.nn.sigmoid(self.learned_threshold)  # (heads,)
            # Smooth gate: values above threshold → 1, below → 0
            gate = tf.nn.sigmoid(
                self.gate_sharpness[:, None, None]
                * (attn_weights - threshold[:, None, None])
            )  # (batch, heads, seq, seq)
            attn_weights = attn_weights * gate

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
                 dropout_rate: float = 0.1, use_attention_rectification: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization()
        self.attn = DynamicTemporalAttention(
            model_dim, num_heads, dropout_rate, use_attention_rectification,
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
        Input (H, W, 2) → CNN Frontend → Patch Embed
        → [Transformer (DTA) + CT-MLP] × N → Global Average Pooling → Dense → RUL

    Each block interleaves a Transformer encoder (with attention rectification)
    and a CT-MLP layer, following the paper's Figure 3.

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

    # Interleaved Transformer + CT-MLP blocks (paper Figure 3)
    for i in range(config.num_layers):
        x = TransformerEncoderBlock(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
            use_attention_rectification=config.attention_rectification,
            name=f"transformer_block_{i}",
        )(x)
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
    outputs = layers.Dense(1, name="rul_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dta_mlp")
    return model


def create_default_dta_mlp() -> keras.Model:
    """Create DTA-MLP with default configuration for XJTU-SY CWT scaleograms.

    Input shape: (64, 128, 2) matching pre-generated CWT scaleograms.
    """
    return build_dta_mlp(DTAMLPConfig())
