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

"""MDSCT model implementation.

Reproduces the architecture from Li et al. (2024):
    Raw Signal → Conv1D stem → MixerBlock (MDSC + ECA) × 3
    → PatchEmbedding → ProbSparse Transformer → AdaptiveAvgPool → FC → RUL

Implementation assumptions (paper does not fully specify):
    - MDSC kernel sizes: [8, 16, 32] for multi-scale temporal feature extraction
    - ECA: Efficient Channel Attention with adaptive kernel size k
    - ProbSparse attention: from Informer (Zhou et al. 2021), sampling factor c=5
    - AdaptH_Swish: x * relu6(x + a) / 6 with trainable a initialized to 3.0
    - Patch embedding: length 16, stride 8 (50% overlap)
    - 4 attention heads, model dimension matching stem output channels
    - Stem Conv1D: kernel=64, stride=2, reduces sequence from 32768 to 16384
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class MDSCTConfig:
    """Configuration for the MDSCT model.

    Attributes:
        input_length: Length of raw vibration signal.
        input_channels: Number of input channels (2 for H/V).
        stem_filters: Filters in initial Conv1D stem.
        stem_kernel: Kernel size for stem Conv1D.
        stem_stride: Stride for stem Conv1D (downsamples the signal).
        mdsc_kernels: Kernel sizes for multi-scale depthwise separable conv branches.
        num_mixer_blocks: Number of MDSC + ECA mixer blocks.
        model_dim: Dimension for Transformer/patch embedding.
        num_heads: Number of attention heads in ProbSparse Transformer.
        ff_dim: Feed-forward dimension in Transformer block.
        num_transformer_layers: Number of Transformer encoder layers.
        patch_length: Length of each patch for embedding.
        patch_stride: Stride between patches.
        probsparse_factor: Sampling factor c for ProbSparse attention (u = c * ln(L)).
        dropout_rate: Dropout rate.
        adapt_hswish_init: Initial value for AdaptH_Swish trainable parameter.
    """

    input_length: int = 32768
    input_channels: int = 2
    stem_filters: int = 64
    stem_kernel: int = 64
    stem_stride: int = 2
    mdsc_kernels: tuple[int, ...] = (8, 16, 32)
    num_mixer_blocks: int = 3
    model_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 512
    num_transformer_layers: int = 2
    patch_length: int = 16
    patch_stride: int = 8
    probsparse_factor: int = 5
    dropout_rate: float = 0.05
    adapt_hswish_init: float = 3.0


class AdaptHSwish(layers.Layer):
    """Adaptive H-Swish activation with trainable shift parameter.

    Formula: AdaptHSwish(x) = x * relu6(x + a) / 6
    where a is a trainable scalar, initialized to 3.0 (standard H-Swish).

    When a=3.0, this is equivalent to standard H-Swish:
        x * relu6(x + 3) / 6

    Making a trainable allows the network to learn the optimal activation shape.
    """

    def __init__(self, init_value: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.a = self.add_weight(
            name="adapt_shift",
            shape=(),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x * tf.nn.relu6(x + self.a) / 6.0

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


class EfficientChannelAttention(layers.Layer):
    """Efficient Channel Attention (ECA) module.

    Uses a 1D convolution across channels (after GAP) instead of FC layers,
    making it extremely lightweight. The kernel size k is computed adaptively
    based on the number of channels.

    Paper: ECA-Net (Wang et al. 2020)
    Formula: k = |log2(C) / 2 + 0.5| rounded to nearest odd integer

    Architecture: GAP → 1D Conv(k) → Sigmoid → Channel reweighting
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        # Adaptive kernel size based on channel count
        k = int(abs(math.log2(channels) / 2 + 0.5))
        k = k if k % 2 == 1 else k + 1  # Ensure odd
        k = max(k, 3)  # Minimum kernel size of 3

        self.avg_pool = layers.GlobalAveragePooling1D(keepdims=True)
        # Conv1D across channels: reshape to (batch, channels, 1) and apply
        self.conv = layers.Conv1D(1, kernel_size=k, padding="same", use_bias=False)
        super().build(input_shape)

    def call(self, x):
        # x: (batch, seq_len, channels)
        # Global average pool: (batch, 1, channels)
        attn = self.avg_pool(x)
        # Transpose to (batch, channels, 1) for Conv1D across channels
        attn = tf.transpose(attn, [0, 2, 1])
        attn = self.conv(attn)
        attn = tf.nn.sigmoid(attn)
        # Transpose back to (batch, 1, channels) and broadcast multiply
        attn = tf.transpose(attn, [0, 2, 1])
        return x * attn


class MultiScaleDepthwiseSeparableConv(layers.Layer):
    """Multi-scale Depthwise Separable Convolution (MDSC) module.

    Processes input through parallel depthwise separable convolution branches
    at different kernel sizes, capturing features at multiple temporal scales.
    Outputs are summed and passed through a pointwise convolution.

    Each branch: DepthwiseConv1D(k) → BatchNorm → Pointwise Conv1D → BatchNorm
    """

    def __init__(self, filters: int, kernel_sizes: tuple[int, ...] = (8, 16, 32),
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.branches = []

    def build(self, input_shape):
        in_channels = input_shape[-1]
        for k in self.kernel_sizes:
            branch = [
                layers.DepthwiseConv1D(kernel_size=k, padding="same"),
                layers.BatchNormalization(),
                layers.Conv1D(self.filters, kernel_size=1),  # pointwise
                layers.BatchNormalization(),
            ]
            self.branches.append(branch)

        # Final pointwise to unify after summation
        self.proj = layers.Conv1D(self.filters, kernel_size=1)
        self.proj_bn = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        branch_outputs = []
        for branch in self.branches:
            out = x
            for layer in branch:
                if isinstance(layer, layers.BatchNormalization):
                    out = layer(out, training=training)
                else:
                    out = layer(out)
            branch_outputs.append(out)

        # Sum multi-scale features
        combined = tf.add_n(branch_outputs)
        combined = self.proj(combined)
        combined = self.proj_bn(combined, training=training)
        return combined


class MixerBlock(layers.Layer):
    """MDSC + ECA mixer block with residual connection and AdaptH_Swish.

    Architecture: Input → MDSC → AdaptH_Swish → ECA → Residual Add → Output

    If input channels differ from MDSC output, a 1x1 conv projection is used
    for the residual path.
    """

    def __init__(self, filters: int, kernel_sizes: tuple[int, ...] = (8, 16, 32),
                 adapt_hswish_init: float = 3.0, dropout_rate: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.mdsc = MultiScaleDepthwiseSeparableConv(
            filters=filters, kernel_sizes=kernel_sizes,
        )
        self.activation = AdaptHSwish(init_value=adapt_hswish_init)
        self.eca = EfficientChannelAttention()
        self.dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels != self.filters:
            self.residual_proj = layers.Conv1D(self.filters, kernel_size=1)
        else:
            self.residual_proj = None
        super().build(input_shape)

    def call(self, x, training=None):
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        x = self.mdsc(x, training=training)
        x = self.activation(x)
        x = self.eca(x)
        x = self.dropout(x, training=training)
        return x + residual


class PatchEmbedding(layers.Layer):
    """Segment input into overlapping patches and project to embedding dim.

    Splits the 1D sequence into patches of fixed length with a given stride,
    then projects each patch to the model dimension via a Dense layer.

    Example: seq_len=16384, patch_length=16, stride=8
             → num_patches = (16384 - 16) / 8 + 1 = 2047 patches
    """

    def __init__(self, patch_length: int, patch_stride: int, model_dim: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.model_dim = model_dim
        self.projection = layers.Dense(model_dim)
        self.ln = layers.LayerNormalization()

    def call(self, x):
        # x: (batch, seq_len, channels)
        batch_size = tf.shape(x)[0]
        # Extract patches using tf.signal.frame
        # frame operates on the last axis by default, we need it on axis 1
        # Transpose to (batch, channels, seq_len), frame, then reshape
        patches = tf.signal.frame(x, self.patch_length, self.patch_stride, axis=1)
        # patches shape: (batch, num_patches, patch_length, channels)
        # Flatten each patch: (batch, num_patches, patch_length * channels)
        num_patches = tf.shape(patches)[1]
        patches = tf.reshape(patches, [batch_size, num_patches,
                                       self.patch_length * tf.shape(x)[-1]])
        # Project to model_dim
        patches = self.projection(patches)
        patches = self.ln(patches)
        return patches


class ProbSparseAttention(layers.Layer):
    """ProbSparse Self-Attention from the Informer paper (Zhou et al. 2021).

    Instead of computing attention for all queries, ProbSparse selects the
    top-u most informative queries based on a KL-divergence-like measure
    between each query's attention distribution and a uniform distribution.
    Queries with spikier distributions (further from uniform) are more
    informative and are selected.

    u = c * ln(L_Q) where c is the sampling factor and L_Q is sequence length.

    For non-selected queries, the output defaults to the mean of values (V),
    providing a reasonable fallback.

    Note: For short sequences (< 64 tokens), this falls back to standard
    attention since the overhead of selection exceeds any savings.
    """

    def __init__(self, model_dim: int, num_heads: int, factor: int = 5,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.factor = factor

        self.wq = layers.Dense(model_dim)
        self.wk = layers.Dense(model_dim)
        self.wv = layers.Dense(model_dim)
        self.wo = layers.Dense(model_dim)
        self.attn_dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to multi-head: (batch, heads, seq, head_dim)
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))

        # Compute number of top queries to keep
        # u = min(factor * ceil(ln(L)), L)
        u = tf.minimum(
            self.factor * tf.cast(
                tf.math.ceil(tf.math.log(tf.cast(seq_len, tf.float32))), tf.int32
            ),
            seq_len,
        )

        # For short sequences, use full attention
        use_full = tf.less(seq_len, 64)

        def full_attention():
            attn_logits = tf.matmul(q, k, transpose_b=True) / scale
            attn_weights = tf.nn.softmax(attn_logits, axis=-1)
            attn_weights = self.attn_dropout(attn_weights, training=training)
            return tf.matmul(attn_weights, v)

        def sparse_attention():
            # Compute Q·K^T scores for all queries
            attn_logits = tf.matmul(q, k, transpose_b=True) / scale
            # (batch, heads, seq, seq)

            # Measure "sparsity" of each query's attention distribution
            # M(q_i) = max(q_i · K^T) - mean(q_i · K^T)
            # Higher M means more informative (spikier) query
            M = tf.reduce_max(attn_logits, axis=-1) - tf.reduce_mean(
                attn_logits, axis=-1
            )
            # M shape: (batch, heads, seq)

            # Select top-u queries per head
            _, top_indices = tf.math.top_k(M, k=u)
            # top_indices: (batch, heads, u)

            # Default output: mean of values (for non-selected queries)
            v_mean = tf.reduce_mean(v, axis=2, keepdims=True)  # (batch, heads, 1, d)
            v_mean_broadcast = tf.broadcast_to(
                v_mean, tf.shape(v)
            )  # (batch, heads, seq, d)

            # Gather selected queries
            # Expand indices for gather: (batch, heads, u, 1)
            top_idx_expanded = tf.expand_dims(top_indices, -1)
            top_idx_q = tf.broadcast_to(
                top_idx_expanded, [batch_size, self.num_heads, u, self.head_dim]
            )

            # Gather Q for top queries
            q_top = tf.gather(q, top_indices, axis=2, batch_dims=2)
            # q_top: (batch, heads, u, head_dim)

            # Compute attention only for top queries against all keys
            top_attn_logits = tf.matmul(q_top, k, transpose_b=True) / scale
            top_attn_weights = tf.nn.softmax(top_attn_logits, axis=-1)
            top_attn_weights = self.attn_dropout(
                top_attn_weights, training=training
            )
            top_output = tf.matmul(top_attn_weights, v)
            # top_output: (batch, heads, u, head_dim)

            # Scatter top outputs back into full output
            output = tf.identity(v_mean_broadcast)
            # Use tensor_scatter_nd_update to place top outputs
            # Create indices for scatter: (batch, heads, u) → need (batch*heads*u, 4)
            batch_idx = tf.repeat(
                tf.range(batch_size), self.num_heads * u
            )
            head_idx = tf.tile(
                tf.repeat(tf.range(self.num_heads), u), [batch_size]
            )
            seq_idx = tf.reshape(top_indices, [-1])
            scatter_indices = tf.stack([batch_idx, head_idx, seq_idx], axis=1)
            flat_top_output = tf.reshape(top_output, [-1, self.head_dim])

            output = tf.tensor_scatter_nd_update(
                output, scatter_indices, flat_top_output
            )
            return output

        output = tf.cond(use_full, full_attention, sparse_attention)

        # Reshape back: (batch, seq, model_dim)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, seq_len, self.model_dim])
        return self.wo(output)


class TransformerBlock(layers.Layer):
    """Transformer encoder block with ProbSparse attention.

    Pre-norm architecture: LayerNorm → ProbSparse Attention → Residual
                          → LayerNorm → FFN → Residual
    """

    def __init__(self, model_dim: int, num_heads: int, ff_dim: int,
                 probsparse_factor: int = 5, dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization()
        self.attn = ProbSparseAttention(
            model_dim, num_heads, factor=probsparse_factor,
            dropout_rate=dropout_rate,
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
        residual = x
        x = self.ln1(x)
        x = self.attn(x, training=training)
        x = self.drop1(x, training=training)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.ffn(x, training=training)
        x = x + residual
        return x


def build_mdsct(config: MDSCTConfig | None = None) -> keras.Model:
    """Build the full MDSCT model.

    Architecture:
        Input (L, 2) → Conv1D stem → MixerBlock × 3 → PatchEmbedding
        → Transformer × N → GlobalAvgPool → Dense → Dropout → Dense → RUL

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Keras model with input shape (None, L, 2) and output (None, 1).
    """
    if config is None:
        config = MDSCTConfig()

    inputs = keras.Input(
        shape=(config.input_length, config.input_channels),
        name="signal_input",
    )

    # Stem: initial Conv1D to downsample and expand channels
    x = layers.Conv1D(
        config.stem_filters, kernel_size=config.stem_kernel,
        strides=config.stem_stride, padding="same",
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = AdaptHSwish(init_value=config.adapt_hswish_init, name="stem_act")(x)

    # Mixer blocks: MDSC + ECA with residual
    for i in range(config.num_mixer_blocks):
        x = MixerBlock(
            filters=config.stem_filters,
            kernel_sizes=config.mdsc_kernels,
            adapt_hswish_init=config.adapt_hswish_init,
            dropout_rate=config.dropout_rate,
            name=f"mixer_block_{i}",
        )(x)

    # Patch embedding: segment into overlapping patches and project
    x = PatchEmbedding(
        patch_length=config.patch_length,
        patch_stride=config.patch_stride,
        model_dim=config.model_dim,
        name="patch_embed",
    )(x)

    # Add positional embedding
    # Compute expected number of patches
    stem_out_len = config.input_length // config.stem_stride
    num_patches = (stem_out_len - config.patch_length) // config.patch_stride + 1
    pos_embedding = layers.Embedding(
        num_patches, config.model_dim, name="pos_embedding",
    )
    positions = tf.range(num_patches)
    x = x + pos_embedding(positions)
    x = layers.Dropout(config.dropout_rate)(x)

    # ProbSparse Transformer encoder
    for i in range(config.num_transformer_layers):
        x = TransformerBlock(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            probsparse_factor=config.probsparse_factor,
            dropout_rate=config.dropout_rate,
            name=f"transformer_{i}",
        )(x)

    # Final layer norm
    x = layers.LayerNormalization(name="final_ln")(x)

    # Global average pooling over patch dimension
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # RUL head
    x = layers.Dense(64, activation="gelu", name="rul_dense1")(x)
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(1, activation="sigmoid", name="rul_output")(x)

    model = keras.Model(inputs=inputs, outputs=x, name="mdsct")
    return model


def create_default_mdsct() -> keras.Model:
    """Create MDSCT with default configuration for XJTU-SY raw signals.

    Input shape: (32768, 2) matching raw vibration signal files.
    """
    return build_mdsct(MDSCTConfig())
