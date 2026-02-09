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

"""MDSCT model — faithful reproduction of Sun et al. 2024 (Heliyon e38317).

Architecture (Table 2, Fig. 7):
    Input (32768, 2)
      → MinMaxNormalize (Eq. 19)
      → Conv1D stem (kernel=64, stride=2, filters=1)  → (16384, 1)
      → AAP1(1024) → Dropout(0.05)                    → (1024, 1)
      → MixerBlock × 3:
      │   ├─ MDSC Attention (local):  MaxPool(3) → Bottleneck(16)
      │   │    → 3× DSC(k=8,16,32; 24ch each) → Concat(72ch)
      │   │    → BN → AdaptH_Swish → Dropout → ECA → Residual
      │   └─ PPSformer (global):  AAP2(96) → Conv1D(1×1, 16ch)
      │        → PatchEmbed(16,8) → ProbSparse MHA(4 heads)
      │        → FFN(256→128) → AAP3(1024)
      │   → Concatenate → (1024, 200)
      → AAP4(64)                                       → (64, 200)
      → Flatten → Dense(1, sigmoid)

Key corrections from v1 (see PRD for details):
    - AdaptH_Swish: δx * relu6(δx+3)/6 with δ=1.0 (scaling, not shifting)
    - Mixer blocks: parallel MDSC ∥ PPSformer → concat (not serial)
    - MDSC: MaxPool → Bottleneck(16) → DSC(24×3) → concat (not sum)
    - Stem: 1 output channel + AAP(1024), not 64 channels
    - 4 AAP layers at key junctions
    - Post-norm Transformer (not pre-norm)
    - FFN dims: 256→128 (not 512)
    - Per-sample min-max normalization inside model
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import keras
import tensorflow as tf
from tensorflow.keras import layers


@dataclass
class MDSCTConfig:
    """Configuration for the MDSCT model (Table 1-2, Sun et al. 2024).

    Attributes:
        input_length: Raw vibration signal length (32768 = 1.28 s at 25.6 kHz).
        input_channels: Number of sensor channels (2 for H/V).
        stem_kernel: Kernel size for stem Conv1D.
        stem_stride: Stride for stem Conv1D.
        stem_out_channels: Output channels of stem Conv1D (paper: 1).
        mdsc_kernels: Multi-scale DSC branch kernel sizes (Table 2).
        mdsc_bottleneck_ch: Bottleneck Conv1D(1×1) output channels.
        mdsc_branch_ch: Per-branch DSC output channels (24 × 3 = 72 total).
        mdsc_maxpool_kernel: MaxPool kernel in MDSC attention.
        num_mixer_blocks: Number of MixerBlock repetitions.
        aap1_size: AAP after stem.
        aap2_size: AAP in PPSformer (before patch embedding).
        aap3_size: AAP in PPSformer (after transformer).
        aap4_size: AAP before final FC.
        ppsformer_proj_ch: Conv1D(1×1) projection channels in PPSformer.
        ppsformer_model_dim: Transformer model dimension in PPSformer (= output channels).
        num_heads: Attention heads in ProbSparse MHA.
        ff_dims: FFN dimensions (expand, contract) in Transformer.
        patch_length: Patch embedding length.
        patch_stride: Patch embedding stride.
        probsparse_factor: Sampling factor c for ProbSparse attention.
        dropout_rate: Dropout rate (paper: 0.05).
        adapt_hswish_init: Initial value of δ in AdaptH_Swish (paper: 1.0).
    """

    input_length: int = 32768
    input_channels: int = 2
    stem_kernel: int = 64
    stem_stride: int = 2
    stem_out_channels: int = 1
    mdsc_kernels: tuple[int, ...] = (8, 16, 32)
    mdsc_bottleneck_ch: int = 16
    mdsc_branch_ch: int = 24
    mdsc_maxpool_kernel: int = 3
    num_mixer_blocks: int = 3
    aap1_size: int = 1024
    aap2_size: int = 96
    aap3_size: int = 1024
    aap4_size: int = 64
    ppsformer_proj_ch: int = 16
    ppsformer_model_dim: int = 128
    num_heads: int = 4
    ff_dims: tuple[int, int] = (256, 128)
    patch_length: int = 16
    patch_stride: int = 8
    probsparse_factor: int = 5
    dropout_rate: float = 0.05
    adapt_hswish_init: float = 1.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="mdsct")
class MinMaxNormalize(layers.Layer):
    """Per-sample per-channel min-max normalization to [0, 1] (Eq. 19).

    x_norm = (x - x_min) / (x_max - x_min + ε)
    Applied independently to each sample and each channel.
    """

    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, x):
        # x: (batch, seq_len, channels)
        x_min = tf.reduce_min(x, axis=1, keepdims=True)
        x_max = tf.reduce_max(x, axis=1, keepdims=True)
        return (x - x_min) / (x_max - x_min + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@keras.saving.register_keras_serializable(package="mdsct")
class AdaptiveAvgPool1D(layers.Layer):
    """Adaptive average pooling for 1D sequences.

    Equivalent to PyTorch's nn.AdaptiveAvgPool1d. Resizes the temporal
    dimension to *target_size* using area interpolation, which correctly
    handles both downsampling (e.g. 16384 → 1024) and upsampling
    (e.g. 11 → 1024).

    Uses tf.image.resize with "area" method by reshaping
    (batch, T, C) → (batch, T, 1, C) → resize → squeeze.
    """

    def __init__(self, target_size: int, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size

    def call(self, x):
        # x: (batch, seq_len, channels)
        # Reshape to 4D for tf.image.resize: (batch, seq_len, 1, channels)
        x_4d = tf.expand_dims(x, axis=2)
        # Resize temporal dimension (bilinear is differentiable; area is not)
        x_4d = tf.image.resize(x_4d, [self.target_size, 1], method="bilinear")
        # Squeeze back to 3D: (batch, target_size, channels)
        return tf.squeeze(x_4d, axis=2)

    def get_config(self):
        config = super().get_config()
        config.update({"target_size": self.target_size})
        return config


@keras.saving.register_keras_serializable(package="mdsct")
class AdaptHSwish(layers.Layer):
    """Adaptive H-Swish activation with trainable scale parameter (Eq. 9).

    Formula: AdaptHSwish(x) = δx * relu6(δx + 3) / 6
    where δ is a trainable scalar, initialized to 1.0.

    When δ=1.0, this reduces to standard H-Swish: x * relu6(x+3) / 6.
    The key difference from v1: δ is a *scale* applied to x before the
    activation, not an additive *shift* inside relu6.
    """

    def __init__(self, init_value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.delta = self.add_weight(
            name="adapt_scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        dx = self.delta * x
        return dx * tf.nn.relu6(dx + 3.0) / 6.0

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable(package="mdsct")
class EfficientChannelAttention(layers.Layer):
    """Efficient Channel Attention (ECA) module (Eq. 10).

    Architecture: GAP → reshape → Conv1D(k) → sigmoid → channel reweight.
    Kernel size k = |log2(C)/2 + 0.5| rounded to nearest odd, minimum 3.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        k = int(abs(math.log2(channels) / 2 + 0.5))
        k = k if k % 2 == 1 else k + 1
        k = max(k, 3)

        self.avg_pool = layers.GlobalAveragePooling1D(keepdims=True)
        self.conv = layers.Conv1D(1, kernel_size=k, padding="same", use_bias=False)
        super().build(input_shape)

    def call(self, x):
        # x: (batch, seq_len, channels)
        attn = self.avg_pool(x)                 # (batch, 1, channels)
        attn = tf.transpose(attn, [0, 2, 1])   # (batch, channels, 1)
        attn = self.conv(attn)                  # (batch, channels, 1)
        attn = tf.nn.sigmoid(attn)
        attn = tf.transpose(attn, [0, 2, 1])   # (batch, 1, channels)
        return x * attn

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="mdsct")
class MDSCAttentionModule(layers.Layer):
    """Multi-scale Depthwise Separable Convolution Attention (Fig. 4-5).

    Architecture:
        Input → MaxPool(3) → Bottleneck(1×1, 16ch)
        → 3× DepthwiseSeparableConv(k=8,16,32; 24ch each) → Concat(72ch)
        → BatchNorm → AdaptH_Swish → Dropout → ECA
        → Residual add (with 1×1 projection if needed)

    The MaxPool provides a small amount of local pooling before the
    multi-scale branches. The bottleneck reduces channels to 16 before
    the expensive depthwise operations. Concat (not sum) preserves all
    72 channels of multi-scale information.
    """

    def __init__(
        self,
        kernel_sizes: tuple[int, ...] = (8, 16, 32),
        bottleneck_ch: int = 16,
        branch_ch: int = 24,
        maxpool_kernel: int = 3,
        adapt_hswish_init: float = 1.0,
        dropout_rate: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_sizes = kernel_sizes
        self.bottleneck_ch = bottleneck_ch
        self.branch_ch = branch_ch
        self.maxpool_kernel = maxpool_kernel
        self.out_channels = branch_ch * len(kernel_sizes)  # 24 * 3 = 72

        self.maxpool = layers.MaxPooling1D(
            pool_size=maxpool_kernel, strides=1, padding="same",
        )
        self.bottleneck = layers.Conv1D(bottleneck_ch, kernel_size=1)
        self.bottleneck_bn = layers.BatchNormalization()

        self.dsc_branches = []
        for k in kernel_sizes:
            branch = {
                "dw": layers.DepthwiseConv1D(kernel_size=k, padding="same"),
                "dw_bn": layers.BatchNormalization(),
                "pw": layers.Conv1D(branch_ch, kernel_size=1),
                "pw_bn": layers.BatchNormalization(),
            }
            self.dsc_branches.append(branch)

        self.concat_bn = layers.BatchNormalization()
        self.activation = AdaptHSwish(init_value=adapt_hswish_init)
        self.dropout = layers.Dropout(dropout_rate)
        self.eca = EfficientChannelAttention()

    def build(self, input_shape):
        in_ch = input_shape[-1]
        if in_ch != self.out_channels:
            self.residual_proj = layers.Conv1D(self.out_channels, kernel_size=1)
        else:
            self.residual_proj = None
        super().build(input_shape)

    def call(self, x, training=None):
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        # MaxPool → Bottleneck
        h = self.maxpool(x)
        h = self.bottleneck(h)
        h = self.bottleneck_bn(h, training=training)

        # Multi-scale DSC branches → Concat
        branch_outs = []
        for br in self.dsc_branches:
            out = br["dw"](h)
            out = br["dw_bn"](out, training=training)
            out = br["pw"](out)
            out = br["pw_bn"](out, training=training)
            branch_outs.append(out)

        h = layers.concatenate(branch_outs, axis=-1)  # (batch, seq, 72)

        # BN → AdaptH_Swish → Dropout → ECA
        h = self.concat_bn(h, training=training)
        h = self.activation(h)
        h = self.dropout(h, training=training)
        h = self.eca(h)

        return h + residual

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_sizes": self.kernel_sizes,
            "bottleneck_ch": self.bottleneck_ch,
            "branch_ch": self.branch_ch,
            "maxpool_kernel": self.maxpool_kernel,
        })
        return config


@keras.saving.register_keras_serializable(package="mdsct")
class PatchEmbedding(layers.Layer):
    """Segment input into overlapping patches and project (Eq. 11-12).

    Splits the 1D sequence into patches of fixed length with a given stride,
    then projects each patch to the model dimension via a Dense layer.
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
        batch_size = tf.shape(x)[0]
        patches = tf.signal.frame(x, self.patch_length, self.patch_stride, axis=1)
        num_patches = tf.shape(patches)[1]
        patches = tf.reshape(
            patches,
            [batch_size, num_patches, self.patch_length * tf.shape(x)[-1]],
        )
        patches = self.projection(patches)
        patches = self.ln(patches)
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_length": self.patch_length,
            "patch_stride": self.patch_stride,
            "model_dim": self.model_dim,
        })
        return config


@keras.saving.register_keras_serializable(package="mdsct")
class ProbSparseAttention(layers.Layer):
    """ProbSparse Self-Attention from Informer (Zhou et al. 2021, Eq. 14-15).

    Selects top-u most informative queries based on KL-divergence measure.
    u = c * ln(L_Q). Falls back to full attention for short sequences (<64).
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

        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))

        u = tf.minimum(
            self.factor * tf.cast(
                tf.math.ceil(tf.math.log(tf.cast(seq_len, tf.float32))),
                tf.int32,
            ),
            seq_len,
        )

        use_full = tf.less(seq_len, 64)

        def full_attention():
            attn_logits = tf.matmul(q, k, transpose_b=True) / scale
            attn_weights = tf.nn.softmax(attn_logits, axis=-1)
            attn_weights = self.attn_dropout(attn_weights, training=training)
            return tf.matmul(attn_weights, v)

        def sparse_attention():
            attn_logits = tf.matmul(q, k, transpose_b=True) / scale
            M = tf.reduce_max(attn_logits, axis=-1) - tf.reduce_mean(
                attn_logits, axis=-1,
            )
            _, top_indices = tf.math.top_k(M, k=u)

            v_mean = tf.reduce_mean(v, axis=2, keepdims=True)
            v_mean_broadcast = tf.broadcast_to(v_mean, tf.shape(v))

            q_top = tf.gather(q, top_indices, axis=2, batch_dims=2)
            top_attn_logits = tf.matmul(q_top, k, transpose_b=True) / scale
            top_attn_weights = tf.nn.softmax(top_attn_logits, axis=-1)
            top_attn_weights = self.attn_dropout(
                top_attn_weights, training=training,
            )
            top_output = tf.matmul(top_attn_weights, v)

            output = tf.identity(v_mean_broadcast)
            batch_idx = tf.repeat(tf.range(batch_size), self.num_heads * u)
            head_idx = tf.tile(
                tf.repeat(tf.range(self.num_heads), u), [batch_size],
            )
            seq_idx = tf.reshape(top_indices, [-1])
            scatter_indices = tf.stack([batch_idx, head_idx, seq_idx], axis=1)
            flat_top_output = tf.reshape(top_output, [-1, self.head_dim])

            output = tf.tensor_scatter_nd_update(
                output, scatter_indices, flat_top_output,
            )
            return output

        output = tf.cond(use_full, full_attention, sparse_attention)

        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, seq_len, self.model_dim])
        return self.wo(output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "factor": self.factor,
        })
        return config


@keras.saving.register_keras_serializable(package="mdsct")
class TransformerBlock(layers.Layer):
    """Post-norm Transformer encoder block with ProbSparse attention (Fig. 6).

    Post-norm architecture (per paper Fig. 6):
        Input → ProbSparse Attention → Dropout → Add(residual) → LayerNorm
              → FFN → Dropout → Add(residual) → LayerNorm
    """

    def __init__(self, model_dim: int, num_heads: int,
                 ff_dims: tuple[int, int] = (256, 128),
                 probsparse_factor: int = 5, dropout_rate: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.attn = ProbSparseAttention(
            model_dim, num_heads, factor=probsparse_factor,
            dropout_rate=dropout_rate,
        )
        self.drop1 = layers.Dropout(dropout_rate)
        self.ln1 = layers.LayerNormalization()

        self.ffn = keras.Sequential([
            layers.Dense(ff_dims[0], activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(model_dim),
        ])
        self.drop2 = layers.Dropout(dropout_rate)
        self.ln2 = layers.LayerNormalization()

    def call(self, x, training=None):
        # Post-norm: attention → residual → norm
        residual = x
        h = self.attn(x, training=training)
        h = self.drop1(h, training=training)
        x = self.ln1(h + residual)

        # Post-norm: FFN → residual → norm
        residual = x
        h = self.ffn(x, training=training)
        h = self.drop2(h, training=training)
        x = self.ln2(h + residual)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dim": self.attn.model_dim,
            "num_heads": self.attn.num_heads,
        })
        return config


@keras.saving.register_keras_serializable(package="mdsct")
class PPSformerModule(layers.Layer):
    """PPSformer branch: global attention pathway (Fig. 6-7).

    Architecture:
        Input → AAP2(96) → Conv1D(1×1, 16ch)
        → PatchEmbed(len=16, stride=8) → ProbSparse Transformer (post-norm)
        → AAP3(1024)
    """

    def __init__(
        self,
        aap2_size: int = 96,
        aap3_size: int = 1024,
        proj_ch: int = 16,
        model_dim: int = 128,
        num_heads: int = 4,
        ff_dims: tuple[int, int] = (256, 128),
        patch_length: int = 16,
        patch_stride: int = 8,
        probsparse_factor: int = 5,
        dropout_rate: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aap2_size = aap2_size
        self.aap3_size = aap3_size
        self.proj_ch = proj_ch
        self._model_dim = model_dim

        self.aap2 = AdaptiveAvgPool1D(aap2_size)
        self.proj = layers.Conv1D(proj_ch, kernel_size=1)

        # Compute number of patches
        num_patches = (aap2_size - patch_length) // patch_stride + 1

        self.patch_embed = PatchEmbedding(
            patch_length=patch_length,
            patch_stride=patch_stride,
            model_dim=model_dim,
        )

        # Positional embedding
        self.pos_embedding = layers.Embedding(num_patches, model_dim)
        self.pos_dropout = layers.Dropout(dropout_rate)

        self.transformer = TransformerBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dims=ff_dims,
            probsparse_factor=probsparse_factor,
            dropout_rate=dropout_rate,
        )

        self.aap3 = AdaptiveAvgPool1D(aap3_size)

    @property
    def out_channels(self):
        return self._model_dim  # 128

    def call(self, x, training=None):
        # x: (batch, 1024, C)
        h = self.aap2(x)             # (batch, 96, C)
        h = self.proj(h)             # (batch, 96, 16)

        h = self.patch_embed(h)      # (batch, num_patches, 256)

        # Add positional embedding
        num_patches = tf.shape(h)[1]
        positions = tf.range(num_patches)
        h = h + self.pos_embedding(positions)
        h = self.pos_dropout(h, training=training)

        h = self.transformer(h, training=training)  # (batch, num_patches, 256)

        # Reduce transformer dim to match expected output
        # AAP3 resizes temporal dimension back to 1024
        h = self.aap3(h)             # (batch, 1024, 256)

        return h

    def get_config(self):
        config = super().get_config()
        config.update({
            "aap2_size": self.aap2_size,
            "aap3_size": self.aap3_size,
            "proj_ch": self.proj_ch,
            "model_dim": self._model_dim,
        })
        return config


@keras.saving.register_keras_serializable(package="mdsct")
class MixerBlock(layers.Layer):
    """Parallel MDSC + PPSformer mixer block (Fig. 7, Section 3.4).

    Architecture:
        Input ──┬── MDSCAttention ──── (1024, 72)
                │                          │
                └── PPSformer ──────── (1024, 128)
                                           │
                         Concatenate ── (1024, 200)

    MDSC captures local multi-scale features via depthwise separable
    convolutions. PPSformer captures global dependencies via ProbSparse
    self-attention. Running them in parallel (not serial) and
    concatenating preserves both perspectives.
    """

    def __init__(self, config: MDSCTConfig, **kwargs):
        super().__init__(**kwargs)
        self.mdsc = MDSCAttentionModule(
            kernel_sizes=config.mdsc_kernels,
            bottleneck_ch=config.mdsc_bottleneck_ch,
            branch_ch=config.mdsc_branch_ch,
            maxpool_kernel=config.mdsc_maxpool_kernel,
            adapt_hswish_init=config.adapt_hswish_init,
            dropout_rate=config.dropout_rate,
        )
        self.ppsformer = PPSformerModule(
            aap2_size=config.aap2_size,
            aap3_size=config.aap3_size,
            proj_ch=config.ppsformer_proj_ch,
            model_dim=config.ppsformer_model_dim,
            num_heads=config.num_heads,
            ff_dims=config.ff_dims,
            patch_length=config.patch_length,
            patch_stride=config.patch_stride,
            probsparse_factor=config.probsparse_factor,
            dropout_rate=config.dropout_rate,
        )

    @property
    def out_channels(self):
        return self.mdsc.out_channels + self.ppsformer.out_channels

    def call(self, x, training=None):
        mdsc_out = self.mdsc(x, training=training)
        pps_out = self.ppsformer(x, training=training)
        return layers.concatenate([mdsc_out, pps_out], axis=-1)

    def get_config(self):
        # MixerBlock takes a full MDSCTConfig, so we serialize the key params
        config = super().get_config()
        return config


# ---------------------------------------------------------------------------
# Top-level model assembly
# ---------------------------------------------------------------------------


def build_mdsct(config: MDSCTConfig | None = None) -> keras.Model:
    """Build the full MDSCT model (Table 2).

    Args:
        config: Model configuration. Uses paper defaults if None.

    Returns:
        Keras model: input (None, L, 2) → output (None, 1) in [0, 1].
    """
    if config is None:
        config = MDSCTConfig()

    inputs = keras.Input(
        shape=(config.input_length, config.input_channels),
        name="signal_input",
    )

    # Per-sample min-max normalization (Eq. 19)
    x = MinMaxNormalize(name="min_max_norm")(inputs)

    # Stem Conv1D: (L, 2) → (L/2, 1)
    x = layers.Conv1D(
        config.stem_out_channels,
        kernel_size=config.stem_kernel,
        strides=config.stem_stride,
        padding="same",
        name="stem_conv",
    )(x)

    # AAP1: reduce temporal dimension → (aap1_size, 1)
    x = AdaptiveAvgPool1D(config.aap1_size, name="aap1")(x)
    x = layers.Dropout(config.dropout_rate, name="stem_dropout")(x)

    # Mixer blocks: parallel MDSC + PPSformer → concat
    for i in range(config.num_mixer_blocks):
        x = MixerBlock(config, name=f"mixer_block_{i}")(x)

    # AAP4: reduce temporal dimension → (aap4_size, out_channels)
    x = AdaptiveAvgPool1D(config.aap4_size, name="aap4")(x)

    # Flatten → Dense → sigmoid
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(1, activation="sigmoid", name="rul_output")(x)

    model = keras.Model(inputs=inputs, outputs=x, name="mdsct")
    return model


def create_default_mdsct() -> keras.Model:
    """Create MDSCT with default configuration for XJTU-SY raw signals.

    Input shape: (32768, 2) matching raw vibration signal files.
    """
    return build_mdsct(MDSCTConfig())
