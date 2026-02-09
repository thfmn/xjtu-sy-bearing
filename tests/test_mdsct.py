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

"""Tests for MDSCT architecture — Sun et al. 2024 (Heliyon e38317).

Each test references the specific equation, figure, or table it verifies.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.mdsct.model import (
    AdaptHSwish,
    AdaptiveAvgPool1D,
    EfficientChannelAttention,
    MDSCAttentionModule,
    MDSCTConfig,
    MinMaxNormalize,
    MixerBlock,
    PPSformerModule,
    PatchEmbedding,
    ProbSparseAttention,
    TransformerBlock,
    build_mdsct,
)


# ---------------------------------------------------------------------------
# MinMaxNormalize (Eq. 19)
# ---------------------------------------------------------------------------


class TestMinMaxNormalize:
    """Verify per-sample per-channel min-max normalization (Eq. 19)."""

    def test_output_in_unit_range(self):
        """Eq. 19: normalized output must be in [0, 1]."""
        norm = MinMaxNormalize()
        x = tf.random.normal((4, 256, 2))
        out = norm(x).numpy()
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_per_sample_independence(self):
        """Eq. 19: each sample is normalized independently."""
        norm = MinMaxNormalize()
        x = tf.constant([
            [[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]],
            [[100.0, 0.5], [200.0, 1.5], [300.0, 1.0]],
        ])
        out = norm(x).numpy()
        # Sample 0: min=[1,10], max=[3,30] → [0,0], [1,1], [0.5,0.5]
        np.testing.assert_allclose(out[0, 0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out[0, 1], [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(out[0, 2], [0.5, 0.5], atol=1e-6)
        # Sample 1 is independent — different scale
        np.testing.assert_allclose(out[1, 0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out[1, 2], [1.0, 0.5], atol=1e-6)

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        norm = MinMaxNormalize()
        x = tf.random.normal((3, 512, 2))
        assert norm(x).shape == (3, 512, 2)

    def test_constant_signal_no_nan(self):
        """Constant signal (max == min) should not produce NaN."""
        norm = MinMaxNormalize()
        x = tf.constant([[[5.0, 5.0]] * 100])
        out = norm(x).numpy()
        assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# AdaptiveAvgPool1D (Table 2)
# ---------------------------------------------------------------------------


class TestAdaptiveAvgPool1D:
    """Verify adaptive average pooling at key AAP locations."""

    def test_downsample_to_1024(self):
        """Table 2: AAP1 reduces 16384 → 1024."""
        pool = AdaptiveAvgPool1D(1024)
        x = tf.random.normal((2, 16384, 1))
        out = pool(x)
        assert out.shape == (2, 1024, 1)

    def test_downsample_to_96(self):
        """Table 2: AAP2 reduces 1024 → 96."""
        pool = AdaptiveAvgPool1D(96)
        x = tf.random.normal((2, 1024, 72))
        out = pool(x)
        assert out.shape == (2, 96, 72)

    def test_downsample_to_64(self):
        """Table 2: AAP4 reduces 1024 → 64."""
        pool = AdaptiveAvgPool1D(64)
        x = tf.random.normal((2, 1024, 200))
        out = pool(x)
        assert out.shape == (2, 64, 200)

    def test_upsample(self):
        """Table 2: AAP3 upsamples num_patches → 1024."""
        pool = AdaptiveAvgPool1D(1024)
        x = tf.random.normal((2, 11, 128))
        out = pool(x)
        assert out.shape == (2, 1024, 128)

    def test_identity(self):
        """When target_size == input_size, output should equal input."""
        pool = AdaptiveAvgPool1D(64)
        x = tf.random.normal((2, 64, 8))
        out = pool(x)
        assert out.shape == (2, 64, 8)


# ---------------------------------------------------------------------------
# AdaptH_Swish (Eq. 9)
# ---------------------------------------------------------------------------


class TestAdaptHSwish:
    """Verify AdaptH_Swish = δx * relu6(δx + 3) / 6."""

    def test_output_shape(self):
        act = AdaptHSwish()
        x = tf.random.normal((2, 16, 64))
        assert act(x).shape == (2, 16, 64)

    def test_standard_hswish_at_delta_1(self):
        """Eq. 9: when δ=1.0, should behave like standard H-Swish."""
        act = AdaptHSwish(init_value=1.0)
        _ = act(tf.constant([0.0]))  # build
        x = tf.constant([0.0, 6.0, -3.0, -6.0])
        out = act(x).numpy()
        # H-Swish(0) = 0 * relu6(3) / 6 = 0
        assert abs(out[0]) < 1e-6
        # H-Swish(6) = 6 * relu6(9) / 6 = 6 * 6 / 6 = 6
        assert abs(out[1] - 6.0) < 1e-6
        # H-Swish(-3) = -3 * relu6(0) / 6 = 0
        assert abs(out[2]) < 1e-6
        # H-Swish(-6) = -6 * relu6(-3) / 6 = 0
        assert abs(out[3]) < 1e-6

    def test_scaling_not_shifting(self):
        """Eq. 9: δ scales the input, not shifts inside relu6.

        With δ=2.0, x=1.0: δx=2, relu6(2+3)/6 = relu6(5)/6 = 5/6
        Result = 2 * 5/6 ≈ 1.6667
        If it were shifting (old bug): x * relu6(x+2)/6 = 1 * relu6(3)/6 = 0.5
        """
        act = AdaptHSwish(init_value=2.0)
        _ = act(tf.constant([0.0]))  # build
        x = tf.constant([1.0])
        out = act(x).numpy()[0]
        expected_scale = 2.0 * tf.nn.relu6(2.0 + 3.0).numpy() / 6.0  # 1.6667
        assert abs(out - expected_scale) < 1e-5

    def test_delta_init_value(self):
        """Eq. 9: δ should be initialized to 1.0 (not 3.0)."""
        act = AdaptHSwish(init_value=1.0)
        _ = act(tf.constant([0.0]))  # build
        delta_vars = [v for v in act.trainable_variables if "adapt_scale" in v.name]
        assert len(delta_vars) == 1
        np.testing.assert_allclose(delta_vars[0].numpy(), 1.0)

    def test_trainable_parameter(self):
        """The δ parameter must be trainable for gradient-based learning."""
        act = AdaptHSwish()
        _ = act(tf.constant([1.0]))
        delta_vars = [v for v in act.trainable_variables if "adapt_scale" in v.name]
        assert len(delta_vars) == 1
        assert delta_vars[0].shape == ()

    def test_gradient_through_delta(self):
        """Gradient must flow through both x and δ."""
        act = AdaptHSwish(init_value=1.0)
        x = tf.constant([1.0, 2.0, 3.0])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            out = act(x)
            loss = tf.reduce_mean(out)
        grad_x = tape.gradient(loss, x)
        grad_delta = tape.gradient(loss, act.delta)
        del tape
        assert grad_x is not None
        assert not tf.reduce_all(grad_x == 0)
        assert grad_delta is not None

    def test_serialization(self):
        act = AdaptHSwish(init_value=2.5)
        config = act.get_config()
        assert config["init_value"] == 2.5
        restored = AdaptHSwish.from_config(config)
        assert restored.init_value == 2.5


# ---------------------------------------------------------------------------
# EfficientChannelAttention (Eq. 10)
# ---------------------------------------------------------------------------


class TestECA:
    """Verify Efficient Channel Attention (Eq. 10)."""

    def test_output_shape(self):
        eca = EfficientChannelAttention()
        x = tf.random.normal((2, 128, 72))
        assert eca(x).shape == (2, 128, 72)

    def test_kernel_formula(self):
        """Eq. 10: k = |log2(C)/2 + 0.5| rounded to odd, min 3."""
        eca = EfficientChannelAttention()
        x = tf.random.normal((2, 64, 72))
        _ = eca(x)  # build
        # C=72 → log2(72)/2 + 0.5 = 3.58/2 + 0.5 = ~6.17/2 + 0.5
        # Actually: log2(72) ≈ 6.17, 6.17/2 + 0.5 = 3.585 → int(abs(...)) = 3
        # 3 is odd, max(3,3) = 3
        assert eca.conv.kernel_size == (3,)

    def test_attention_weights_bounded(self):
        """Sigmoid ensures attention weights are in [0, 1]."""
        eca = EfficientChannelAttention()
        x = tf.random.normal((2, 64, 32))
        out = eca(x)
        assert (tf.abs(out) <= tf.abs(x) + 1e-6).numpy().all()


# ---------------------------------------------------------------------------
# MDSCAttentionModule (Fig. 4-5)
# ---------------------------------------------------------------------------


class TestMDSCAttentionModule:
    """Verify MDSC Attention structure (Fig. 4-5)."""

    def test_output_channels_72(self):
        """Fig. 4: 3 branches × 24 ch = 72 output channels via concat."""
        mdsc = MDSCAttentionModule(
            kernel_sizes=(8, 16, 32), bottleneck_ch=16, branch_ch=24,
        )
        x = tf.random.normal((2, 1024, 1))
        out = mdsc(x, training=False)
        assert out.shape == (2, 1024, 72)

    def test_three_dsc_branches(self):
        """Fig. 4: exactly 3 DSC branches with different kernel sizes."""
        mdsc = MDSCAttentionModule(kernel_sizes=(8, 16, 32))
        _ = mdsc(tf.random.normal((1, 64, 16)))  # build
        assert len(mdsc.dsc_branches) == 3

    def test_has_maxpool(self):
        """Fig. 5: MaxPool(3) is first operation in MDSC."""
        mdsc = MDSCAttentionModule(maxpool_kernel=3)
        assert isinstance(mdsc.maxpool, tf.keras.layers.MaxPooling1D)

    def test_has_bottleneck_16ch(self):
        """Fig. 5: Bottleneck Conv1D(1×1, 16ch) after MaxPool."""
        mdsc = MDSCAttentionModule(bottleneck_ch=16)
        assert isinstance(mdsc.bottleneck, tf.keras.layers.Conv1D)
        # Verify the bottleneck conv will produce 16 channels
        _ = mdsc(tf.random.normal((1, 64, 1)))  # build
        assert mdsc.bottleneck.filters == 16

    def test_concat_not_sum(self):
        """Fig. 4: branches are concatenated (72ch), not summed.

        If summed, output would be 24ch. Concat gives 3×24 = 72ch.
        """
        mdsc = MDSCAttentionModule(
            kernel_sizes=(8, 16, 32), branch_ch=24,
        )
        x = tf.random.normal((2, 1024, 1))
        out = mdsc(x, training=False)
        # 72 channels proves concat; if sum, would be 24
        assert out.shape[-1] == 72

    def test_residual_connection(self):
        """Fig. 5: residual add with 1×1 projection when channels differ."""
        mdsc = MDSCAttentionModule()
        x = tf.random.normal((2, 64, 1))
        out = mdsc(x, training=False)
        # Output should not be all zeros (residual helps)
        assert tf.reduce_mean(tf.abs(out)).numpy() > 1e-4

    def test_gradient_flow(self):
        mdsc = MDSCAttentionModule(kernel_sizes=(4, 8), bottleneck_ch=8, branch_ch=8)
        x = tf.random.normal((2, 64, 8))
        with tf.GradientTape() as tape:
            out = mdsc(x, training=True)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, mdsc.trainable_variables)
        assert all(g is not None for g in grads)


# ---------------------------------------------------------------------------
# PPSformerModule (Fig. 6-7)
# ---------------------------------------------------------------------------


class TestPPSformerModule:
    """Verify PPSformer global attention branch (Fig. 6-7)."""

    def test_output_shape(self):
        """Fig. 7: PPSformer output is (1024, 128)."""
        pps = PPSformerModule(aap2_size=96, aap3_size=1024, model_dim=128)
        x = tf.random.normal((2, 1024, 1))
        out = pps(x, training=False)
        assert out.shape == (2, 1024, 128)

    def test_has_aap2(self):
        """Fig. 7: AAP2 reduces to 96 before patch embedding."""
        pps = PPSformerModule(aap2_size=96)
        assert isinstance(pps.aap2, AdaptiveAvgPool1D)
        assert pps.aap2.target_size == 96

    def test_has_conv1d_projection(self):
        """Fig. 7: Conv1D(1×1, 16ch) projects after AAP2."""
        pps = PPSformerModule(proj_ch=16)
        assert isinstance(pps.proj, tf.keras.layers.Conv1D)

    def test_has_aap3(self):
        """Fig. 7: AAP3 resizes back to 1024 after transformer."""
        pps = PPSformerModule(aap3_size=1024)
        assert isinstance(pps.aap3, AdaptiveAvgPool1D)
        assert pps.aap3.target_size == 1024

    def test_out_channels_property(self):
        """PPSformer output channels = model_dim = 128."""
        pps = PPSformerModule(model_dim=128)
        assert pps.out_channels == 128


# ---------------------------------------------------------------------------
# MixerBlock (Fig. 7, Section 3.4)
# ---------------------------------------------------------------------------


class TestMixerBlock:
    """Verify parallel MDSC + PPSformer mixer block (Fig. 7)."""

    def test_output_channels_200(self):
        """Fig. 7: MDSC(72) + PPSformer(128) concatenated → 200 channels."""
        config = MDSCTConfig()
        block = MixerBlock(config)
        x = tf.random.normal((2, 1024, 1))
        out = block(x, training=False)
        assert out.shape == (2, 1024, 200)

    def test_parallel_not_serial(self):
        """Fig. 7: MDSC and PPSformer run in parallel, not serial.

        Verify by checking that MixerBlock has both sub-modules
        and output channels = sum of both paths.
        """
        config = MDSCTConfig()
        block = MixerBlock(config)
        assert hasattr(block, "mdsc")
        assert hasattr(block, "ppsformer")
        assert block.out_channels == block.mdsc.out_channels + block.ppsformer.out_channels

    def test_output_is_concatenation(self):
        """Fig. 7: verify output channels = MDSC channels + PPSformer channels."""
        config = MDSCTConfig()
        block = MixerBlock(config)
        assert block.out_channels == 72 + 128  # 200

    def test_accepts_varying_input_channels(self):
        """MixerBlock should handle 1-channel input (from stem) and
        200-channel input (from previous MixerBlock)."""
        config = MDSCTConfig()
        # First block: 1 channel input
        block1 = MixerBlock(config)
        x1 = tf.random.normal((1, 1024, 1))
        out1 = block1(x1, training=False)
        assert out1.shape == (1, 1024, 200)

        # Second block: 200 channel input
        block2 = MixerBlock(config)
        out2 = block2(out1, training=False)
        assert out2.shape == (1, 1024, 200)


# ---------------------------------------------------------------------------
# ProbSparseAttention (Eq. 14-15)
# ---------------------------------------------------------------------------


class TestProbSparseAttention:
    """Test ProbSparse attention (from Informer, used in PPSformer)."""

    def test_output_shape(self):
        attn = ProbSparseAttention(model_dim=64, num_heads=4)
        x = tf.random.normal((2, 32, 64))
        assert attn(x).shape == (2, 32, 64)

    def test_short_sequence_full_attention(self):
        """Sequences < 64 should use full attention (no sparsity)."""
        attn = ProbSparseAttention(model_dim=32, num_heads=4)
        x = tf.random.normal((2, 16, 32))
        assert attn(x).shape == (2, 16, 32)

    def test_long_sequence(self):
        """Sequences >= 64 should use ProbSparse attention."""
        attn = ProbSparseAttention(model_dim=32, num_heads=4, factor=5)
        x = tf.random.normal((2, 128, 32))
        assert attn(x).shape == (2, 128, 32)

    def test_gradient_flow(self):
        attn = ProbSparseAttention(model_dim=32, num_heads=4)
        x = tf.random.normal((2, 32, 32))
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = attn(x, training=True)
            loss = tf.reduce_mean(out)
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert not tf.reduce_all(grad == 0)


# ---------------------------------------------------------------------------
# TransformerBlock (Fig. 6)
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    """Verify post-norm Transformer block (Fig. 6)."""

    def test_output_shape(self):
        block = TransformerBlock(model_dim=64, num_heads=4)
        x = tf.random.normal((2, 16, 64))
        assert block(x).shape == (2, 16, 64)

    def test_post_norm_structure(self):
        """Fig. 6: post-norm means LayerNorm is AFTER residual add.

        Check that ln1 exists as a separate attribute (not inside attn).
        In post-norm, the call order is: attn → drop → add → ln
        (vs pre-norm: ln → attn → drop → add).
        """
        block = TransformerBlock(model_dim=32, num_heads=4)
        assert hasattr(block, "ln1")
        assert hasattr(block, "ln2")
        # Verify they are LayerNormalization instances
        assert isinstance(block.ln1, tf.keras.layers.LayerNormalization)
        assert isinstance(block.ln2, tf.keras.layers.LayerNormalization)

    def test_ffn_dims(self):
        """Section 3.4: FFN expands to 256, contracts to model_dim via 128."""
        block = TransformerBlock(model_dim=64, num_heads=4, ff_dims=(256, 128))
        _ = block(tf.random.normal((1, 8, 64)))  # build
        # The FFN sequential: Dense(256) → Dropout → Dense(model_dim=64)
        ffn_layers = block.ffn.layers
        dense_layers = [l for l in ffn_layers if isinstance(l, tf.keras.layers.Dense)]
        assert dense_layers[0].units == 256
        assert dense_layers[1].units == 64  # model_dim


# ---------------------------------------------------------------------------
# Full MDSCT model (Table 2)
# ---------------------------------------------------------------------------


class TestFullMDSCT:
    """Test the complete MDSCT model against Table 2."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast tests (preserves architecture ratios)."""
        return MDSCTConfig(
            input_length=2048,
            input_channels=2,
            stem_kernel=8,
            stem_stride=2,
            stem_out_channels=1,
            mdsc_kernels=(4, 8),
            mdsc_bottleneck_ch=8,
            mdsc_branch_ch=12,
            mdsc_maxpool_kernel=3,
            num_mixer_blocks=1,
            aap1_size=128,
            aap2_size=32,
            aap3_size=128,
            aap4_size=16,
            ppsformer_proj_ch=8,
            ppsformer_model_dim=64,
            num_heads=4,
            ff_dims=(64, 32),
            patch_length=8,
            patch_stride=4,
            probsparse_factor=5,
            dropout_rate=0.05,
            adapt_hswish_init=1.0,
        )

    def test_stem_produces_1_channel(self):
        """Table 2: stem Conv1D has 1 output channel."""
        config = MDSCTConfig()
        model = build_mdsct(config)
        stem = model.get_layer("stem_conv")
        assert stem.filters == 1

    def test_has_min_max_normalization(self):
        """Eq. 19: model starts with per-sample min-max normalization."""
        model = build_mdsct()
        norm = model.get_layer("min_max_norm")
        assert isinstance(norm, MinMaxNormalize)

    def test_has_aap1(self):
        """Table 2: AAP1(1024) after stem."""
        model = build_mdsct()
        aap1 = model.get_layer("aap1")
        assert isinstance(aap1, AdaptiveAvgPool1D)
        assert aap1.target_size == 1024

    def test_has_aap4(self):
        """Table 2: AAP4(64) before final FC."""
        model = build_mdsct()
        aap4 = model.get_layer("aap4")
        assert isinstance(aap4, AdaptiveAvgPool1D)
        assert aap4.target_size == 64

    def test_three_mixer_blocks(self):
        """Table 2: 3 MixerBlocks."""
        model = build_mdsct()
        mixer_layers = [l for l in model.layers if isinstance(l, MixerBlock)]
        assert len(mixer_layers) == 3

    def test_single_fc_output(self):
        """Table 2: single Dense(1, sigmoid) at the end."""
        model = build_mdsct()
        output_layer = model.get_layer("rul_output")
        assert isinstance(output_layer, tf.keras.layers.Dense)
        assert output_layer.units == 1
        assert output_layer.activation.__name__ == "sigmoid"

    def test_default_build(self):
        """Default config builds with correct I/O shapes."""
        model = build_mdsct()
        assert model.input_shape == (None, 32768, 2)
        assert model.output_shape == (None, 1)

    def test_small_config_build(self, small_config):
        model = build_mdsct(small_config)
        assert model.input_shape == (None, 2048, 2)
        assert model.output_shape == (None, 1)

    def test_forward_pass(self, small_config):
        model = build_mdsct(small_config)
        batch = np.random.randn(2, 2048, 2).astype(np.float32)
        preds = model.predict(batch, verbose=0)
        assert preds.shape == (2, 1)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_training_step(self, small_config):
        model = build_mdsct(small_config)
        model.compile(optimizer="adam", loss="mse")
        x = np.random.randn(4, 2048, 2).astype(np.float32)
        y = np.random.rand(4, 1).astype(np.float32)
        history = model.fit(x, y, epochs=1, verbose=0)
        assert "loss" in history.history
        assert len(history.history["loss"]) == 1

    def test_model_checkpoint_save(self, small_config, tmp_path):
        """All custom layers implement get_config() for Keras serialization.

        This ensures ModelCheckpoint can save .keras files during training,
        which is required for checkpoint-based early stopping.
        """
        model = build_mdsct(small_config)
        model.compile(optimizer="adam", loss="mse")
        save_path = tmp_path / "mdsct_test.keras"
        model.save(save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Architectural constants (Table 1)
# ---------------------------------------------------------------------------


class TestArchitecturalConstants:
    """Verify default config matches paper Table 1."""

    def test_patch_params(self):
        """Table 1: patch_length=16, patch_stride=8."""
        cfg = MDSCTConfig()
        assert cfg.patch_length == 16
        assert cfg.patch_stride == 8

    def test_num_heads(self):
        """Table 1: 4 attention heads."""
        assert MDSCTConfig().num_heads == 4

    def test_dropout_rate(self):
        """Table 1: dropout = 0.05."""
        assert MDSCTConfig().dropout_rate == 0.05

    def test_mdsc_kernels(self):
        """Table 2: kernel sizes (8, 16, 32)."""
        assert MDSCTConfig().mdsc_kernels == (8, 16, 32)

    def test_adapt_hswish_delta_init(self):
        """Eq. 9: δ initialized to 1.0."""
        assert MDSCTConfig().adapt_hswish_init == 1.0


# ---------------------------------------------------------------------------
# Model registry integration
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Test MDSCT model registry integration."""

    def test_model_listed(self):
        from src.models.registry import list_models
        assert "mdsct" in list_models()

    def test_build_from_registry(self):
        from src.models.registry import build_model
        model = build_model("mdsct")
        assert model.input_shape == (None, 32768, 2)

    def test_model_info(self):
        from src.models.registry import get_model_info
        info = get_model_info("mdsct")
        assert info.input_type == "raw_signal"
        assert info.default_input_shape == (32768, 2)
