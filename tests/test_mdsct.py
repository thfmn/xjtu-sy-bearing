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

"""Tests for MDSCT architecture and its components."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.mdsct.model import (
    AdaptHSwish,
    EfficientChannelAttention,
    MDSCTConfig,
    MixerBlock,
    MultiScaleDepthwiseSeparableConv,
    PatchEmbedding,
    ProbSparseAttention,
    TransformerBlock,
    build_mdsct,
)


class TestAdaptHSwish:
    """Test the AdaptH_Swish activation function."""

    def test_output_shape(self):
        act = AdaptHSwish()
        x = tf.random.normal((2, 16, 64))
        out = act(x)
        assert out.shape == (2, 16, 64)

    def test_standard_hswish_at_default(self):
        """At init a=3.0, should behave like standard H-Swish."""
        act = AdaptHSwish(init_value=3.0)
        # Build the layer
        _ = act(tf.constant([0.0]))
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

    def test_trainable_parameter(self):
        act = AdaptHSwish()
        _ = act(tf.constant([1.0]))  # build
        a_vars = [v for v in act.trainable_variables if "adapt_shift" in v.name]
        assert len(a_vars) == 1
        assert a_vars[0].shape == ()

    def test_gradient_flow(self):
        act = AdaptHSwish()
        x = tf.constant([1.0, 2.0, 3.0])
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = act(x)
            loss = tf.reduce_mean(out)
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert not tf.reduce_all(grad == 0)

    def test_serialization(self):
        act = AdaptHSwish(init_value=4.0)
        config = act.get_config()
        assert config["init_value"] == 4.0
        restored = AdaptHSwish.from_config(config)
        assert restored.init_value == 4.0


class TestEfficientChannelAttention:
    """Test the ECA attention module."""

    def test_output_shape(self):
        eca = EfficientChannelAttention()
        x = tf.random.normal((2, 128, 64))
        out = eca(x)
        assert out.shape == (2, 128, 64)

    def test_channel_reweighting(self):
        """Output should differ from input (channels are reweighted)."""
        eca = EfficientChannelAttention()
        x = tf.random.normal((2, 64, 32))
        out = eca(x)
        # Not identical to input
        assert not np.allclose(out.numpy(), x.numpy(), atol=1e-3)

    def test_attention_weights_bounded(self):
        """Sigmoid ensures attention weights are in [0, 1]."""
        eca = EfficientChannelAttention()
        x = tf.random.normal((2, 64, 32))
        out = eca(x)
        # Output magnitude should be <= input magnitude per element
        # (since weights are in [0, 1])
        assert (tf.abs(out) <= tf.abs(x) + 1e-6).numpy().all()


class TestMultiScaleDepthwiseSeparableConv:
    """Test the MDSC module."""

    def test_output_shape(self):
        mdsc = MultiScaleDepthwiseSeparableConv(filters=64, kernel_sizes=(4, 8, 16))
        x = tf.random.normal((2, 128, 32))
        out = mdsc(x, training=False)
        assert out.shape == (2, 128, 64)

    def test_multiple_branches(self):
        """Verify all kernel-size branches are created."""
        kernels = (8, 16, 32)
        mdsc = MultiScaleDepthwiseSeparableConv(filters=64, kernel_sizes=kernels)
        x = tf.random.normal((2, 128, 32))
        _ = mdsc(x)  # build
        assert len(mdsc.branches) == len(kernels)

    def test_gradient_flow(self):
        mdsc = MultiScaleDepthwiseSeparableConv(filters=32, kernel_sizes=(4, 8))
        x = tf.random.normal((2, 64, 16))
        with tf.GradientTape() as tape:
            out = mdsc(x, training=True)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, mdsc.trainable_variables)
        assert all(g is not None for g in grads)
        assert any(not tf.reduce_all(g == 0) for g in grads)


class TestMixerBlock:
    """Test the MDSC + ECA mixer block."""

    def test_output_shape(self):
        block = MixerBlock(filters=64, kernel_sizes=(4, 8))
        x = tf.random.normal((2, 128, 64))
        out = block(x, training=False)
        assert out.shape == (2, 128, 64)

    def test_residual_projection(self):
        """When input channels != filters, residual projection is applied."""
        block = MixerBlock(filters=64, kernel_sizes=(4, 8))
        x = tf.random.normal((2, 128, 32))
        out = block(x, training=False)
        assert out.shape == (2, 128, 64)

    def test_residual_connection(self):
        """Output should be close to input for random init (residual path)."""
        block = MixerBlock(filters=32, kernel_sizes=(4,))
        x = tf.random.normal((2, 64, 32))
        out = block(x, training=False)
        # With residual, output shouldn't be zero
        assert tf.reduce_mean(tf.abs(out)).numpy() > 0.01


class TestPatchEmbedding:
    """Test the patch embedding module."""

    def test_output_shape(self):
        pe = PatchEmbedding(patch_length=8, patch_stride=4, model_dim=64)
        x = tf.random.normal((2, 256, 32))
        out = pe(x)
        # num_patches = (256 - 8) / 4 + 1 = 63
        assert out.shape == (2, 63, 64)

    def test_num_patches_calculation(self):
        pe = PatchEmbedding(patch_length=16, patch_stride=8, model_dim=32)
        x = tf.random.normal((2, 512, 16))
        out = pe(x)
        expected_patches = (512 - 16) // 8 + 1  # 63
        assert out.shape[1] == expected_patches

    def test_no_overlap(self):
        """Non-overlapping patches (stride == length)."""
        pe = PatchEmbedding(patch_length=16, patch_stride=16, model_dim=32)
        x = tf.random.normal((2, 256, 8))
        out = pe(x)
        # 256 / 16 = 16 patches
        assert out.shape == (2, 16, 32)


class TestProbSparseAttention:
    """Test the ProbSparse attention mechanism."""

    def test_output_shape(self):
        attn = ProbSparseAttention(model_dim=64, num_heads=4)
        x = tf.random.normal((2, 32, 64))
        out = attn(x)
        assert out.shape == (2, 32, 64)

    def test_short_sequence_full_attention(self):
        """Sequences < 64 should use full attention (no sparsity)."""
        attn = ProbSparseAttention(model_dim=32, num_heads=4)
        x = tf.random.normal((2, 16, 32))
        out = attn(x)
        assert out.shape == (2, 16, 32)

    def test_long_sequence(self):
        """Sequences >= 64 should use ProbSparse attention."""
        attn = ProbSparseAttention(model_dim=32, num_heads=4, factor=5)
        x = tf.random.normal((2, 128, 32))
        out = attn(x)
        assert out.shape == (2, 128, 32)

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


class TestTransformerBlock:
    """Test the Transformer encoder block."""

    def test_output_shape(self):
        block = TransformerBlock(model_dim=64, num_heads=4, ff_dim=128)
        x = tf.random.normal((2, 16, 64))
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_residual_connection(self):
        block = TransformerBlock(model_dim=32, num_heads=4, ff_dim=64)
        x = tf.random.normal((2, 8, 32))
        out = block(x, training=False)
        diff = tf.reduce_mean(tf.abs(out - x))
        assert diff < 5.0  # reasonable for random init


class TestFullMDSCT:
    """Test the complete MDSCT model."""

    def test_small_config_build(self):
        config = MDSCTConfig(
            input_length=512,
            stem_filters=16,
            stem_kernel=8,
            stem_stride=2,
            mdsc_kernels=(4, 8),
            num_mixer_blocks=1,
            model_dim=32,
            num_heads=4,
            ff_dim=64,
            num_transformer_layers=1,
            patch_length=8,
            patch_stride=4,
        )
        model = build_mdsct(config)
        assert model.input_shape == (None, 512, 2)
        assert model.output_shape == (None, 1)

    def test_default_build(self):
        model = build_mdsct()
        assert model.input_shape == (None, 32768, 2)
        assert model.output_shape == (None, 1)

    def test_forward_pass(self):
        config = MDSCTConfig(
            input_length=512,
            stem_filters=16,
            stem_kernel=8,
            stem_stride=2,
            mdsc_kernels=(4, 8),
            num_mixer_blocks=1,
            model_dim=32,
            num_heads=4,
            ff_dim=64,
            num_transformer_layers=1,
            patch_length=8,
            patch_stride=4,
        )
        model = build_mdsct(config)
        batch = np.random.randn(2, 512, 2).astype(np.float32)
        preds = model.predict(batch, verbose=0)
        assert preds.shape == (2, 1)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_training_step(self):
        config = MDSCTConfig(
            input_length=256,
            stem_filters=8,
            stem_kernel=8,
            stem_stride=2,
            mdsc_kernels=(4,),
            num_mixer_blocks=1,
            model_dim=16,
            num_heads=4,
            ff_dim=32,
            num_transformer_layers=1,
            patch_length=8,
            patch_stride=4,
        )
        model = build_mdsct(config)
        model.compile(optimizer="adam", loss="mse")

        x = np.random.randn(4, 256, 2).astype(np.float32)
        y = np.random.rand(4, 1).astype(np.float32)

        history = model.fit(x, y, epochs=1, verbose=0)
        assert "loss" in history.history
        assert len(history.history["loss"]) == 1

    def test_parameter_count_reasonable(self):
        """Default model should have a reasonable parameter count."""
        model = build_mdsct()
        params = model.count_params()
        # We don't have a specific target from the paper, but it should be
        # under 5M for this architecture
        assert 100_000 < params < 5_000_000, f"Unexpected params: {params:,}"


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
