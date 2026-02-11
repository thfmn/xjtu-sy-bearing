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

"""Tests for DTA-MLP architecture and its components."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.dta_mlp.model import (
    CTMixedMLP,
    CWTFeatureExtractor,
    DTAMLPConfig,
    DynamicTemporalAttention,
    TransformerEncoderBlock,
    build_dta_mlp,
)


class TestCWTFeatureExtractor:
    """Test the CNN frontend for CWT scaleograms."""

    def test_output_shape(self):
        frontend = CWTFeatureExtractor(filters=(32, 64, 128))
        x = tf.random.normal((2, 64, 128, 2))
        out = frontend(x)
        # After 3 MaxPool2D(2,2): 64/8=8, 128/8=16 → 128 patches, 128 channels
        assert out.shape == (2, 128, 128)

    def test_custom_filters(self):
        frontend = CWTFeatureExtractor(filters=(16, 32, 64))
        x = tf.random.normal((2, 64, 128, 2))
        out = frontend(x)
        # After 3 MaxPool2D: 64/8=8, 128/8=16 → 128 patches, 64 channels
        assert out.shape == (2, 128, 64)

    def test_training_mode(self):
        """BatchNorm should behave differently in training vs inference."""
        frontend = CWTFeatureExtractor(filters=(16,))
        x = tf.random.normal((4, 64, 128, 2))
        out_train = frontend(x, training=True)
        out_infer = frontend(x, training=False)
        assert out_train.shape == out_infer.shape


class TestDynamicTemporalAttention:
    """Test the DTA attention mechanism."""

    def test_output_shape(self):
        dta = DynamicTemporalAttention(model_dim=64, num_heads=4)
        x = tf.random.normal((2, 16, 64))
        out = dta(x)
        assert out.shape == (2, 16, 64)

    def test_attention_rectification_parameter_exists(self):
        dta = DynamicTemporalAttention(model_dim=64, num_heads=4, use_attention_rectification=True)
        x = tf.random.normal((2, 16, 64))
        _ = dta(x)  # build
        threshold_vars = [v for v in dta.trainable_variables if "learned_threshold" in v.name]
        sharpness_vars = [v for v in dta.trainable_variables if "gate_sharpness" in v.name]
        assert len(threshold_vars) == 1
        assert threshold_vars[0].shape == (4,)
        assert len(sharpness_vars) == 1
        assert sharpness_vars[0].shape == (4,)

    def test_no_attention_rectification(self):
        dta = DynamicTemporalAttention(model_dim=64, num_heads=4, use_attention_rectification=False)
        x = tf.random.normal((2, 16, 64))
        out = dta(x)
        assert out.shape == (2, 16, 64)
        rect_vars = [v for v in dta.trainable_variables
                     if "learned_threshold" in v.name or "gate_sharpness" in v.name]
        assert len(rect_vars) == 0

    def test_gradient_flow(self):
        """Verify gradients flow through DTA."""
        dta = DynamicTemporalAttention(model_dim=64, num_heads=4)
        x = tf.random.normal((2, 8, 64))
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = dta(x, training=True)
            loss = tf.reduce_mean(out)
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert not tf.reduce_all(grad == 0)


class TestTransformerEncoderBlock:
    """Test the Transformer encoder block."""

    def test_output_shape(self):
        block = TransformerEncoderBlock(
            model_dim=64, num_heads=4, ff_dim=256,
        )
        x = tf.random.normal((2, 16, 64))
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_residual_connection(self):
        """Output should be different from input (non-identity) but close for init."""
        block = TransformerEncoderBlock(
            model_dim=64, num_heads=4, ff_dim=256,
        )
        x = tf.random.normal((2, 8, 64))
        out = block(x, training=False)
        # With residual connections and near-zero initial weights,
        # output should be close to input
        diff = tf.reduce_mean(tf.abs(out - x))
        assert diff < 5.0  # reasonable bound for random init


class TestCTMixedMLP:
    """Test the Channel-Temporal Mixed MLP."""

    def test_output_shape(self):
        mlp = CTMixedMLP(model_dim=64, seq_len=16)
        x = tf.random.normal((2, 16, 64))
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_gradient_flow(self):
        mlp = CTMixedMLP(model_dim=64, seq_len=8)
        x = tf.random.normal((2, 8, 64))
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = mlp(x, training=True)
            loss = tf.reduce_mean(out)
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert not tf.reduce_all(grad == 0)


class TestFullDTAMLP:
    """Test the complete DTA-MLP model."""

    def test_default_build(self):
        model = build_dta_mlp()
        assert model.input_shape == (None, 64, 128, 2)
        assert model.output_shape == (None, 1)

    def test_forward_pass(self):
        model = build_dta_mlp()
        batch = np.random.randn(2, 64, 128, 2).astype(np.float32)
        preds = model.predict(batch, verbose=0)
        assert preds.shape == (2, 1)
        # Linear output — predictions are NOT constrained to [0, 1]

    def test_parameter_count(self):
        model = build_dta_mlp()
        params = model.count_params()
        # With 3-block CNN, 4 interleaved transformer+CT-MLP, expect ~5.7M
        assert 4_000_000 < params < 8_000_000, f"Expected ~5.7M params, got {params:,}"

    def test_training_step(self):
        """Verify the model can complete a single training step."""
        config = DTAMLPConfig(
            cnn_filters=(16, 32),  # smaller for speed
            model_dim=32,
            num_heads=4,
            ff_dim=64,
            num_layers=1,
            input_height=16,
            input_width=32,
        )
        model = build_dta_mlp(config)
        model.compile(optimizer="adam", loss="mse")

        x = np.random.randn(4, 16, 32, 2).astype(np.float32)
        y = np.random.rand(4, 1).astype(np.float32)

        history = model.fit(x, y, epochs=1, verbose=0)
        assert "loss" in history.history
        assert len(history.history["loss"]) == 1

    def test_custom_config(self):
        config = DTAMLPConfig(
            input_height=32,
            input_width=64,
            cnn_filters=(16, 32, 64),
            model_dim=64,
            num_heads=4,
            ff_dim=128,
            num_layers=2,
        )
        model = build_dta_mlp(config)
        assert model.input_shape == (None, 32, 64, 2)
        assert model.output_shape == (None, 1)


class TestModelRegistry:
    """Test DTA-MLP model registry integration."""

    def test_model_listed(self):
        from src.models.registry import list_models
        assert "dta_mlp" in list_models()

    def test_build_from_registry(self):
        from src.models.registry import build_model
        model = build_model("dta_mlp")
        assert model.input_shape == (None, 64, 128, 2)

    def test_model_info(self):
        from src.models.registry import get_model_info
        info = get_model_info("dta_mlp")
        assert info.input_type == "cwt_scaleogram"
        assert info.default_input_shape == (64, 128, 2)
