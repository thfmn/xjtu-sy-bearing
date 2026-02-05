"""Tests for onset classifier model (ONSET-15 acceptance criteria).

Verifies:
- Model compiles without errors
- Input shape: (None, window_size, 4), output: (None, 1)
- Total parameters under 10K (lightweight)
- Model supports predict_proba() equivalent via sigmoid output
"""

from __future__ import annotations

import numpy as np
import pytest


class TestModelCompilesWithoutErrors:
    """ONSET-15 Acceptance: Model compiles without errors."""

    def test_build_onset_classifier_returns_model(self):
        from src.onset.models import build_onset_classifier

        model = build_onset_classifier()
        assert model is not None
        assert model.name == "onset_classifier"

    def test_compile_onset_classifier_no_error(self):
        from src.onset.models import build_onset_classifier, compile_onset_classifier

        model = build_onset_classifier()
        compiled = compile_onset_classifier(model)
        # compile returns the same model object
        assert compiled is model
        # model should have an optimizer after compilation
        assert model.optimizer is not None

    def test_create_onset_classifier_factory(self):
        from src.onset.models import create_onset_classifier

        model = create_onset_classifier()
        assert model is not None

    def test_compiled_model_can_train_one_step(self):
        from src.onset.models import build_onset_classifier, compile_onset_classifier

        model = build_onset_classifier()
        compile_onset_classifier(model)

        # Create dummy data: 8 samples, window_size=10, 4 features
        x = np.random.randn(8, 10, 4).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)

        history = model.fit(x, y, epochs=1, verbose=0)
        assert "loss" in history.history
        assert len(history.history["loss"]) == 1


class TestInputOutputShape:
    """ONSET-15 Acceptance: Input shape: (None, window_size, 4), output: (None, 1)."""

    def test_default_input_shape(self):
        from src.onset.models import build_onset_classifier

        model = build_onset_classifier()
        # input_shape includes batch dim as None
        assert model.input_shape == (None, 10, 4)

    def test_default_output_shape(self):
        from src.onset.models import build_onset_classifier

        model = build_onset_classifier()
        assert model.output_shape == (None, 1)

    def test_custom_window_size_input_shape(self):
        from src.onset.models import OnsetClassifierConfig, build_onset_classifier

        config = OnsetClassifierConfig(window_size=20)
        model = build_onset_classifier(config)
        assert model.input_shape == (None, 20, 4)
        assert model.output_shape == (None, 1)

    def test_create_onset_classifier_shapes(self):
        from src.onset.models import create_onset_classifier

        model = create_onset_classifier(input_dim=4, window_size=10)
        assert model.input_shape == (None, 10, 4)
        assert model.output_shape == (None, 1)

    def test_forward_pass_shapes(self):
        from src.onset.models import build_onset_classifier

        model = build_onset_classifier()
        x = np.random.randn(5, 10, 4).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (5, 1)

    def test_custom_features_input_shape(self):
        from src.onset.models import OnsetClassifierConfig, build_onset_classifier

        config = OnsetClassifierConfig(n_features=6, window_size=15)
        model = build_onset_classifier(config)
        assert model.input_shape == (None, 15, 6)
        assert model.output_shape == (None, 1)


class TestTotalParametersUnder10K:
    """ONSET-15 Acceptance: Total parameters under 10K (lightweight)."""

    def test_default_config_under_10k(self):
        from src.onset.models import build_onset_classifier

        model = build_onset_classifier()
        total_params = model.count_params()
        assert total_params < 10_000, (
            f"Model has {total_params} params, expected <10,000"
        )

    def test_exact_param_count_default(self):
        """Verify known param count for regression detection."""
        from src.onset.models import build_onset_classifier

        model = build_onset_classifier()
        total_params = model.count_params()
        # LSTM(4->32): 4*32 + 32*32 + 32*4 = 128+1024+128 = 4,736 (with bias)
        # Dense(32->16): 32*16+16 = 528
        # Dense(16->1): 16*1+1 = 17
        # Total = 5,281
        assert total_params == 5_281, (
            f"Expected 5,281 params but got {total_params}; "
            "architecture may have changed unintentionally"
        )

    def test_factory_function_under_10k(self):
        from src.onset.models import create_onset_classifier

        model = create_onset_classifier()
        assert model.count_params() < 10_000

    def test_no_trainable_vs_total_discrepancy(self):
        from src.onset.models import build_onset_classifier

        model = build_onset_classifier()
        trainable = sum(
            int(np.prod(w.shape)) for w in model.trainable_weights
        )
        total = model.count_params()
        # All params should be trainable (no frozen layers)
        assert trainable == total


class TestPredictProba:
    """ONSET-15 Acceptance: Model supports predict_proba() equivalent via sigmoid output."""

    def test_returns_two_columns(self):
        from src.onset.models import build_onset_classifier, predict_proba

        model = build_onset_classifier()
        x = np.random.randn(8, 10, 4).astype(np.float32)
        proba = predict_proba(model, x, verbose=0)
        assert proba.shape == (8, 2)

    def test_columns_sum_to_one(self):
        from src.onset.models import build_onset_classifier, predict_proba

        model = build_onset_classifier()
        x = np.random.randn(10, 10, 4).astype(np.float32)
        proba = predict_proba(model, x, verbose=0)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_probabilities_in_zero_one_range(self):
        from src.onset.models import build_onset_classifier, predict_proba

        model = build_onset_classifier()
        x = np.random.randn(10, 10, 4).astype(np.float32)
        proba = predict_proba(model, x, verbose=0)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_class1_matches_model_predict(self):
        """Column 1 (P(degraded)) should match raw model.predict() output."""
        from src.onset.models import build_onset_classifier, predict_proba

        model = build_onset_classifier()
        x = np.random.randn(5, 10, 4).astype(np.float32)
        raw = model.predict(x, verbose=0).flatten()
        proba = predict_proba(model, x, verbose=0)
        np.testing.assert_allclose(proba[:, 1], raw, atol=1e-6)

    def test_class0_is_complement_of_class1(self):
        from src.onset.models import build_onset_classifier, predict_proba

        model = build_onset_classifier()
        x = np.random.randn(5, 10, 4).astype(np.float32)
        proba = predict_proba(model, x, verbose=0)
        np.testing.assert_allclose(proba[:, 0], 1.0 - proba[:, 1], atol=1e-6)

    def test_single_sample(self):
        from src.onset.models import build_onset_classifier, predict_proba

        model = build_onset_classifier()
        x = np.random.randn(1, 10, 4).astype(np.float32)
        proba = predict_proba(model, x, verbose=0)
        assert proba.shape == (1, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
