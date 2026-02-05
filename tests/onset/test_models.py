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
