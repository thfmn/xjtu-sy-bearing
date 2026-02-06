"""Tests for the two-stage pipeline module.

Tests the TwoStagePipeline class which chains onset detection (Stage 1)
with RUL prediction (Stage 2).

ONSET-18 Acceptance Criteria Tests:
- Pipeline correctly chains onset detection and RUL prediction
- Pre-onset samples receive max_rul (125) prediction
- Post-onset samples receive model predictions
- Supports swapping onset detector without changing RUL model
"""

from __future__ import annotations

import numpy as np
import pytest

from src.onset.detectors import (
    BaseOnsetDetector,
    CUSUMOnsetDetector,
    OnsetResult,
    ThresholdOnsetDetector,
)
from src.onset.pipeline import TwoStagePipeline


# ============================================================================
# Mock classes for isolated testing
# ============================================================================


class MockRulModel:
    """Mock RUL regression model that returns linearly decreasing RUL."""

    def __init__(self, start_rul: float = 50.0):
        self.start_rul = start_rul
        self.predict_called = False
        self.predict_input_shape = None

    def predict(self, x, verbose=0):
        self.predict_called = True
        self.predict_input_shape = x.shape
        n = x.shape[0]
        return np.linspace(self.start_rul, 0, n).reshape(-1, 1)


class MockOnsetClassifier:
    """Mock ML onset classifier that returns fixed probabilities."""

    def __init__(self, probs: np.ndarray):
        self.probs = probs

    def predict(self, x, verbose=0):
        return self.probs.reshape(-1, 1)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def healthy_then_degraded_series():
    """1-D HI series: 50 healthy samples, then 50 degraded samples."""
    np.random.seed(42)
    healthy = np.random.normal(0.0, 0.1, 50)
    degraded = np.random.normal(3.0, 0.3, 50)
    return np.concatenate([healthy, degraded])


@pytest.fixture
def fitted_threshold_detector():
    """ThresholdOnsetDetector fitted on healthy baseline."""
    detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
    healthy = np.random.RandomState(42).normal(0.0, 0.1, 50)
    detector.fit(healthy)
    return detector


@pytest.fixture
def mock_rul_model():
    """Mock RUL model returning linearly decreasing values."""
    return MockRulModel(start_rul=50.0)


# ============================================================================
# ONSET-18 Acceptance: Pipeline correctly chains onset detection and RUL prediction
# ============================================================================


class TestPipelineChainsOnsetAndRul:
    """Verify pipeline correctly chains Stage 1 (onset) and Stage 2 (RUL)."""

    def test_predict_calls_detect_onset_then_predict_rul(
        self, fitted_threshold_detector, mock_rul_model, healthy_then_degraded_series
    ):
        """predict() should call detect_onset first, then predict_rul."""
        pipeline = TwoStagePipeline(
            onset_detector=fitted_threshold_detector,
            rul_model=mock_rul_model,
        )
        result = pipeline.predict(healthy_then_degraded_series)

        # RUL model was called (Stage 2 executed)
        assert mock_rul_model.predict_called
        # Result covers all 100 samples
        assert result.shape == (100,)

    def test_detect_onset_result_feeds_into_predict_rul(
        self, fitted_threshold_detector, mock_rul_model, healthy_then_degraded_series
    ):
        """The onset_idx from detect_onset should be used by predict_rul."""
        pipeline = TwoStagePipeline(
            onset_detector=fitted_threshold_detector,
            rul_model=mock_rul_model,
        )

        # Run detect_onset independently to get onset_idx
        onset_result = pipeline.detect_onset(healthy_then_degraded_series)
        assert onset_result.onset_idx is not None

        # Run full pipeline
        full_result = pipeline.predict(healthy_then_degraded_series)

        # Run predict_rul with same onset_idx
        manual_result = pipeline.predict_rul(
            healthy_then_degraded_series, onset_result.onset_idx
        )

        # Both should produce identical results
        np.testing.assert_array_equal(full_result, manual_result)

    def test_predict_returns_correct_length(
        self, fitted_threshold_detector, mock_rul_model, healthy_then_degraded_series
    ):
        """predict() returns one RUL value per input sample."""
        pipeline = TwoStagePipeline(
            onset_detector=fitted_threshold_detector,
            rul_model=mock_rul_model,
        )
        result = pipeline.predict(healthy_then_degraded_series)
        assert len(result) == len(healthy_then_degraded_series)

    def test_predict_with_separate_rul_signals(
        self, fitted_threshold_detector, mock_rul_model, healthy_then_degraded_series
    ):
        """predict() uses rul_signals for Stage 2 when provided."""
        pipeline = TwoStagePipeline(
            onset_detector=fitted_threshold_detector,
            rul_model=mock_rul_model,
        )

        # rul_signals has same number of samples but different shape
        rul_signals = np.random.rand(100, 10)

        result = pipeline.predict(healthy_then_degraded_series, rul_signals=rul_signals)

        # Result still covers all samples
        assert result.shape == (100,)
        # RUL model received multi-dim input (from rul_signals, not onset_signals)
        assert mock_rul_model.predict_input_shape is not None
        assert mock_rul_model.predict_input_shape[1] == 10

    def test_predict_with_ml_classifier_chains_correctly(self, mock_rul_model):
        """Pipeline chains ML classifier (Stage 1) with RUL model (Stage 2)."""
        # ML classifier: first 40 windows healthy, last 60 degraded
        probs = np.concatenate([
            np.full(40, 0.1),  # healthy
            np.full(60, 0.9),  # degraded
        ])
        onset_classifier = MockOnsetClassifier(probs)

        # Still need a detector (required by __init__)
        detector = ThresholdOnsetDetector(threshold_sigma=3.0)
        detector.fit(np.zeros(10))

        pipeline = TwoStagePipeline(
            onset_detector=detector,
            rul_model=mock_rul_model,
            onset_model=onset_classifier,
        )

        onset_signals = np.random.rand(100, 10, 4)  # windowed for ML classifier
        rul_signals = np.random.rand(100, 5)  # different shape for RUL model

        result = pipeline.predict(onset_signals, rul_signals=rul_signals)

        assert result.shape == (100,)
        # Pre-onset (first 40) should be max_rul=125
        np.testing.assert_array_equal(result[:40], 125.0)
        # Post-onset (last 60) should be model predictions (not 125)
        assert not np.all(result[40:] == 125.0)

    def test_pipeline_output_is_float32(
        self, fitted_threshold_detector, mock_rul_model, healthy_then_degraded_series
    ):
        """Pipeline output should be float32 dtype."""
        pipeline = TwoStagePipeline(
            onset_detector=fitted_threshold_detector,
            rul_model=mock_rul_model,
        )
        result = pipeline.predict(healthy_then_degraded_series)
        assert result.dtype == np.float32


# ============================================================================
# ONSET-18 Acceptance: Pre-onset samples receive max_rul (125) prediction
# ============================================================================


class TestPreOnsetSamplesReceiveMaxRul:
    """Verify that all pre-onset samples receive max_rul (125) prediction."""

    def test_pre_onset_samples_are_exactly_max_rul(
        self, fitted_threshold_detector, mock_rul_model, healthy_then_degraded_series
    ):
        """Samples before onset_idx must be exactly max_rul=125."""
        pipeline = TwoStagePipeline(
            onset_detector=fitted_threshold_detector,
            rul_model=mock_rul_model,
        )

        onset_result = pipeline.detect_onset(healthy_then_degraded_series)
        assert onset_result.onset_idx is not None
        onset_idx = onset_result.onset_idx

        result = pipeline.predict(healthy_then_degraded_series)

        # Every sample before onset_idx must be exactly 125.0
        pre_onset = result[:onset_idx]
        assert len(pre_onset) > 0, "Need at least one pre-onset sample"
        np.testing.assert_array_equal(pre_onset, 125.0)

    def test_no_onset_all_samples_are_max_rul(self, mock_rul_model):
        """When no onset is detected, ALL samples should be max_rul=125."""
        # Healthy-only signal: low variance, no degradation
        healthy_only = np.random.RandomState(42).normal(0.0, 0.1, 100)

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(healthy_only[:50])

        pipeline = TwoStagePipeline(
            onset_detector=detector,
            rul_model=mock_rul_model,
        )

        # Verify no onset detected
        onset_result = pipeline.detect_onset(healthy_only)
        assert onset_result.onset_idx is None

        result = pipeline.predict(healthy_only)

        # All 100 samples should be exactly 125.0
        np.testing.assert_array_equal(result, 125.0)
        # RUL model should NOT have been called
        assert not mock_rul_model.predict_called

    def test_predict_rul_with_none_onset_returns_max_rul(self, mock_rul_model):
        """predict_rul(signals, onset_idx=None) returns all max_rul."""
        pipeline = TwoStagePipeline(
            onset_detector=ThresholdOnsetDetector(),
            rul_model=mock_rul_model,
        )

        signals = np.random.rand(80)
        result = pipeline.predict_rul(signals, onset_idx=None)

        np.testing.assert_array_equal(result, 125.0)
        assert result.shape == (80,)
        assert not mock_rul_model.predict_called

    def test_custom_max_rul_value_applied_to_pre_onset(
        self, fitted_threshold_detector, healthy_then_degraded_series
    ):
        """Custom max_rul value should be used for pre-onset samples."""
        custom_max = 200
        rul_model = MockRulModel(start_rul=50.0)

        pipeline = TwoStagePipeline(
            onset_detector=fitted_threshold_detector,
            rul_model=rul_model,
            max_rul=custom_max,
        )

        onset_result = pipeline.detect_onset(healthy_then_degraded_series)
        assert onset_result.onset_idx is not None

        result = pipeline.predict(healthy_then_degraded_series)
        pre_onset = result[: onset_result.onset_idx]

        # Pre-onset samples use the custom max_rul, not default 125
        np.testing.assert_array_equal(pre_onset, float(custom_max))

    def test_pre_onset_max_rul_with_ml_classifier(self, mock_rul_model):
        """Pre-onset samples get max_rul when using ML classifier for Stage 1."""
        # ML classifier: first 30 healthy, last 70 degraded
        probs = np.concatenate([
            np.full(30, 0.1),  # healthy
            np.full(70, 0.9),  # degraded
        ])
        onset_classifier = MockOnsetClassifier(probs)

        detector = ThresholdOnsetDetector()
        detector.fit(np.zeros(10))

        pipeline = TwoStagePipeline(
            onset_detector=detector,
            rul_model=mock_rul_model,
            onset_model=onset_classifier,
        )

        onset_signals = np.random.rand(100, 10, 4)
        rul_signals = np.random.rand(100, 5)

        result = pipeline.predict(onset_signals, rul_signals=rul_signals)

        # First 30 samples (pre-onset) must be exactly 125.0
        np.testing.assert_array_equal(result[:30], 125.0)
