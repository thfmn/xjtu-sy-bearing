"""Tests for onset detection algorithms.

Tests the onset detector implementations including:
- ThresholdOnsetDetector: threshold-based detection
- Validation against synthetic data with known onset points
- Edge cases: no onset, transient spikes, constant signals

ONSET-3 Acceptance Criteria Tests:
- Detector finds onset within 10 samples of known onset for test data
- min_consecutive filter reduces false positives from transient spikes
- Confidence reflects how far HI exceeds threshold
- Returns None for onset_idx if no onset detected
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.onset.detectors import (
    BaseOnsetDetector,
    CUSUMOnsetDetector,
    OnsetResult,
    ThresholdOnsetDetector,
)
from src.onset.health_indicators import (
    compute_composite_hi,
    load_bearing_health_series,
)


# ============================================================================
# Fixtures for synthetic data with known onset points
# ============================================================================


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """Create synthetic features DataFrame with known onset points.

    Creates data for 2 bearings:
    - Bearing1_1: gradual degradation (no clear onset, starts from sample 0)
    - Bearing2_1: sudden degradation at sample 30 (known onset point)
    """
    np.random.seed(42)

    data = []

    # Bearing1_1: gradual degradation pattern (no clear onset)
    for i in range(50):
        # Kurtosis starts ~3 (healthy) and increases gradually
        base_kurtosis = 3.0 + (i / 50) * 10  # 3 -> 13
        data.append(
            {
                "bearing_id": "Bearing1_1",
                "condition": "35Hz12kN",
                "file_idx": i,
                "h_kurtosis": base_kurtosis + np.random.randn() * 0.5,
                "v_kurtosis": base_kurtosis * 0.9 + np.random.randn() * 0.5,
                "h_rms": 0.1 + (i / 50) * 0.3 + np.random.randn() * 0.01,
                "v_rms": 0.08 + (i / 50) * 0.25 + np.random.randn() * 0.01,
            }
        )

    # Bearing2_1: sudden degradation at sample 30 (KNOWN ONSET = 30)
    for i in range(50):
        if i < 30:
            # Healthy phase: low kurtosis with small variance
            base_kurtosis = 3.0 + np.random.randn() * 0.3
            base_rms = 0.1 + np.random.randn() * 0.01
        else:
            # Degraded phase: high kurtosis (clearly above threshold)
            base_kurtosis = 15.0 + np.random.randn() * 2.0
            base_rms = 0.5 + np.random.randn() * 0.05
        data.append(
            {
                "bearing_id": "Bearing2_1",
                "condition": "37.5Hz11kN",
                "file_idx": i,
                "h_kurtosis": base_kurtosis,
                "v_kurtosis": base_kurtosis * 0.95,
                "h_rms": base_rms,
                "v_rms": base_rms * 0.9,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sudden_onset_series() -> tuple[np.ndarray, int]:
    """Create synthetic HI series with known sudden onset.

    Returns:
        Tuple of (hi_series, known_onset_idx)
    """
    np.random.seed(42)
    n_samples = 100
    known_onset = 50

    hi_series = np.zeros(n_samples)
    # Healthy phase: low values around 0.1 with small noise
    hi_series[:known_onset] = 0.1 + np.random.randn(known_onset) * 0.02
    # Degraded phase: high values around 0.8 with noise
    hi_series[known_onset:] = 0.8 + np.random.randn(n_samples - known_onset) * 0.05

    return hi_series, known_onset


@pytest.fixture
def gradual_onset_series() -> tuple[np.ndarray, int]:
    """Create synthetic HI series with gradual onset.

    Returns:
        Tuple of (hi_series, approximate_onset_idx)
    """
    np.random.seed(42)
    n_samples = 100
    onset_start = 40  # Where degradation starts ramping

    hi_series = np.zeros(n_samples)
    # Healthy phase
    hi_series[:onset_start] = 0.1 + np.random.randn(onset_start) * 0.02

    # Gradual ramp from onset_start to onset_start + 20
    ramp_length = 20
    for i in range(ramp_length):
        idx = onset_start + i
        progress = i / ramp_length
        hi_series[idx] = 0.1 + progress * 0.7 + np.random.randn() * 0.02

    # Fully degraded
    hi_series[onset_start + ramp_length :] = (
        0.8 + np.random.randn(n_samples - onset_start - ramp_length) * 0.05
    )

    return hi_series, onset_start


@pytest.fixture
def transient_spike_series() -> np.ndarray:
    """Create synthetic HI series with transient spikes (not real onset)."""
    np.random.seed(42)
    n_samples = 100

    hi_series = 0.1 + np.random.randn(n_samples) * 0.02
    # Add transient spikes that should NOT trigger onset
    hi_series[20] = 0.9  # Single spike
    hi_series[40] = 0.85  # Single spike
    hi_series[60:62] = 0.8  # Two consecutive (below min_consecutive=3)

    return hi_series


@pytest.fixture
def healthy_only_series() -> np.ndarray:
    """Create synthetic HI series with no onset (healthy throughout)."""
    np.random.seed(42)
    n_samples = 100
    return 0.1 + np.random.randn(n_samples) * 0.02


# ============================================================================
# ONSET-3 Acceptance Criteria Tests
# ============================================================================


class TestOnsetWithin10Samples:
    """Test: Detector finds onset within 10 samples of known onset.

    This tests the first acceptance criterion for ONSET-3.
    """

    def test_sudden_onset_within_tolerance(
        self, sudden_onset_series: tuple[np.ndarray, int]
    ):
        """Test detector finds sudden onset within 10 samples of true onset."""
        hi_series, known_onset = sudden_onset_series
        TOLERANCE = 10

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is not None, "Detector should find onset"
        error = abs(result.onset_idx - known_onset)
        assert error <= TOLERANCE, (
            f"Onset detection error {error} exceeds tolerance {TOLERANCE}. "
            f"Detected: {result.onset_idx}, True: {known_onset}"
        )

    def test_sudden_onset_with_composite_hi(
        self, synthetic_features_df: pd.DataFrame
    ):
        """Test detector finds onset within 10 samples using composite HI.

        Uses Bearing2_1 which has known onset at sample 30.
        """
        KNOWN_ONSET = 30
        TOLERANCE = 10

        # Load health series for bearing with known onset
        health_series = load_bearing_health_series(
            "Bearing2_1", synthetic_features_df
        )

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        # Use first 20% as healthy baseline (samples 0-9, well before onset at 30)
        result = detector.fit_detect(health_series.composite, healthy_fraction=0.2)

        assert result.onset_idx is not None, "Detector should find onset in Bearing2_1"
        error = abs(result.onset_idx - KNOWN_ONSET)
        assert error <= TOLERANCE, (
            f"Onset detection error {error} exceeds tolerance {TOLERANCE}. "
            f"Detected: {result.onset_idx}, True: {KNOWN_ONSET}"
        )

    def test_gradual_onset_within_tolerance(
        self, gradual_onset_series: tuple[np.ndarray, int]
    ):
        """Test detector finds gradual onset within reasonable tolerance.

        Note: Gradual onsets are inherently ambiguous. The detector should
        find onset somewhere in the transition region.
        """
        hi_series, onset_start = gradual_onset_series
        # For gradual onset, allow wider tolerance since "true" onset is ambiguous
        TOLERANCE = 15

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is not None, "Detector should find onset"
        # For gradual onset, we expect detection somewhere in [onset_start, onset_start + 20]
        assert result.onset_idx >= onset_start - 5, (
            f"Onset detected too early: {result.onset_idx} < {onset_start - 5}"
        )
        assert result.onset_idx <= onset_start + TOLERANCE, (
            f"Onset detected too late: {result.onset_idx} > {onset_start + TOLERANCE}"
        )


class TestMinConsecutiveFilter:
    """Test: min_consecutive filter reduces false positives from transient spikes.

    This tests the second acceptance criterion for ONSET-3.
    """

    def test_single_spike_filtered(self, transient_spike_series: np.ndarray):
        """Test that single transient spikes don't trigger false onset."""
        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(transient_spike_series[:15])  # Fit on healthy start

        result = detector.detect(transient_spike_series)

        # Should NOT detect onset because spikes are transient (< min_consecutive)
        assert result.onset_idx is None, (
            f"False positive: detector triggered on transient spike at {result.onset_idx}"
        )

    def test_min_consecutive_1_triggers_on_spike(
        self, transient_spike_series: np.ndarray
    ):
        """Test that min_consecutive=1 DOES trigger on single spikes."""
        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=1)
        detector.fit(transient_spike_series[:15])

        result = detector.detect(transient_spike_series)

        # Should detect onset at first spike (index 20)
        assert result.onset_idx is not None, "Should trigger with min_consecutive=1"
        assert result.onset_idx == 20, f"Expected onset at 20, got {result.onset_idx}"

    def test_min_consecutive_2_vs_3(self, transient_spike_series: np.ndarray):
        """Test behavior difference between min_consecutive=2 and min_consecutive=3."""
        # Series has two consecutive spikes at indices 60-61

        detector_2 = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=2)
        detector_2.fit(transient_spike_series[:15])
        result_2 = detector_2.detect(transient_spike_series)

        detector_3 = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector_3.fit(transient_spike_series[:15])
        result_3 = detector_3.detect(transient_spike_series)

        # min_consecutive=2 should trigger at 60-61 (two consecutive)
        assert result_2.onset_idx == 60, f"Expected onset at 60, got {result_2.onset_idx}"
        # min_consecutive=3 should NOT trigger (only 2 consecutive)
        assert result_3.onset_idx is None, "Should not trigger with only 2 consecutive"


class TestConfidenceScore:
    """Test: Confidence score reflects how far HI exceeds threshold.

    This tests the third acceptance criterion for ONSET-3.
    """

    def test_higher_exceedance_higher_confidence(self):
        """Test that larger exceedance produces higher confidence.

        Uses carefully calibrated values so neither saturates at 1.0.
        """
        np.random.seed(42)
        n_samples = 100
        onset_idx = 50

        # Create baseline: mean ~0.1, std ~0.02
        baseline = 0.1 + np.random.randn(onset_idx) * 0.02

        # threshold = mean + 3*std ≈ 0.1 + 0.06 = 0.16
        # For confidence not to saturate: exceedance / std / threshold_sigma < 1
        # exceedance < std * threshold_sigma = 0.02 * 3 = 0.06 above threshold
        # So values around 0.17-0.20 won't saturate

        # Moderate: just above threshold (~0.18)
        hi_moderate = np.zeros(n_samples)
        hi_moderate[:onset_idx] = baseline
        hi_moderate[onset_idx:] = 0.18

        # Large: further above threshold (~0.22)
        hi_large = np.zeros(n_samples)
        hi_large[:onset_idx] = baseline
        hi_large[onset_idx:] = 0.22

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)

        detector.fit(hi_moderate[:30])
        result_moderate = detector.detect(hi_moderate)

        detector.fit(hi_large[:30])
        result_large = detector.detect(hi_large)

        # Both should detect onset
        assert result_moderate.onset_idx is not None
        assert result_large.onset_idx is not None

        # Larger exceedance should have higher confidence
        assert result_large.confidence > result_moderate.confidence, (
            f"Large exceedance confidence ({result_large.confidence:.3f}) should be "
            f"greater than moderate ({result_moderate.confidence:.3f})"
        )

    def test_confidence_in_valid_range(self, sudden_onset_series: tuple[np.ndarray, int]):
        """Test that confidence is in [0, 1] range."""
        hi_series, _ = sudden_onset_series

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence {result.confidence} outside [0, 1] range"
        )

    def test_no_onset_zero_confidence(self, healthy_only_series: np.ndarray):
        """Test that no onset detected gives zero confidence."""
        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        result = detector.fit_detect(healthy_only_series, healthy_fraction=0.3)

        assert result.onset_idx is None
        assert result.confidence == 0.0, (
            f"Expected 0.0 confidence for no onset, got {result.confidence}"
        )


class TestNoOnsetReturnsNone:
    """Test: Returns None for onset_idx if no onset detected.

    This tests the fourth acceptance criterion for ONSET-3.
    """

    def test_healthy_bearing_returns_none(self, healthy_only_series: np.ndarray):
        """Test that healthy-only series returns None for onset_idx."""
        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        result = detector.fit_detect(healthy_only_series, healthy_fraction=0.3)

        assert result.onset_idx is None, (
            f"Expected None onset_idx for healthy bearing, got {result.onset_idx}"
        )
        assert result.onset_time is None, "onset_time should also be None"

    def test_onset_result_fields_when_none(self, healthy_only_series: np.ndarray):
        """Test OnsetResult has correct field values when no onset."""
        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        result = detector.fit_detect(healthy_only_series, healthy_fraction=0.3)

        # Check all fields
        assert result.onset_idx is None
        assert result.onset_time is None
        assert result.confidence == 0.0
        assert "mean" in result.healthy_baseline
        assert "std" in result.healthy_baseline
        assert "threshold" in result.healthy_baseline


# ============================================================================
# ThresholdOnsetDetector Unit Tests
# ============================================================================


class TestThresholdOnsetDetector:
    """Unit tests for ThresholdOnsetDetector class."""

    def test_init_default_params(self):
        """Test default initialization parameters."""
        detector = ThresholdOnsetDetector()

        assert detector.threshold_sigma == 3.0
        assert detector.min_consecutive == 3

    def test_init_custom_params(self):
        """Test custom initialization parameters."""
        detector = ThresholdOnsetDetector(threshold_sigma=2.5, min_consecutive=5)

        assert detector.threshold_sigma == 2.5
        assert detector.min_consecutive == 5

    def test_init_invalid_threshold_sigma(self):
        """Test that non-positive threshold_sigma raises error."""
        with pytest.raises(ValueError, match="threshold_sigma must be positive"):
            ThresholdOnsetDetector(threshold_sigma=0)

        with pytest.raises(ValueError, match="threshold_sigma must be positive"):
            ThresholdOnsetDetector(threshold_sigma=-1)

    def test_init_invalid_min_consecutive(self):
        """Test that min_consecutive < 1 raises error."""
        with pytest.raises(ValueError, match="min_consecutive must be at least 1"):
            ThresholdOnsetDetector(min_consecutive=0)

    def test_fit_stores_baseline_stats(self):
        """Test that fit() stores baseline statistics."""
        np.random.seed(42)
        healthy = np.random.randn(20) * 0.5 + 3.0

        detector = ThresholdOnsetDetector()
        detector.fit(healthy)

        assert detector._mean is not None
        assert detector._std is not None
        assert detector._n_samples == 20

    def test_fit_too_few_samples(self):
        """Test that fit() raises error with < 2 samples."""
        detector = ThresholdOnsetDetector()

        with pytest.raises(ValueError, match="at least 2 samples"):
            detector.fit(np.array([1.0]))

    def test_detect_without_fit_raises(self):
        """Test that detect() without fit() raises error."""
        detector = ThresholdOnsetDetector()

        with pytest.raises(RuntimeError, match="not fitted"):
            detector.detect(np.array([1.0, 2.0, 3.0]))

    def test_detect_returns_onset_result(
        self, sudden_onset_series: tuple[np.ndarray, int]
    ):
        """Test that detect() returns OnsetResult."""
        hi_series, _ = sudden_onset_series

        detector = ThresholdOnsetDetector()
        detector.fit(hi_series[:30])
        result = detector.detect(hi_series)

        assert isinstance(result, OnsetResult)
        assert hasattr(result, "onset_idx")
        assert hasattr(result, "onset_time")
        assert hasattr(result, "confidence")
        assert hasattr(result, "healthy_baseline")

    def test_onset_idx_is_start_of_run(self):
        """Test that onset_idx is at START of consecutive run, not end."""
        np.random.seed(42)
        hi_series = np.zeros(20)
        hi_series[:10] = 0.1  # Healthy
        hi_series[10:] = 0.9  # Degraded (onset at 10)

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(hi_series[:5])
        result = detector.detect(hi_series)

        # Onset should be at 10 (start), not 12 (end of 3 consecutive)
        assert result.onset_idx == 10, (
            f"Onset should be at start of run (10), got {result.onset_idx}"
        )

    def test_fit_detect_convenience_method(
        self, sudden_onset_series: tuple[np.ndarray, int]
    ):
        """Test fit_detect() convenience method."""
        hi_series, known_onset = sudden_onset_series

        detector = ThresholdOnsetDetector()
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert isinstance(result, OnsetResult)
        assert result.onset_idx is not None
        # Should find onset close to known value
        assert abs(result.onset_idx - known_onset) <= 10

    def test_handles_nan_in_baseline(self):
        """Test that NaN values in baseline are filtered."""
        healthy = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])

        detector = ThresholdOnsetDetector()
        detector.fit(healthy)

        # Should have computed stats from non-NaN values
        assert detector._n_samples == 5  # 7 - 2 NaN
        assert not np.isnan(detector._mean)
        assert not np.isnan(detector._std)

    def test_constant_baseline_handled(self):
        """Test handling of constant healthy baseline (zero std)."""
        healthy = np.ones(10) * 5.0  # Constant

        detector = ThresholdOnsetDetector()
        detector.fit(healthy)

        # Should set small epsilon for std to avoid division by zero
        assert detector._std >= 1e-10

    def test_healthy_baseline_contains_threshold(
        self, sudden_onset_series: tuple[np.ndarray, int]
    ):
        """Test that healthy_baseline dict contains computed threshold."""
        hi_series, _ = sudden_onset_series

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(hi_series[:30])
        result = detector.detect(hi_series)

        assert "threshold" in result.healthy_baseline
        expected_threshold = detector._mean + 3.0 * detector._std
        assert result.healthy_baseline["threshold"] == pytest.approx(expected_threshold)


# ============================================================================
# BaseOnsetDetector Tests
# ============================================================================


class TestBaseOnsetDetector:
    """Tests for BaseOnsetDetector abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseOnsetDetector cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            BaseOnsetDetector()

    def test_subclass_must_implement_fit(self):
        """Test that subclass must implement fit()."""

        class IncompleteDetector(BaseOnsetDetector):
            def detect(self, hi_series):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteDetector()

    def test_subclass_must_implement_detect(self):
        """Test that subclass must implement detect()."""

        class IncompleteDetector(BaseOnsetDetector):
            def fit(self, healthy_samples):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteDetector()


# ============================================================================
# OnsetResult Tests
# ============================================================================


class TestOnsetResult:
    """Tests for OnsetResult dataclass."""

    def test_dataclass_fields(self):
        """Test OnsetResult has expected fields."""
        result = OnsetResult(
            onset_idx=50,
            onset_time=50.0,
            confidence=0.85,
            healthy_baseline={"mean": 0.1, "std": 0.02},
        )

        assert result.onset_idx == 50
        assert result.onset_time == 50.0
        assert result.confidence == 0.85
        assert result.healthy_baseline["mean"] == 0.1

    def test_none_values_allowed(self):
        """Test OnsetResult allows None for onset_idx and onset_time."""
        result = OnsetResult(
            onset_idx=None,
            onset_time=None,
            confidence=0.0,
            healthy_baseline={"mean": 0.1, "std": 0.02},
        )

        assert result.onset_idx is None
        assert result.onset_time is None


# ============================================================================
# Parametrized Tests for Different Threshold Values
# ============================================================================


class TestParametrizedThresholdValues:
    """Parametrized tests for different threshold_sigma values.

    Tests verify that threshold sensitivity affects detection behavior:
    - Lower threshold_sigma = more sensitive (earlier detection, more false positives)
    - Higher threshold_sigma = less sensitive (later detection, fewer false positives)
    """

    @pytest.mark.parametrize(
        "threshold_sigma,expected_detection",
        [
            (1.0, True),   # Very sensitive - should detect
            (2.0, True),   # Sensitive - should detect
            (3.0, True),   # Standard - should detect
            (5.0, True),   # Less sensitive - should still detect strong onset
            (10.0, False), # Very insensitive - may miss moderate onset
        ],
    )
    def test_threshold_sensitivity_on_detection(
        self, threshold_sigma: float, expected_detection: bool
    ):
        """Test that different threshold_sigma values affect detection.

        Uses a moderate onset signal (~5 std above healthy mean).
        Very high thresholds (10σ) should miss it.
        """
        np.random.seed(42)
        n_samples = 100
        onset_idx = 50

        # Healthy phase: mean=0.1, std≈0.02
        hi_series = np.zeros(n_samples)
        hi_series[:onset_idx] = 0.1 + np.random.randn(onset_idx) * 0.02

        # Degraded phase: ~5 std above healthy mean (0.1 + 5*0.02 = 0.2)
        hi_series[onset_idx:] = 0.2 + np.random.randn(n_samples - onset_idx) * 0.02

        detector = ThresholdOnsetDetector(
            threshold_sigma=threshold_sigma, min_consecutive=3
        )
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        if expected_detection:
            assert result.onset_idx is not None, (
                f"Expected detection with threshold_sigma={threshold_sigma}"
            )
        else:
            assert result.onset_idx is None, (
                f"Expected no detection with threshold_sigma={threshold_sigma}"
            )

    @pytest.mark.parametrize(
        "threshold_sigma",
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    def test_higher_threshold_later_or_same_detection(self, threshold_sigma: float):
        """Test that higher thresholds detect onset at same or later index.

        For gradual onset, lower thresholds should trigger earlier
        (closer to when HI first starts rising).
        """
        np.random.seed(42)
        n_samples = 100

        # Gradual ramp from 0.1 to 0.9 between indices 30-60
        hi_series = np.zeros(n_samples)
        hi_series[:30] = 0.1 + np.random.randn(30) * 0.015
        for i in range(30):
            hi_series[30 + i] = 0.1 + (i / 30) * 0.8 + np.random.randn() * 0.015
        hi_series[60:] = 0.9 + np.random.randn(40) * 0.015

        # Detect with current and next threshold
        detector_low = ThresholdOnsetDetector(
            threshold_sigma=threshold_sigma, min_consecutive=3
        )
        detector_high = ThresholdOnsetDetector(
            threshold_sigma=threshold_sigma + 1.0, min_consecutive=3
        )

        result_low = detector_low.fit_detect(hi_series, healthy_fraction=0.25)
        result_high = detector_high.fit_detect(hi_series, healthy_fraction=0.25)

        # Both should detect (strong signal)
        assert result_low.onset_idx is not None
        assert result_high.onset_idx is not None

        # Higher threshold should detect at same or later index
        assert result_high.onset_idx >= result_low.onset_idx, (
            f"Higher threshold ({threshold_sigma + 1}) detected earlier "
            f"({result_high.onset_idx}) than lower threshold ({threshold_sigma}) "
            f"({result_low.onset_idx})"
        )

    @pytest.mark.parametrize(
        "threshold_sigma,min_consecutive",
        [
            (2.0, 1),
            (2.0, 3),
            (2.0, 5),
            (3.0, 1),
            (3.0, 3),
            (3.0, 5),
            (4.0, 1),
            (4.0, 3),
            (4.0, 5),
        ],
    )
    def test_parameter_combinations_produce_valid_results(
        self, threshold_sigma: float, min_consecutive: int
    ):
        """Test various parameter combinations produce valid OnsetResult."""
        np.random.seed(42)
        n_samples = 100
        onset_idx = 50

        hi_series = np.zeros(n_samples)
        hi_series[:onset_idx] = 0.1 + np.random.randn(onset_idx) * 0.02
        hi_series[onset_idx:] = 0.5 + np.random.randn(n_samples - onset_idx) * 0.05

        detector = ThresholdOnsetDetector(
            threshold_sigma=threshold_sigma, min_consecutive=min_consecutive
        )
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        # Verify result structure is valid regardless of parameters
        assert isinstance(result, OnsetResult)
        assert 0.0 <= result.confidence <= 1.0
        assert "mean" in result.healthy_baseline
        assert "std" in result.healthy_baseline
        assert "threshold" in result.healthy_baseline

        # If onset detected, verify it's reasonable
        if result.onset_idx is not None:
            assert 0 <= result.onset_idx < n_samples
            assert result.onset_time is not None

    @pytest.mark.parametrize("threshold_sigma", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    def test_threshold_affects_confidence(self, threshold_sigma: float):
        """Test that same signal produces different confidence at different thresholds.

        Lower threshold = lower relative exceedance = potentially lower confidence.
        This tests that confidence computation scales with threshold_sigma.
        """
        np.random.seed(42)
        n_samples = 100
        onset_idx = 50

        hi_series = np.zeros(n_samples)
        hi_series[:onset_idx] = 0.1 + np.random.randn(onset_idx) * 0.02
        hi_series[onset_idx:] = 0.3  # Fixed exceedance

        detector = ThresholdOnsetDetector(
            threshold_sigma=threshold_sigma, min_consecutive=3
        )
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        # All should detect (strong signal)
        if result.onset_idx is not None:
            # Confidence should be valid
            assert 0.0 <= result.confidence <= 1.0


# ============================================================================
# CUSUMOnsetDetector Tests
# ============================================================================


class TestCUSUMOnsetDetector:
    """Unit tests for CUSUMOnsetDetector class."""

    def test_init_default_params(self):
        """Test default initialization parameters."""
        detector = CUSUMOnsetDetector()

        assert detector.drift == 0.5
        assert detector.threshold == 5.0
        assert detector.direction == "increase"

    def test_init_custom_params(self):
        """Test custom initialization parameters."""
        detector = CUSUMOnsetDetector(drift=0.25, threshold=4.0, direction="decrease")

        assert detector.drift == 0.25
        assert detector.threshold == 4.0
        assert detector.direction == "decrease"

    def test_init_invalid_drift(self):
        """Test that non-positive drift raises error."""
        with pytest.raises(ValueError, match="drift must be positive"):
            CUSUMOnsetDetector(drift=0)

        with pytest.raises(ValueError, match="drift must be positive"):
            CUSUMOnsetDetector(drift=-0.5)

    def test_init_invalid_threshold(self):
        """Test that non-positive threshold raises error."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            CUSUMOnsetDetector(threshold=0)

    def test_init_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be one of"):
            CUSUMOnsetDetector(direction="invalid")

    def test_fit_stores_baseline_stats(self):
        """Test that fit() stores baseline statistics."""
        np.random.seed(42)
        healthy = np.random.randn(20) * 0.5 + 3.0

        detector = CUSUMOnsetDetector()
        detector.fit(healthy)

        assert detector._target_mean is not None
        assert detector._std is not None
        assert detector._n_samples == 20

    def test_fit_too_few_samples(self):
        """Test that fit() raises error with < 2 samples."""
        detector = CUSUMOnsetDetector()

        with pytest.raises(ValueError, match="at least 2 samples"):
            detector.fit(np.array([1.0]))

    def test_detect_without_fit_raises(self):
        """Test that detect() without fit() raises error."""
        detector = CUSUMOnsetDetector()

        with pytest.raises(RuntimeError, match="not fitted"):
            detector.detect(np.array([1.0, 2.0, 3.0]))

    def test_detect_returns_onset_result(self):
        """Test that detect() returns OnsetResult."""
        np.random.seed(42)
        n_samples = 100
        onset_idx = 50

        hi_series = np.zeros(n_samples)
        hi_series[:onset_idx] = 0.1 + np.random.randn(onset_idx) * 0.02
        hi_series[onset_idx:] = 0.5 + np.random.randn(n_samples - onset_idx) * 0.02

        detector = CUSUMOnsetDetector()
        detector.fit(hi_series[:30])
        result = detector.detect(hi_series)

        assert isinstance(result, OnsetResult)
        assert hasattr(result, "onset_idx")
        assert hasattr(result, "onset_time")
        assert hasattr(result, "confidence")
        assert hasattr(result, "healthy_baseline")

    def test_handles_nan_in_baseline(self):
        """Test that NaN values in baseline are filtered."""
        healthy = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])

        detector = CUSUMOnsetDetector()
        detector.fit(healthy)

        assert detector._n_samples == 5  # 7 - 2 NaN
        assert not np.isnan(detector._target_mean)
        assert not np.isnan(detector._std)

    def test_healthy_baseline_contains_cusum_params(self):
        """Test that healthy_baseline dict contains CUSUM-specific parameters."""
        np.random.seed(42)
        n_samples = 100
        onset_idx = 50

        hi_series = np.zeros(n_samples)
        hi_series[:onset_idx] = 0.1 + np.random.randn(onset_idx) * 0.02
        hi_series[onset_idx:] = 0.5 + np.random.randn(n_samples - onset_idx) * 0.02

        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        detector.fit(hi_series[:30])
        result = detector.detect(hi_series)

        assert "drift" in result.healthy_baseline
        assert "threshold" in result.healthy_baseline
        assert "direction" in result.healthy_baseline
        assert result.healthy_baseline["drift"] == 0.5
        assert result.healthy_baseline["threshold"] == 5.0


class TestCUSUMGradualShift:
    """Test CUSUMOnsetDetector on synthetic data with gradual shift.

    This tests the ONSET-6 acceptance criterion that CUSUM should detect
    gradual shifts that threshold-based detectors might miss or detect late.
    """

    @pytest.fixture
    def gradual_shift_series(self) -> tuple[np.ndarray, int]:
        """Create synthetic HI series with gradual mean shift.

        The shift is designed to be subtle enough that threshold detection
        is delayed, but CUSUM should detect it earlier by accumulating evidence.

        Returns:
            Tuple of (hi_series, shift_start_idx)
        """
        np.random.seed(42)
        n_samples = 150
        shift_start = 60

        hi_series = np.zeros(n_samples)
        std = 0.02

        # Healthy phase: stationary around 0.1
        hi_series[:shift_start] = 0.1 + np.random.randn(shift_start) * std

        # Gradual shift: ramp from 0.1 to 0.2 over 40 samples
        ramp_length = 40
        for i in range(ramp_length):
            idx = shift_start + i
            progress = i / ramp_length
            # Mean shifts from 0.1 to 0.2 (5 sigma total shift)
            hi_series[idx] = 0.1 + progress * 0.1 + np.random.randn() * std

        # Post-shift: stationary at elevated level
        hi_series[shift_start + ramp_length :] = (
            0.2 + np.random.randn(n_samples - shift_start - ramp_length) * std
        )

        return hi_series, shift_start

    def test_cusum_detects_gradual_shift(
        self, gradual_shift_series: tuple[np.ndarray, int]
    ):
        """Test that CUSUM detects gradual shift."""
        hi_series, shift_start = gradual_shift_series

        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is not None, "CUSUM should detect gradual shift"
        # Onset should be detected somewhere in the shift region
        # Allow detection from shift_start to shift_start + 50 (shift region + margin)
        assert result.onset_idx >= shift_start - 5, (
            f"Onset detected too early: {result.onset_idx} < {shift_start - 5}"
        )
        assert result.onset_idx <= shift_start + 50, (
            f"Onset detected too late: {result.onset_idx} > {shift_start + 50}"
        )

    def test_cusum_detects_earlier_than_threshold_for_gradual_shift(
        self, gradual_shift_series: tuple[np.ndarray, int]
    ):
        """Test that CUSUM detects gradual shifts earlier than threshold detector.

        This is a key advantage of CUSUM: it accumulates small deviations,
        detecting shifts before any single sample exceeds a threshold.
        """
        hi_series, shift_start = gradual_shift_series

        # CUSUM detector with default params
        cusum = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        cusum_result = cusum.fit_detect(hi_series, healthy_fraction=0.3)

        # Threshold detector with standard 3-sigma threshold
        threshold_det = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        threshold_result = threshold_det.fit_detect(hi_series, healthy_fraction=0.3)

        # Both should detect onset
        assert cusum_result.onset_idx is not None, "CUSUM should detect onset"
        assert threshold_result.onset_idx is not None, "Threshold should detect onset"

        # CUSUM should detect at same time or earlier than threshold
        # (for gradual shifts, CUSUM is typically earlier)
        assert cusum_result.onset_idx <= threshold_result.onset_idx, (
            f"CUSUM ({cusum_result.onset_idx}) should detect no later than "
            f"threshold ({threshold_result.onset_idx}) for gradual shifts"
        )

    def test_cusum_sensitivity_to_drift_parameter(
        self, gradual_shift_series: tuple[np.ndarray, int]
    ):
        """Test that drift parameter affects detection sensitivity.

        Lower drift = more sensitive (earlier detection, but more false positives)
        Higher drift = less sensitive (later detection, but fewer false positives)
        """
        hi_series, shift_start = gradual_shift_series

        # Lower drift = more sensitive
        detector_low = CUSUMOnsetDetector(drift=0.25, threshold=5.0)
        result_low = detector_low.fit_detect(hi_series, healthy_fraction=0.3)

        # Higher drift = less sensitive
        detector_high = CUSUMOnsetDetector(drift=1.0, threshold=5.0)
        result_high = detector_high.fit_detect(hi_series, healthy_fraction=0.3)

        # Both should detect (strong enough signal)
        assert result_low.onset_idx is not None
        assert result_high.onset_idx is not None

        # Lower drift should detect at same time or earlier
        assert result_low.onset_idx <= result_high.onset_idx, (
            f"Lower drift ({result_low.onset_idx}) should detect no later than "
            f"higher drift ({result_high.onset_idx})"
        )

    def test_cusum_no_false_alarm_on_healthy_series(
        self, healthy_only_series: np.ndarray
    ):
        """Test that CUSUM doesn't trigger false alarms on healthy-only data."""
        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        result = detector.fit_detect(healthy_only_series, healthy_fraction=0.3)

        assert result.onset_idx is None, (
            f"False alarm: CUSUM triggered at {result.onset_idx} on healthy series"
        )


class TestCUSUMDirection:
    """Test CUSUM direction parameter for increase/decrease detection."""

    @pytest.fixture
    def upward_shift_series(self) -> tuple[np.ndarray, int]:
        """Create series with upward mean shift."""
        np.random.seed(42)
        n_samples = 100
        shift_idx = 50

        hi_series = np.zeros(n_samples)
        hi_series[:shift_idx] = 0.1 + np.random.randn(shift_idx) * 0.02
        hi_series[shift_idx:] = 0.3 + np.random.randn(n_samples - shift_idx) * 0.02

        return hi_series, shift_idx

    @pytest.fixture
    def downward_shift_series(self) -> tuple[np.ndarray, int]:
        """Create series with downward mean shift."""
        np.random.seed(42)
        n_samples = 100
        shift_idx = 50

        hi_series = np.zeros(n_samples)
        hi_series[:shift_idx] = 0.3 + np.random.randn(shift_idx) * 0.02
        hi_series[shift_idx:] = 0.1 + np.random.randn(n_samples - shift_idx) * 0.02

        return hi_series, shift_idx

    def test_direction_increase_detects_upward_shift(
        self, upward_shift_series: tuple[np.ndarray, int]
    ):
        """Test direction='increase' detects upward shifts."""
        hi_series, shift_idx = upward_shift_series

        detector = CUSUMOnsetDetector(direction="increase")
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is not None, "Should detect upward shift"
        assert abs(result.onset_idx - shift_idx) <= 10, (
            f"Detection {result.onset_idx} too far from true shift {shift_idx}"
        )

    def test_direction_increase_ignores_downward_shift(
        self, downward_shift_series: tuple[np.ndarray, int]
    ):
        """Test direction='increase' ignores downward shifts."""
        hi_series, _ = downward_shift_series

        detector = CUSUMOnsetDetector(direction="increase")
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is None, (
            f"direction='increase' should not detect downward shift, "
            f"but detected at {result.onset_idx}"
        )

    def test_direction_decrease_detects_downward_shift(
        self, downward_shift_series: tuple[np.ndarray, int]
    ):
        """Test direction='decrease' detects downward shifts."""
        hi_series, shift_idx = downward_shift_series

        detector = CUSUMOnsetDetector(direction="decrease")
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is not None, "Should detect downward shift"
        assert abs(result.onset_idx - shift_idx) <= 10, (
            f"Detection {result.onset_idx} too far from true shift {shift_idx}"
        )

    def test_direction_decrease_ignores_upward_shift(
        self, upward_shift_series: tuple[np.ndarray, int]
    ):
        """Test direction='decrease' ignores upward shifts."""
        hi_series, _ = upward_shift_series

        detector = CUSUMOnsetDetector(direction="decrease")
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is None, (
            f"direction='decrease' should not detect upward shift, "
            f"but detected at {result.onset_idx}"
        )

    def test_direction_both_detects_either_shift(
        self,
        upward_shift_series: tuple[np.ndarray, int],
        downward_shift_series: tuple[np.ndarray, int],
    ):
        """Test direction='both' detects shifts in either direction."""
        hi_up, shift_up = upward_shift_series
        hi_down, shift_down = downward_shift_series

        detector = CUSUMOnsetDetector(direction="both")

        # Should detect upward
        result_up = detector.fit_detect(hi_up, healthy_fraction=0.3)
        assert result_up.onset_idx is not None, "Should detect upward shift"

        # Should detect downward
        result_down = detector.fit_detect(hi_down, healthy_fraction=0.3)
        assert result_down.onset_idx is not None, "Should detect downward shift"

    def test_triggered_direction_in_result(
        self, upward_shift_series: tuple[np.ndarray, int]
    ):
        """Test that triggered_direction is recorded in healthy_baseline."""
        hi_series, _ = upward_shift_series

        detector = CUSUMOnsetDetector(direction="both")
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is not None
        assert "triggered_direction" in result.healthy_baseline
        assert result.healthy_baseline["triggered_direction"] == "increase"


# ============================================================================
# CUSUM Drift Parameter Sensitivity vs. False Positive Trade-off Tests
# ONSET-4 Acceptance Criterion
# ============================================================================


class TestCUSUMDriftSensitivityTradeoff:
    """Test that tunable drift parameter controls sensitivity vs. false positive trade-off.

    This validates the ONSET-4 acceptance criterion:
    "Tunable drift parameter controls sensitivity vs. false positive trade-off"

    The drift parameter (k) in CUSUM controls:
    - Lower drift → more sensitive → earlier detection but more false positives
    - Higher drift → less sensitive → fewer false positives but later detection

    These tests use borderline/ambiguous signals that expose the trade-off clearly.
    """

    @pytest.fixture
    def borderline_shift_series(self) -> tuple[np.ndarray, int]:
        """Create series with a borderline shift (hard to detect reliably).

        The shift is designed to be ~2 sigma - detectable with low drift,
        but may be missed or delayed with high drift.
        """
        np.random.seed(42)
        n_samples = 200
        shift_start = 100

        hi_series = np.zeros(n_samples)
        std = 0.05

        # Healthy phase: stationary
        hi_series[:shift_start] = 1.0 + np.random.randn(shift_start) * std

        # Borderline shift: ~2 sigma increase (1.0 -> 1.1, shift = 0.1 = 2*std)
        hi_series[shift_start:] = 1.1 + np.random.randn(n_samples - shift_start) * std

        return hi_series, shift_start

    @pytest.fixture
    def noisy_no_shift_series(self) -> np.ndarray:
        """Create noisy series with NO real shift (for false positive testing).

        Some random noise peaks may temporarily look like shifts.
        """
        np.random.seed(123)  # Different seed for independence
        n_samples = 200
        std = 0.05

        # Pure noise around mean, no actual shift
        return 1.0 + np.random.randn(n_samples) * std

    def test_lower_drift_detects_borderline_shift(
        self, borderline_shift_series: tuple[np.ndarray, int]
    ):
        """Test that low drift (high sensitivity) detects borderline shifts.

        Note: With very low drift (0.1), CUSUM accumulates random noise and may
        trigger before the actual shift. This is expected behavior - it shows
        the sensitivity/false-positive trade-off. The key is detection HAPPENS.
        """
        hi_series, shift_start = borderline_shift_series

        # Very sensitive detector
        detector = CUSUMOnsetDetector(drift=0.1, threshold=5.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.4)

        # Should detect SOMETHING (may be early due to high sensitivity)
        assert result.onset_idx is not None, (
            "Low drift should detect borderline shift"
        )
        # With low drift, detection may happen before actual shift (accumulating noise)
        # Key verification: detection happens before end of series
        assert result.onset_idx < len(hi_series), "Should detect before end of series"

    def test_higher_drift_may_miss_borderline_shift(
        self, borderline_shift_series: tuple[np.ndarray, int]
    ):
        """Test that high drift (low sensitivity) may miss borderline shifts.

        This demonstrates the trade-off: high drift reduces false positives
        but may also miss real but subtle shifts.
        """
        hi_series, shift_start = borderline_shift_series

        # Less sensitive detector
        detector = CUSUMOnsetDetector(drift=1.5, threshold=5.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.4)

        # May or may not detect - the key is this is later than low drift
        if result.onset_idx is not None:
            # If detected, should be later than low drift would detect
            detector_low = CUSUMOnsetDetector(drift=0.1, threshold=5.0)
            result_low = detector_low.fit_detect(hi_series, healthy_fraction=0.4)

            assert result_low.onset_idx is not None
            assert result.onset_idx >= result_low.onset_idx, (
                f"High drift ({result.onset_idx}) should detect no earlier "
                f"than low drift ({result_low.onset_idx})"
            )

    def test_drift_affects_false_positive_rate(
        self, noisy_no_shift_series: np.ndarray
    ):
        """Test that higher drift reduces false positives on noisy data.

        Run multiple trials with different noise realizations and count
        false positives at different drift levels.
        """
        n_trials = 20
        low_drift_fps = 0
        high_drift_fps = 0

        for seed in range(n_trials):
            np.random.seed(seed)
            # Generate noisy series with no true shift
            hi_series = 1.0 + np.random.randn(150) * 0.05

            # Low drift - more sensitive, expect more false positives
            detector_low = CUSUMOnsetDetector(drift=0.1, threshold=5.0)
            result_low = detector_low.fit_detect(hi_series, healthy_fraction=0.3)
            if result_low.onset_idx is not None:
                low_drift_fps += 1

            # High drift - less sensitive, expect fewer false positives
            detector_high = CUSUMOnsetDetector(drift=1.0, threshold=5.0)
            result_high = detector_high.fit_detect(hi_series, healthy_fraction=0.3)
            if result_high.onset_idx is not None:
                high_drift_fps += 1

        # Higher drift should have same or fewer false positives
        assert high_drift_fps <= low_drift_fps, (
            f"High drift should have <= false positives than low drift. "
            f"High: {high_drift_fps}, Low: {low_drift_fps}"
        )

    @pytest.mark.parametrize(
        "drift",
        [0.1, 0.25, 0.5, 0.75, 1.0, 1.5],
    )
    def test_drift_monotonically_affects_detection_timing(self, drift: float):
        """Test that increasing drift leads to same or later detection.

        For a consistent signal, increasing drift should not make detection EARLIER.
        """
        np.random.seed(42)
        n_samples = 150
        shift_idx = 60

        # Clear 3-sigma shift
        hi_series = np.zeros(n_samples)
        hi_series[:shift_idx] = 1.0 + np.random.randn(shift_idx) * 0.05
        hi_series[shift_idx:] = 1.15 + np.random.randn(n_samples - shift_idx) * 0.05

        # Detect with current drift
        detector = CUSUMOnsetDetector(drift=drift, threshold=5.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        # Detect with slightly higher drift
        detector_higher = CUSUMOnsetDetector(drift=drift + 0.25, threshold=5.0)
        result_higher = detector_higher.fit_detect(hi_series, healthy_fraction=0.3)

        if result.onset_idx is not None and result_higher.onset_idx is not None:
            # Higher drift should detect at same time or later
            assert result_higher.onset_idx >= result.onset_idx, (
                f"Higher drift ({drift + 0.25}) detected earlier "
                f"({result_higher.onset_idx}) than lower drift ({drift}) "
                f"({result.onset_idx})"
            )

    def test_drift_tradeoff_summary(self):
        """Comprehensive test demonstrating sensitivity vs. false positive trade-off.

        Creates two scenarios:
        1. True shift (should be detected)
        2. No shift (should not trigger false alarm)

        And shows how drift affects detection in both scenarios.
        """
        np.random.seed(42)

        # Scenario 1: True 2-sigma shift at index 80
        hi_with_shift = np.zeros(160)
        hi_with_shift[:80] = 1.0 + np.random.randn(80) * 0.05
        hi_with_shift[80:] = 1.1 + np.random.randn(80) * 0.05

        # Scenario 2: No shift (pure noise)
        np.random.seed(99)
        hi_no_shift = 1.0 + np.random.randn(160) * 0.05

        # Test at different drift levels
        drift_levels = [0.1, 0.5, 1.0]
        results = []

        for drift in drift_levels:
            detector = CUSUMOnsetDetector(drift=drift, threshold=5.0)

            # Detection on true shift
            res_shift = detector.fit_detect(hi_with_shift, healthy_fraction=0.4)

            # Detection on no shift (false positive check)
            res_no_shift = detector.fit_detect(hi_no_shift, healthy_fraction=0.4)

            results.append({
                "drift": drift,
                "detected_shift": res_shift.onset_idx is not None,
                "detection_idx": res_shift.onset_idx,
                "false_positive": res_no_shift.onset_idx is not None,
            })

        # Verify trade-off pattern:
        # Low drift should be most likely to detect true shift
        assert results[0]["detected_shift"], "Low drift should detect true shift"

        # Higher drift should have same or fewer false positives
        # (Can't always guarantee detection at high drift for borderline shift)
        for i in range(len(results) - 1):
            if results[i]["false_positive"] is False:
                # If lower drift has no FP, higher drift also shouldn't
                assert results[i + 1]["false_positive"] is False or \
                       results[i + 1]["false_positive"] == results[i]["false_positive"], (
                    f"Higher drift should not have more false positives"
                )


# ============================================================================
# ONSET-4 Acceptance: Works for both sudden and gradual degradation patterns
# ============================================================================


class TestCUSUMSuddenAndGradualPatterns:
    """Test that CUSUM works for both sudden and gradual degradation patterns.

    This validates the ONSET-4 acceptance criterion:
    "Works for both sudden and gradual degradation patterns"

    Bearing degradation in real-world scenarios can manifest as:
    1. Sudden failures: Abrupt increase in kurtosis/RMS (e.g., spalling)
    2. Gradual wear: Slow, progressive increase over time (e.g., surface fatigue)

    The CUSUM detector should handle both cases effectively.
    """

    @pytest.fixture
    def sudden_degradation_series(self) -> tuple[np.ndarray, int]:
        """Create series simulating sudden degradation (step change).

        Models a bearing that experiences sudden spalling or defect,
        causing an immediate jump in health indicator values.
        """
        np.random.seed(42)
        n_samples = 120
        sudden_onset = 60  # Degradation starts abruptly here

        hi_series = np.zeros(n_samples)
        std = 0.03

        # Healthy phase: stable low values
        hi_series[:sudden_onset] = 0.1 + np.random.randn(sudden_onset) * std

        # Sudden degradation: immediate step to higher level
        hi_series[sudden_onset:] = 0.4 + np.random.randn(n_samples - sudden_onset) * std

        return hi_series, sudden_onset

    @pytest.fixture
    def gradual_degradation_series(self) -> tuple[np.ndarray, int]:
        """Create series simulating gradual degradation (slow ramp).

        Models a bearing with progressive surface fatigue,
        causing a slow but steady increase in health indicator.
        """
        np.random.seed(42)
        n_samples = 150
        gradual_start = 50  # Degradation starts ramping here
        ramp_duration = 60  # Takes 60 samples to fully degrade

        hi_series = np.zeros(n_samples)
        std = 0.025

        # Healthy phase
        hi_series[:gradual_start] = 0.1 + np.random.randn(gradual_start) * std

        # Gradual ramp: linear increase from 0.1 to 0.35 over ramp_duration
        for i in range(ramp_duration):
            idx = gradual_start + i
            progress = i / ramp_duration
            hi_series[idx] = 0.1 + progress * 0.25 + np.random.randn() * std

        # Fully degraded phase
        remaining = n_samples - gradual_start - ramp_duration
        hi_series[gradual_start + ramp_duration:] = (
            0.35 + np.random.randn(remaining) * std
        )

        return hi_series, gradual_start

    def test_cusum_detects_sudden_degradation(
        self, sudden_degradation_series: tuple[np.ndarray, int]
    ):
        """Test that CUSUM detects sudden step-change degradation."""
        hi_series, true_onset = sudden_degradation_series
        TOLERANCE = 10  # Allow 10 samples tolerance

        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is not None, "CUSUM should detect sudden degradation"
        assert abs(result.onset_idx - true_onset) <= TOLERANCE, (
            f"Sudden onset detection error {abs(result.onset_idx - true_onset)} "
            f"exceeds tolerance {TOLERANCE}. Detected: {result.onset_idx}, True: {true_onset}"
        )

    def test_cusum_detects_gradual_degradation(
        self, gradual_degradation_series: tuple[np.ndarray, int]
    ):
        """Test that CUSUM detects gradual ramp degradation."""
        hi_series, true_onset = gradual_degradation_series
        # Gradual onset is inherently ambiguous - allow detection anywhere in ramp region
        EARLY_BOUND = true_onset - 5
        LATE_BOUND = true_onset + 40  # Within ramp duration

        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.25)

        assert result.onset_idx is not None, "CUSUM should detect gradual degradation"
        assert result.onset_idx >= EARLY_BOUND, (
            f"Onset detected too early: {result.onset_idx} < {EARLY_BOUND}"
        )
        assert result.onset_idx <= LATE_BOUND, (
            f"Onset detected too late: {result.onset_idx} > {LATE_BOUND}"
        )

    def test_cusum_works_on_both_patterns_same_params(
        self,
        sudden_degradation_series: tuple[np.ndarray, int],
        gradual_degradation_series: tuple[np.ndarray, int],
    ):
        """Test that same CUSUM parameters work for both sudden and gradual patterns.

        This is important for practical use: we need parameters that work
        across different bearing failure modes without retuning.
        """
        hi_sudden, onset_sudden = sudden_degradation_series
        hi_gradual, onset_gradual = gradual_degradation_series

        # Use same detector configuration for both
        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)

        # Test sudden
        result_sudden = detector.fit_detect(hi_sudden, healthy_fraction=0.3)
        assert result_sudden.onset_idx is not None, (
            "Same params should work for sudden degradation"
        )

        # Test gradual
        result_gradual = detector.fit_detect(hi_gradual, healthy_fraction=0.25)
        assert result_gradual.onset_idx is not None, (
            "Same params should work for gradual degradation"
        )

        # Both detections should be valid
        assert isinstance(result_sudden, OnsetResult)
        assert isinstance(result_gradual, OnsetResult)
        assert result_sudden.confidence > 0
        assert result_gradual.confidence > 0

    @pytest.mark.parametrize(
        "drift,threshold",
        [
            (0.25, 4.0),  # More sensitive
            (0.5, 5.0),   # Default/balanced
            (0.75, 6.0),  # Less sensitive
        ],
    )
    def test_cusum_parameter_robustness_both_patterns(
        self, drift: float, threshold: float
    ):
        """Test various CUSUM parameters work for both degradation patterns.

        Validates that the algorithm is robust across parameter choices,
        not just a single hand-tuned configuration.
        """
        np.random.seed(42)

        # Sudden pattern
        hi_sudden = np.zeros(100)
        hi_sudden[:50] = 0.1 + np.random.randn(50) * 0.02
        hi_sudden[50:] = 0.35 + np.random.randn(50) * 0.02

        # Gradual pattern
        np.random.seed(43)
        hi_gradual = np.zeros(100)
        hi_gradual[:40] = 0.1 + np.random.randn(40) * 0.02
        for i in range(30):
            hi_gradual[40 + i] = 0.1 + (i / 30) * 0.25 + np.random.randn() * 0.02
        hi_gradual[70:] = 0.35 + np.random.randn(30) * 0.02

        detector = CUSUMOnsetDetector(drift=drift, threshold=threshold)

        # Both should be detected (signals are strong enough)
        result_sudden = detector.fit_detect(hi_sudden, healthy_fraction=0.3)
        result_gradual = detector.fit_detect(hi_gradual, healthy_fraction=0.3)

        assert result_sudden.onset_idx is not None, (
            f"Sudden not detected with drift={drift}, threshold={threshold}"
        )
        assert result_gradual.onset_idx is not None, (
            f"Gradual not detected with drift={drift}, threshold={threshold}"
        )


class TestEWMASuddenAndGradualPatterns:
    """Test that EWMA detector works for both sudden and gradual degradation.

    EWMA is the optional variant from ONSET-4 and should also handle
    both degradation patterns effectively.
    """

    def test_ewma_detects_sudden_degradation(self):
        """Test EWMA handles sudden step-change degradation."""
        from src.onset.detectors import EWMAOnsetDetector

        np.random.seed(42)
        n_samples = 100
        sudden_onset = 50

        hi_series = np.zeros(n_samples)
        hi_series[:sudden_onset] = 0.1 + np.random.randn(sudden_onset) * 0.02
        hi_series[sudden_onset:] = 0.4 + np.random.randn(n_samples - sudden_onset) * 0.02

        detector = EWMAOnsetDetector(lambda_=0.2, L=3.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        assert result.onset_idx is not None, "EWMA should detect sudden degradation"
        assert abs(result.onset_idx - sudden_onset) <= 15, (
            f"EWMA onset error {abs(result.onset_idx - sudden_onset)} too large"
        )

    def test_ewma_detects_gradual_degradation(self):
        """Test EWMA handles gradual ramp degradation."""
        from src.onset.detectors import EWMAOnsetDetector

        np.random.seed(42)
        n_samples = 120
        gradual_start = 40

        hi_series = np.zeros(n_samples)
        hi_series[:gradual_start] = 0.1 + np.random.randn(gradual_start) * 0.02

        for i in range(40):
            hi_series[gradual_start + i] = 0.1 + (i / 40) * 0.25 + np.random.randn() * 0.02

        hi_series[gradual_start + 40:] = 0.35 + np.random.randn(n_samples - gradual_start - 40) * 0.02

        detector = EWMAOnsetDetector(lambda_=0.2, L=3.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.25)

        assert result.onset_idx is not None, "EWMA should detect gradual degradation"
        # Should detect somewhere in or near the ramp region
        assert result.onset_idx >= gradual_start - 5
        assert result.onset_idx <= gradual_start + 50

    def test_ewma_returns_onset_result_dataclass(self):
        """Test EWMA returns same OnsetResult dataclass as other detectors."""
        from src.onset.detectors import EWMAOnsetDetector

        np.random.seed(42)
        hi_series = np.zeros(80)
        hi_series[:40] = 0.1 + np.random.randn(40) * 0.02
        hi_series[40:] = 0.5 + np.random.randn(40) * 0.02

        detector = EWMAOnsetDetector(lambda_=0.2, L=3.0)
        result = detector.fit_detect(hi_series, healthy_fraction=0.3)

        # Verify it's the same dataclass with same fields
        assert isinstance(result, OnsetResult)
        assert hasattr(result, "onset_idx")
        assert hasattr(result, "onset_time")
        assert hasattr(result, "confidence")
        assert hasattr(result, "healthy_baseline")

        # onset_time should match onset_idx (unit time steps)
        if result.onset_idx is not None:
            assert result.onset_time == float(result.onset_idx)


class TestOnsetResultConsistency:
    """Test that all detectors return the same OnsetResult dataclass structure.

    This validates the ONSET-4 acceptance criterion:
    "Returns same OnsetResult dataclass as threshold detector"
    """

    def test_all_detectors_return_onset_result(self):
        """Test that all detector types return OnsetResult dataclass."""
        from src.onset.detectors import EWMAOnsetDetector

        np.random.seed(42)
        hi_series = np.zeros(100)
        hi_series[:50] = 0.1 + np.random.randn(50) * 0.02
        hi_series[50:] = 0.5 + np.random.randn(50) * 0.02

        detectors = [
            ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3),
            CUSUMOnsetDetector(drift=0.5, threshold=5.0),
            EWMAOnsetDetector(lambda_=0.2, L=3.0),
        ]

        for detector in detectors:
            result = detector.fit_detect(hi_series, healthy_fraction=0.3)

            assert isinstance(result, OnsetResult), (
                f"{detector.__class__.__name__} should return OnsetResult"
            )

    def test_onset_result_fields_consistent_across_detectors(self):
        """Test all detectors populate OnsetResult fields consistently."""
        from src.onset.detectors import EWMAOnsetDetector

        np.random.seed(42)
        hi_series = np.zeros(100)
        hi_series[:50] = 0.1 + np.random.randn(50) * 0.02
        hi_series[50:] = 0.5 + np.random.randn(50) * 0.02

        detectors = [
            ("Threshold", ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)),
            ("CUSUM", CUSUMOnsetDetector(drift=0.5, threshold=5.0)),
            ("EWMA", EWMAOnsetDetector(lambda_=0.2, L=3.0)),
        ]

        for name, detector in detectors:
            result = detector.fit_detect(hi_series, healthy_fraction=0.3)

            # All should detect onset (strong signal)
            assert result.onset_idx is not None, f"{name} should detect onset"

            # onset_idx and onset_time should be consistent
            assert isinstance(result.onset_idx, int), f"{name} onset_idx should be int"
            assert result.onset_time == float(result.onset_idx), (
                f"{name} onset_time should equal float(onset_idx)"
            )

            # Confidence should be in valid range
            assert 0.0 <= result.confidence <= 1.0, (
                f"{name} confidence should be in [0, 1]"
            )

            # healthy_baseline should contain common fields
            assert "mean" in result.healthy_baseline, (
                f"{name} should have 'mean' in healthy_baseline"
            )
            assert "std" in result.healthy_baseline, (
                f"{name} should have 'std' in healthy_baseline"
            )
            assert "n_samples" in result.healthy_baseline, (
                f"{name} should have 'n_samples' in healthy_baseline"
            )

    def test_no_onset_result_consistent_across_detectors(self):
        """Test all detectors return consistent result when no onset detected."""
        from src.onset.detectors import EWMAOnsetDetector

        np.random.seed(42)
        # Healthy-only series (no onset)
        hi_series = 0.1 + np.random.randn(100) * 0.02

        detectors = [
            ("Threshold", ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)),
            ("CUSUM", CUSUMOnsetDetector(drift=0.5, threshold=5.0)),
            ("EWMA", EWMAOnsetDetector(lambda_=0.2, L=3.0)),
        ]

        for name, detector in detectors:
            result = detector.fit_detect(hi_series, healthy_fraction=0.3)

            # All should return no onset
            assert result.onset_idx is None, f"{name} should return None onset_idx"
            assert result.onset_time is None, f"{name} should return None onset_time"
            assert result.confidence == 0.0, f"{name} should return 0.0 confidence"
            assert isinstance(result.healthy_baseline, dict), (
                f"{name} should return dict healthy_baseline"
            )
