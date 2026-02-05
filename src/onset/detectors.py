"""Onset detection algorithms for bearing degradation.

This module provides multiple algorithms for detecting the onset of degradation
in bearing health indicator time series. Detection is the first stage in a
two-stage RUL prediction pipeline.

Algorithms:
- ThresholdOnsetDetector: Simple threshold-based detection (mean + k*sigma)
- CUSUMOnsetDetector: Cumulative Sum change-point detection (planned)
- BayesianOnsetDetector: Bayesian Online Change-Point Detection (planned)
- EnsembleOnsetDetector: Combine multiple detectors with voting (planned)

Reference:
- Threshold/CUSUM: Standard Statistical Process Control methods
- Bayesian: Adams & MacKay (2007), "Bayesian Online Changepoint Detection"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class OnsetResult:
    """Result container for onset detection.

    Attributes:
        onset_idx: Index in the time series where onset was detected.
            None if no onset detected (healthy throughout).
        onset_time: Time/file index corresponding to onset_idx.
            None if no onset detected.
        confidence: Confidence score in [0, 1] range.
            Higher values indicate stronger evidence for onset.
        healthy_baseline: Dictionary with healthy region statistics:
            - 'mean': Mean of healthy samples
            - 'std': Standard deviation of healthy samples
            - 'n_samples': Number of samples used for baseline
    """

    onset_idx: int | None
    onset_time: float | None
    confidence: float
    healthy_baseline: dict[str, float]


class BaseOnsetDetector(ABC):
    """Abstract base class for onset detection algorithms.

    All onset detectors must implement:
    - fit(): Learn healthy baseline from early samples
    - detect(): Find onset point in a health indicator series
    """

    @abstractmethod
    def fit(self, healthy_samples: np.ndarray) -> "BaseOnsetDetector":
        """Learn healthy baseline statistics from known healthy samples.

        Args:
            healthy_samples: Array of health indicator values from healthy period.

        Returns:
            Self, for method chaining.
        """
        pass

    @abstractmethod
    def detect(self, hi_series: np.ndarray) -> OnsetResult:
        """Detect onset point in a health indicator time series.

        Args:
            hi_series: Full health indicator time series for a bearing.

        Returns:
            OnsetResult with onset index and confidence.
        """
        pass

    def fit_detect(
        self,
        hi_series: np.ndarray,
        healthy_fraction: float = 0.1,
    ) -> OnsetResult:
        """Convenience method: fit on early samples and detect onset.

        Uses the first `healthy_fraction` of samples to establish baseline,
        then detects onset in the full series.

        Args:
            hi_series: Full health indicator time series.
            healthy_fraction: Fraction of samples to use as healthy baseline.
                Default 0.1 (first 10% assumed healthy).

        Returns:
            OnsetResult with onset index and confidence.
        """
        n_healthy = max(3, int(len(hi_series) * healthy_fraction))
        healthy_samples = hi_series[:n_healthy]
        self.fit(healthy_samples)
        return self.detect(hi_series)


class ThresholdOnsetDetector(BaseOnsetDetector):
    """Threshold-based onset detector using mean + k*sigma rule.

    Detects degradation onset when the health indicator exceeds a threshold
    defined as `mean + threshold_sigma * std` of the healthy baseline.
    Requires `min_consecutive` consecutive exceedances to confirm onset,
    reducing false positives from transient spikes.

    Attributes:
        threshold_sigma: Number of standard deviations above mean for threshold.
        min_consecutive: Minimum consecutive exceedances to confirm onset.
        _mean: Fitted healthy baseline mean.
        _std: Fitted healthy baseline standard deviation.
        _n_samples: Number of samples used for baseline fitting.
    """

    def __init__(
        self,
        threshold_sigma: float = 3.0,
        min_consecutive: int = 3,
    ) -> None:
        """Initialize the threshold-based onset detector.

        Args:
            threshold_sigma: Number of standard deviations above baseline mean
                to set the threshold. Default 3.0 (99.7% of normal distribution).
            min_consecutive: Minimum number of consecutive samples exceeding
                threshold to confirm onset. Default 3. Must be >= 1.
        """
        if threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be positive")
        if min_consecutive < 1:
            raise ValueError("min_consecutive must be at least 1")

        self.threshold_sigma = threshold_sigma
        self.min_consecutive = min_consecutive
        self._mean: float | None = None
        self._std: float | None = None
        self._n_samples: int = 0

    def fit(self, healthy_samples: np.ndarray) -> "ThresholdOnsetDetector":
        """Learn healthy baseline statistics from known healthy samples.

        Args:
            healthy_samples: Array of health indicator values from healthy period.
                Should be at least 3 samples for meaningful statistics.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If healthy_samples is empty or has fewer than 2 samples.
        """
        healthy_samples = np.asarray(healthy_samples)
        if healthy_samples.size < 2:
            raise ValueError("Need at least 2 samples to compute baseline statistics")

        # Filter out NaN values for robust estimation
        valid_samples = healthy_samples[~np.isnan(healthy_samples)]
        if valid_samples.size < 2:
            raise ValueError("Need at least 2 non-NaN samples for baseline")

        self._mean = float(np.mean(valid_samples))
        self._std = float(np.std(valid_samples, ddof=1))  # Sample std with Bessel correction
        self._n_samples = int(valid_samples.size)

        # Handle zero or near-zero std (constant signal)
        if self._std < 1e-10:
            self._std = 1e-10  # Small epsilon to avoid division issues

        return self

    def detect(self, hi_series: np.ndarray) -> OnsetResult:
        """Detect onset point in a health indicator time series.

        Scans the series for the first point where `min_consecutive` consecutive
        samples exceed the threshold (mean + threshold_sigma * std).

        Args:
            hi_series: Full health indicator time series for a bearing.

        Returns:
            OnsetResult with:
            - onset_idx: First index of the consecutive exceedance run, or None
            - onset_time: Same as onset_idx (assuming unit time steps)
            - confidence: How far the onset point exceeds threshold (normalized)
            - healthy_baseline: Dict with mean, std, n_samples

        Raises:
            RuntimeError: If detector has not been fitted.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        hi_series = np.asarray(hi_series)
        threshold = self._mean + self.threshold_sigma * self._std

        # Find consecutive exceedances
        exceeds = hi_series > threshold
        consecutive_count = 0
        onset_idx = None

        for i, exc in enumerate(exceeds):
            if exc:
                consecutive_count += 1
                if consecutive_count >= self.min_consecutive:
                    # Onset is at the START of the consecutive run
                    onset_idx = i - self.min_consecutive + 1
                    break
            else:
                consecutive_count = 0

        # Build result
        healthy_baseline = {
            "mean": self._mean,
            "std": self._std,
            "n_samples": self._n_samples,
            "threshold": threshold,
        }

        if onset_idx is None:
            # No onset detected
            return OnsetResult(
                onset_idx=None,
                onset_time=None,
                confidence=0.0,
                healthy_baseline=healthy_baseline,
            )

        # Compute confidence: how far onset value exceeds threshold
        # Normalized to [0, 1] using sigmoid-like transformation
        onset_value = hi_series[onset_idx]
        exceedance = (onset_value - threshold) / self._std
        # Confidence: saturates at ~1 for large exceedances
        confidence = min(1.0, exceedance / self.threshold_sigma) if exceedance > 0 else 0.0

        return OnsetResult(
            onset_idx=onset_idx,
            onset_time=float(onset_idx),  # Assuming unit time steps
            confidence=confidence,
            healthy_baseline=healthy_baseline,
        )
