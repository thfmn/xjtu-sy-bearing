"""Onset detection algorithms for bearing degradation.

This module provides multiple algorithms for detecting the onset of degradation
in bearing health indicator time series. Detection is the first stage in a
two-stage RUL prediction pipeline.

Algorithms:
- ThresholdOnsetDetector: Simple threshold-based detection (mean + k*sigma)
- CUSUMOnsetDetector: Cumulative Sum change-point detection
- EWMAOnsetDetector: Exponentially Weighted Moving Average change-point detection
- BayesianOnsetDetector: Bayesian Online Change-Point Detection (Adams & MacKay, 2007)
- EnsembleOnsetDetector: Combine multiple detectors with voting

Reference:
- Threshold/CUSUM: Standard Statistical Process Control methods
- Bayesian: Adams & MacKay (2007), "Bayesian Online Changepoint Detection"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

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


class CUSUMOnsetDetector(BaseOnsetDetector):
    """CUSUM (Cumulative Sum) change-point detector for onset detection.

    Implements the tabular CUSUM algorithm for detecting mean shifts in time series.
    CUSUM accumulates deviations from a target mean, making it sensitive to small
    but persistent shifts - ideal for detecting gradual bearing degradation.

    The algorithm maintains upper and lower cumulative sums:
    - S_high(t) = max(0, S_high(t-1) + (x_t - μ - k))  # Detects increase
    - S_low(t)  = max(0, S_low(t-1) + (μ - k - x_t))   # Detects decrease

    The `direction` parameter controls which shift triggers onset:
    - "increase": Only S_high triggers onset (default, for kurtosis/RMS)
    - "decrease": Only S_low triggers onset (for features that drop with degradation)
    - "both": Either S_high or S_low can trigger onset (whichever exceeds threshold first)

    Attributes:
        drift: Allowable drift (k) in units of standard deviation. Controls
            sensitivity: smaller k detects smaller shifts but increases false alarms.
        threshold: Decision threshold (h) in units of standard deviation.
            Onset triggered when cumulative sum exceeds h * std.
        direction: Which shift direction triggers onset ("increase", "decrease", "both").
        _target_mean: Fitted target mean from healthy samples.
        _std: Fitted standard deviation from healthy samples.
        _n_samples: Number of samples used for baseline fitting.

    Reference:
        Page, E.S. (1954). "Continuous Inspection Schemes".
        Montgomery, D.C. (2009). "Statistical Quality Control", 6th ed.
    """

    VALID_DIRECTIONS = ("increase", "decrease", "both")

    def __init__(
        self,
        drift: float = 0.5,
        threshold: float = 5.0,
        direction: str = "increase",
    ) -> None:
        """Initialize the CUSUM onset detector.

        Args:
            drift: Allowable drift parameter (k) in std units. Controls the
                reference value shift from target mean. Typical range: 0.25-1.0.
                Default 0.5 is a common choice for detecting 1-sigma shifts.
            threshold: Decision threshold (h) in std units. Onset detected when
                cumulative sum exceeds threshold * std. Typical range: 4-6.
                Default 5.0 provides reasonable ARL (Average Run Length) properties.
            direction: Which shift direction triggers onset detection:
                - "increase": Detect upward shifts (S_high > h). Default.
                - "decrease": Detect downward shifts (S_low > h).
                - "both": Detect either direction (first to exceed threshold).

        Raises:
            ValueError: If drift or threshold is not positive, or direction is invalid.
        """
        if drift <= 0:
            raise ValueError("drift must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self.VALID_DIRECTIONS}, got '{direction}'"
            )

        self.drift = drift
        self.threshold = threshold
        self.direction = direction
        self._target_mean: float | None = None
        self._std: float | None = None
        self._n_samples: int = 0

    def fit(self, healthy_samples: np.ndarray) -> "CUSUMOnsetDetector":
        """Learn target mean and standard deviation from healthy samples.

        Args:
            healthy_samples: Array of health indicator values from healthy period.
                Should be at least 3 samples for meaningful statistics.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If healthy_samples has fewer than 2 valid samples.
        """
        healthy_samples = np.asarray(healthy_samples)
        if healthy_samples.size < 2:
            raise ValueError("Need at least 2 samples to compute baseline statistics")

        # Filter out NaN values for robust estimation
        valid_samples = healthy_samples[~np.isnan(healthy_samples)]
        if valid_samples.size < 2:
            raise ValueError("Need at least 2 non-NaN samples for baseline")

        self._target_mean = float(np.mean(valid_samples))
        self._std = float(np.std(valid_samples, ddof=1))  # Sample std
        self._n_samples = int(valid_samples.size)

        # Handle zero or near-zero std (constant signal)
        if self._std < 1e-10:
            self._std = 1e-10

        return self

    def detect(self, hi_series: np.ndarray) -> OnsetResult:
        """Detect onset point using CUSUM algorithm.

        Computes upper and lower CUSUM statistics and returns the first index
        where the specified direction's statistic exceeds the threshold.

        Args:
            hi_series: Full health indicator time series for a bearing.

        Returns:
            OnsetResult with:
            - onset_idx: First index where CUSUM exceeds threshold, or None
            - onset_time: Same as onset_idx (assuming unit time steps)
            - confidence: Normalized CUSUM value at onset (how far it exceeded)
            - healthy_baseline: Dict with target_mean, std, n_samples, drift, threshold, direction

        Raises:
            RuntimeError: If detector has not been fitted.
        """
        if self._target_mean is None or self._std is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        hi_series = np.asarray(hi_series)
        n = len(hi_series)

        # CUSUM parameters in original units
        k = self.drift * self._std  # Reference value (allowable slack)
        h = self.threshold * self._std  # Decision threshold

        # Initialize CUSUM statistics
        s_high = 0.0  # Upper CUSUM (detects increase)
        s_low = 0.0   # Lower CUSUM (detects decrease)

        onset_idx = None
        onset_direction = None  # Track which direction triggered onset
        max_cusum = 0.0

        for i in range(n):
            x = hi_series[i]

            # Skip NaN values (don't update CUSUM)
            if np.isnan(x):
                continue

            # Update CUSUM statistics
            s_high = max(0.0, s_high + (x - self._target_mean - k))
            s_low = max(0.0, s_low + (self._target_mean - k - x))

            # Track maximum for confidence calculation
            current_max = max(s_high, s_low)
            if current_max > max_cusum:
                max_cusum = current_max

            # Check for onset based on direction setting
            if onset_idx is None:
                if self.direction == "increase" and s_high > h:
                    onset_idx = i
                    onset_direction = "increase"
                elif self.direction == "decrease" and s_low > h:
                    onset_idx = i
                    onset_direction = "decrease"
                elif self.direction == "both":
                    if s_high > h:
                        onset_idx = i
                        onset_direction = "increase"
                    elif s_low > h:
                        onset_idx = i
                        onset_direction = "decrease"

        # Build result
        healthy_baseline = {
            "mean": self._target_mean,
            "std": self._std,
            "n_samples": self._n_samples,
            "drift": self.drift,
            "threshold": self.threshold,
            "direction": self.direction,
            "decision_threshold_h": h,
        }

        if onset_idx is None:
            return OnsetResult(
                onset_idx=None,
                onset_time=None,
                confidence=0.0,
                healthy_baseline=healthy_baseline,
            )

        # Add which direction actually triggered onset
        healthy_baseline["triggered_direction"] = onset_direction

        # Confidence: how far CUSUM exceeded threshold, normalized to [0, 1]
        # Use the max CUSUM value reached, normalized by threshold
        confidence = min(1.0, max_cusum / (2 * h)) if h > 0 else 1.0

        return OnsetResult(
            onset_idx=onset_idx,
            onset_time=float(onset_idx),
            confidence=confidence,
            healthy_baseline=healthy_baseline,
        )


class EWMAOnsetDetector(BaseOnsetDetector):
    """EWMA (Exponentially Weighted Moving Average) change-point detector.

    Implements the EWMA control chart for detecting mean shifts. EWMA applies
    exponential smoothing to the time series, making it effective at detecting
    small but persistent shifts while filtering out noise.

    The EWMA statistic is computed as:
        Z_t = λ × X_t + (1 - λ) × Z_{t-1}

    Control limits (time-varying) are:
        UCL_t = μ + L × σ × √(λ/(2-λ) × [1 - (1-λ)^{2t}])
        LCL_t = μ - L × σ × √(λ/(2-λ) × [1 - (1-λ)^{2t}])

    As t → ∞, limits converge to:
        UCL = μ + L × σ × √(λ/(2-λ))
        LCL = μ - L × σ × √(λ/(2-λ))

    Attributes:
        lambda_: Smoothing parameter (0 < λ ≤ 1). Smaller values give more
            smoothing and slower detection. Typical range: 0.05-0.3.
        L: Control limit width in standard deviations. Default 3.0.
        direction: Which shift direction triggers onset ("increase", "decrease", "both").
        _target_mean: Fitted target mean from healthy samples.
        _std: Fitted standard deviation from healthy samples.
        _n_samples: Number of samples used for baseline fitting.

    Reference:
        Roberts, S.W. (1959). "Control Chart Tests Based on Geometric Moving Averages".
        Montgomery, D.C. (2009). "Statistical Quality Control", 6th ed., Chapter 9.
    """

    VALID_DIRECTIONS = ("increase", "decrease", "both")

    def __init__(
        self,
        lambda_: float = 0.2,
        L: float = 3.0,
        direction: str = "increase",
    ) -> None:
        """Initialize the EWMA onset detector.

        Args:
            lambda_: Smoothing parameter (0 < λ ≤ 1). Controls weight given to
                recent observations. Smaller λ = more smoothing, better for
                detecting small shifts. Default 0.2 is a common choice.
            L: Control limit multiplier (number of sigma for control limits).
                Typical range: 2.5-3.5. Default 3.0.
            direction: Which shift direction triggers onset detection:
                - "increase": Detect upward shifts (Z > UCL). Default.
                - "decrease": Detect downward shifts (Z < LCL).
                - "both": Detect either direction (first to exceed limits).

        Raises:
            ValueError: If lambda_ not in (0, 1], L not positive, or direction invalid.
        """
        if not (0 < lambda_ <= 1):
            raise ValueError("lambda_ must be in (0, 1]")
        if L <= 0:
            raise ValueError("L must be positive")
        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self.VALID_DIRECTIONS}, got '{direction}'"
            )

        self.lambda_ = lambda_
        self.L = L
        self.direction = direction
        self._target_mean: float | None = None
        self._std: float | None = None
        self._n_samples: int = 0

    def fit(self, healthy_samples: np.ndarray) -> "EWMAOnsetDetector":
        """Learn target mean and standard deviation from healthy samples.

        Args:
            healthy_samples: Array of health indicator values from healthy period.
                Should be at least 3 samples for meaningful statistics.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If healthy_samples has fewer than 2 valid samples.
        """
        healthy_samples = np.asarray(healthy_samples)
        if healthy_samples.size < 2:
            raise ValueError("Need at least 2 samples to compute baseline statistics")

        # Filter out NaN values for robust estimation
        valid_samples = healthy_samples[~np.isnan(healthy_samples)]
        if valid_samples.size < 2:
            raise ValueError("Need at least 2 non-NaN samples for baseline")

        self._target_mean = float(np.mean(valid_samples))
        self._std = float(np.std(valid_samples, ddof=1))  # Sample std
        self._n_samples = int(valid_samples.size)

        # Handle zero or near-zero std (constant signal)
        if self._std < 1e-10:
            self._std = 1e-10

        return self

    def detect(self, hi_series: np.ndarray) -> OnsetResult:
        """Detect onset point using EWMA algorithm.

        Computes EWMA statistic and checks against time-varying control limits.
        Returns the first index where the statistic exceeds the specified
        direction's control limit.

        Args:
            hi_series: Full health indicator time series for a bearing.

        Returns:
            OnsetResult with:
            - onset_idx: First index where EWMA exceeds control limit, or None
            - onset_time: Same as onset_idx (assuming unit time steps)
            - confidence: How far EWMA exceeded the limit (normalized)
            - healthy_baseline: Dict with target_mean, std, lambda_, L, direction

        Raises:
            RuntimeError: If detector has not been fitted.
        """
        if self._target_mean is None or self._std is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        hi_series = np.asarray(hi_series)
        n = len(hi_series)

        # Initialize EWMA statistic at target mean
        z = self._target_mean
        onset_idx = None
        onset_direction = None
        max_exceedance = 0.0

        # Asymptotic control limit half-width
        asymptotic_sigma = self._std * np.sqrt(self.lambda_ / (2 - self.lambda_))

        for t in range(n):
            x = hi_series[t]

            # Skip NaN values (keep z unchanged)
            if np.isnan(x):
                continue

            # Update EWMA statistic
            z = self.lambda_ * x + (1 - self.lambda_) * z

            # Compute time-varying control limit factor
            # σ_z(t) = σ × √(λ/(2-λ) × [1 - (1-λ)^{2(t+1)}])
            time_factor = np.sqrt(1 - (1 - self.lambda_) ** (2 * (t + 1)))
            sigma_z = asymptotic_sigma * time_factor

            # Control limits
            ucl = self._target_mean + self.L * sigma_z
            lcl = self._target_mean - self.L * sigma_z

            # Track exceedance for confidence calculation
            if z > ucl:
                exceedance = (z - ucl) / sigma_z if sigma_z > 0 else 0
                if exceedance > max_exceedance:
                    max_exceedance = exceedance
            elif z < lcl:
                exceedance = (lcl - z) / sigma_z if sigma_z > 0 else 0
                if exceedance > max_exceedance:
                    max_exceedance = exceedance

            # Check for onset based on direction setting
            if onset_idx is None:
                if self.direction == "increase" and z > ucl:
                    onset_idx = t
                    onset_direction = "increase"
                elif self.direction == "decrease" and z < lcl:
                    onset_idx = t
                    onset_direction = "decrease"
                elif self.direction == "both":
                    if z > ucl:
                        onset_idx = t
                        onset_direction = "increase"
                    elif z < lcl:
                        onset_idx = t
                        onset_direction = "decrease"

        # Build result
        healthy_baseline = {
            "mean": self._target_mean,
            "std": self._std,
            "n_samples": self._n_samples,
            "lambda": self.lambda_,
            "L": self.L,
            "direction": self.direction,
            "asymptotic_sigma_z": asymptotic_sigma,
        }

        if onset_idx is None:
            return OnsetResult(
                onset_idx=None,
                onset_time=None,
                confidence=0.0,
                healthy_baseline=healthy_baseline,
            )

        # Add which direction actually triggered onset
        healthy_baseline["triggered_direction"] = onset_direction

        # Confidence: how far beyond limit, normalized to [0, 1]
        confidence = min(1.0, max_exceedance / self.L) if self.L > 0 else 1.0

        return OnsetResult(
            onset_idx=onset_idx,
            onset_time=float(onset_idx),
            confidence=confidence,
            healthy_baseline=healthy_baseline,
        )


class BayesianOnsetDetector(BaseOnsetDetector):
    """Bayesian Online Change-Point Detection (Adams & MacKay, 2007).

    Maintains a posterior distribution over the current run length (time since
    the last changepoint). At each timestep, computes the probability that a
    changepoint has occurred by marginalizing over all possible run lengths.

    Uses a Normal-Inverse-Gamma (NIG) conjugate prior for Gaussian observations,
    allowing exact Bayesian inference. The sufficient statistics are updated
    incrementally for each possible run length, making the algorithm O(n^2) in
    total computation and O(n) in memory per timestep.

    The algorithm:
        1. At each timestep t, observe x_t
        2. Evaluate predictive probability P(x_t | r_{t-1}) for each run length
        3. Compute growth probabilities: P(r_t = r_{t-1}+1) ∝ π(x_t|r) × (1 - H(r))
        4. Compute changepoint probability: P(r_t = 0) ∝ Σ π(x_t|r) × H(r)
        5. Normalize to get P(r_t | x_{1:t})
        6. Detect onset where P(r_t = 0) is large (changepoint posterior spike)

    Onset is detected when the posterior run-length distribution shifts
    dramatically — the MAP (most likely) run length drops, indicating
    that probability mass has moved from a long established run to a
    new short run after a changepoint. Specifically, onset is declared
    when the cumulative probability on "short" run lengths (r < threshold)
    exceeds `cp_threshold`.

    Attributes:
        hazard_rate: Prior rate of changepoints (1/expected_run_length).
            Higher = more changepoints expected. Default 1/200.
        cp_threshold: Probability mass threshold on short run lengths
            to declare a changepoint. Default 0.5.
        prior_mean: Prior mean for the NIG prior. If None, estimated from fit().
        prior_var: Prior variance scale for the NIG prior. If None, estimated from fit().
        _mu0: NIG prior: prior mean.
        _kappa0: NIG prior: prior precision scale (number of pseudo-observations for mean).
        _alpha0: NIG prior: prior shape (half the pseudo-observations for variance).
        _beta0: NIG prior: prior rate (half the pseudo sum of squares).
        _fitted: Whether the detector has been fitted.

    Reference:
        Adams, R.P. & MacKay, D.J.C. (2007).
        "Bayesian Online Changepoint Detection". arXiv:0710.3742.
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 200,
        cp_threshold: float = 0.5,
        prior_mean: float | None = None,
        prior_var: float | None = None,
    ) -> None:
        """Initialize the Bayesian Online Change-Point Detector.

        Args:
            hazard_rate: Prior probability of a changepoint at each timestep.
                Equals 1/expected_run_length. Default 1/200 (expect changepoint
                every ~200 samples). Must be in (0, 1).
            cp_threshold: Minimum cumulative probability on short run lengths
                to declare onset. When probability mass on r < short_window
                exceeds this threshold, a changepoint is declared. Default 0.5.
                Must be in (0, 1).
            prior_mean: Prior mean for observations. If None, estimated from
                fit() healthy samples.
            prior_var: Prior variance for observations. If None, estimated from
                fit() healthy samples.

        Raises:
            ValueError: If hazard_rate or cp_threshold not in (0, 1).
        """
        if not (0 < hazard_rate < 1):
            raise ValueError("hazard_rate must be in (0, 1)")
        if not (0 < cp_threshold < 1):
            raise ValueError("cp_threshold must be in (0, 1)")

        self.hazard_rate = hazard_rate
        self.cp_threshold = cp_threshold
        self.prior_mean = prior_mean
        self.prior_var = prior_var

        # NIG prior hyperparameters (set during fit)
        self._mu0: float | None = None
        self._kappa0: float = 1.0
        self._alpha0: float = 1.0
        self._beta0: float | None = None
        self._fitted = False

        # Stored for baseline reporting
        self._healthy_mean: float | None = None
        self._healthy_std: float | None = None
        self._n_samples: int = 0

    def fit(self, healthy_samples: np.ndarray) -> "BayesianOnsetDetector":
        """Learn NIG prior hyperparameters from healthy samples.

        Sets the prior mean to the sample mean and the prior variance scale
        based on the sample variance, so the model expects observations
        similar to the healthy baseline.

        Args:
            healthy_samples: Array of health indicator values from healthy period.
                Should be at least 2 samples.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If healthy_samples has fewer than 2 valid samples.
        """
        healthy_samples = np.asarray(healthy_samples)
        if healthy_samples.size < 2:
            raise ValueError("Need at least 2 samples to compute baseline statistics")

        valid_samples = healthy_samples[~np.isnan(healthy_samples)]
        if valid_samples.size < 2:
            raise ValueError("Need at least 2 non-NaN samples for baseline")

        sample_mean = float(np.mean(valid_samples))
        sample_var = float(np.var(valid_samples, ddof=1))

        self._healthy_mean = sample_mean
        self._healthy_std = float(np.sqrt(sample_var)) if sample_var > 0 else 1e-10
        self._n_samples = int(valid_samples.size)

        # Set NIG prior hyperparameters
        # mu0: prior mean (use sample mean or user-provided)
        self._mu0 = self.prior_mean if self.prior_mean is not None else sample_mean

        # kappa0: prior precision scale (number of pseudo-observations for mean)
        # Small value = weak prior, lets data dominate quickly
        self._kappa0 = 1.0

        # alpha0: prior shape for variance (half pseudo-observations)
        # alpha0 = 1 gives a weakly informative prior
        self._alpha0 = 1.0

        # beta0: prior rate for variance
        # Set from sample variance or user-provided prior_var
        prior_variance = self.prior_var if self.prior_var is not None else sample_var
        if prior_variance < 1e-10:
            prior_variance = 1e-10
        # beta0 = alpha0 * prior_variance (so that E[sigma^2] = beta0/alpha0 = prior_var)
        self._beta0 = self._alpha0 * prior_variance

        self._fitted = True
        return self

    def detect(self, hi_series: np.ndarray) -> OnsetResult:
        """Detect onset using Bayesian Online Change-Point Detection.

        Runs the BOCPD algorithm over the entire series, computing the
        posterior run-length distribution at each timestep. Detects onset
        when the cumulative probability on short run lengths (indicating
        a recent changepoint) exceeds cp_threshold.

        Under constant hazard, P(r_t=0) is always equal to the hazard rate.
        Instead, we detect changepoints by monitoring the cumulative mass
        on short run lengths: sum P(r_t < short_window). When this exceeds
        cp_threshold, the model believes a changepoint recently occurred.

        Args:
            hi_series: Full health indicator time series for a bearing.

        Returns:
            OnsetResult with:
            - onset_idx: First changepoint index, or None
            - onset_time: Same as onset_idx
            - confidence: Posterior probability mass on short runs at onset
            - healthy_baseline: Dict with prior params and posterior cp_probs

        Raises:
            RuntimeError: If detector has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        hi_series = np.asarray(hi_series, dtype=np.float64)
        n = len(hi_series)

        if n == 0:
            return OnsetResult(
                onset_idx=None,
                onset_time=None,
                confidence=0.0,
                healthy_baseline=self._build_baseline(None),
            )

        # Constant hazard function: H(r) = hazard_rate for all r
        H = self.hazard_rate

        # Window for "short" run lengths — if most probability mass is on
        # run lengths < short_window, a changepoint likely occurred recently
        short_window = max(3, int(1 / H * 0.1))  # 10% of expected run length

        # Initialize: at t=0 before seeing data, P(r_0=0) = 1
        msg = np.array([1.0])

        # NIG sufficient statistics arrays (one entry per possible run length)
        mu = np.array([self._mu0])
        kappa = np.array([self._kappa0])
        alpha = np.array([self._alpha0])
        beta = np.array([self._beta0])

        # Store posterior probability of recent changepoint at each timestep
        # cp_probs[t] = sum P(r_t < short_window)
        cp_probs = np.zeros(n)

        for t in range(n):
            x = hi_series[t]

            # Skip NaN: carry forward unchanged
            if np.isnan(x):
                cp_probs[t] = cp_probs[t - 1] if t > 0 else 0.0
                continue

            # 1. Evaluate predictive probability P(x_t | r_{t-1})
            # Under NIG, the predictive is Student-t
            pred_probs = self._student_t_pdf(x, mu, kappa, alpha, beta)

            # 2. Compute growth probabilities
            growth = pred_probs * (1.0 - H) * msg

            # 3. Compute changepoint probability (mass flowing to r=0)
            cp = np.sum(pred_probs * H * msg)

            # 4. Assemble new run-length distribution: [cp, growth_0, growth_1, ...]
            new_msg = np.empty(len(growth) + 1)
            new_msg[0] = cp
            new_msg[1:] = growth

            # 5. Normalize
            evidence = np.sum(new_msg)
            if evidence > 0:
                new_msg /= evidence

            # 6. Compute cumulative probability on short run lengths
            sw = min(short_window, len(new_msg))
            cp_probs[t] = float(np.sum(new_msg[:sw]))

            # 7. Update NIG sufficient statistics
            new_mu = np.empty(len(mu) + 1)
            new_kappa = np.empty(len(kappa) + 1)
            new_alpha = np.empty(len(alpha) + 1)
            new_beta = np.empty(len(beta) + 1)

            # Run length 0: reset to prior
            new_mu[0] = self._mu0
            new_kappa[0] = self._kappa0
            new_alpha[0] = self._alpha0
            new_beta[0] = self._beta0

            # Run lengths 1, 2, ...: update from previous
            new_kappa[1:] = kappa + 1
            new_mu[1:] = (kappa * mu + x) / new_kappa[1:]
            new_alpha[1:] = alpha + 0.5
            new_beta[1:] = beta + 0.5 * kappa * (x - mu) ** 2 / new_kappa[1:]

            # Pruning: keep only run lengths up to n (can't exceed series length)
            max_rl = min(len(new_msg), n)
            if len(new_msg) > max_rl:
                new_msg = new_msg[:max_rl]
                new_mu = new_mu[:max_rl]
                new_kappa = new_kappa[:max_rl]
                new_alpha = new_alpha[:max_rl]
                new_beta = new_beta[:max_rl]
                total = np.sum(new_msg)
                if total > 0:
                    new_msg /= total

            msg = new_msg
            mu = new_mu
            kappa = new_kappa
            alpha = new_alpha
            beta = new_beta

        # Detect onset: first timestep where short-run-length mass exceeds threshold
        # Must skip the initial transient where all run lengths are naturally short
        # (the model needs time to establish a stable run before detecting a break)
        onset_idx = None
        max_cp_prob = 0.0
        established = False  # Whether model has established a stable run

        # The model is "established" once cp_probs drops below threshold
        # (meaning most mass is on longer run lengths, i.e. stable regime)
        skip = max(short_window + 1, int(0.05 * n))

        for t in range(skip, n):
            if not established and cp_probs[t] < self.cp_threshold:
                established = True

            if cp_probs[t] > max_cp_prob:
                max_cp_prob = cp_probs[t]

            if established and onset_idx is None and cp_probs[t] > self.cp_threshold:
                onset_idx = t

        # Build result
        healthy_baseline = self._build_baseline(cp_probs)

        if onset_idx is None:
            return OnsetResult(
                onset_idx=None,
                onset_time=None,
                confidence=float(max_cp_prob),
                healthy_baseline=healthy_baseline,
            )

        confidence = float(cp_probs[onset_idx])

        return OnsetResult(
            onset_idx=onset_idx,
            onset_time=float(onset_idx),
            confidence=confidence,
            healthy_baseline=healthy_baseline,
        )

    def _student_t_pdf(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Compute Student-t predictive probability under NIG posterior.

        The predictive distribution for a new observation x given NIG
        parameters (mu, kappa, alpha, beta) is:
            x ~ t_{2*alpha}(mu, beta*(kappa+1)/(alpha*kappa))

        Args:
            x: New observation.
            mu: Array of posterior means (one per run length).
            kappa: Array of posterior precision scales.
            alpha: Array of posterior shapes.
            beta: Array of posterior rates.

        Returns:
            Array of predictive probabilities P(x | params) for each run length.
        """
        # Student-t parameters
        df = 2 * alpha  # degrees of freedom
        scale_sq = beta * (kappa + 1) / (alpha * kappa)  # variance scale

        # Guard against numerical issues
        scale_sq = np.maximum(scale_sq, 1e-30)

        # Student-t pdf: Γ((ν+1)/2) / (Γ(ν/2) √(νπσ²)) × (1 + (x-μ)²/(νσ²))^(-(ν+1)/2)
        # Use log-space for numerical stability
        z = (x - mu) ** 2 / (df * scale_sq)
        log_pdf = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi * scale_sq)
            - ((df + 1) / 2) * np.log1p(z)
        )

        return np.exp(log_pdf)

    def _build_baseline(self, cp_probs: np.ndarray | None) -> dict:
        """Build healthy_baseline dictionary for OnsetResult.

        Args:
            cp_probs: Array of changepoint probabilities, or None.

        Returns:
            Dictionary with prior parameters and optionally cp_probs summary.
        """
        baseline: dict = {
            "mean": self._healthy_mean,
            "std": self._healthy_std,
            "n_samples": self._n_samples,
            "hazard_rate": self.hazard_rate,
            "cp_threshold": self.cp_threshold,
            "prior_mu0": self._mu0,
            "prior_kappa0": self._kappa0,
            "prior_alpha0": self._alpha0,
            "prior_beta0": self._beta0,
        }
        if cp_probs is not None:
            baseline["cp_probs"] = cp_probs
        return baseline


class EnsembleOnsetDetector(BaseOnsetDetector):
    """Ensemble detector combining multiple onset detectors with voting.

    Combines predictions from multiple onset detectors using configurable
    voting strategies. This provides more robust onset detection than any
    single detector, especially when detectors have complementary strengths.

    Voting Strategies:
    - "majority": Onset at index where >50% of detectors agree (within tolerance).
        Good balance between sensitivity and false positive reduction.
    - "unanimous": Onset only where ALL detectors agree (within tolerance).
        Most conservative, lowest false positive rate, may miss some onsets.
    - "earliest": Onset at the earliest detected onset across all detectors.
        Most sensitive, highest recall, but also highest false positive rate.
    - "weighted": Onset determined by confidence-weighted voting.
        Uses individual detector confidences to weight the votes.

    Attributes:
        detectors: List of onset detectors to combine.
        voting: Voting strategy ("majority", "unanimous", "earliest", "weighted").
        tolerance: Index tolerance for considering two onsets as "agreeing".
        _fitted: Whether all detectors have been fitted.

    Example:
        >>> threshold_det = ThresholdOnsetDetector(threshold_sigma=3.0)
        >>> cusum_det = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        >>> ensemble = EnsembleOnsetDetector(
        ...     detectors=[threshold_det, cusum_det],
        ...     voting="majority"
        ... )
        >>> result = ensemble.fit_detect(hi_series)
    """

    VALID_VOTING_STRATEGIES = ("majority", "unanimous", "earliest", "weighted")

    def __init__(
        self,
        detectors: list[BaseOnsetDetector],
        voting: str = "majority",
        tolerance: int = 5,
    ) -> None:
        """Initialize the ensemble onset detector.

        Args:
            detectors: List of onset detectors to combine. Must contain at
                least 2 detectors. All detectors must inherit from BaseOnsetDetector.
            voting: Voting strategy to use:
                - "majority": >50% agreement required (default)
                - "unanimous": All detectors must agree
                - "earliest": Use first detected onset
                - "weighted": Confidence-weighted voting
            tolerance: Maximum index difference to consider two onsets as
                "agreeing" for majority/unanimous voting. Default 5 samples.

        Raises:
            ValueError: If fewer than 2 detectors provided, invalid voting
                strategy, or tolerance is negative.
            TypeError: If any detector doesn't inherit from BaseOnsetDetector.
        """
        if len(detectors) < 2:
            raise ValueError("Ensemble requires at least 2 detectors")
        if voting not in self.VALID_VOTING_STRATEGIES:
            raise ValueError(
                f"voting must be one of {self.VALID_VOTING_STRATEGIES}, got '{voting}'"
            )
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        for i, det in enumerate(detectors):
            if not isinstance(det, BaseOnsetDetector):
                raise TypeError(
                    f"Detector at index {i} must inherit from BaseOnsetDetector, "
                    f"got {type(det).__name__}"
                )

        self.detectors = list(detectors)  # Copy to avoid external mutation
        self.voting = voting
        self.tolerance = tolerance
        self._fitted = False

    def add_detector(self, detector: BaseOnsetDetector) -> "EnsembleOnsetDetector":
        """Add a detector to the ensemble.

        Args:
            detector: Onset detector to add. Must inherit from BaseOnsetDetector.

        Returns:
            Self, for method chaining.

        Raises:
            TypeError: If detector doesn't inherit from BaseOnsetDetector.
        """
        if not isinstance(detector, BaseOnsetDetector):
            raise TypeError(
                f"Detector must inherit from BaseOnsetDetector, got {type(detector).__name__}"
            )
        self.detectors.append(detector)
        self._fitted = False  # New detector needs fitting
        return self

    def remove_detector(self, index: int) -> BaseOnsetDetector:
        """Remove a detector from the ensemble by index.

        Args:
            index: Index of detector to remove.

        Returns:
            The removed detector.

        Raises:
            ValueError: If removal would leave fewer than 2 detectors.
            IndexError: If index is out of range.
        """
        if len(self.detectors) <= 2:
            raise ValueError("Cannot remove detector: ensemble requires at least 2")
        return self.detectors.pop(index)

    def fit(self, healthy_samples: np.ndarray) -> "EnsembleOnsetDetector":
        """Fit all detectors on the healthy samples.

        Args:
            healthy_samples: Array of health indicator values from healthy period.

        Returns:
            Self, for method chaining.
        """
        healthy_samples = np.asarray(healthy_samples)
        for detector in self.detectors:
            detector.fit(healthy_samples)
        self._fitted = True
        return self

    def detect(self, hi_series: np.ndarray) -> OnsetResult:
        """Detect onset using ensemble voting.

        Runs all detectors and combines their predictions using the specified
        voting strategy.

        Args:
            hi_series: Full health indicator time series for a bearing.

        Returns:
            OnsetResult with:
            - onset_idx: Ensemble onset index based on voting strategy, or None
            - onset_time: Same as onset_idx
            - confidence: Aggregated confidence (weighted average with disagreement penalty)
            - healthy_baseline: Dict with ensemble info and individual results

        Raises:
            RuntimeError: If ensemble has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        hi_series = np.asarray(hi_series)

        # Collect individual detector results
        results: list[OnsetResult] = []
        for detector in self.detectors:
            result = detector.detect(hi_series)
            results.append(result)

        # Apply voting strategy to determine onset_idx
        if self.voting == "earliest":
            onset_idx, _ = self._vote_earliest(results)
        elif self.voting == "unanimous":
            onset_idx, _ = self._vote_unanimous(results)
        elif self.voting == "weighted":
            onset_idx, _ = self._vote_weighted(results)
        else:  # majority
            onset_idx, _ = self._vote_majority(results)

        # Compute aggregated confidence (weighted average with disagreement penalty)
        confidence, disagreement_factor = self._aggregate_confidence(results, onset_idx)

        # Build healthy_baseline with ensemble info
        healthy_baseline = {
            "voting_strategy": self.voting,
            "n_detectors": len(self.detectors),
            "tolerance": self.tolerance,
            "disagreement_factor": disagreement_factor,
            "individual_results": [
                {
                    "detector": type(d).__name__,
                    "onset_idx": r.onset_idx,
                    "confidence": r.confidence,
                }
                for d, r in zip(self.detectors, results)
            ],
        }

        if onset_idx is None:
            return OnsetResult(
                onset_idx=None,
                onset_time=None,
                confidence=confidence,
                healthy_baseline=healthy_baseline,
            )

        return OnsetResult(
            onset_idx=onset_idx,
            onset_time=float(onset_idx),
            confidence=confidence,
            healthy_baseline=healthy_baseline,
        )

    def _aggregate_confidence(
        self, results: list[OnsetResult], ensemble_onset_idx: int | None
    ) -> tuple[float, float]:
        """Compute weighted average confidence with disagreement penalty.

        Aggregates individual detector confidences into an ensemble confidence
        that reflects both the average certainty and the level of agreement.

        Args:
            results: List of OnsetResult from each detector.
            ensemble_onset_idx: The onset index determined by voting, or None.

        Returns:
            Tuple of (aggregated_confidence, disagreement_factor).
            - aggregated_confidence: Weighted average confidence, reduced by disagreement
            - disagreement_factor: 1.0 if perfect agreement, decreasing with disagreement
        """
        confidences = [r.confidence for r in results]
        onset_indices = [r.onset_idx for r in results]

        # Weighted average of all confidences
        # Weight by confidence itself (higher confidence detectors have more influence)
        total_conf = sum(confidences)
        if total_conf < 1e-10:
            return 0.0, 1.0  # No detector has confidence

        weighted_avg = sum(c * c for c in confidences) / total_conf

        # Compute disagreement factor based on onset index spread
        valid_indices = [idx for idx in onset_indices if idx is not None]

        if len(valid_indices) <= 1:
            # Not enough detections to compute disagreement
            # If no detections, disagreement is 0 (they agree on no onset)
            # If one detection, we can't measure disagreement
            if ensemble_onset_idx is None and all(idx is None for idx in onset_indices):
                # Perfect agreement: all say no onset
                disagreement_factor = 1.0
            else:
                # Only one detector found onset, others didn't - moderate disagreement
                n_detecting = len(valid_indices)
                disagreement_factor = n_detecting / len(results)
        else:
            # Multiple detections: measure spread relative to tolerance
            idx_spread = max(valid_indices) - min(valid_indices)

            # Agreement factor: 1.0 if within tolerance, decreasing if spread is larger
            if idx_spread <= self.tolerance:
                disagreement_factor = 1.0  # Within tolerance = full agreement
            else:
                # Penalize based on how much spread exceeds tolerance
                # Factor decreases as spread grows beyond tolerance
                # At 2x tolerance, factor is 0.5; at 3x, factor is 0.33, etc.
                disagreement_factor = self.tolerance / idx_spread

        # Also penalize if some detectors detected onset and others didn't
        n_detecting = sum(1 for idx in onset_indices if idx is not None)
        n_not_detecting = len(results) - n_detecting

        if n_detecting > 0 and n_not_detecting > 0:
            # Mixed: some detected, some didn't
            # Agreement is lower when split is more even
            detection_agreement = abs(n_detecting - n_not_detecting) / len(results)
            # Combine with index-based disagreement
            disagreement_factor *= (0.5 + 0.5 * detection_agreement)

        # Final confidence: weighted average scaled by disagreement factor
        aggregated_confidence = weighted_avg * disagreement_factor

        return aggregated_confidence, disagreement_factor

    def _vote_earliest(self, results: list[OnsetResult]) -> tuple[int | None, float]:
        """Return the earliest detected onset across all detectors.

        Args:
            results: List of OnsetResult from each detector.

        Returns:
            Tuple of (onset_idx, confidence). Confidence is the confidence of
            the detector that detected the earliest onset.
        """
        earliest_idx = None
        earliest_confidence = 0.0

        for result in results:
            if result.onset_idx is not None:
                if earliest_idx is None or result.onset_idx < earliest_idx:
                    earliest_idx = result.onset_idx
                    earliest_confidence = result.confidence

        return earliest_idx, earliest_confidence

    def _vote_unanimous(self, results: list[OnsetResult]) -> tuple[int | None, float]:
        """Return onset only if ALL detectors agree within tolerance.

        Args:
            results: List of OnsetResult from each detector.

        Returns:
            Tuple of (onset_idx, confidence). Returns None if any detector
            didn't detect onset or if disagreement exceeds tolerance.
            Confidence is the minimum of all detector confidences.
        """
        onset_indices = [r.onset_idx for r in results if r.onset_idx is not None]

        # All detectors must detect an onset
        if len(onset_indices) != len(results):
            return None, 0.0

        # Check if all indices are within tolerance of each other
        min_idx = min(onset_indices)
        max_idx = max(onset_indices)

        if max_idx - min_idx > self.tolerance:
            # Disagreement too large - reduce confidence significantly
            avg_conf = np.mean([r.confidence for r in results])
            return None, avg_conf * 0.5  # Low confidence due to disagreement

        # Unanimous agreement - return median index
        median_idx = int(np.median(onset_indices))
        min_confidence = min(r.confidence for r in results)

        return median_idx, min_confidence

    def _vote_majority(self, results: list[OnsetResult]) -> tuple[int | None, float]:
        """Return onset if >50% of detectors agree within tolerance.

        Uses a clustering approach: finds groups of detectors that agree
        (within tolerance) and returns the consensus of the largest group.

        Args:
            results: List of OnsetResult from each detector.

        Returns:
            Tuple of (onset_idx, confidence). Returns None if no majority.
            Confidence is the average of agreeing detectors' confidences.
        """
        onset_indices = [(i, r.onset_idx, r.confidence) for i, r in enumerate(results)]
        valid_onsets = [(i, idx, conf) for i, idx, conf in onset_indices if idx is not None]

        # If half or fewer detected anything, no majority (>50% required)
        majority_threshold = len(results) / 2
        if len(valid_onsets) <= majority_threshold:
            return None, 0.0

        # Cluster valid onsets by tolerance
        # Simple greedy clustering: assign each to nearest existing cluster
        clusters: list[list[tuple[int, int, float]]] = []

        for det_i, idx, conf in valid_onsets:
            assigned = False
            for cluster in clusters:
                # Check if this onset is within tolerance of cluster center
                cluster_indices = [c[1] for c in cluster]
                cluster_center = int(np.median(cluster_indices))
                if abs(idx - cluster_center) <= self.tolerance:
                    cluster.append((det_i, idx, conf))
                    assigned = True
                    break
            if not assigned:
                clusters.append([(det_i, idx, conf)])

        # Find largest cluster
        largest_cluster = max(clusters, key=len)

        # Check if largest cluster has majority (>50%, strictly greater)
        if len(largest_cluster) <= majority_threshold:
            # No majority - return low confidence
            return None, np.mean([r.confidence for r in results]) * 0.3

        # Return median of largest cluster
        cluster_indices = [c[1] for c in largest_cluster]
        cluster_confs = [c[2] for c in largest_cluster]
        median_idx = int(np.median(cluster_indices))
        avg_confidence = float(np.mean(cluster_confs))

        return median_idx, avg_confidence

    def _vote_weighted(self, results: list[OnsetResult]) -> tuple[int | None, float]:
        """Return confidence-weighted average onset index.

        Detectors with higher confidence have more influence on the final
        onset index. If total weight is too low, returns None.

        Args:
            results: List of OnsetResult from each detector.

        Returns:
            Tuple of (onset_idx, confidence). Confidence is the normalized
            sum of weights for detectors that detected onset.
        """
        valid_onsets = [(r.onset_idx, r.confidence) for r in results if r.onset_idx is not None]

        if not valid_onsets:
            return None, 0.0

        total_confidence = sum(r.confidence for r in results)
        if total_confidence < 1e-10:
            return None, 0.0

        # Weighted average of onset indices
        weighted_sum = sum(idx * conf for idx, conf in valid_onsets)
        valid_weight = sum(conf for _, conf in valid_onsets)

        if valid_weight < 1e-10:
            return None, 0.0

        weighted_idx = int(round(weighted_sum / valid_weight))

        # Confidence: proportion of total confidence from detecting detectors
        confidence = valid_weight / total_confidence

        return weighted_idx, confidence
