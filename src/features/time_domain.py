"""Time-Domain Feature Extraction for Bearing Vibration Signals.

This module implements 18 time-domain features per channel plus cross-channel
correlation for the XJTU-SY bearing dataset.

Features extracted per channel (18 total):
    Statistical: mean, std, variance (3)
    Amplitude: RMS, peak, peak-to-peak (3)
    Shape factors: crest factor, shape factor, impulse factor, clearance factor (4)
    Distribution: kurtosis, skewness (2)
    Other: line integral, zero crossing rate, Shannon entropy (3)
    Percentiles: 5th, 50th (median), 95th (3)

Total: 18 features × 2 channels + 1 cross-correlation = 37 features
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Feature names for reference
CHANNEL_FEATURES = [
    "mean",
    "std",
    "variance",
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "kurtosis",
    "skewness",
    "line_integral",
    "zero_crossing_rate",
    "entropy",
    "percentile_5",
    "percentile_50",
    "percentile_95",
]

NUM_FEATURES_PER_CHANNEL = len(CHANNEL_FEATURES)


def _safe_divide(numerator: NDArray, denominator: NDArray, fill: float = 0.0) -> NDArray:
    """Safely divide arrays, replacing inf/nan with fill value."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        result = np.where(np.isfinite(result), result, fill)
    return result


def extract_mean(signal: NDArray) -> NDArray:
    """Extract mean value."""
    return np.mean(signal, axis=-1)


def extract_std(signal: NDArray) -> NDArray:
    """Extract standard deviation."""
    return np.std(signal, axis=-1)


def extract_variance(signal: NDArray) -> NDArray:
    """Extract variance."""
    return np.var(signal, axis=-1)


def extract_rms(signal: NDArray) -> NDArray:
    """Extract Root Mean Square (RMS)."""
    return np.sqrt(np.mean(signal ** 2, axis=-1))


def extract_peak(signal: NDArray) -> NDArray:
    """Extract peak value (maximum absolute value)."""
    return np.max(np.abs(signal), axis=-1)


def extract_peak_to_peak(signal: NDArray) -> NDArray:
    """Extract peak-to-peak amplitude."""
    return np.max(signal, axis=-1) - np.min(signal, axis=-1)


def extract_crest_factor(signal: NDArray) -> NDArray:
    """Extract crest factor (peak / RMS)."""
    peak = extract_peak(signal)
    rms = extract_rms(signal)
    return _safe_divide(peak, rms)


def extract_shape_factor(signal: NDArray) -> NDArray:
    """Extract shape factor (RMS / mean absolute value)."""
    rms = extract_rms(signal)
    mean_abs = np.mean(np.abs(signal), axis=-1)
    return _safe_divide(rms, mean_abs)


def extract_impulse_factor(signal: NDArray) -> NDArray:
    """Extract impulse factor (peak / mean absolute value)."""
    peak = extract_peak(signal)
    mean_abs = np.mean(np.abs(signal), axis=-1)
    return _safe_divide(peak, mean_abs)


def extract_clearance_factor(signal: NDArray) -> NDArray:
    """Extract clearance factor (peak / mean square root of absolute value)."""
    peak = extract_peak(signal)
    mean_sqrt = np.mean(np.sqrt(np.abs(signal)), axis=-1) ** 2
    return _safe_divide(peak, mean_sqrt)


def extract_kurtosis(signal: NDArray) -> NDArray:
    """Extract kurtosis (excess kurtosis, Fisher's definition)."""
    return stats.kurtosis(signal, axis=-1, fisher=True)


def extract_skewness(signal: NDArray) -> NDArray:
    """Extract skewness."""
    return stats.skew(signal, axis=-1)


def extract_line_integral(signal: NDArray) -> NDArray:
    """Extract line integral (sum of absolute differences)."""
    return np.sum(np.abs(np.diff(signal, axis=-1)), axis=-1)


def extract_zero_crossing_rate(signal: NDArray) -> NDArray:
    """Extract zero crossing rate."""
    # Count sign changes
    n_samples = signal.shape[-1]
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal), axis=-1)) > 0, axis=-1)
    return zero_crossings / (n_samples - 1)


def extract_entropy(signal: NDArray) -> NDArray:
    """Extract Shannon entropy of the signal amplitude distribution.

    Uses histogram-based probability estimation with 100 bins.
    """
    def _entropy_1d(x: NDArray) -> float:
        hist, _ = np.histogram(x, bins=100, density=True)
        hist = hist[hist > 0]  # Remove zeros to avoid log(0)
        if len(hist) == 0:
            return 0.0
        # Normalize to probability
        hist = hist / hist.sum()
        return float(-np.sum(hist * np.log2(hist + 1e-12)))

    if signal.ndim == 1:
        return np.array(_entropy_1d(signal))
    else:
        # Batch processing
        return np.array([_entropy_1d(s) for s in signal])


def extract_percentiles(signal: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Extract 5th, 50th (median), and 95th percentiles."""
    p5 = np.percentile(signal, 5, axis=-1)
    p50 = np.percentile(signal, 50, axis=-1)
    p95 = np.percentile(signal, 95, axis=-1)
    return p5, p50, p95


def extract_cross_correlation(horizontal: NDArray, vertical: NDArray) -> NDArray:
    """Extract Pearson correlation coefficient between channels.

    Args:
        horizontal: Horizontal channel signal(s), shape (samples,) or (batch, samples).
        vertical: Vertical channel signal(s), same shape as horizontal.

    Returns:
        Correlation coefficient(s), scalar or shape (batch,).
    """
    def _corr_1d(h: NDArray, v: NDArray) -> float:
        return float(np.corrcoef(h, v)[0, 1])

    if horizontal.ndim == 1:
        return np.array(_corr_1d(horizontal, vertical))
    else:
        return np.array([_corr_1d(h, v) for h, v in zip(horizontal, vertical)])


def extract_channel_features(signal: NDArray) -> NDArray:
    """Extract all 17 time-domain features for a single channel.

    Args:
        signal: Signal array of shape (samples,) or (batch, samples).

    Returns:
        Feature array of shape (17,) or (batch, 17).
    """
    is_single = signal.ndim == 1
    if is_single:
        signal = signal.reshape(1, -1)

    # Extract all features
    mean = extract_mean(signal)
    std = extract_std(signal)
    variance = extract_variance(signal)
    rms = extract_rms(signal)
    peak = extract_peak(signal)
    peak_to_peak = extract_peak_to_peak(signal)
    crest_factor = extract_crest_factor(signal)
    shape_factor = extract_shape_factor(signal)
    impulse_factor = extract_impulse_factor(signal)
    clearance_factor = extract_clearance_factor(signal)
    kurtosis = extract_kurtosis(signal)
    skewness = extract_skewness(signal)
    line_integral = extract_line_integral(signal)
    zcr = extract_zero_crossing_rate(signal)
    entropy = extract_entropy(signal)
    p5, p50, p95 = extract_percentiles(signal)

    # Stack features
    features = np.stack([
        mean, std, variance, rms, peak, peak_to_peak,
        crest_factor, shape_factor, impulse_factor, clearance_factor,
        kurtosis, skewness, line_integral, zcr, entropy,
        p5, p50, p95
    ], axis=-1)

    if is_single:
        features = features.squeeze(0)

    return features.astype(np.float32)


def extract_features(signal: NDArray) -> NDArray:
    """Extract all time-domain features from a dual-channel signal.

    Extracts 17 features per channel (horizontal, vertical) plus
    cross-channel correlation for a total of 35 features.

    Args:
        signal: Signal array of shape (samples, 2) or (batch, samples, 2).
                Channel 0 = horizontal, Channel 1 = vertical.

    Returns:
        Feature array of shape (35,) or (batch, 35).
        Feature order: horizontal features (17), vertical features (17), correlation (1).
    """
    is_single = signal.ndim == 2
    if is_single:
        signal = signal.reshape(1, signal.shape[0], signal.shape[1])

    # Separate channels
    horizontal = signal[:, :, 0]  # Shape: (batch, samples)
    vertical = signal[:, :, 1]    # Shape: (batch, samples)

    # Extract features for each channel
    h_features = extract_channel_features(horizontal)  # (batch, 17)
    v_features = extract_channel_features(vertical)    # (batch, 17)

    # Cross-channel correlation
    correlation = extract_cross_correlation(horizontal, vertical)  # (batch,)
    correlation = correlation.reshape(-1, 1)  # (batch, 1)

    # Concatenate all features
    all_features = np.concatenate([h_features, v_features, correlation], axis=-1)

    if is_single:
        all_features = all_features.squeeze(0)

    return all_features.astype(np.float32)


def get_feature_names() -> list[str]:
    """Get ordered list of all 35 feature names.

    Returns:
        List of feature names in the order they appear in extract_features output.
    """
    h_names = [f"h_{name}" for name in CHANNEL_FEATURES]
    v_names = [f"v_{name}" for name in CHANNEL_FEATURES]
    return h_names + v_names + ["cross_correlation"]


def extract_features_dict(signal: NDArray) -> dict[str, float]:
    """Extract features and return as a named dictionary.

    Args:
        signal: Signal array of shape (samples, 2).

    Returns:
        Dictionary mapping feature names to values.
    """
    features = extract_features(signal)
    names = get_feature_names()
    return dict(zip(names, features))
