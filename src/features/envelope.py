"""Envelope Analysis for Bearing Fault Detection.

This module implements envelope analysis techniques for bearing vibration signals,
which is a key method for detecting bearing faults by extracting amplitude
modulation caused by defects.

Envelope analysis workflow:
1. Bandpass filter around fault frequency band of interest
2. Compute envelope using Hilbert transform
3. Analyze envelope spectrum for fault signatures

Features extracted:
    - Envelope peak amplitude
    - Envelope RMS
    - Envelope crest factor
    - Envelope kurtosis
    - Characteristic frequency amplitudes in envelope spectrum (BPFO, BPFI, BSF, FTF)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

from src.features.frequency_domain import (
    BEARING_GEOMETRY,
    SAMPLING_RATE,
    calculate_bearing_frequencies,
    get_characteristic_frequencies_for_condition,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FilterType(Enum):
    """Bandpass filter types."""
    BUTTERWORTH = "butterworth"
    CHEBYSHEV1 = "chebyshev1"
    CHEBYSHEV2 = "chebyshev2"


@dataclass
class EnvelopeConfig:
    """Configuration for envelope analysis.

    Attributes:
        low_freq: Lower cutoff frequency for bandpass filter (Hz).
        high_freq: Upper cutoff frequency for bandpass filter (Hz).
        filter_order: Filter order (typically 4-8).
        filter_type: Type of bandpass filter.
        sampling_rate: Signal sampling rate (Hz).
        freq_tolerance: Tolerance for characteristic frequency peak detection (Hz).
    """
    low_freq: float = 2000.0  # High frequency band for bearing resonance
    high_freq: float = 10000.0
    filter_order: int = 5
    filter_type: FilterType = FilterType.BUTTERWORTH
    sampling_rate: float = SAMPLING_RATE
    freq_tolerance: float = 5.0  # Hz tolerance for peak detection


def get_default_config() -> EnvelopeConfig:
    """Get default envelope analysis configuration."""
    return EnvelopeConfig()


def design_bandpass_filter(
    low_freq: float,
    high_freq: float,
    sampling_rate: float = SAMPLING_RATE,
    order: int = 5,
    filter_type: FilterType = FilterType.BUTTERWORTH,
) -> tuple[NDArray, NDArray]:
    """Design a bandpass filter.

    Args:
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.
        filter_type: Type of filter.

    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Ensure frequencies are within valid range
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))

    if filter_type == FilterType.BUTTERWORTH:
        b, a = signal.butter(order, [low, high], btype='band')
    elif filter_type == FilterType.CHEBYSHEV1:
        b, a = signal.cheby1(order, 0.5, [low, high], btype='band')  # 0.5dB ripple
    elif filter_type == FilterType.CHEBYSHEV2:
        b, a = signal.cheby2(order, 40, [low, high], btype='band')  # 40dB stopband
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return b, a


def apply_bandpass_filter(
    signal_data: NDArray,
    low_freq: float,
    high_freq: float,
    sampling_rate: float = SAMPLING_RATE,
    order: int = 5,
    filter_type: FilterType = FilterType.BUTTERWORTH,
) -> NDArray:
    """Apply bandpass filter to signal.

    Args:
        signal_data: Input signal, shape (samples,) or (batch, samples).
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.
        filter_type: Type of filter.

    Returns:
        Filtered signal with same shape as input.
    """
    b, a = design_bandpass_filter(
        low_freq, high_freq, sampling_rate, order, filter_type
    )

    is_single = signal_data.ndim == 1
    if is_single:
        signal_data = signal_data.reshape(1, -1)

    # Use filtfilt for zero-phase filtering
    filtered = np.zeros_like(signal_data)
    for i in range(signal_data.shape[0]):
        filtered[i] = signal.filtfilt(b, a, signal_data[i])

    if is_single:
        filtered = filtered.squeeze(0)

    return filtered


def compute_envelope(signal_data: NDArray) -> NDArray:
    """Compute signal envelope using Hilbert transform.

    The envelope is the magnitude of the analytic signal.

    Args:
        signal_data: Input signal, shape (samples,) or (batch, samples).

    Returns:
        Envelope signal with same shape as input.
    """
    is_single = signal_data.ndim == 1
    if is_single:
        signal_data = signal_data.reshape(1, -1)

    envelopes = []
    for sig in signal_data:
        # Compute analytic signal using Hilbert transform
        analytic = signal.hilbert(sig)
        # Envelope is the magnitude of the analytic signal
        env = np.abs(analytic)
        envelopes.append(env)

    result = np.stack(envelopes)

    if is_single:
        result = result.squeeze(0)

    return result


def compute_envelope_spectrum(
    envelope: NDArray,
    sampling_rate: float = SAMPLING_RATE,
) -> tuple[NDArray, NDArray]:
    """Compute frequency spectrum of envelope signal.

    Args:
        envelope: Envelope signal, shape (samples,) or (batch, samples).
        sampling_rate: Sampling rate in Hz.

    Returns:
        Tuple of (frequencies, magnitudes).
    """
    is_single = envelope.ndim == 1
    if is_single:
        envelope = envelope.reshape(1, -1)

    n_samples = envelope.shape[-1]

    # Remove DC component (mean)
    envelope_centered = envelope - envelope.mean(axis=-1, keepdims=True)

    # Compute FFT
    fft_result = rfft(envelope_centered, axis=-1)
    magnitudes = np.abs(fft_result) * 2 / n_samples

    # Frequency bins
    freqs = rfftfreq(n_samples, d=1.0 / sampling_rate)

    if is_single:
        magnitudes = magnitudes.squeeze(0)

    return freqs, magnitudes


def extract_envelope_statistics(envelope: NDArray) -> dict[str, NDArray]:
    """Extract statistical features from envelope signal.

    Args:
        envelope: Envelope signal, shape (samples,) or (batch, samples).

    Returns:
        Dictionary of envelope statistics.
    """
    is_single = envelope.ndim == 1
    if is_single:
        envelope = envelope.reshape(1, -1)

    # Peak amplitude
    peak = np.max(envelope, axis=-1)

    # RMS
    rms = np.sqrt(np.mean(envelope ** 2, axis=-1))

    # Crest factor (peak / RMS)
    crest_factor = np.where(rms > 0, peak / rms, 0.0)

    # Kurtosis
    mean = np.mean(envelope, axis=-1, keepdims=True)
    std = np.std(envelope, axis=-1, keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    normalized = (envelope - mean) / std
    kurtosis = np.mean(normalized ** 4, axis=-1) - 3  # Excess kurtosis

    # Mean envelope level
    mean_level = np.mean(envelope, axis=-1)

    # Variance
    variance = np.var(envelope, axis=-1)

    if is_single:
        peak = peak.item()
        rms = rms.item()
        crest_factor = crest_factor.item()
        kurtosis = kurtosis.item()
        mean_level = mean_level.item()
        variance = variance.item()

    return {
        "env_peak": peak,
        "env_rms": rms,
        "env_crest_factor": crest_factor,
        "env_kurtosis": kurtosis,
        "env_mean": mean_level,
        "env_variance": variance,
    }


def find_peak_at_frequency(
    freqs: NDArray,
    magnitudes: NDArray,
    target_freq: float,
    tolerance: float = 5.0,
) -> tuple[float, float]:
    """Find peak amplitude near a target frequency.

    Args:
        freqs: Frequency bins.
        magnitudes: Spectrum magnitudes.
        target_freq: Target frequency to search around.
        tolerance: Frequency tolerance in Hz.

    Returns:
        Tuple of (peak_frequency, peak_amplitude).
    """
    # Find indices within tolerance
    mask = np.abs(freqs - target_freq) <= tolerance

    if not np.any(mask):
        return target_freq, 0.0

    masked_mags = magnitudes.copy()
    masked_mags[~mask] = 0

    peak_idx = np.argmax(masked_mags)
    return freqs[peak_idx], magnitudes[peak_idx]


def extract_characteristic_freq_amplitudes(
    freqs: NDArray,
    magnitudes: NDArray,
    characteristic_freqs: dict[str, float],
    tolerance: float = 5.0,
    n_harmonics: int = 2,
) -> dict[str, float]:
    """Extract amplitudes at bearing characteristic frequencies from envelope spectrum.

    Args:
        freqs: Frequency bins of envelope spectrum.
        magnitudes: Envelope spectrum magnitudes.
        characteristic_freqs: Dict with BPFO, BPFI, BSF, FTF frequencies.
        tolerance: Frequency tolerance for peak detection (Hz).
        n_harmonics: Number of harmonics to include (1 = fundamental only).

    Returns:
        Dictionary with amplitudes at characteristic frequencies.
    """
    result = {}

    for name, fund_freq in characteristic_freqs.items():
        # Fundamental frequency amplitude
        _, amp = find_peak_at_frequency(freqs, magnitudes, fund_freq, tolerance)
        result[f"env_{name}_1x"] = amp

        # Harmonics (2x, 3x, etc.)
        for h in range(2, n_harmonics + 1):
            harmonic_freq = fund_freq * h
            if harmonic_freq < freqs.max():
                _, harm_amp = find_peak_at_frequency(
                    freqs, magnitudes, harmonic_freq, tolerance
                )
                result[f"env_{name}_{h}x"] = harm_amp
            else:
                result[f"env_{name}_{h}x"] = 0.0

        # Total energy (sum of harmonics)
        total = sum(result[f"env_{name}_{h}x"] for h in range(1, n_harmonics + 1))
        result[f"env_{name}_total"] = total

    return result


def detect_fault_signatures(
    envelope_spectrum_amplitudes: dict[str, float],
    threshold_factor: float = 3.0,
    noise_floor: float | None = None,
) -> dict[str, dict]:
    """Detect bearing fault signatures from envelope spectrum features.

    Fault detection is based on comparing characteristic frequency amplitudes
    to the noise floor or a threshold.

    Args:
        envelope_spectrum_amplitudes: Dict from extract_characteristic_freq_amplitudes.
        threshold_factor: Factor above noise floor to consider as fault.
        noise_floor: Noise floor amplitude. If None, estimated from minimum.

    Returns:
        Dictionary with fault detection results for each defect type.
    """
    # Extract total amplitudes for each characteristic frequency
    totals = {
        "outer_race": envelope_spectrum_amplitudes.get("env_bpfo_total", 0),
        "inner_race": envelope_spectrum_amplitudes.get("env_bpfi_total", 0),
        "rolling_element": envelope_spectrum_amplitudes.get("env_bsf_total", 0),
        "cage": envelope_spectrum_amplitudes.get("env_ftf_total", 0),
    }

    # Estimate noise floor from minimum if not provided
    if noise_floor is None:
        # Use the minimum of 1x amplitudes as noise estimate
        one_x_amps = [
            envelope_spectrum_amplitudes.get("env_bpfo_1x", 0),
            envelope_spectrum_amplitudes.get("env_bpfi_1x", 0),
            envelope_spectrum_amplitudes.get("env_bsf_1x", 0),
            envelope_spectrum_amplitudes.get("env_ftf_1x", 0),
        ]
        noise_floor = min(one_x_amps) if one_x_amps else 1e-6

    noise_floor = max(noise_floor, 1e-10)  # Avoid division by zero

    result = {}
    for fault_type, amplitude in totals.items():
        snr = amplitude / noise_floor if noise_floor > 0 else 0
        detected = snr > threshold_factor
        result[fault_type] = {
            "amplitude": amplitude,
            "snr": snr,
            "detected": detected,
            "confidence": min(snr / threshold_factor, 1.0) if threshold_factor > 0 else 0,
        }

    return result


def extract_envelope_features(
    signal_data: NDArray,
    config: EnvelopeConfig | None = None,
    condition: str | None = None,
) -> NDArray:
    """Extract all envelope analysis features from a single channel.

    Features extracted (21 total):
        - 6 envelope statistics (peak, rms, crest factor, kurtosis, mean, variance)
        - 12 characteristic frequency amplitudes (BPFO/BPFI/BSF/FTF × 1x, 2x, total)
        - 3 additional: dominant envelope frequency, envelope spectral centroid, flatness

    Args:
        signal_data: Input signal, shape (samples,) or (batch, samples).
        config: Envelope analysis configuration.
        condition: Operating condition for characteristic frequencies.

    Returns:
        Feature array of shape (21,) or (batch, 21).
    """
    if config is None:
        config = get_default_config()

    is_single = signal_data.ndim == 1
    if is_single:
        signal_data = signal_data.reshape(1, -1)

    batch_size = signal_data.shape[0]

    # Get characteristic frequencies
    char_freqs = None
    if condition is not None:
        char_freqs = get_characteristic_frequencies_for_condition(condition)
    else:
        # Use default 35Hz shaft frequency
        char_freqs = calculate_bearing_frequencies(35.0, **BEARING_GEOMETRY)

    all_features = []

    for sig in signal_data:
        # 1. Bandpass filter
        filtered = apply_bandpass_filter(
            sig,
            config.low_freq,
            config.high_freq,
            config.sampling_rate,
            config.filter_order,
            config.filter_type,
        )

        # 2. Compute envelope
        envelope = compute_envelope(filtered)

        # 3. Extract envelope statistics
        stats = extract_envelope_statistics(envelope)

        # 4. Compute envelope spectrum
        freqs, magnitudes = compute_envelope_spectrum(envelope, config.sampling_rate)

        # 5. Extract characteristic frequency amplitudes
        char_amps = extract_characteristic_freq_amplitudes(
            freqs, magnitudes, char_freqs, config.freq_tolerance, n_harmonics=2
        )

        # 6. Additional envelope spectrum features
        # Dominant frequency in envelope spectrum
        if magnitudes.size > 0:
            # Skip DC component
            valid_idx = freqs > 1.0
            if np.any(valid_idx):
                dom_idx = np.argmax(magnitudes[valid_idx])
                dom_freq = freqs[valid_idx][dom_idx]
            else:
                dom_freq = 0.0
        else:
            dom_freq = 0.0

        # Spectral centroid of envelope spectrum
        mag_sum = magnitudes.sum()
        if mag_sum > 0:
            centroid = np.sum(freqs * magnitudes) / mag_sum
        else:
            centroid = 0.0

        # Spectral flatness of envelope spectrum
        eps = 1e-10
        mag_nonzero = magnitudes + eps
        geo_mean = np.exp(np.mean(np.log(mag_nonzero)))
        arith_mean = np.mean(mag_nonzero)
        flatness = geo_mean / arith_mean if arith_mean > 0 else 0

        # Collect all features in order
        features = [
            # Envelope statistics (6)
            stats["env_peak"],
            stats["env_rms"],
            stats["env_crest_factor"],
            stats["env_kurtosis"],
            stats["env_mean"],
            stats["env_variance"],
            # BPFO features (3)
            char_amps["env_bpfo_1x"],
            char_amps["env_bpfo_2x"],
            char_amps["env_bpfo_total"],
            # BPFI features (3)
            char_amps["env_bpfi_1x"],
            char_amps["env_bpfi_2x"],
            char_amps["env_bpfi_total"],
            # BSF features (3)
            char_amps["env_bsf_1x"],
            char_amps["env_bsf_2x"],
            char_amps["env_bsf_total"],
            # FTF features (3)
            char_amps["env_ftf_1x"],
            char_amps["env_ftf_2x"],
            char_amps["env_ftf_total"],
            # Envelope spectrum features (3)
            dom_freq,
            centroid,
            flatness,
        ]

        all_features.append(features)

    result = np.array(all_features, dtype=np.float32)

    if is_single:
        result = result.squeeze(0)

    return result


# Feature names for single channel
ENVELOPE_FEATURE_NAMES = [
    "env_peak",
    "env_rms",
    "env_crest_factor",
    "env_kurtosis",
    "env_mean",
    "env_variance",
    "env_bpfo_1x",
    "env_bpfo_2x",
    "env_bpfo_total",
    "env_bpfi_1x",
    "env_bpfi_2x",
    "env_bpfi_total",
    "env_bsf_1x",
    "env_bsf_2x",
    "env_bsf_total",
    "env_ftf_1x",
    "env_ftf_2x",
    "env_ftf_total",
    "env_dominant_freq",
    "env_spectral_centroid",
    "env_spectral_flatness",
]

NUM_ENVELOPE_FEATURES = len(ENVELOPE_FEATURE_NAMES)


def get_envelope_feature_names(prefix: str = "") -> list[str]:
    """Get ordered list of envelope feature names.

    Args:
        prefix: Prefix for feature names (e.g., "h_" or "v_").

    Returns:
        List of feature names.
    """
    return [f"{prefix}{name}" for name in ENVELOPE_FEATURE_NAMES]


def extract_envelope_features_dual_channel(
    signal_data: NDArray,
    config: EnvelopeConfig | None = None,
    condition: str | None = None,
) -> NDArray:
    """Extract envelope features from dual-channel signal.

    Args:
        signal_data: Signal array of shape (samples, 2) or (batch, samples, 2).
            Channel 0 = horizontal, Channel 1 = vertical.
        config: Envelope analysis configuration.
        condition: Operating condition for characteristic frequencies.

    Returns:
        Feature array of shape (42,) or (batch, 42).
        21 features per channel × 2 channels.
    """
    is_single = signal_data.ndim == 2
    if is_single:
        signal_data = signal_data.reshape(1, signal_data.shape[0], signal_data.shape[1])

    # Separate channels
    horizontal = signal_data[:, :, 0]
    vertical = signal_data[:, :, 1]

    # Extract features for each channel
    h_features = extract_envelope_features(horizontal, config, condition)
    v_features = extract_envelope_features(vertical, config, condition)

    # Concatenate
    all_features = np.concatenate([h_features, v_features], axis=-1)

    if is_single:
        all_features = all_features.squeeze(0)

    return all_features


def get_dual_channel_feature_names() -> list[str]:
    """Get ordered list of all 42 envelope feature names for dual channel.

    Returns:
        List of feature names in order: h_* (21), v_* (21).
    """
    h_names = get_envelope_feature_names("h_")
    v_names = get_envelope_feature_names("v_")
    return h_names + v_names


def extract_envelope_features_dict(
    signal_data: NDArray,
    config: EnvelopeConfig | None = None,
    condition: str | None = None,
) -> dict[str, float]:
    """Extract envelope features and return as named dictionary.

    Args:
        signal_data: Signal array of shape (samples, 2).
        config: Envelope analysis configuration.
        condition: Operating condition for characteristic frequencies.

    Returns:
        Dictionary mapping feature names to values.
    """
    features = extract_envelope_features_dual_channel(signal_data, config, condition)
    names = get_dual_channel_feature_names()
    return dict(zip(names, features))


def analyze_bearing_health(
    signal_data: NDArray,
    config: EnvelopeConfig | None = None,
    condition: str | None = None,
    threshold_factor: float = 3.0,
) -> dict:
    """Comprehensive bearing health analysis using envelope analysis.

    This is a high-level function that performs full envelope analysis
    and fault detection for bearing diagnostics.

    Args:
        signal_data: Signal array of shape (samples, 2) for dual channel.
        config: Envelope analysis configuration.
        condition: Operating condition for characteristic frequencies.
        threshold_factor: SNR threshold for fault detection.

    Returns:
        Dictionary with:
        - features: All extracted envelope features
        - fault_detection: Fault detection results for each channel
        - health_score: Overall health score (0-1, 1=healthy)
    """
    if config is None:
        config = get_default_config()

    # Get characteristic frequencies
    char_freqs = None
    if condition is not None:
        char_freqs = get_characteristic_frequencies_for_condition(condition)
    else:
        char_freqs = calculate_bearing_frequencies(35.0, **BEARING_GEOMETRY)

    # Extract features
    features = extract_envelope_features_dict(signal_data, config, condition)

    # Analyze each channel for fault signatures
    fault_results = {}
    health_scores = []

    for ch, prefix in [("horizontal", "h_"), ("vertical", "v_")]:
        ch_idx = 0 if ch == "horizontal" else 1
        channel_data = signal_data[:, ch_idx]

        # Compute envelope spectrum
        filtered = apply_bandpass_filter(
            channel_data,
            config.low_freq,
            config.high_freq,
            config.sampling_rate,
            config.filter_order,
            config.filter_type,
        )
        envelope = compute_envelope(filtered)
        freqs, magnitudes = compute_envelope_spectrum(envelope, config.sampling_rate)

        # Get characteristic frequency amplitudes
        char_amps = extract_characteristic_freq_amplitudes(
            freqs, magnitudes, char_freqs, config.freq_tolerance, n_harmonics=2
        )

        # Detect faults
        faults = detect_fault_signatures(char_amps, threshold_factor)
        fault_results[ch] = faults

        # Calculate channel health score
        # Lower score = more faults detected
        confidence_sum = sum(f["confidence"] for f in faults.values())
        max_confidence = len(faults)
        ch_health = 1.0 - (confidence_sum / max_confidence) if max_confidence > 0 else 1.0
        health_scores.append(ch_health)

    # Overall health score (average of channels)
    overall_health = np.mean(health_scores)

    return {
        "features": features,
        "fault_detection": fault_results,
        "health_score": float(overall_health),
        "channel_health_scores": {
            "horizontal": health_scores[0],
            "vertical": health_scores[1],
        },
    }
