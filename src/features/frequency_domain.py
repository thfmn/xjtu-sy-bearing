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

"""Frequency-Domain Feature Extraction for Bearing Vibration Signals.

This module implements frequency-domain features for the XJTU-SY bearing dataset,
including spectral statistics and bearing characteristic frequency band powers.

Features extracted per channel:
    Spectral statistics: centroid, bandwidth, rolloff, flatness (4)
    Band powers: 0-1kHz, 1-3kHz, 3-6kHz, 6-12kHz (4)
    Frequency measures: dominant frequency, mean frequency (2)

Bearing characteristic frequency bands (4):
    - BPFO (Ball Pass Frequency Outer)
    - BPFI (Ball Pass Frequency Inner)
    - BSF (Ball Spin Frequency)
    - FTF (Fundamental Train Frequency)

Total: 10 base features × 2 channels + 4 characteristic bands × 2 channels = 28 features
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Dataset constants
SAMPLING_RATE = 25600  # 25.6 kHz
NYQUIST_FREQ = SAMPLING_RATE / 2  # 12.8 kHz

# LDK UER204 bearing geometry (from XJTU-SY dataset documentation)
# These are used to calculate characteristic frequencies
BEARING_GEOMETRY = {
    "n_balls": 8,           # Number of rolling elements
    "ball_diameter": 7.92,  # mm
    "pitch_diameter": 34.5, # mm
    "contact_angle": 0.0,   # degrees (assume 0 for deep groove ball bearing)
}

# Feature names for reference
SPECTRAL_FEATURES = [
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "band_power_0_1k",
    "band_power_1_3k",
    "band_power_3_6k",
    "band_power_6_12k",
    "dominant_frequency",
    "mean_frequency",
]

CHARACTERISTIC_FEATURES = [
    "bpfo_band_power",
    "bpfi_band_power",
    "bsf_band_power",
    "ftf_band_power",
]

NUM_SPECTRAL_FEATURES = len(SPECTRAL_FEATURES)
NUM_CHARACTERISTIC_FEATURES = len(CHARACTERISTIC_FEATURES)


def calculate_bearing_frequencies(
    shaft_freq_hz: float,
    n_balls: int = 8,
    ball_diameter: float = 7.92,
    pitch_diameter: float = 34.5,
    contact_angle: float = 0.0,
) -> dict[str, float]:
    """Calculate bearing characteristic frequencies.

    These frequencies are where defects on different bearing components
    manifest in the vibration spectrum.

    Args:
        shaft_freq_hz: Shaft rotation frequency in Hz.
        n_balls: Number of rolling elements.
        ball_diameter: Ball diameter in mm.
        pitch_diameter: Pitch diameter in mm.
        contact_angle: Contact angle in degrees.

    Returns:
        Dictionary with BPFO, BPFI, BSF, and FTF in Hz.

    Reference:
        Standard bearing defect frequency formulas from vibration analysis literature.
    """
    d = ball_diameter
    D = pitch_diameter
    phi = math.radians(contact_angle)
    n = n_balls
    f_s = shaft_freq_hz

    # Fundamental Train Frequency (cage rotation)
    ftf = (f_s / 2) * (1 - (d / D) * math.cos(phi))

    # Ball Pass Frequency Outer (outer race defect)
    bpfo = (n * f_s / 2) * (1 - (d / D) * math.cos(phi))

    # Ball Pass Frequency Inner (inner race defect)
    bpfi = (n * f_s / 2) * (1 + (d / D) * math.cos(phi))

    # Ball Spin Frequency (ball defect)
    bsf = (D * f_s / (2 * d)) * (1 - ((d / D) * math.cos(phi)) ** 2)

    return {
        "bpfo": bpfo,
        "bpfi": bpfi,
        "bsf": bsf,
        "ftf": ftf,
    }


def get_characteristic_frequencies_for_condition(condition: str) -> dict[str, float]:
    """Get bearing characteristic frequencies for a given operating condition.

    Args:
        condition: Operating condition string (e.g., "35Hz12kN").

    Returns:
        Dictionary with characteristic frequencies in Hz.
    """
    # Parse shaft frequency from condition string
    condition_to_shaft_hz = {
        "35Hz12kN": 35.0,
        "37.5Hz11kN": 37.5,
        "40Hz10kN": 40.0,
    }

    if condition not in condition_to_shaft_hz:
        raise ValueError(f"Unknown condition: {condition}")

    shaft_freq = condition_to_shaft_hz[condition]
    return calculate_bearing_frequencies(shaft_freq, **BEARING_GEOMETRY)


def compute_fft(
    signal_data: NDArray,
    sampling_rate: float = SAMPLING_RATE,
    window: str = "hann",
) -> tuple[NDArray, NDArray]:
    """Compute FFT magnitude spectrum with windowing.

    Args:
        signal_data: Time-domain signal, shape (samples,) or (batch, samples).
        sampling_rate: Sampling rate in Hz.
        window: Window function name (e.g., "hann", "hamming").

    Returns:
        Tuple of (frequencies, magnitudes) where magnitudes are normalized.
    """
    is_single = signal_data.ndim == 1
    if is_single:
        signal_data = signal_data.reshape(1, -1)

    n_samples = signal_data.shape[-1]

    # Apply window
    win = signal.get_window(window, n_samples)
    windowed = signal_data * win

    # Compute FFT
    fft_result = rfft(windowed, axis=-1)
    magnitudes = np.abs(fft_result) * 2 / n_samples  # Normalize

    # Frequency bins
    freqs = rfftfreq(n_samples, d=1.0 / sampling_rate)

    if is_single:
        magnitudes = magnitudes.squeeze(0)

    return freqs, magnitudes


def compute_psd(
    signal_data: NDArray,
    sampling_rate: float = SAMPLING_RATE,
    nperseg: int = 1024,
) -> tuple[NDArray, NDArray]:
    """Compute Power Spectral Density using Welch's method.

    Args:
        signal_data: Time-domain signal, shape (samples,) or (batch, samples).
        sampling_rate: Sampling rate in Hz.
        nperseg: Length of each segment for Welch's method.

    Returns:
        Tuple of (frequencies, power spectral density).
    """
    is_single = signal_data.ndim == 1
    if is_single:
        signal_data = signal_data.reshape(1, -1)

    # Compute PSD for each sample in batch
    psds = []
    for sig in signal_data:
        f, pxx = signal.welch(sig, fs=sampling_rate, nperseg=nperseg)
        psds.append(pxx)

    psd_array = np.stack(psds)
    if is_single:
        psd_array = psd_array.squeeze(0)

    return f, psd_array


def extract_spectral_centroid(freqs: NDArray, magnitudes: NDArray) -> NDArray:
    """Extract spectral centroid (center of mass of spectrum).

    Args:
        freqs: Frequency bins.
        magnitudes: FFT magnitudes, shape (n_freqs,) or (batch, n_freqs).

    Returns:
        Spectral centroid in Hz.
    """
    if magnitudes.ndim == 1:
        mag_sum = magnitudes.sum()
        if mag_sum == 0:
            return np.array(0.0)
        return np.sum(freqs * magnitudes) / mag_sum
    else:
        mag_sum = magnitudes.sum(axis=-1, keepdims=True)
        mag_sum = np.where(mag_sum == 0, 1, mag_sum)  # Avoid division by zero
        return np.sum(freqs * magnitudes, axis=-1) / mag_sum.squeeze()


def extract_spectral_bandwidth(
    freqs: NDArray, magnitudes: NDArray, centroid: NDArray
) -> NDArray:
    """Extract spectral bandwidth (spread around centroid).

    Args:
        freqs: Frequency bins.
        magnitudes: FFT magnitudes.
        centroid: Pre-computed spectral centroid.

    Returns:
        Spectral bandwidth in Hz.
    """
    if magnitudes.ndim == 1:
        mag_sum = magnitudes.sum()
        if mag_sum == 0:
            return np.array(0.0)
        variance = np.sum(magnitudes * (freqs - centroid) ** 2) / mag_sum
        return np.sqrt(variance)
    else:
        mag_sum = magnitudes.sum(axis=-1, keepdims=True)
        mag_sum = np.where(mag_sum == 0, 1, mag_sum)
        centroid_2d = centroid.reshape(-1, 1)
        variance = np.sum(magnitudes * (freqs - centroid_2d) ** 2, axis=-1) / mag_sum.squeeze()
        return np.sqrt(variance)


def extract_spectral_rolloff(
    freqs: NDArray, magnitudes: NDArray, threshold: float = 0.85
) -> NDArray:
    """Extract spectral rolloff frequency.

    The frequency below which a threshold percentage of total energy is contained.

    Args:
        freqs: Frequency bins.
        magnitudes: FFT magnitudes.
        threshold: Energy threshold (default 0.85 = 85%).

    Returns:
        Rolloff frequency in Hz.
    """
    if magnitudes.ndim == 1:
        total_energy = magnitudes.sum()
        if total_energy == 0:
            return np.array(0.0)
        cumsum = np.cumsum(magnitudes)
        idx = np.searchsorted(cumsum, threshold * total_energy)
        return freqs[min(idx, len(freqs) - 1)]
    else:
        rolloffs = []
        for mag in magnitudes:
            total = mag.sum()
            if total == 0:
                rolloffs.append(0.0)
            else:
                cumsum = np.cumsum(mag)
                idx = np.searchsorted(cumsum, threshold * total)
                rolloffs.append(freqs[min(idx, len(freqs) - 1)])
        return np.array(rolloffs)


def extract_spectral_flatness(magnitudes: NDArray) -> NDArray:
    """Extract spectral flatness (tonality measure).

    Ratio of geometric mean to arithmetic mean of spectrum.
    Values close to 1 indicate noise-like, close to 0 indicate tonal.

    Args:
        magnitudes: FFT magnitudes.

    Returns:
        Spectral flatness (0 to 1).
    """
    eps = 1e-10  # Small value to avoid log(0)

    if magnitudes.ndim == 1:
        mag_nonzero = magnitudes + eps
        geometric_mean = np.exp(np.mean(np.log(mag_nonzero)))
        arithmetic_mean = np.mean(mag_nonzero)
        return geometric_mean / arithmetic_mean
    else:
        mag_nonzero = magnitudes + eps
        geometric_mean = np.exp(np.mean(np.log(mag_nonzero), axis=-1))
        arithmetic_mean = np.mean(mag_nonzero, axis=-1)
        return geometric_mean / arithmetic_mean


def extract_band_power(
    freqs: NDArray,
    magnitudes: NDArray,
    low_freq: float,
    high_freq: float,
) -> NDArray:
    """Extract power in a specific frequency band.

    Args:
        freqs: Frequency bins.
        magnitudes: FFT magnitudes.
        low_freq: Lower frequency bound in Hz.
        high_freq: Upper frequency bound in Hz.

    Returns:
        Total power in the specified band.
    """
    mask = (freqs >= low_freq) & (freqs < high_freq)

    if magnitudes.ndim == 1:
        return np.sum(magnitudes[mask] ** 2)
    else:
        return np.sum(magnitudes[:, mask] ** 2, axis=-1)


def extract_dominant_frequency(freqs: NDArray, magnitudes: NDArray) -> NDArray:
    """Extract dominant (peak) frequency.

    Args:
        freqs: Frequency bins.
        magnitudes: FFT magnitudes.

    Returns:
        Frequency with maximum magnitude in Hz.
    """
    if magnitudes.ndim == 1:
        return freqs[np.argmax(magnitudes)]
    else:
        return freqs[np.argmax(magnitudes, axis=-1)]


def extract_mean_frequency(freqs: NDArray, psd: NDArray) -> NDArray:
    """Extract mean frequency from PSD.

    Args:
        freqs: Frequency bins from PSD.
        psd: Power spectral density.

    Returns:
        Mean frequency in Hz.
    """
    if psd.ndim == 1:
        total_power = psd.sum()
        if total_power == 0:
            return np.array(0.0)
        return np.sum(freqs * psd) / total_power
    else:
        total_power = psd.sum(axis=-1, keepdims=True)
        total_power = np.where(total_power == 0, 1, total_power)
        return np.sum(freqs * psd, axis=-1) / total_power.squeeze()


def extract_characteristic_band_powers(
    freqs: NDArray,
    magnitudes: NDArray,
    characteristic_freqs: dict[str, float],
    bandwidth_ratio: float = 0.1,
) -> dict[str, NDArray]:
    """Extract power in bands around bearing characteristic frequencies.

    Args:
        freqs: Frequency bins.
        magnitudes: FFT magnitudes.
        characteristic_freqs: Dict with BPFO, BPFI, BSF, FTF frequencies.
        bandwidth_ratio: Band width as ratio of center frequency.

    Returns:
        Dictionary with band powers for each characteristic frequency.
    """
    result = {}
    for name, center_freq in characteristic_freqs.items():
        half_bw = center_freq * bandwidth_ratio
        low = max(0, center_freq - half_bw)
        high = center_freq + half_bw
        result[f"{name}_band_power"] = extract_band_power(freqs, magnitudes, low, high)
    return result


def extract_channel_frequency_features(
    signal_data: NDArray,
    sampling_rate: float = SAMPLING_RATE,
    characteristic_freqs: dict[str, float] | None = None,
) -> NDArray:
    """Extract all frequency-domain features for a single channel.

    Args:
        signal_data: Signal array of shape (samples,) or (batch, samples).
        sampling_rate: Sampling rate in Hz.
        characteristic_freqs: Optional bearing characteristic frequencies.
            If None, characteristic band powers are set to 0.

    Returns:
        Feature array of shape (14,) or (batch, 14).
        Order: 10 spectral features + 4 characteristic band powers.
    """
    is_single = signal_data.ndim == 1
    if is_single:
        signal_data = signal_data.reshape(1, -1)

    batch_size = signal_data.shape[0]

    # Compute FFT and PSD
    freqs, magnitudes = compute_fft(signal_data, sampling_rate)
    psd_freqs, psd = compute_psd(signal_data, sampling_rate)

    # Extract spectral features
    centroid = extract_spectral_centroid(freqs, magnitudes)
    bandwidth = extract_spectral_bandwidth(freqs, magnitudes, centroid)
    rolloff = extract_spectral_rolloff(freqs, magnitudes)
    flatness = extract_spectral_flatness(magnitudes)

    # Band powers (standard frequency ranges)
    bp_0_1k = extract_band_power(freqs, magnitudes, 0, 1000)
    bp_1_3k = extract_band_power(freqs, magnitudes, 1000, 3000)
    bp_3_6k = extract_band_power(freqs, magnitudes, 3000, 6000)
    bp_6_12k = extract_band_power(freqs, magnitudes, 6000, 12000)

    # Frequency measures
    dominant = extract_dominant_frequency(freqs, magnitudes)
    mean_freq = extract_mean_frequency(psd_freqs, psd)

    # Ensure all are 1D arrays for stacking
    def ensure_1d(x: NDArray) -> NDArray:
        x = np.atleast_1d(x)
        if x.shape == ():
            x = x.reshape(1)
        return x

    spectral_features = np.stack([
        ensure_1d(centroid),
        ensure_1d(bandwidth),
        ensure_1d(rolloff),
        ensure_1d(flatness),
        ensure_1d(bp_0_1k),
        ensure_1d(bp_1_3k),
        ensure_1d(bp_3_6k),
        ensure_1d(bp_6_12k),
        ensure_1d(dominant),
        ensure_1d(mean_freq),
    ], axis=-1)

    # Characteristic frequency band powers
    if characteristic_freqs is not None:
        char_powers = extract_characteristic_band_powers(
            freqs, magnitudes, characteristic_freqs
        )
        char_features = np.stack([
            ensure_1d(char_powers["bpfo_band_power"]),
            ensure_1d(char_powers["bpfi_band_power"]),
            ensure_1d(char_powers["bsf_band_power"]),
            ensure_1d(char_powers["ftf_band_power"]),
        ], axis=-1)
    else:
        char_features = np.zeros((batch_size, 4))

    all_features = np.concatenate([spectral_features, char_features], axis=-1)

    if is_single:
        all_features = all_features.squeeze(0)

    return all_features.astype(np.float32)


def extract_features(
    signal_data: NDArray,
    condition: str | None = None,
    sampling_rate: float = SAMPLING_RATE,
) -> NDArray:
    """Extract all frequency-domain features from a dual-channel signal.

    Extracts 14 features per channel (10 spectral + 4 characteristic) for
    a total of 28 features.

    Args:
        signal_data: Signal array of shape (samples, 2) or (batch, samples, 2).
            Channel 0 = horizontal, Channel 1 = vertical.
        condition: Operating condition (e.g., "35Hz12kN") for characteristic
            frequency calculation. If None, characteristic features are 0.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Feature array of shape (28,) or (batch, 28).
        Feature order: horizontal features (14), vertical features (14).
    """
    is_single = signal_data.ndim == 2
    if is_single:
        signal_data = signal_data.reshape(1, signal_data.shape[0], signal_data.shape[1])

    # Get characteristic frequencies if condition provided
    char_freqs = None
    if condition is not None:
        char_freqs = get_characteristic_frequencies_for_condition(condition)

    # Separate channels
    horizontal = signal_data[:, :, 0]  # Shape: (batch, samples)
    vertical = signal_data[:, :, 1]    # Shape: (batch, samples)

    # Extract features for each channel
    h_features = extract_channel_frequency_features(
        horizontal, sampling_rate, char_freqs
    )
    v_features = extract_channel_frequency_features(
        vertical, sampling_rate, char_freqs
    )

    # Concatenate all features
    all_features = np.concatenate([h_features, v_features], axis=-1)

    if is_single:
        all_features = all_features.squeeze(0)

    return all_features.astype(np.float32)


def get_feature_names() -> list[str]:
    """Get ordered list of all 28 feature names.

    Returns:
        List of feature names in the order they appear in extract_features output.
    """
    all_channel_features = SPECTRAL_FEATURES + CHARACTERISTIC_FEATURES
    h_names = [f"h_{name}" for name in all_channel_features]
    v_names = [f"v_{name}" for name in all_channel_features]
    return h_names + v_names


def extract_features_dict(
    signal_data: NDArray,
    condition: str | None = None,
) -> dict[str, float]:
    """Extract features and return as a named dictionary.

    Args:
        signal_data: Signal array of shape (samples, 2).
        condition: Operating condition for characteristic frequencies.

    Returns:
        Dictionary mapping feature names to values.
    """
    features = extract_features(signal_data, condition=condition)
    names = get_feature_names()
    return dict(zip(names, features))
