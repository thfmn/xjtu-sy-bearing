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

"""Continuous Wavelet Transform (CWT) Scalogram Generation for Bearing Vibration Signals.

This module implements CWT-based time-frequency representations for the XJTU-SY
bearing dataset using the Morlet wavelet.

Key features:
    - CWT scalogram generation using PyWavelets (pywt.cwt)
    - Morlet wavelet optimized for bearing fault frequency detection
    - Scale configuration covering bearing fault frequency range (FTF to harmonics)
    - Log-scale power conversion with numerical stability
    - Per-sample and global normalization options
    - Output shape: (64, 128, 2) for dual-channel signals
    - Memory-efficient batch processing

Reference:
    Dataset: XJTU-SY Bearing Dataset, 25.6 kHz sampling rate
    Output: CWT scalograms suitable for 2D CNN input (Pattern 2 models)

Note on CWT vs STFT:
    - CWT provides better frequency resolution at low frequencies
    - CWT provides better time resolution at high frequencies
    - Complementary to STFT spectrograms for bearing fault detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pywt

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Dataset constants
SAMPLING_RATE = 25600  # 25.6 kHz
NYQUIST_FREQ = SAMPLING_RATE / 2  # 12.8 kHz
DEFAULT_SIGNAL_LENGTH = 32768  # Samples per file

# Bearing fault frequency ranges (approximate, across all operating conditions)
# FTF: 13-16 Hz, BSF: 69-80 Hz, BPFO: 107-123 Hz, BPFI: 173-198 Hz
# We want scales that capture these frequencies and their harmonics up to ~3 kHz
MIN_FREQ_OF_INTEREST = 10.0  # Hz (below FTF)
MAX_FREQ_OF_INTEREST = 6000.0  # Hz (covers harmonics)


class NormalizationMode(Enum):
    """Normalization modes for scalograms."""

    NONE = "none"  # No normalization
    PER_SAMPLE = "per_sample"  # Normalize each scalogram independently
    GLOBAL = "global"  # Normalize using global statistics


@dataclass
class CWTConfig:
    """Configuration for CWT scalogram generation.

    Attributes:
        wavelet: Wavelet name for CWT. Default 'morl' (Morlet).
        num_scales: Number of scales (frequency bins) in output.
        min_freq: Minimum frequency of interest in Hz.
        max_freq: Maximum frequency of interest in Hz.
        log_scale: Whether to apply log scaling to magnitude.
        log_offset: Small constant added before log to avoid log(0).
        normalization: Normalization mode.
        sampling_rate: Sampling rate in Hz.
        target_time_bins: Number of time bins in output (for resizing).
    """

    wavelet: str = "morl"  # Morlet wavelet
    num_scales: int = 64  # PRD specifies 64 frequency bins
    min_freq: float = MIN_FREQ_OF_INTEREST
    max_freq: float = MAX_FREQ_OF_INTEREST
    log_scale: bool = True
    log_offset: float = 1e-10
    normalization: NormalizationMode = NormalizationMode.PER_SAMPLE
    sampling_rate: float = SAMPLING_RATE
    target_time_bins: int = 128  # PRD specifies 128 time bins

    # Computed attributes (set in __post_init__)
    scales: NDArray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        """Validate configuration and compute scales."""
        if self.min_freq <= 0:
            raise ValueError(f"min_freq must be positive, got {self.min_freq}")
        if self.max_freq <= self.min_freq:
            raise ValueError(f"max_freq ({self.max_freq}) must be > min_freq ({self.min_freq})")
        if self.max_freq > NYQUIST_FREQ:
            raise ValueError(f"max_freq ({self.max_freq}) cannot exceed Nyquist ({NYQUIST_FREQ})")
        if self.num_scales <= 0:
            raise ValueError(f"num_scales must be positive, got {self.num_scales}")

        # Compute scales from frequency range
        self.scales = self._compute_scales()

    def _compute_scales(self) -> NDArray:
        """Compute wavelet scales corresponding to frequency range.

        For the Morlet wavelet, the relationship between scale (a) and
        pseudo-frequency (f) is: f = Fc / (a * Ts)
        where Fc is the wavelet center frequency and Ts is the sampling period.

        For 'morl', Fc ≈ 0.8125 (can be obtained via pywt.central_frequency)

        Rearranging: a = Fc / (f * Ts) = Fc * fs / f
        """
        # Get wavelet center frequency
        wavelet = pywt.ContinuousWavelet(self.wavelet)
        # central_frequency returns frequency for scale=1, sampling_period=1
        fc = pywt.central_frequency(wavelet)

        # Compute scales for logarithmically spaced frequencies
        # We want higher resolution at lower frequencies (larger scales)
        frequencies = np.logspace(
            np.log10(self.min_freq),
            np.log10(self.max_freq),
            self.num_scales
        )

        # Convert frequencies to scales
        # scale = fc * fs / f
        scales = fc * self.sampling_rate / frequencies

        # Reverse so that lower scales (higher frequencies) come first
        # This makes the scalogram orientation similar to spectrograms
        scales = scales[::-1]

        return scales.astype(np.float64)

    def scale_to_frequency(self, scale: float) -> float:
        """Convert a scale value to pseudo-frequency in Hz.

        Args:
            scale: Wavelet scale value.

        Returns:
            Corresponding pseudo-frequency in Hz.
        """
        wavelet = pywt.ContinuousWavelet(self.wavelet)
        fc = pywt.central_frequency(wavelet)
        return fc * self.sampling_rate / scale


def compute_cwt(
    signal_data: NDArray,
    scales: NDArray,
    wavelet: str = "morl",
    sampling_rate: float = SAMPLING_RATE,
) -> tuple[NDArray, NDArray]:
    """Compute Continuous Wavelet Transform.

    Args:
        signal_data: Time-domain signal, shape (samples,) or (batch, samples).
        scales: Array of scales for the CWT.
        wavelet: Wavelet name.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Tuple of (coefficients, frequencies) where:
            - coefficients: CWT magnitude, shape (n_scales, n_samples) or
              (batch, n_scales, n_samples)
            - frequencies: Pseudo-frequencies corresponding to scales, shape (n_scales,)
    """
    is_single = signal_data.ndim == 1
    if is_single:
        signal_data = signal_data.reshape(1, -1)

    batch_size = signal_data.shape[0]
    n_samples = signal_data.shape[1]
    n_scales = len(scales)

    # Pre-allocate output array
    coefficients = np.zeros((batch_size, n_scales, n_samples), dtype=np.float64)

    # Compute CWT for each sample
    for i in range(batch_size):
        coef, freqs = pywt.cwt(
            signal_data[i],
            scales,
            wavelet,
            sampling_period=1.0 / sampling_rate,
        )
        # Take magnitude of complex coefficients
        coefficients[i] = np.abs(coef)

    if is_single:
        coefficients = coefficients.squeeze(0)

    return coefficients, freqs


def power_to_db(
    scalogram: NDArray,
    power: float = 2.0,
    log_offset: float = 1e-10,
) -> NDArray:
    """Convert scalogram to decibel scale.

    Args:
        scalogram: Scalogram (magnitude).
        power: Exponent to apply (2.0 for power, 1.0 for magnitude).
        log_offset: Small constant for numerical stability.

    Returns:
        Log-scaled scalogram in dB.
    """
    if power != 1.0:
        scalogram = scalogram ** power

    # Log scale with offset for numerical stability
    log_spec = 10.0 * np.log10(scalogram + log_offset)

    return log_spec


def normalize_scalogram(
    scalogram: NDArray,
    mode: NormalizationMode = NormalizationMode.PER_SAMPLE,
    global_mean: float | None = None,
    global_std: float | None = None,
) -> NDArray:
    """Normalize scalogram.

    Args:
        scalogram: Scalogram to normalize, shape (..., height, width).
        mode: Normalization mode.
        global_mean: Mean for global normalization.
        global_std: Std for global normalization.

    Returns:
        Normalized scalogram.
    """
    if mode == NormalizationMode.NONE:
        return scalogram

    if mode == NormalizationMode.PER_SAMPLE:
        if scalogram.ndim == 2:
            mean = scalogram.mean()
            std = scalogram.std()
            std = std if std > 0 else 1.0
            return (scalogram - mean) / std
        else:
            # Normalize each sample independently
            batch_size = scalogram.shape[0]
            result = np.zeros_like(scalogram)
            for i in range(batch_size):
                mean = scalogram[i].mean()
                std = scalogram[i].std()
                std = std if std > 0 else 1.0
                result[i] = (scalogram[i] - mean) / std
            return result

    elif mode == NormalizationMode.GLOBAL:
        if global_mean is None or global_std is None:
            raise ValueError("Global normalization requires global_mean and global_std")
        std = global_std if global_std > 0 else 1.0
        return (scalogram - global_mean) / std

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def resize_scalogram(
    scalogram: NDArray,
    target_height: int = 64,
    target_width: int = 128,
) -> NDArray:
    """Resize scalogram to target dimensions using bilinear interpolation.

    Args:
        scalogram: Scalogram, shape (height, width) or (batch, height, width).
        target_height: Target height (scale/frequency bins).
        target_width: Target width (time bins).

    Returns:
        Resized scalogram.
    """
    from scipy.ndimage import zoom

    is_single = scalogram.ndim == 2
    if is_single:
        scalogram = scalogram.reshape(1, scalogram.shape[0], scalogram.shape[1])

    batch_size, height, width = scalogram.shape

    # Calculate zoom factors
    height_factor = target_height / height
    width_factor = target_width / width

    # Resize each sample
    resized = np.zeros((batch_size, target_height, target_width), dtype=scalogram.dtype)
    for i in range(batch_size):
        resized[i] = zoom(scalogram[i], (height_factor, width_factor), order=1)

    if is_single:
        resized = resized.squeeze(0)

    return resized


def generate_scalogram(
    signal_data: NDArray,
    config: CWTConfig | None = None,
    target_shape: tuple[int, int] = (64, 128),
) -> NDArray:
    """Generate CWT scalogram from time-domain signal.

    This is the main function for generating CWT scalograms with the
    standard configuration for the XJTU-SY dataset.

    Args:
        signal_data: Time-domain signal, shape (samples,) or (batch, samples).
        config: CWT configuration. If None, uses default config.
        target_shape: Target (height, width) for output scalogram.

    Returns:
        Scalogram, shape (height, width) or (batch, height, width).
    """
    if config is None:
        config = CWTConfig()

    # Compute CWT
    coefficients, freqs = compute_cwt(
        signal_data,
        scales=config.scales,
        wavelet=config.wavelet,
        sampling_rate=config.sampling_rate,
    )

    # Apply power scaling and log
    if config.log_scale:
        coefficients = power_to_db(coefficients, power=2.0, log_offset=config.log_offset)

    # Normalize
    coefficients = normalize_scalogram(coefficients, mode=config.normalization)

    # Resize to target shape
    target_height, target_width = target_shape
    scalogram = resize_scalogram(coefficients, target_height, target_width)

    return scalogram.astype(np.float32)


def generate_scalogram_dual_channel(
    signal_data: NDArray,
    config: CWTConfig | None = None,
    target_shape: tuple[int, int] = (64, 128),
) -> NDArray:
    """Generate CWT scalogram for dual-channel signal.

    This produces the standard (64, 128, 2) output shape for Pattern 2 models.

    Args:
        signal_data: Dual-channel signal, shape (samples, 2) or (batch, samples, 2).
            Channel 0 = horizontal, Channel 1 = vertical.
        config: CWT configuration. If None, uses default config.
        target_shape: Target (height, width) for output scalogram.

    Returns:
        Scalogram, shape (height, width, 2) or (batch, height, width, 2).
    """
    is_single = signal_data.ndim == 2
    if is_single:
        signal_data = signal_data.reshape(1, signal_data.shape[0], signal_data.shape[1])

    batch_size = signal_data.shape[0]
    height, width = target_shape

    # Separate channels
    horizontal = signal_data[:, :, 0]  # (batch, samples)
    vertical = signal_data[:, :, 1]  # (batch, samples)

    # Generate scalograms for each channel
    h_scalogram = generate_scalogram(
        horizontal, config=config, target_shape=target_shape
    )
    v_scalogram = generate_scalogram(
        vertical, config=config, target_shape=target_shape
    )

    # Stack channels: (batch, height, width) -> (batch, height, width, 2)
    result = np.stack([h_scalogram, v_scalogram], axis=-1)

    if is_single:
        result = result.squeeze(0)

    return result


def get_default_config() -> CWTConfig:
    """Get the default CWT configuration for XJTU-SY dataset.

    Returns:
        Default CWTConfig optimized for 25.6 kHz bearing vibration signals.
    """
    return CWTConfig(
        wavelet="morl",
        num_scales=64,
        min_freq=MIN_FREQ_OF_INTEREST,
        max_freq=MAX_FREQ_OF_INTEREST,
        log_scale=True,
        log_offset=1e-10,
        normalization=NormalizationMode.PER_SAMPLE,
        sampling_rate=SAMPLING_RATE,
        target_time_bins=128,
    )


# Convenience function matching PRD specification
def extract_scalogram(
    signal_data: NDArray,
    sampling_rate: float = SAMPLING_RATE,
) -> NDArray:
    """Extract CWT scalogram with standard PRD configuration.

    This is the primary interface for generating scalograms matching
    the PRD specification: output shape (64, 128, 2).

    Args:
        signal_data: Dual-channel signal, shape (samples, 2) or (batch, samples, 2).
        sampling_rate: Sampling rate in Hz.

    Returns:
        Scalogram, shape (64, 128, 2) or (batch, 64, 128, 2).
    """
    config = get_default_config()
    config.sampling_rate = sampling_rate
    # Recompute scales with new sampling rate if different
    if sampling_rate != SAMPLING_RATE:
        config.scales = config._compute_scales()

    return generate_scalogram_dual_channel(
        signal_data,
        config=config,
        target_shape=(64, 128),
    )


def get_scale_frequencies(config: CWTConfig | None = None) -> NDArray:
    """Get the pseudo-frequencies corresponding to each scale.

    Useful for annotating scalogram plots with frequency labels.

    Args:
        config: CWT configuration. If None, uses default config.

    Returns:
        Array of frequencies in Hz, shape (num_scales,).
    """
    if config is None:
        config = get_default_config()

    frequencies = np.array([config.scale_to_frequency(s) for s in config.scales])
    return frequencies


def estimate_memory_usage(
    batch_size: int = 32,
    signal_length: int = DEFAULT_SIGNAL_LENGTH,
    num_scales: int = 64,
    dtype_bytes: int = 8,  # float64 for CWT computation
) -> dict[str, float]:
    """Estimate memory usage for CWT computation.

    Args:
        batch_size: Number of signals in batch.
        signal_length: Length of each signal in samples.
        num_scales: Number of wavelet scales.
        dtype_bytes: Bytes per element (8 for float64).

    Returns:
        Dictionary with memory estimates in MB and GB.
    """
    # Input signal memory
    input_mem = batch_size * signal_length * 2 * dtype_bytes  # 2 channels

    # CWT output memory (complex coefficients, then magnitude)
    # PyWavelets returns complex array of shape (num_scales, signal_length)
    cwt_complex_mem = batch_size * num_scales * signal_length * 16  # complex128
    cwt_magnitude_mem = batch_size * num_scales * signal_length * dtype_bytes

    # Total for one channel
    single_channel_mem = cwt_complex_mem + cwt_magnitude_mem

    # Total for both channels (processed sequentially in our implementation)
    total_mem = input_mem + single_channel_mem

    return {
        "input_mb": input_mem / (1024 ** 2),
        "cwt_per_channel_mb": single_channel_mem / (1024 ** 2),
        "total_mb": total_mem / (1024 ** 2),
        "total_gb": total_mem / (1024 ** 3),
    }


# Memory-efficient batch processing for large datasets
def generate_scalograms_batched(
    signal_data: NDArray,
    config: CWTConfig | None = None,
    target_shape: tuple[int, int] = (64, 128),
    batch_size: int = 8,
) -> NDArray:
    """Generate scalograms in batches to manage memory.

    For large datasets or limited memory, this processes signals in smaller
    batches to avoid memory issues.

    Args:
        signal_data: Dual-channel signals, shape (n_samples, samples, 2).
        config: CWT configuration. If None, uses default config.
        target_shape: Target (height, width) for output scalograms.
        batch_size: Number of signals to process at once.

    Returns:
        Scalograms, shape (n_samples, height, width, 2).
    """
    n_samples = signal_data.shape[0]
    height, width = target_shape

    # Pre-allocate output
    output = np.zeros((n_samples, height, width, 2), dtype=np.float32)

    # Process in batches
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = signal_data[start:end]
        output[start:end] = generate_scalogram_dual_channel(
            batch, config=config, target_shape=target_shape
        )

    return output
