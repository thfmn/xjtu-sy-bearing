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

"""STFT Spectrogram and Mel-Spectrogram Generation for Bearing Vibration Signals.

This module implements Short-Time Fourier Transform (STFT) based time-frequency
representations for the XJTU-SY bearing dataset.

Key features:
    - STFT spectrogram generation using scipy.signal.spectrogram
    - Mel-spectrogram with 128 mel bins spanning 0 to Nyquist (12.8 kHz)
    - Log-scale power conversion with numerical stability
    - Per-sample and global normalization options
    - Output shape: (128, 128, 2) for dual-channel signals

Reference:
    Dataset: XJTU-SY Bearing Dataset, 25.6 kHz sampling rate
    Output: Mel-spectrograms suitable for 2D CNN input (Pattern 2 models)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Dataset constants
SAMPLING_RATE = 25600  # 25.6 kHz
NYQUIST_FREQ = SAMPLING_RATE / 2  # 12.8 kHz
DEFAULT_SIGNAL_LENGTH = 32768  # Samples per file


class NormalizationMode(Enum):
    """Normalization modes for spectrograms."""

    NONE = "none"  # No normalization
    PER_SAMPLE = "per_sample"  # Normalize each spectrogram independently
    GLOBAL = "global"  # Normalize using global statistics


@dataclass
class STFTConfig:
    """Configuration for STFT spectrogram generation.

    Attributes:
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel filterbank bins (set to 0 for linear spectrogram).
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank (None = Nyquist).
        power: Exponent for magnitude (1=amplitude, 2=power).
        log_scale: Whether to apply log scaling.
        log_offset: Small constant added before log to avoid log(0).
        normalization: Normalization mode.
        window: Window function name.
        center: Whether to pad signal for centered frames.
    """

    n_fft: int = 512
    hop_length: int = 256
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None  # None means Nyquist
    power: float = 2.0
    log_scale: bool = True
    log_offset: float = 1e-10
    normalization: NormalizationMode = NormalizationMode.PER_SAMPLE
    window: str = "hann"
    center: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.fmax is None:
            self.fmax = NYQUIST_FREQ
        if self.fmax > NYQUIST_FREQ:
            raise ValueError(f"fmax ({self.fmax}) cannot exceed Nyquist ({NYQUIST_FREQ})")
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")


def hz_to_mel(frequencies: NDArray) -> NDArray:
    """Convert Hz to mel scale.

    Uses the O'Shaughnessy formula: m = 2595 * log10(1 + f/700)

    Args:
        frequencies: Array of frequencies in Hz.

    Returns:
        Frequencies in mel scale.
    """
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)


def mel_to_hz(mels: NDArray) -> NDArray:
    """Convert mel scale to Hz.

    Args:
        mels: Array of mel values.

    Returns:
        Frequencies in Hz.
    """
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def create_mel_filterbank(
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = NYQUIST_FREQ,
    sampling_rate: float = SAMPLING_RATE,
) -> NDArray:
    """Create a mel filterbank matrix.

    Args:
        n_fft: FFT size (determines number of frequency bins).
        n_mels: Number of mel filterbank bins.
        fmin: Minimum frequency in Hz.
        fmax: Maximum frequency in Hz.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    # Number of frequency bins
    n_freqs = n_fft // 2 + 1

    # Frequency bins
    fft_freqs = np.linspace(0, sampling_rate / 2, n_freqs)

    # Mel scale points
    mel_min = hz_to_mel(np.array([fmin]))[0]
    mel_max = hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Create filterbank
    filterbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        # Left slope
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Left ramp
        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        if center > left:
            filterbank[i, left_mask] = (fft_freqs[left_mask] - left) / (center - left)

        # Right ramp
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        if right > center:
            filterbank[i, right_mask] = (right - fft_freqs[right_mask]) / (right - center)

    return filterbank.astype(np.float32)


def compute_stft(
    signal_data: NDArray,
    n_fft: int = 512,
    hop_length: int = 256,
    window: str = "hann",
    sampling_rate: float = SAMPLING_RATE,
    center: bool = True,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Short-Time Fourier Transform.

    Args:
        signal_data: Time-domain signal, shape (samples,) or (batch, samples).
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        window: Window function name.
        sampling_rate: Sampling rate in Hz.
        center: Whether to pad signal for centered frames.

    Returns:
        Tuple of (frequencies, times, spectrogram) where:
            - frequencies: shape (n_fft // 2 + 1,)
            - times: shape (n_frames,)
            - spectrogram: shape (n_fft // 2 + 1, n_frames) or (batch, n_fft // 2 + 1, n_frames)
    """
    is_single = signal_data.ndim == 1
    if is_single:
        signal_data = signal_data.reshape(1, -1)

    batch_size = signal_data.shape[0]
    n_samples = signal_data.shape[1]

    # Compute expected number of frames
    if center:
        padded_length = n_samples + n_fft
        n_frames = 1 + (padded_length - n_fft) // hop_length
    else:
        n_frames = 1 + (n_samples - n_fft) // hop_length

    n_freqs = n_fft // 2 + 1

    # Compute STFT for each sample
    spectrograms = np.zeros((batch_size, n_freqs, n_frames), dtype=np.float64)

    for i in range(batch_size):
        sig = signal_data[i]

        # Pad signal if centering
        if center:
            sig = np.pad(sig, (n_fft // 2, n_fft // 2), mode="reflect")

        # Use scipy.signal.spectrogram
        freqs, times, Sxx = signal.spectrogram(
            sig,
            fs=sampling_rate,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            mode="magnitude",
            scaling="spectrum",
        )

        # Handle size mismatch due to different padding
        min_frames = min(Sxx.shape[1], n_frames)
        spectrograms[i, :, :min_frames] = Sxx[:, :min_frames]

    if is_single:
        spectrograms = spectrograms.squeeze(0)

    # Return frequency and time arrays from last computation
    return freqs, times[:n_frames] if len(times) >= n_frames else times, spectrograms


def apply_mel_filterbank(
    spectrogram: NDArray,
    mel_filterbank: NDArray,
) -> NDArray:
    """Apply mel filterbank to a linear spectrogram.

    Args:
        spectrogram: Linear spectrogram, shape (n_freqs, n_frames) or (batch, n_freqs, n_frames).
        mel_filterbank: Mel filterbank matrix, shape (n_mels, n_freqs).

    Returns:
        Mel spectrogram, shape (n_mels, n_frames) or (batch, n_mels, n_frames).
    """
    is_single = spectrogram.ndim == 2
    if is_single:
        spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1])

    # Apply filterbank: (n_mels, n_freqs) @ (batch, n_freqs, n_frames) -> (batch, n_mels, n_frames)
    mel_spec = np.einsum("mf,bft->bmt", mel_filterbank, spectrogram)

    if is_single:
        mel_spec = mel_spec.squeeze(0)

    return mel_spec


def power_to_db(
    spectrogram: NDArray,
    power: float = 2.0,
    log_offset: float = 1e-10,
) -> NDArray:
    """Convert power spectrogram to decibel scale.

    Args:
        spectrogram: Spectrogram (magnitude or power).
        power: Exponent to apply (2.0 for power, 1.0 for amplitude).
        log_offset: Small constant for numerical stability.

    Returns:
        Log-scaled spectrogram in dB.
    """
    if power != 1.0:
        spectrogram = spectrogram ** power

    # Log scale with offset for numerical stability
    log_spec = 10.0 * np.log10(spectrogram + log_offset)

    return log_spec


def normalize_spectrogram(
    spectrogram: NDArray,
    mode: NormalizationMode = NormalizationMode.PER_SAMPLE,
    global_mean: float | None = None,
    global_std: float | None = None,
) -> NDArray:
    """Normalize spectrogram.

    Args:
        spectrogram: Spectrogram to normalize, shape (..., height, width).
        mode: Normalization mode.
        global_mean: Mean for global normalization.
        global_std: Std for global normalization.

    Returns:
        Normalized spectrogram.
    """
    if mode == NormalizationMode.NONE:
        return spectrogram

    if mode == NormalizationMode.PER_SAMPLE:
        if spectrogram.ndim == 2:
            mean = spectrogram.mean()
            std = spectrogram.std()
            std = std if std > 0 else 1.0
            return (spectrogram - mean) / std
        else:
            # Normalize each sample independently
            batch_size = spectrogram.shape[0]
            result = np.zeros_like(spectrogram)
            for i in range(batch_size):
                mean = spectrogram[i].mean()
                std = spectrogram[i].std()
                std = std if std > 0 else 1.0
                result[i] = (spectrogram[i] - mean) / std
            return result

    elif mode == NormalizationMode.GLOBAL:
        if global_mean is None or global_std is None:
            raise ValueError("Global normalization requires global_mean and global_std")
        std = global_std if global_std > 0 else 1.0
        return (spectrogram - global_mean) / std

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def resize_spectrogram(
    spectrogram: NDArray,
    target_height: int = 128,
    target_width: int = 128,
) -> NDArray:
    """Resize spectrogram to target dimensions using bilinear interpolation.

    Args:
        spectrogram: Spectrogram, shape (height, width) or (batch, height, width).
        target_height: Target height (frequency bins).
        target_width: Target width (time frames).

    Returns:
        Resized spectrogram.
    """
    from scipy.ndimage import zoom

    is_single = spectrogram.ndim == 2
    if is_single:
        spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1])

    batch_size, height, width = spectrogram.shape

    # Calculate zoom factors
    height_factor = target_height / height
    width_factor = target_width / width

    # Resize each sample
    resized = np.zeros((batch_size, target_height, target_width), dtype=spectrogram.dtype)
    for i in range(batch_size):
        resized[i] = zoom(spectrogram[i], (height_factor, width_factor), order=1)

    if is_single:
        resized = resized.squeeze(0)

    return resized


def generate_mel_spectrogram(
    signal_data: NDArray,
    config: STFTConfig | None = None,
    sampling_rate: float = SAMPLING_RATE,
    target_shape: tuple[int, int] = (128, 128),
) -> NDArray:
    """Generate mel spectrogram from time-domain signal.

    This is the main function for generating mel spectrograms with the
    standard configuration for the XJTU-SY dataset.

    Args:
        signal_data: Time-domain signal, shape (samples,) or (batch, samples).
        config: STFT configuration. If None, uses default config.
        sampling_rate: Sampling rate in Hz.
        target_shape: Target (height, width) for output spectrogram.

    Returns:
        Mel spectrogram, shape (height, width) or (batch, height, width).
    """
    if config is None:
        config = STFTConfig()

    # Compute STFT
    freqs, times, stft_spec = compute_stft(
        signal_data,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=config.window,
        sampling_rate=sampling_rate,
        center=config.center,
    )

    # Create mel filterbank
    mel_fb = create_mel_filterbank(
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax if config.fmax is not None else sampling_rate / 2,
        sampling_rate=sampling_rate,
    )

    # Apply mel filterbank
    mel_spec = apply_mel_filterbank(stft_spec, mel_fb)

    # Apply power scaling and log
    if config.log_scale:
        mel_spec = power_to_db(mel_spec, power=config.power, log_offset=config.log_offset)
    elif config.power != 1.0:
        mel_spec = mel_spec ** config.power

    # Normalize
    mel_spec = normalize_spectrogram(mel_spec, mode=config.normalization)

    # Resize to target shape
    target_height, target_width = target_shape
    mel_spec = resize_spectrogram(mel_spec, target_height, target_width)

    return mel_spec.astype(np.float32)


def generate_spectrogram_dual_channel(
    signal_data: NDArray,
    config: STFTConfig | None = None,
    sampling_rate: float = SAMPLING_RATE,
    target_shape: tuple[int, int] = (128, 128),
) -> NDArray:
    """Generate mel spectrogram for dual-channel signal.

    This produces the standard (128, 128, 2) output shape for Pattern 2 models.

    Args:
        signal_data: Dual-channel signal, shape (samples, 2) or (batch, samples, 2).
            Channel 0 = horizontal, Channel 1 = vertical.
        config: STFT configuration. If None, uses default config.
        sampling_rate: Sampling rate in Hz.
        target_shape: Target (height, width) for output spectrogram.

    Returns:
        Mel spectrogram, shape (height, width, 2) or (batch, height, width, 2).
    """
    is_single = signal_data.ndim == 2
    if is_single:
        signal_data = signal_data.reshape(1, signal_data.shape[0], signal_data.shape[1])

    batch_size = signal_data.shape[0]
    height, width = target_shape

    # Separate channels
    horizontal = signal_data[:, :, 0]  # (batch, samples)
    vertical = signal_data[:, :, 1]    # (batch, samples)

    # Generate mel spectrograms for each channel
    h_spec = generate_mel_spectrogram(
        horizontal, config=config, sampling_rate=sampling_rate, target_shape=target_shape
    )
    v_spec = generate_mel_spectrogram(
        vertical, config=config, sampling_rate=sampling_rate, target_shape=target_shape
    )

    # Stack channels: (batch, height, width) -> (batch, height, width, 2)
    result = np.stack([h_spec, v_spec], axis=-1)

    if is_single:
        result = result.squeeze(0)

    return result


def get_default_config() -> STFTConfig:
    """Get the default STFT configuration for XJTU-SY dataset.

    Returns:
        Default STFTConfig optimized for 25.6 kHz bearing vibration signals.
    """
    return STFTConfig(
        n_fft=512,
        hop_length=256,
        n_mels=128,
        fmin=0.0,
        fmax=NYQUIST_FREQ,
        power=2.0,
        log_scale=True,
        log_offset=1e-10,
        normalization=NormalizationMode.PER_SAMPLE,
        window="hann",
        center=True,
    )


# Convenience function matching PRD specification
def extract_spectrogram(
    signal_data: NDArray,
    sampling_rate: float = SAMPLING_RATE,
) -> NDArray:
    """Extract mel spectrogram with standard PRD configuration.

    This is the primary interface for generating spectrograms matching
    the PRD specification: output shape (128, 128, 2).

    Args:
        signal_data: Dual-channel signal, shape (samples, 2) or (batch, samples, 2).
        sampling_rate: Sampling rate in Hz.

    Returns:
        Mel spectrogram, shape (128, 128, 2) or (batch, 128, 128, 2).
    """
    return generate_spectrogram_dual_channel(
        signal_data,
        config=get_default_config(),
        sampling_rate=sampling_rate,
        target_shape=(128, 128),
    )


# Linear spectrogram (non-mel) for comparison
def generate_linear_spectrogram(
    signal_data: NDArray,
    n_fft: int = 512,
    hop_length: int = 256,
    sampling_rate: float = SAMPLING_RATE,
    log_scale: bool = True,
    normalize: bool = True,
    target_shape: tuple[int, int] | None = (128, 128),
) -> NDArray:
    """Generate linear (non-mel) spectrogram.

    Args:
        signal_data: Time-domain signal, shape (samples,) or (batch, samples).
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        sampling_rate: Sampling rate in Hz.
        log_scale: Whether to apply log scaling.
        normalize: Whether to normalize per sample.
        target_shape: Target shape for resizing. None to skip resizing.

    Returns:
        Linear spectrogram.
    """
    freqs, times, spec = compute_stft(
        signal_data,
        n_fft=n_fft,
        hop_length=hop_length,
        sampling_rate=sampling_rate,
    )

    # Power spectrogram
    spec = spec ** 2

    if log_scale:
        spec = power_to_db(spec, power=1.0)  # Already squared

    if normalize:
        spec = normalize_spectrogram(spec, mode=NormalizationMode.PER_SAMPLE)

    if target_shape is not None:
        spec = resize_spectrogram(spec, target_shape[0], target_shape[1])

    return spec.astype(np.float32)
