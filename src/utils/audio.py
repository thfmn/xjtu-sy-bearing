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

"""Audio conversion utilities for XJTU-SY bearing vibration signals (FEAT-12).

This module provides functions to convert vibration signals to WAV audio files
for auditory analysis of bearing degradation patterns.

Features:
    - Resampling from 25.6kHz to 44.1kHz (standard audio)
    - Amplitude normalization to prevent clipping
    - WAV file generation (16-bit PCM, mono or stereo)
    - Batch conversion utilities

Example:
    >>> from src.utils.audio import resample_signal, normalize_audio, signal_to_wav
    >>> from src.data.loader import XJTUBearingLoader
    >>> loader = XJTUBearingLoader()
    >>> signal, _ = loader.load_file('35Hz12kN', 'Bearing1_1', '1.csv')
    >>> # Resample horizontal channel to 44.1kHz
    >>> resampled = resample_signal(signal[:, 0])
    >>> # Normalize and write to WAV
    >>> signal_to_wav(resampled, 'bearing_audio.wav')
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Union

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile

# Audio configuration constants
SOURCE_SAMPLE_RATE = 25600  # Hz - XJTU-SY dataset sampling rate
TARGET_SAMPLE_RATE = 44100  # Hz - Standard audio sample rate
DEFAULT_PEAK_AMPLITUDE = 0.95  # Leave headroom for DAC


@dataclass
class AudioConfig:
    """Configuration for audio conversion.

    Attributes:
        source_rate: Source sampling rate in Hz (default: 25600).
        target_rate: Target sampling rate in Hz (default: 44100).
        normalize: Whether to normalize amplitude (default: True).
        target_peak: Target peak amplitude after normalization (default: 0.95).
        bit_depth: Output bit depth, 16 or 24 (default: 16).
    """

    source_rate: int = SOURCE_SAMPLE_RATE
    target_rate: int = TARGET_SAMPLE_RATE
    normalize: bool = True
    target_peak: float = DEFAULT_PEAK_AMPLITUDE
    bit_depth: int = 16


def resample_signal(
    data: np.ndarray,
    source_rate: int = SOURCE_SAMPLE_RATE,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample signal from source to target sample rate.

    Uses scipy.signal.resample_poly for efficient polyphase resampling,
    which provides high-quality anti-aliasing filtering.

    Args:
        data: Input signal array. Can be:
            - 1D array of shape (samples,) for single channel
            - 2D array of shape (samples, channels) for multi-channel
        source_rate: Original sampling rate in Hz (default: 25600).
        target_rate: Target sampling rate in Hz (default: 44100).

    Returns:
        Resampled signal array with same number of dimensions as input.

    Example:
        >>> signal = np.sin(2 * np.pi * 1000 * np.arange(25600) / 25600)  # 1kHz tone
        >>> resampled = resample_signal(signal)  # Now at 44.1kHz
        >>> len(resampled)  # ~44100 samples
        44100
    """
    # Calculate resampling ratio using GCD for rational approximation
    g = gcd(source_rate, target_rate)
    up = target_rate // g
    down = source_rate // g

    # Handle single channel (1D array)
    if data.ndim == 1:
        return scipy_signal.resample_poly(data, up, down)

    # Handle multi-channel (2D array)
    resampled_channels = []
    for ch in range(data.shape[1]):
        resampled = scipy_signal.resample_poly(data[:, ch], up, down)
        resampled_channels.append(resampled)
    return np.column_stack(resampled_channels)


def normalize_audio(
    data: np.ndarray,
    target_peak: float = DEFAULT_PEAK_AMPLITUDE,
) -> np.ndarray:
    """Normalize audio amplitude to prevent clipping.

    Scales the signal so that the maximum absolute value equals target_peak.
    This ensures the audio doesn't clip when converted to integer PCM format.

    Args:
        data: Input signal array (any shape).
        target_peak: Target peak amplitude in range (0, 1).
            Default 0.95 leaves headroom for DAC and downstream processing.

    Returns:
        Normalized signal in range [-target_peak, target_peak].

    Example:
        >>> loud_signal = np.array([0.1, 0.5, 2.0, -1.5])
        >>> normalized = normalize_audio(loud_signal)
        >>> np.abs(normalized).max()  # Now peaked at 0.95
        0.95
    """
    max_val = np.abs(data).max()
    if max_val > 0:
        return data * (target_peak / max_val)
    return data


def normalize_amplitude(data: np.ndarray) -> np.ndarray:
    """Normalize signal amplitude to [-1, 1] range.

    Scales the signal so that the maximum absolute value equals 1.0.
    Handles all-zeros input gracefully (returns as-is).

    Args:
        data: Input 1D signal array.

    Returns:
        Signal scaled to [-1.0, 1.0] range.
    """
    return normalize_audio(data, target_peak=1.0)


def signal_to_wav(
    data: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
    target_peak: float = DEFAULT_PEAK_AMPLITUDE,
    bit_depth: int = 16,
) -> Path:
    """Convert signal array to WAV file.

    Writes a standard WAV file with PCM encoding. Supports mono and stereo.

    Args:
        data: Signal array. Can be:
            - 1D array of shape (samples,) for mono
            - 2D array of shape (samples, channels) for stereo/multi-channel
        output_path: Output WAV file path.
        sample_rate: Sample rate for the WAV file (default: 44100).
        normalize: Whether to normalize amplitude before writing (default: True).
        target_peak: Target peak amplitude when normalizing (default: 0.95).
        bit_depth: Output bit depth, 16 or 24 (default: 16).

    Returns:
        Path to the generated WAV file.

    Raises:
        ValueError: If bit_depth is not 16 or 24.

    Example:
        >>> signal = np.random.randn(44100)  # 1 second of noise at 44.1kHz
        >>> path = signal_to_wav(signal, 'test.wav')
        >>> path.exists()
        True
    """
    if bit_depth not in (16, 24):
        raise ValueError(f"bit_depth must be 16 or 24, got {bit_depth}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize if requested
    if normalize:
        data = normalize_audio(data, target_peak)

    # Convert to integer PCM
    if bit_depth == 16:
        # Scale to int16 range [-32768, 32767]
        data_int = (data * 32767).astype(np.int16)
    else:  # 24-bit
        # Scale to int32, write as 24-bit (scipy handles this)
        data_int = (data * 8388607).astype(np.int32)

    wavfile.write(str(output_path), sample_rate, data_int)
    return output_path


def wav_to_signal(
    wav_path: Union[str, Path],
) -> tuple[np.ndarray, int]:
    """Read WAV file and return normalized signal.

    Args:
        wav_path: Path to WAV file.

    Returns:
        Tuple of (signal_array, sample_rate) where signal is normalized to [-1, 1].

    Example:
        >>> signal, sr = wav_to_signal('test.wav')
        >>> signal.dtype
        dtype('float64')
    """
    sample_rate, data = wavfile.read(str(wav_path))

    # Convert to float in range [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)

    return data, sample_rate


def convert_vibration_to_audio(
    signal_data: np.ndarray,
    output_path: Union[str, Path],
    channel: str = "horizontal",
    config: AudioConfig | None = None,
) -> Path:
    """Convert bearing vibration signal to WAV audio file.

    High-level convenience function that handles the full conversion pipeline:
    channel selection, resampling, normalization, and WAV writing.

    Args:
        signal_data: Raw vibration signal array of shape (samples, 2)
            where channel 0 is horizontal and channel 1 is vertical.
        output_path: Output WAV file path.
        channel: Which channel to use:
            - "horizontal": Use channel 0 (default)
            - "vertical": Use channel 1
            - "both" or "stereo": Use both channels for stereo output
        config: Audio configuration. Uses defaults if None.

    Returns:
        Path to the generated WAV file.

    Example:
        >>> from src.data.loader import XJTUBearingLoader
        >>> loader = XJTUBearingLoader()
        >>> signal, _ = loader.load_file('35Hz12kN', 'Bearing1_1', '1.csv')
        >>> wav_path = convert_vibration_to_audio(signal, 'bearing.wav')
    """
    if config is None:
        config = AudioConfig()

    # Select channel(s)
    if channel == "horizontal":
        audio_data = signal_data[:, 0]
    elif channel == "vertical":
        audio_data = signal_data[:, 1]
    elif channel in ("both", "stereo"):
        audio_data = signal_data
    else:
        raise ValueError(f"Invalid channel '{channel}'. Use 'horizontal', 'vertical', or 'both'.")

    # Resample
    resampled = resample_signal(
        audio_data,
        source_rate=config.source_rate,
        target_rate=config.target_rate,
    )

    # Write WAV
    return signal_to_wav(
        resampled,
        output_path,
        sample_rate=config.target_rate,
        normalize=config.normalize,
        target_peak=config.target_peak,
        bit_depth=config.bit_depth,
    )


def generate_bearing_audio(
    condition: str,
    bearing_id: str,
    file_indices: list[int],
    output_dir: Union[str, Path],
    data_loader=None,
    config: AudioConfig | None = None,
) -> list[Path]:
    """Generate WAV audio files for specific bearing lifecycle stages.

    Loads vibration data for the given file indices and converts each to a
    WAV audio file. By default, normalization is disabled to preserve
    relative amplitude differences between lifecycle stages.

    Args:
        condition: Operating condition (e.g., "35Hz12kN").
        bearing_id: Bearing identifier (e.g., "Bearing1_1").
        file_indices: List of file numbers (CSV filename stems, e.g., [1, 60, 120]).
        output_dir: Directory to write WAV files into.
        data_loader: XJTUBearingLoader instance (required).
        config: Audio configuration. Defaults to AudioConfig(normalize=False).

    Returns:
        List of paths to generated WAV files.

    Raises:
        ValueError: If data_loader is None.
    """
    if data_loader is None:
        raise ValueError("data_loader is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = AudioConfig(normalize=False)

    paths: list[Path] = []
    for file_idx in file_indices:
        file_path = data_loader.data_root / condition / bearing_id / f"{file_idx}.csv"
        signal = data_loader.load_file(file_path)

        output_path = output_dir / f"{bearing_id}_{file_idx}.wav"
        wav_path = convert_vibration_to_audio(signal, output_path, config=config)
        paths.append(wav_path)

    return paths


def get_resampled_duration_ms(
    num_samples: int,
    source_rate: int = SOURCE_SAMPLE_RATE,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> float:
    """Calculate duration in milliseconds after resampling.

    Args:
        num_samples: Number of samples in original signal.
        source_rate: Original sample rate.
        target_rate: Target sample rate.

    Returns:
        Duration in milliseconds (unchanged by resampling, just for convenience).
    """
    return (num_samples / source_rate) * 1000


def get_resampled_num_samples(
    num_samples: int,
    source_rate: int = SOURCE_SAMPLE_RATE,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> int:
    """Calculate number of samples after resampling.

    Args:
        num_samples: Number of samples in original signal.
        source_rate: Original sample rate.
        target_rate: Target sample rate.

    Returns:
        Number of samples in resampled signal.
    """
    g = gcd(source_rate, target_rate)
    up = target_rate // g
    down = source_rate // g
    return int(np.ceil(num_samples * up / down))
