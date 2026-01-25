"""Windowing utilities for bearing vibration signals.

This module provides windowing functionality for segmenting long time-series
signals into smaller windows for batch processing and model training.

Supported window sizes:
    - 2048 samples (80ms @ 25.6kHz)
    - 4096 samples (160ms @ 25.6kHz)
    - 8192 samples (320ms @ 25.6kHz)
    - 32768 samples (1.28s @ 25.6kHz, full file)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

# Standard window sizes as per PRD
WINDOW_SIZES = (2048, 4096, 8192, 32768)
DEFAULT_WINDOW_SIZE = 32768
SAMPLES_PER_FILE = 32768

WindowSize = Literal[2048, 4096, 8192, 32768]


@dataclass
class WindowConfig:
    """Configuration for signal windowing.

    Attributes:
        window_size: Number of samples per window.
        overlap: Fraction of overlap between consecutive windows (0.0 to 0.9).
        drop_last: Whether to drop the last window if it's incomplete.
    """

    window_size: int = DEFAULT_WINDOW_SIZE
    overlap: float = 0.0
    drop_last: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.window_size not in WINDOW_SIZES:
            raise ValueError(
                f"window_size must be one of {WINDOW_SIZES}, got {self.window_size}"
            )
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")

    @property
    def hop_size(self) -> int:
        """Number of samples to advance between windows."""
        return int(self.window_size * (1 - self.overlap))

    def num_windows(self, signal_length: int) -> int:
        """Calculate number of windows for a signal.

        Args:
            signal_length: Total number of samples in the signal.

        Returns:
            Number of windows that can be extracted.
        """
        if signal_length < self.window_size:
            return 0

        if self.overlap == 0.0:
            if self.drop_last:
                return signal_length // self.window_size
            return (signal_length + self.window_size - 1) // self.window_size

        num_full = (signal_length - self.window_size) // self.hop_size + 1
        return num_full


def extract_windows(
    signal: np.ndarray,
    config: WindowConfig | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: float = 0.0,
) -> np.ndarray:
    """Extract windows from a signal using sliding window.

    Args:
        signal: Input signal of shape (samples,) or (samples, channels).
        config: WindowConfig object. If provided, overrides window_size and overlap.
        window_size: Number of samples per window (if config not provided).
        overlap: Fraction of overlap between windows (if config not provided).

    Returns:
        Array of shape (num_windows, window_size) or (num_windows, window_size, channels).

    Raises:
        ValueError: If signal is too short for even one window.
    """
    if config is None:
        config = WindowConfig(window_size=window_size, overlap=overlap)

    signal = np.asarray(signal)
    is_multichannel = signal.ndim == 2

    if is_multichannel:
        signal_length = signal.shape[0]
        num_channels = signal.shape[1]
    else:
        signal_length = len(signal)
        num_channels = 1

    num_windows = config.num_windows(signal_length)

    if num_windows == 0:
        raise ValueError(
            f"Signal length {signal_length} is too short for window size {config.window_size}"
        )

    # Pre-allocate output array
    if is_multichannel:
        windows = np.zeros((num_windows, config.window_size, num_channels), dtype=signal.dtype)
    else:
        windows = np.zeros((num_windows, config.window_size), dtype=signal.dtype)

    # Extract windows
    for i in range(num_windows):
        start = i * config.hop_size
        end = start + config.window_size
        windows[i] = signal[start:end]

    return windows


def iter_windows(
    signal: np.ndarray,
    config: WindowConfig | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: float = 0.0,
) -> Iterator[tuple[np.ndarray, int]]:
    """Iterate over windows from a signal (memory-efficient).

    Args:
        signal: Input signal of shape (samples,) or (samples, channels).
        config: WindowConfig object. If provided, overrides window_size and overlap.
        window_size: Number of samples per window (if config not provided).
        overlap: Fraction of overlap between windows (if config not provided).

    Yields:
        Tuple of (window, start_index) for each extracted window.
    """
    if config is None:
        config = WindowConfig(window_size=window_size, overlap=overlap)

    signal = np.asarray(signal)
    signal_length = signal.shape[0]
    num_windows = config.num_windows(signal_length)

    for i in range(num_windows):
        start = i * config.hop_size
        end = start + config.window_size
        yield signal[start:end], start


def window_with_labels(
    signal: np.ndarray,
    rul: float,
    config: WindowConfig | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract windows from a signal and replicate RUL label.

    For a single file with a single RUL value, all windows from that file
    share the same RUL label.

    Args:
        signal: Input signal of shape (samples, channels).
        rul: RUL label for this signal (shared by all windows).
        config: WindowConfig object. If provided, overrides window_size and overlap.
        window_size: Number of samples per window (if config not provided).
        overlap: Fraction of overlap between windows (if config not provided).

    Returns:
        Tuple of:
            - windows: Array of shape (num_windows, window_size, channels)
            - labels: Array of shape (num_windows,) with replicated RUL values
    """
    windows = extract_windows(signal, config=config, window_size=window_size, overlap=overlap)
    labels = np.full(len(windows), rul, dtype=np.float32)
    return windows, labels


def calculate_num_windows_per_file(
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: float = 0.0,
    file_samples: int = SAMPLES_PER_FILE,
) -> int:
    """Calculate how many windows can be extracted from a single file.

    Args:
        window_size: Number of samples per window.
        overlap: Fraction of overlap between windows.
        file_samples: Total samples in a file (default: 32768).

    Returns:
        Number of windows per file.

    Example:
        >>> calculate_num_windows_per_file(32768, overlap=0.0)
        1
        >>> calculate_num_windows_per_file(8192, overlap=0.0)
        4
        >>> calculate_num_windows_per_file(8192, overlap=0.5)
        7
    """
    config = WindowConfig(window_size=window_size, overlap=overlap)
    return config.num_windows(file_samples)


def get_window_duration_ms(window_size: int, sampling_rate: int = 25600) -> float:
    """Get the duration of a window in milliseconds.

    Args:
        window_size: Number of samples in the window.
        sampling_rate: Sampling rate in Hz (default: 25600).

    Returns:
        Window duration in milliseconds.
    """
    return (window_size / sampling_rate) * 1000
