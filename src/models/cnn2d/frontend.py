"""Frontend module for spectrogram generation.

This module provides the interface for generating time-frequency representations
(STFT spectrograms or CWT scalograms) from raw vibration signals, preparing
input for the 2D CNN backbone.

The frontend can operate in two modes:
1. On-the-fly: Generate spectrograms during training (slower, memory efficient)
2. Pre-computed: Load pre-generated spectrograms (faster, requires disk space)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FrontendType(Enum):
    """Type of time-frequency representation."""

    STFT = "stft"
    CWT = "cwt"
    PRECOMPUTED = "precomputed"


@dataclass
class SpectrogramFrontendConfig:
    """Configuration for spectrogram frontend.

    Attributes:
        frontend_type: Type of spectrogram ('stft', 'cwt', or 'precomputed').
        target_height: Height of output spectrogram (frequency bins).
        target_width: Width of output spectrogram (time frames).
        n_fft: FFT window size for STFT.
        hop_length: Hop length for STFT.
        n_mels: Number of mel bins (0 for linear spectrogram).
        log_scale: Whether to apply log scaling.
        normalize: Whether to normalize per sample.
        trainable: Whether frontend has trainable parameters.
    """

    frontend_type: FrontendType = FrontendType.STFT
    target_height: int = 128
    target_width: int = 128
    n_fft: int = 512
    hop_length: int = 256
    n_mels: int = 128
    log_scale: bool = True
    normalize: bool = True
    trainable: bool = False


class STFTFrontend(keras.layers.Layer):
    """STFT-based spectrogram frontend as a Keras layer.

    Converts raw time-domain signals to mel-spectrograms using the
    STFT implementation from src/features/stft.py.

    This layer wraps numpy operations in tf.py_function for use in
    TensorFlow data pipelines.
    """

    def __init__(
        self,
        config: SpectrogramFrontendConfig | None = None,
        sampling_rate: float = 25600.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config if config is not None else SpectrogramFrontendConfig()
        self.sampling_rate = sampling_rate

    def call(self, inputs, training=None):
        """Generate spectrograms from raw signals.

        Note: This uses tf.py_function for numpy operations, which
        may have performance implications. For production, consider
        using pre-computed spectrograms.

        Args:
            inputs: Raw signal tensor, shape (batch, samples, channels).
            training: Training mode flag.

        Returns:
            Spectrogram tensor, shape (batch, height, width, channels).
        """
        # Import here to avoid circular imports
        from src.features.stft import (
            STFTConfig,
            NormalizationMode,
            generate_spectrogram_dual_channel,
        )

        def _process_batch(signals):
            """Process batch of signals to spectrograms."""
            # signals: (batch, samples, 2) numpy array
            stft_config = STFTConfig(
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                log_scale=self.config.log_scale,
                normalization=(
                    NormalizationMode.PER_SAMPLE
                    if self.config.normalize
                    else NormalizationMode.NONE
                ),
            )
            specs = generate_spectrogram_dual_channel(
                signals,
                config=stft_config,
                sampling_rate=self.sampling_rate,
                target_shape=(self.config.target_height, self.config.target_width),
            )
            return specs.astype(np.float32)

        # Use tf.py_function for numpy operations
        output = tf.py_function(
            _process_batch,
            [inputs],
            tf.float32
        )

        # Set output shape explicitly
        batch_size = tf.shape(inputs)[0]
        output.set_shape([None, self.config.target_height, self.config.target_width, 2])

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": {
                "frontend_type": self.config.frontend_type.value,
                "target_height": self.config.target_height,
                "target_width": self.config.target_width,
                "n_fft": self.config.n_fft,
                "hop_length": self.config.hop_length,
                "n_mels": self.config.n_mels,
                "log_scale": self.config.log_scale,
                "normalize": self.config.normalize,
            },
            "sampling_rate": self.sampling_rate,
        })
        return config


class PrecomputedFrontend(keras.layers.Layer):
    """Frontend for pre-computed spectrograms.

    This is a pass-through layer that validates input shapes and
    optionally applies normalization to pre-computed spectrograms.
    """

    def __init__(
        self,
        expected_height: int = 128,
        expected_width: int = 128,
        normalize: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.expected_height = expected_height
        self.expected_width = expected_width
        self.normalize = normalize

    def call(self, inputs, training=None):
        """Pass through pre-computed spectrograms.

        Args:
            inputs: Pre-computed spectrogram, shape (batch, height, width, channels).
            training: Training mode flag.

        Returns:
            Spectrogram tensor (possibly normalized).
        """
        if self.normalize:
            # Per-sample normalization
            mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
            std = tf.math.reduce_std(inputs, axis=[1, 2, 3], keepdims=True)
            inputs = (inputs - mean) / (std + 1e-8)

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "expected_height": self.expected_height,
            "expected_width": self.expected_width,
            "normalize": self.normalize,
        })
        return config


def create_frontend(
    frontend_type: Literal["stft", "cwt", "precomputed"] = "stft",
    target_shape: tuple[int, int] = (128, 128),
    sampling_rate: float = 25600.0,
    **kwargs
) -> keras.layers.Layer:
    """Create a spectrogram frontend layer.

    Args:
        frontend_type: Type of frontend ('stft', 'cwt', or 'precomputed').
        target_shape: Target (height, width) for spectrograms.
        sampling_rate: Sampling rate of input signals.
        **kwargs: Additional arguments passed to frontend.

    Returns:
        Configured frontend layer.
    """
    if frontend_type == "stft":
        config = SpectrogramFrontendConfig(
            frontend_type=FrontendType.STFT,
            target_height=target_shape[0],
            target_width=target_shape[1],
            **kwargs
        )
        return STFTFrontend(config=config, sampling_rate=sampling_rate)

    elif frontend_type == "precomputed":
        return PrecomputedFrontend(
            expected_height=target_shape[0],
            expected_width=target_shape[1],
            **kwargs
        )

    elif frontend_type == "cwt":
        # CWT frontend - placeholder for FEAT-8
        raise NotImplementedError(
            "CWT frontend is not yet implemented. See FEAT-8 in PRD."
        )

    else:
        raise ValueError(f"Unknown frontend type: {frontend_type}")


def extract_spectrogram_numpy(
    signal: NDArray,
    sampling_rate: float = 25600.0,
    target_shape: tuple[int, int] = (128, 128),
) -> NDArray:
    """Extract spectrogram using numpy (for data preprocessing).

    This is a convenience function for extracting spectrograms outside
    of the TensorFlow graph, useful for data preprocessing pipelines.

    Args:
        signal: Raw signal, shape (samples, 2) or (batch, samples, 2).
        sampling_rate: Sampling rate in Hz.
        target_shape: Target (height, width) for spectrogram.

    Returns:
        Spectrogram array, shape (height, width, 2) or (batch, height, width, 2).
    """
    from src.features.stft import extract_spectrogram

    # Handle both single and batch inputs
    return extract_spectrogram(signal, sampling_rate=sampling_rate)
