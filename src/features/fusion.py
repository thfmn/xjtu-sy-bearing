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

"""Unified Feature Extraction Module for Bearing Vibration Signals.

This module provides a unified interface for extracting all feature types
from the XJTU-SY bearing dataset, combining time-domain and frequency-domain
features into a single pipeline.

Supported extraction modes:
    - "time": Time-domain features only (37 features)
    - "freq": Frequency-domain features only (28 features)
    - "all": All handcrafted features (65 features)

The FeatureExtractor class provides:
    - Unified interface for all feature types
    - Consistent output format (numpy arrays or dictionaries)
    - Feature name mapping for interpretability
    - Batch processing support
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

from src.features import frequency_domain, time_domain

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ExtractionMode(Enum):
    """Feature extraction mode."""

    TIME = "time"
    FREQ = "freq"
    ALL = "all"


@dataclass
class FeatureInfo:
    """Information about extracted features."""

    names: list[str]
    n_features: int
    mode: ExtractionMode

    def __repr__(self) -> str:
        return f"FeatureInfo(mode={self.mode.value}, n_features={self.n_features})"


class FeatureExtractor:
    """Unified feature extractor for bearing vibration signals.

    Provides a single interface to extract time-domain, frequency-domain,
    or all features from dual-channel vibration signals.

    Attributes:
        mode: Extraction mode ("time", "freq", or "all").
        condition: Operating condition for frequency calculations (optional).
        sampling_rate: Signal sampling rate in Hz.

    Example:
        >>> extractor = FeatureExtractor(mode="all", condition="35Hz12kN")
        >>> features = extractor.extract(signal)  # shape (32768, 2)
        >>> print(features.shape)  # (65,)
        >>> feature_dict = extractor.extract_dict(signal)
        >>> print(feature_dict["h_rms"])
    """

    def __init__(
        self,
        mode: Literal["time", "freq", "all"] = "all",
        condition: str | None = None,
        sampling_rate: float = 25600.0,
    ) -> None:
        """Initialize the feature extractor.

        Args:
            mode: Extraction mode - "time" for time-domain only,
                "freq" for frequency-domain only, "all" for both.
            condition: Operating condition (e.g., "35Hz12kN") for bearing
                characteristic frequency calculation. Required for freq/all
                modes to get meaningful characteristic band powers.
            sampling_rate: Signal sampling rate in Hz (default: 25600).
        """
        self.mode = ExtractionMode(mode)
        self.condition = condition
        self.sampling_rate = sampling_rate

        # Cache feature names based on mode
        self._feature_names = self._build_feature_names()
        self._n_features = len(self._feature_names)

    def _build_feature_names(self) -> list[str]:
        """Build the list of feature names based on extraction mode."""
        if self.mode == ExtractionMode.TIME:
            return time_domain.get_feature_names()
        elif self.mode == ExtractionMode.FREQ:
            return frequency_domain.get_feature_names()
        else:  # ALL
            time_names = time_domain.get_feature_names()
            freq_names = frequency_domain.get_feature_names()
            return time_names + freq_names

    @property
    def feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return self._feature_names.copy()

    @property
    def n_features(self) -> int:
        """Get total number of features."""
        return self._n_features

    def get_info(self) -> FeatureInfo:
        """Get information about the feature extraction configuration.

        Returns:
            FeatureInfo object with names, count, and mode.
        """
        return FeatureInfo(
            names=self.feature_names,
            n_features=self.n_features,
            mode=self.mode,
        )

    def extract(self, signal: NDArray) -> NDArray:
        """Extract features from a dual-channel signal.

        Args:
            signal: Signal array of shape (samples, 2) or (batch, samples, 2).
                Channel 0 = horizontal, Channel 1 = vertical.

        Returns:
            Feature array of shape (n_features,) or (batch, n_features).
        """
        is_single = signal.ndim == 2
        if is_single:
            signal = signal.reshape(1, signal.shape[0], signal.shape[1])

        if self.mode == ExtractionMode.TIME:
            features = self._extract_time(signal)
        elif self.mode == ExtractionMode.FREQ:
            features = self._extract_freq(signal)
        else:  # ALL
            time_feats = self._extract_time(signal)
            freq_feats = self._extract_freq(signal)
            features = np.concatenate([time_feats, freq_feats], axis=-1)

        if is_single:
            features = features.squeeze(0)

        return features.astype(np.float32)

    def _extract_time(self, signal: NDArray) -> NDArray:
        """Extract time-domain features.

        Args:
            signal: Signal array of shape (batch, samples, 2).

        Returns:
            Feature array of shape (batch, 37).
        """
        return time_domain.extract_features(signal)

    def _extract_freq(self, signal: NDArray) -> NDArray:
        """Extract frequency-domain features.

        Args:
            signal: Signal array of shape (batch, samples, 2).

        Returns:
            Feature array of shape (batch, 28).
        """
        return frequency_domain.extract_features(
            signal,
            condition=self.condition,
            sampling_rate=self.sampling_rate,
        )

    def extract_dict(self, signal: NDArray) -> dict[str, float]:
        """Extract features and return as a named dictionary.

        Args:
            signal: Signal array of shape (samples, 2).

        Returns:
            Dictionary mapping feature names to values.

        Raises:
            ValueError: If signal has batch dimension (must be single sample).
        """
        if signal.ndim != 2:
            raise ValueError(
                f"extract_dict requires single sample with shape (samples, 2), "
                f"got shape {signal.shape}"
            )
        features = self.extract(signal)
        return dict(zip(self._feature_names, features))

    def extract_dataframe(
        self, signal: NDArray, include_metadata: dict | None = None
    ) -> "pd.DataFrame":
        """Extract features and return as a pandas DataFrame.

        Args:
            signal: Signal array of shape (samples, 2) or (batch, samples, 2).
            include_metadata: Optional dictionary of metadata to prepend as columns.

        Returns:
            DataFrame with feature columns (and optional metadata columns).

        Note:
            Requires pandas to be installed.
        """
        import pandas as pd

        features = self.extract(signal)

        # Handle single vs batch
        if features.ndim == 1:
            features = features.reshape(1, -1)

        df = pd.DataFrame(features, columns=self._feature_names)

        # Prepend metadata columns if provided
        if include_metadata:
            for key, value in reversed(include_metadata.items()):
                if not isinstance(value, (list, np.ndarray)):
                    value = [value] * len(df)
                df.insert(0, key, value)

        return df


def extract_all_features(
    signal: NDArray,
    condition: str | None = None,
    sampling_rate: float = 25600.0,
) -> NDArray:
    """Convenience function to extract all features from a signal.

    This is a stateless wrapper around FeatureExtractor for simple use cases.

    Args:
        signal: Signal array of shape (samples, 2) or (batch, samples, 2).
        condition: Operating condition for characteristic frequencies.
        sampling_rate: Signal sampling rate in Hz.

    Returns:
        Feature array of shape (65,) or (batch, 65).
    """
    extractor = FeatureExtractor(
        mode="all",
        condition=condition,
        sampling_rate=sampling_rate,
    )
    return extractor.extract(signal)


def extract_all_features_dict(
    signal: NDArray,
    condition: str | None = None,
    sampling_rate: float = 25600.0,
) -> dict[str, float]:
    """Convenience function to extract all features as a dictionary.

    Args:
        signal: Signal array of shape (samples, 2).
        condition: Operating condition for characteristic frequencies.
        sampling_rate: Signal sampling rate in Hz.

    Returns:
        Dictionary mapping feature names to values.
    """
    extractor = FeatureExtractor(
        mode="all",
        condition=condition,
        sampling_rate=sampling_rate,
    )
    return extractor.extract_dict(signal)


def get_all_feature_names() -> list[str]:
    """Get names of all features in 'all' mode.

    Returns:
        List of 65 feature names.
    """
    return time_domain.get_feature_names() + frequency_domain.get_feature_names()


def get_feature_groups() -> dict[str, list[str]]:
    """Get feature names organized by group for interpretability.

    Returns:
        Dictionary mapping group names to lists of feature names.
    """
    return {
        "time_horizontal": [
            f"h_{name}" for name in time_domain.CHANNEL_FEATURES
        ],
        "time_vertical": [
            f"v_{name}" for name in time_domain.CHANNEL_FEATURES
        ],
        "time_cross": ["cross_correlation"],
        "freq_horizontal_spectral": [
            f"h_{name}" for name in frequency_domain.SPECTRAL_FEATURES
        ],
        "freq_horizontal_characteristic": [
            f"h_{name}" for name in frequency_domain.CHARACTERISTIC_FEATURES
        ],
        "freq_vertical_spectral": [
            f"v_{name}" for name in frequency_domain.SPECTRAL_FEATURES
        ],
        "freq_vertical_characteristic": [
            f"v_{name}" for name in frequency_domain.CHARACTERISTIC_FEATURES
        ],
    }


# Feature counts for reference
NUM_TIME_FEATURES = 37  # 18 per channel + cross-correlation
NUM_FREQ_FEATURES = 28  # 14 per channel
NUM_ALL_FEATURES = NUM_TIME_FEATURES + NUM_FREQ_FEATURES  # 65 total
