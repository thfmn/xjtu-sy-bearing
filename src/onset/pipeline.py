"""Two-stage pipeline integrating onset detection with RUL prediction.

This module implements the full two-stage approach for bearing remaining
useful life prediction:

1. **Stage 1 (Onset Detection):** Identifies when degradation begins using
   either a rule-based detector (ThresholdOnsetDetector, CUSUMOnsetDetector,
   etc.) or a trained ML onset classifier model.

2. **Stage 2 (RUL Prediction):** Applies a RUL regression model only to
   post-onset samples. Pre-onset samples receive max_rul as their prediction.

Usage:
    pipeline = TwoStagePipeline(
        onset_detector=ThresholdOnsetDetector(threshold_sigma=3.0),
        rul_model=loaded_rul_keras_model,
    )
    rul_predictions = pipeline.predict(bearing_signals)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.onset.detectors import BaseOnsetDetector, OnsetResult

if TYPE_CHECKING:
    pass

# Default maximum RUL value assigned to pre-onset (healthy) samples.
# Matches the piecewise-linear RUL labeling convention used throughout
# the XJTU-SY pipeline.
DEFAULT_MAX_RUL = 125


class TwoStagePipeline:
    """Two-stage pipeline: onset detection followed by RUL prediction.

    Stage 1 uses a rule-based onset detector (any ``BaseOnsetDetector``
    subclass) or an ML onset classifier (Keras model with sigmoid output).
    Stage 2 uses an existing RUL regression model (Keras) applied only to
    samples identified as post-onset.

    Pre-onset samples are assigned ``max_rul`` without querying the RUL
    model.  This avoids noisy predictions on healthy samples and focuses
    the RUL model on the degradation trajectory.

    Args:
        onset_detector: A fitted ``BaseOnsetDetector`` instance for
            rule-based onset detection. Required.
        rul_model: A compiled Keras model for RUL regression. Its
            ``predict()`` method will be called on post-onset samples.
        onset_model: An optional compiled Keras onset classifier model
            (sigmoid output). When provided, ``detect_onset()`` uses
            the ML classifier instead of the rule-based detector.
        max_rul: Maximum RUL value for pre-onset samples. Default 125.
    """

    def __init__(
        self,
        onset_detector: BaseOnsetDetector,
        rul_model,
        onset_model=None,
        max_rul: int = DEFAULT_MAX_RUL,
    ) -> None:
        if not isinstance(onset_detector, BaseOnsetDetector):
            raise TypeError(
                f"onset_detector must be a BaseOnsetDetector, "
                f"got {type(onset_detector).__name__}"
            )
        if rul_model is None:
            raise ValueError("rul_model must not be None")

        self.onset_detector = onset_detector
        self.rul_model = rul_model
        self.onset_model = onset_model
        self.max_rul = max_rul

    def detect_onset(self, bearing_signals: np.ndarray) -> OnsetResult:
        """Detect the onset of degradation in a bearing's signal.

        Uses the ML onset classifier if ``onset_model`` was provided,
        otherwise falls back to the rule-based ``onset_detector``.

        Args:
            bearing_signals: Health indicator time series for a single
                bearing. Shape depends on which detector is used:
                - Rule-based: 1-D array of composite health indicator values.
                - ML classifier: 2-D array (n_samples, n_features) or 3-D
                  (n_windows, window_size, n_features).

        Returns:
            OnsetResult with onset_idx and confidence.
        """
        raise NotImplementedError("detect_onset will be implemented in ONSET-18 checkbox 2")

    def predict_rul(
        self,
        bearing_signals: np.ndarray,
        onset_idx: int | None,
    ) -> np.ndarray:
        """Predict RUL for a bearing given a known onset index.

        Pre-onset samples receive ``max_rul``.  Post-onset samples are
        fed to the RUL model for prediction.

        Args:
            bearing_signals: Input array suitable for ``rul_model.predict()``.
                Shape depends on the RUL model architecture.
            onset_idx: Index where degradation begins. If ``None``, all
                samples receive ``max_rul`` (no degradation detected).

        Returns:
            1-D numpy array of RUL predictions, one per sample.
        """
        raise NotImplementedError("predict_rul will be implemented in ONSET-18 checkbox 3")

    def predict(self, bearing_signals: np.ndarray) -> np.ndarray:
        """Full two-stage prediction: detect onset, then predict RUL.

        Chains ``detect_onset()`` and ``predict_rul()`` for end-to-end
        inference on a single bearing's data.

        Args:
            bearing_signals: Input data for a single bearing.  For
                rule-based detectors this should be a 1-D composite HI
                array; for ML classifiers a windowed feature array.

        Returns:
            1-D numpy array of RUL predictions, one per sample.
        """
        raise NotImplementedError("predict will be implemented in ONSET-18 checkbox 4")
