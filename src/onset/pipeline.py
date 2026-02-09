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

import numpy as np

from src.onset.detectors import BaseOnsetDetector, OnsetResult

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
                - ML classifier: 3-D array
                  (n_windows, window_size, n_features).

        Returns:
            OnsetResult with onset_idx and confidence.
        """
        if self.onset_model is not None:
            return self._detect_onset_ml(bearing_signals)
        return self.onset_detector.detect(bearing_signals)

    def _detect_onset_ml(self, bearing_signals: np.ndarray) -> OnsetResult:
        """Detect onset using the ML onset classifier.

        The classifier outputs P(degraded) for each window. Onset is the
        first window where P(degraded) > 0.5.

        Args:
            bearing_signals: 3-D array (n_windows, window_size, n_features).

        Returns:
            OnsetResult with onset_idx set to the first degraded window index.
        """
        preds = self.onset_model.predict(bearing_signals, verbose=0)
        probs = np.asarray(preds).ravel()  # (n_windows,)

        degraded_mask = probs > 0.5
        if not np.any(degraded_mask):
            return OnsetResult(
                onset_idx=None,
                onset_time=None,
                confidence=0.0,
                healthy_baseline={"method": "ml_classifier"},
            )

        onset_idx = int(np.argmax(degraded_mask))
        confidence = float(probs[onset_idx])
        return OnsetResult(
            onset_idx=onset_idx,
            onset_time=float(onset_idx),
            confidence=confidence,
            healthy_baseline={"method": "ml_classifier"},
        )

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
        n_samples = bearing_signals.shape[0]

        # No onset detected → all samples are healthy
        if onset_idx is None:
            return np.full(n_samples, self.max_rul, dtype=np.float32)

        # Onset at start → all samples go through RUL model
        if onset_idx <= 0:
            preds = self.rul_model.predict(bearing_signals, verbose=0)
            return np.asarray(preds).ravel().astype(np.float32)

        # Mixed: pre-onset gets max_rul, post-onset gets model predictions
        rul = np.full(n_samples, self.max_rul, dtype=np.float32)
        post_onset = bearing_signals[onset_idx:]
        if post_onset.shape[0] > 0:
            preds = self.rul_model.predict(post_onset, verbose=0)
            rul[onset_idx:] = np.asarray(preds).ravel()
        return rul

    def predict(
        self,
        onset_signals: np.ndarray,
        rul_signals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Full two-stage prediction: detect onset, then predict RUL.

        Chains ``detect_onset()`` and ``predict_rul()`` for end-to-end
        inference on a single bearing's data.

        Stage 1 and Stage 2 typically operate on different data
        representations (e.g. 1-D HI series for rule-based detectors,
        3-D windowed features for ML classifiers, spectrograms for RUL
        models).  Use ``rul_signals`` to pass a separate input to the
        RUL model.

        Args:
            onset_signals: Input for Stage 1 onset detection.  For
                rule-based detectors this should be a 1-D composite HI
                array; for ML classifiers a 3-D windowed feature array
                ``(n_windows, window_size, n_features)``.
            rul_signals: Input for Stage 2 RUL prediction.  Shape must
                be compatible with ``rul_model.predict()``.  If ``None``,
                ``onset_signals`` is reused (useful when both stages
                accept the same representation).

        Returns:
            1-D numpy array of RUL predictions, one per sample.
        """
        onset_result = self.detect_onset(onset_signals)
        rul_input = rul_signals if rul_signals is not None else onset_signals
        return self.predict_rul(rul_input, onset_result.onset_idx)
