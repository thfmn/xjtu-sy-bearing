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

"""Two-stage degradation onset detection for bearing RUL prediction.

This module provides tools for detecting when bearing degradation begins,
enabling two-stage RUL prediction approaches that can significantly improve
accuracy over single-stage methods.

Components:
- Health indicators: Aggregate time-domain features into health indicator series
- Detectors: Threshold, CUSUM, Bayesian, and ensemble onset detection algorithms
- Labels: Load and generate onset labels for supervised learning
- Pipeline: Integrate onset detection with RUL prediction models
"""

from src.onset.detectors import (
    BaseOnsetDetector,
    BayesianOnsetDetector,
    CUSUMOnsetDetector,
    EnsembleOnsetDetector,
    EWMAOnsetDetector,
    OnsetResult,
    ThresholdOnsetDetector,
)
from src.onset.health_indicators import (
    BearingHealthSeries,
    compute_composite_hi,
    get_all_bearing_ids,
    load_all_bearings_health_series,
    load_bearing_health_series,
    smooth_health_indicator,
)
from src.onset.dataset import (
    OnsetDatasetResult,
    OnsetSplitResult,
    compute_class_weights,
    create_onset_dataset,
    split_by_bearing,
)
from src.onset.models import (
    OnsetClassifierConfig,
    build_onset_classifier,
    compile_onset_classifier,
    create_onset_classifier,
    predict_proba,
)
from src.onset.labels import (
    OnsetLabelEntry,
    add_onset_column,
    get_onset_label,
    load_onset_labels,
)
from src.onset.pipeline import (
    TwoStagePipeline,
)
from src.onset.visualization import (
    plot_all_bearings_onset,
    plot_bearing_onset,
    plot_onset_comparison,
)

__all__: list[str] = [
    # Health indicators
    "BearingHealthSeries",
    "compute_composite_hi",
    "get_all_bearing_ids",
    "load_all_bearings_health_series",
    "load_bearing_health_series",
    "smooth_health_indicator",
    # Detectors
    "BaseOnsetDetector",
    "BayesianOnsetDetector",
    "CUSUMOnsetDetector",
    "EnsembleOnsetDetector",
    "EWMAOnsetDetector",
    "OnsetResult",
    "ThresholdOnsetDetector",
    # Dataset
    "OnsetDatasetResult",
    "OnsetSplitResult",
    "compute_class_weights",
    "create_onset_dataset",
    "split_by_bearing",
    # Models
    "OnsetClassifierConfig",
    "build_onset_classifier",
    "compile_onset_classifier",
    "create_onset_classifier",
    "predict_proba",
    # Labels
    "OnsetLabelEntry",
    "add_onset_column",
    "get_onset_label",
    "load_onset_labels",
    # Pipeline
    "TwoStagePipeline",
    # Visualization
    "plot_all_bearings_onset",
    "plot_bearing_onset",
    "plot_onset_comparison",
]
