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
from src.onset.labels import (
    OnsetLabelEntry,
    add_onset_column,
    get_onset_label,
    load_onset_labels,
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
    "CUSUMOnsetDetector",
    "EnsembleOnsetDetector",
    "EWMAOnsetDetector",
    "OnsetResult",
    "ThresholdOnsetDetector",
    # Labels
    "OnsetLabelEntry",
    "add_onset_column",
    "get_onset_label",
    "load_onset_labels",
    # Visualization
    "plot_all_bearings_onset",
    "plot_bearing_onset",
    "plot_onset_comparison",
]
