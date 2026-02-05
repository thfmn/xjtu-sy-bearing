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

__all__: list[str] = []
