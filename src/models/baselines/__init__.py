"""Baseline models for RUL prediction."""

from src.models.baselines.trending import (
    HealthIndicatorFusion,
    KurtosisTrendingBaseline,
    RMSThresholdBaseline,
    TrendingPrediction,
    evaluate_trending_baseline,
    predict_all_bearings,
)

__all__ = [
    "HealthIndicatorFusion",
    "KurtosisTrendingBaseline",
    "RMSThresholdBaseline",
    "TrendingPrediction",
    "evaluate_trending_baseline",
    "predict_all_bearings",
]
