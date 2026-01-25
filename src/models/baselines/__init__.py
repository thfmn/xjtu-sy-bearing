"""Baseline models for RUL prediction."""

from src.models.baselines.trending import (
    HealthIndicatorFusion,
    KurtosisTrendingBaseline,
    RMSThresholdBaseline,
    TrendingPrediction,
    evaluate_trending_baseline,
    predict_all_bearings,
)

from src.models.baselines.lgbm_baseline import (
    CVResult,
    LGBMConfig,
    LGBMPrediction,
    LightGBMBaseline,
    evaluate_lgbm_cv,
    get_feature_columns,
    plot_feature_importance,
    plot_shap_summary,
    train_with_cv,
)

__all__ = [
    # Trending baselines
    "HealthIndicatorFusion",
    "KurtosisTrendingBaseline",
    "RMSThresholdBaseline",
    "TrendingPrediction",
    "evaluate_trending_baseline",
    "predict_all_bearings",
    # LightGBM baseline
    "CVResult",
    "LGBMConfig",
    "LGBMPrediction",
    "LightGBMBaseline",
    "evaluate_lgbm_cv",
    "get_feature_columns",
    "plot_feature_importance",
    "plot_shap_summary",
    "train_with_cv",
]
