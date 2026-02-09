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

from src.models.baselines.cnn1d_baseline import (
    CNN1DBaseline,
    CNN1DConfig,
    build_cnn1d_model,
    create_default_cnn1d,
    get_model_summary,
    print_model_summary,
)

from src.models.baselines.feature_lstm import (
    FeatureLSTMConfig,
    build_feature_lstm_model,
    create_default_feature_lstm,
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
    # 1D CNN baseline
    "CNN1DBaseline",
    "CNN1DConfig",
    "build_cnn1d_model",
    "create_default_cnn1d",
    "get_model_summary",
    "print_model_summary",
    # Feature LSTM
    "FeatureLSTMConfig",
    "build_feature_lstm_model",
    "create_default_feature_lstm",
]
