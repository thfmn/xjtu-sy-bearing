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
