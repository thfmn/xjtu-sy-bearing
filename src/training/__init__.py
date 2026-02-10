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

"""Training and evaluation utilities."""

from src.training.cv import (
    CONDITIONS,
    BEARING_IDS,
    BEARINGS_PER_CONDITION,
    CVFold,
    CVSplit,
    generate_cv_folds,
    get_bearing_groups,
    leave_one_bearing_out,
    leave_one_condition_out,
    loco_per_bearing,
    stratified_split,
    time_series_split,
    validate_no_leakage,
)
from src.training.metrics import (
    MetricsSummary,
    aggregate_bearing_metrics,
    compute_summary,
    evaluate_predictions,
    mae,
    mape,
    per_bearing_metrics,
    phm08_score,
    phm08_score_normalized,
    print_evaluation_report,
    rmse,
)
from src.training.config import (
    OptimizerConfig,
    LossConfig,
    CallbackConfig,
    TrainingConfig,
    MLflowCallback,
    build_callbacks,
    compile_model,
    DEFAULT_CONFIG,
)

__all__ = [
    # CV utilities
    "CONDITIONS",
    "BEARING_IDS",
    "BEARINGS_PER_CONDITION",
    "CVFold",
    "CVSplit",
    "generate_cv_folds",
    "get_bearing_groups",
    "leave_one_bearing_out",
    "leave_one_condition_out",
    "loco_per_bearing",
    "stratified_split",
    "time_series_split",
    "validate_no_leakage",
    # Metrics
    "rmse",
    "mae",
    "mape",
    "phm08_score",
    "phm08_score_normalized",
    "MetricsSummary",
    "compute_summary",
    "evaluate_predictions",
    "per_bearing_metrics",
    "aggregate_bearing_metrics",
    "print_evaluation_report",
    # Training config
    "OptimizerConfig",
    "LossConfig",
    "CallbackConfig",
    "TrainingConfig",
    "MLflowCallback",
    "build_callbacks",
    "compile_model",
    "DEFAULT_CONFIG",
]
