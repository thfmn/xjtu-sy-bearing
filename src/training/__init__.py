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
]
