"""RUL prediction metrics for bearing prognostics.

This module implements standard metrics for Remaining Useful Life (RUL) prediction,
including the PHM08 asymmetric scoring function that penalizes late predictions
more heavily than early predictions.

Reference:
    Saxena, A., et al. "Damage propagation modeling for aircraft engine run-to-failure
    simulation." PHM 2008.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        RMSE value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        MAE value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.
        epsilon: Small value to avoid division by zero.

    Returns:
        MAPE value as a percentage (0-100 scale).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Avoid division by zero for RUL=0 samples
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def phm08_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a1: float = 13.0,
    a2: float = 10.0,
) -> float:
    """PHM08 asymmetric scoring function for RUL prediction.

    The PHM08 score penalizes late predictions (underestimating RUL) more heavily
    than early predictions (overestimating RUL). This reflects the real-world
    consequence that predicting failure too late is more dangerous than too early.

    Score formula:
        - If prediction is early (d < 0): exp(-d/a1) - 1
        - If prediction is late (d >= 0): exp(d/a2) - 1

    where d = y_pred - y_true (negative means early, positive means late).

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.
        a1: Parameter for early predictions (higher = less penalty). Default 13.
        a2: Parameter for late predictions (higher = less penalty). Default 10.

    Returns:
        Sum of individual scores (lower is better).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    d = y_pred - y_true  # Negative = early prediction, Positive = late prediction

    scores = np.where(
        d < 0,
        np.exp(-d / a1) - 1,  # Early: less severe penalty
        np.exp(d / a2) - 1,  # Late: more severe penalty
    )
    return float(np.sum(scores))


def phm08_score_normalized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a1: float = 13.0,
    a2: float = 10.0,
) -> float:
    """Normalized PHM08 score (per-sample average).

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.
        a1: Parameter for early predictions.
        a2: Parameter for late predictions.

    Returns:
        Average score per sample (lower is better).
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    return phm08_score(y_true, y_pred, a1, a2) / n


@dataclass
class MetricsSummary:
    """Summary statistics for a set of metric values."""

    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int

    def __repr__(self) -> str:
        return (
            f"MetricsSummary(mean={self.mean:.4f}, std={self.std:.4f}, "
            f"min={self.min:.4f}, max={self.max:.4f}, n={self.count})"
        )


def compute_summary(values: np.ndarray) -> MetricsSummary:
    """Compute summary statistics for a set of values.

    Args:
        values: Array of metric values.

    Returns:
        MetricsSummary with mean, std, min, max, median, count.
    """
    values = np.asarray(values)
    return MetricsSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        median=float(np.median(values)),
        count=len(values),
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all metrics for RUL predictions.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Dictionary with RMSE, MAE, MAPE, PHM08 score, and normalized PHM08 score.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "phm08_score": phm08_score(y_true, y_pred),
        "phm08_score_normalized": phm08_score_normalized(y_true, y_pred),
    }


def per_bearing_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bearing_ids: np.ndarray,
) -> pd.DataFrame:
    """Compute metrics broken down by bearing.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.
        bearing_ids: Array of bearing identifiers for each sample.

    Returns:
        DataFrame with one row per bearing and columns for each metric.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bearing_ids = np.asarray(bearing_ids)

    results = []
    unique_bearings = np.unique(bearing_ids)

    for bearing_id in unique_bearings:
        mask = bearing_ids == bearing_id
        y_true_b = y_true[mask]
        y_pred_b = y_pred[mask]

        metrics = evaluate_predictions(y_true_b, y_pred_b)
        metrics["bearing_id"] = bearing_id
        metrics["n_samples"] = int(np.sum(mask))
        results.append(metrics)

    df = pd.DataFrame(results)
    # Reorder columns to put bearing_id first
    cols = ["bearing_id", "n_samples", "rmse", "mae", "mape", "phm08_score", "phm08_score_normalized"]
    return df[cols]


def aggregate_bearing_metrics(
    per_bearing_df: pd.DataFrame,
) -> dict[str, MetricsSummary]:
    """Aggregate per-bearing metrics into summary statistics.

    Args:
        per_bearing_df: DataFrame from per_bearing_metrics().

    Returns:
        Dictionary mapping metric names to MetricsSummary objects.
    """
    metric_cols = ["rmse", "mae", "mape", "phm08_score", "phm08_score_normalized"]
    summaries = {}

    for col in metric_cols:
        if col in per_bearing_df.columns:
            summaries[col] = compute_summary(per_bearing_df[col].values)

    return summaries


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bearing_ids: Optional[np.ndarray] = None,
) -> None:
    """Print a formatted evaluation report.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.
        bearing_ids: Optional array of bearing identifiers for per-bearing breakdown.
    """
    print("=" * 60)
    print("RUL Prediction Evaluation Report")
    print("=" * 60)

    # Overall metrics
    overall = evaluate_predictions(y_true, y_pred)
    print("\nOverall Metrics:")
    print(f"  RMSE:                 {overall['rmse']:.4f}")
    print(f"  MAE:                  {overall['mae']:.4f}")
    print(f"  MAPE:                 {overall['mape']:.2f}%")
    print(f"  PHM08 Score:          {overall['phm08_score']:.4f}")
    print(f"  PHM08 Score (norm):   {overall['phm08_score_normalized']:.4f}")
    print(f"  N samples:            {len(y_true)}")

    # Per-bearing breakdown if bearing_ids provided
    if bearing_ids is not None:
        print("\n" + "-" * 60)
        print("Per-Bearing Breakdown:")
        print("-" * 60)

        per_bearing_df = per_bearing_metrics(y_true, y_pred, bearing_ids)
        print(per_bearing_df.to_string(index=False))

        # Summary statistics
        print("\n" + "-" * 60)
        print("Summary Statistics Across Bearings:")
        print("-" * 60)

        summaries = aggregate_bearing_metrics(per_bearing_df)
        for metric_name, summary in summaries.items():
            print(f"\n  {metric_name}:")
            print(f"    Mean:   {summary.mean:.4f}")
            print(f"    Std:    {summary.std:.4f}")
            print(f"    Min:    {summary.min:.4f}")
            print(f"    Max:    {summary.max:.4f}")
            print(f"    Median: {summary.median:.4f}")

    print("\n" + "=" * 60)
