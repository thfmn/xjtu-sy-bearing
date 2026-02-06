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


def onset_detection_metrics(
    y_true_onset: np.ndarray,
    y_pred_onset: np.ndarray,
) -> dict[str, float]:
    """Compute metrics for onset detection (binary classification).

    Args:
        y_true_onset: Ground truth binary onset labels (0=healthy, 1=degraded).
        y_pred_onset: Predicted binary onset labels (0=healthy, 1=degraded).

    Returns:
        Dictionary with precision, recall, f1, and accuracy.
    """
    y_true = np.asarray(y_true_onset, dtype=int)
    y_pred = np.asarray(y_pred_onset, dtype=int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def onset_timing_mae(
    true_onset_indices: np.ndarray,
    pred_onset_indices: np.ndarray,
    samples_per_minute: float = 1.0,
) -> dict[str, float]:
    """Compute Mean Absolute Error of onset timing across bearings.

    Args:
        true_onset_indices: Per-bearing true onset indices (file index).
        pred_onset_indices: Per-bearing predicted onset indices (file index).
            Use np.nan or -1 for "no onset detected".
        samples_per_minute: Conversion factor from samples to minutes.
            Default 1.0 returns MAE in sample units.

    Returns:
        Dictionary with onset_timing_mae_samples, onset_timing_mae_minutes,
        and n_valid (number of bearings where both true and pred onset exist).
    """
    true_idx = np.asarray(true_onset_indices, dtype=float)
    pred_idx = np.asarray(pred_onset_indices, dtype=float)

    # Replace -1 with NaN for uniform handling
    true_idx = np.where(true_idx < 0, np.nan, true_idx)
    pred_idx = np.where(pred_idx < 0, np.nan, pred_idx)

    # Only compare where both have valid onset indices
    valid = ~np.isnan(true_idx) & ~np.isnan(pred_idx)
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return {
            "onset_timing_mae_samples": float("nan"),
            "onset_timing_mae_minutes": float("nan"),
            "n_valid": 0,
        }

    abs_errors = np.abs(true_idx[valid] - pred_idx[valid])
    mae_samples = float(np.mean(abs_errors))

    return {
        "onset_timing_mae_samples": mae_samples,
        "onset_timing_mae_minutes": mae_samples / samples_per_minute if samples_per_minute > 0 else float("nan"),
        "n_valid": n_valid,
    }


def conditional_rul_metrics(
    y_true_rul: np.ndarray,
    y_pred_rul: np.ndarray,
    onset_mask: np.ndarray,
) -> dict[str, float]:
    """Compute RUL metrics conditioned on onset detection.

    Computes MAE, RMSE, and PHM08 score only on post-onset samples
    (where onset_mask == True/1), excluding pre-onset samples that
    trivially receive max_rul predictions.

    Args:
        y_true_rul: Ground truth RUL values for all samples.
        y_pred_rul: Predicted RUL values for all samples.
        onset_mask: Boolean or binary array where True/1 indicates
            post-onset (degraded) samples.

    Returns:
        Dictionary with post_onset_mae, post_onset_rmse,
        post_onset_phm08, post_onset_n_samples, and total_n_samples.
    """
    y_true = np.asarray(y_true_rul)
    y_pred = np.asarray(y_pred_rul)
    mask = np.asarray(onset_mask, dtype=bool)

    n_total = len(y_true)
    n_post = int(np.sum(mask))

    if n_post == 0:
        return {
            "post_onset_mae": float("nan"),
            "post_onset_rmse": float("nan"),
            "post_onset_phm08": float("nan"),
            "post_onset_phm08_normalized": float("nan"),
            "post_onset_n_samples": 0,
            "total_n_samples": n_total,
        }

    y_true_post = y_true[mask]
    y_pred_post = y_pred[mask]

    return {
        "post_onset_mae": mae(y_true_post, y_pred_post),
        "post_onset_rmse": rmse(y_true_post, y_pred_post),
        "post_onset_phm08": phm08_score(y_true_post, y_pred_post),
        "post_onset_phm08_normalized": phm08_score_normalized(y_true_post, y_pred_post),
        "post_onset_n_samples": n_post,
        "total_n_samples": n_total,
    }


def twostage_combined_score(
    onset_metrics: dict[str, float],
    rul_metrics: dict[str, float],
    onset_weight: float = 0.3,
    rul_weight: float = 0.7,
) -> float:
    """Compute a combined score from onset detection and RUL metrics.

    Combines onset F1 (higher is better) with post-onset MAE (lower is better)
    into a single score for model comparison. Lower is better.

    Score = rul_weight * post_onset_mae + onset_weight * (1 - onset_f1) * 125

    The (1 - F1) * 125 term scales onset detection error to be in the same
    units as MAE (RUL scale), making the weighting meaningful.

    Args:
        onset_metrics: Dictionary from onset_detection_metrics().
        rul_metrics: Dictionary from conditional_rul_metrics().
        onset_weight: Weight for onset detection quality (default 0.3).
        rul_weight: Weight for RUL prediction quality (default 0.7).

    Returns:
        Combined score (lower is better). Returns nan if either metric is nan.
    """
    f1 = onset_metrics.get("f1", 0.0)
    post_mae = rul_metrics.get("post_onset_mae", float("nan"))

    if np.isnan(post_mae):
        return float("nan")

    # Scale onset error to RUL units (max_rul=125 scale)
    onset_error = (1.0 - f1) * 125.0

    return rul_weight * post_mae + onset_weight * onset_error


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
