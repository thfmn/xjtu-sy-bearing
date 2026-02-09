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

"""Onset classification dataset for two-stage RUL prediction.

This module provides functions to create TensorFlow datasets for training
an onset classifier (Stage 1 of the two-stage pipeline). The classifier
learns to predict whether a bearing is in a healthy or degraded state
based on sliding windows of health indicator features.

Input features (per timestep): kurtosis_h, kurtosis_v, rms_h, rms_v
Label: 0 (healthy) or 1 (degraded/post-onset)

Functions:
    create_onset_dataset: Create sliding window dataset from features DataFrame
    build_onset_tf_dataset: Build tf.data.Dataset generator for training
    compute_class_weights: Compute class weights for imbalanced classification
    split_by_bearing: Train/val split respecting bearing boundaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import tensorflow as tf

    from src.onset.labels import OnsetLabelEntry

# Default tf.data pipeline parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_SHUFFLE_BUFFER = 1024

# Health indicator columns used as input features for onset classification
FEATURE_COLUMNS = ["h_kurtosis", "v_kurtosis", "h_rms", "v_rms"]
# After z-score + abs(z-score) augmentation: 4 signed + 4 absolute = 8
N_FEATURES = len(FEATURE_COLUMNS) * 2


@dataclass
class OnsetDatasetResult:
    """Result from create_onset_dataset.

    Attributes:
        windows: float32 array of shape (n_windows, window_size, n_features)
        labels: int32 array of shape (n_windows,), 0=healthy, 1=degraded
        bearing_ids: list of bearing_id for each window (for split tracking)
    """

    windows: np.ndarray
    labels: np.ndarray
    bearing_ids: list[str]


def create_onset_dataset(
    features_df: pd.DataFrame,
    onset_labels: dict[str, OnsetLabelEntry],
    window_size: int = 10,
) -> OnsetDatasetResult:
    """Create sliding window dataset from features DataFrame for onset classification.

    For each bearing, extracts sliding windows of health indicator features
    (h_kurtosis, v_kurtosis, h_rms, v_rms) and labels each window based on
    whether the last sample in the window is at or past the onset point.

    Windows do not cross bearing boundaries. Each bearing's time series is
    processed independently, sorted by file_idx.

    Args:
        features_df: DataFrame with columns: bearing_id, file_idx,
            h_kurtosis, v_kurtosis, h_rms, v_rms.
        onset_labels: Dictionary mapping bearing_id to OnsetLabelEntry,
            as returned by load_onset_labels().
        window_size: Number of consecutive samples per window. Default 10.

    Returns:
        OnsetDatasetResult with:
            - windows: (n_windows, window_size, 4) float32 array
            - labels: (n_windows,) int32 array, 0=healthy, 1=degraded
            - bearing_ids: list of bearing_id strings (one per window)

    Raises:
        ValueError: If required columns are missing or window_size < 1.
    """
    import pandas as pd

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    required_cols = ["bearing_id", "file_idx"] + FEATURE_COLUMNS
    missing_cols = [c for c in required_cols if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []
    all_bearing_ids: list[str] = []

    for bearing_id, group in features_df.groupby("bearing_id"):
        if bearing_id not in onset_labels:
            continue

        # Sort by file_idx to ensure temporal ordering
        group = group.sort_values("file_idx").reset_index(drop=True)

        # Extract feature matrix for this bearing: (n_samples, 4)
        features = group[FEATURE_COLUMNS].values.astype(np.float32)
        file_indices = group["file_idx"].values
        onset_idx = onset_labels[bearing_id].onset_file_idx

        n_samples = len(features)
        if n_samples < window_size:
            continue

        # Per-bearing z-score normalization using healthy baseline.
        # This makes the model see how features deviate from their healthy
        # baseline, regardless of absolute scale differences across
        # operating conditions (35Hz vs 37.5Hz vs 40Hz).
        healthy_mask = file_indices < onset_idx
        n_healthy = int(np.sum(healthy_mask))
        if n_healthy >= 2:
            baseline_mean = features[healthy_mask].mean(axis=0)
            baseline_std = features[healthy_mask].std(axis=0)
        else:
            # Fallback: use first 20% as pseudo-baseline
            n_baseline = max(2, n_samples // 5)
            baseline_mean = features[:n_baseline].mean(axis=0)
            baseline_std = features[:n_baseline].std(axis=0)
        # Avoid division by zero
        baseline_std = np.where(baseline_std < 1e-8, 1.0, baseline_std)
        features = (features - baseline_mean) / baseline_std

        # Append absolute z-scores as additional features. This helps the
        # model detect degradation where kurtosis may decrease (e.g. Bearing3_2)
        # instead of the typical increase. The model sees both the signed
        # deviation (direction) and the magnitude (absolute value), giving it
        # robustness to different degradation patterns.
        features = np.concatenate([features, np.abs(features)], axis=1)

        # Create sliding windows for this bearing
        for i in range(n_samples - window_size + 1):
            window = features[i : i + window_size]  # (window_size, 8)
            # Label by the last sample in the window
            last_file_idx = file_indices[i + window_size - 1]
            label = 0 if last_file_idx < onset_idx else 1

            all_windows.append(window)
            all_labels.append(label)
            all_bearing_ids.append(bearing_id)

    if not all_windows:
        return OnsetDatasetResult(
            windows=np.empty((0, window_size, N_FEATURES), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int32),
            bearing_ids=[],
        )

    return OnsetDatasetResult(
        windows=np.stack(all_windows, axis=0),
        labels=np.array(all_labels, dtype=np.int32),
        bearing_ids=all_bearing_ids,
    )


def build_onset_tf_dataset(
    dataset_result: OnsetDatasetResult,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from an OnsetDatasetResult for training/evaluation.

    Creates a tf.data.Dataset pipeline with optional shuffling, batching,
    and prefetching. Follows the same pattern as the existing RUL dataset
    builders in src/data/dataset.py.

    Args:
        dataset_result: Output from create_onset_dataset().
        batch_size: Number of samples per batch. Default 32.
        shuffle: Whether to shuffle the dataset. Set True for training,
            False for evaluation. Default True.
        shuffle_buffer: Buffer size for tf.data.Dataset.shuffle().
            Default 1024.

    Returns:
        tf.data.Dataset yielding (windows, labels) batches.
            windows shape: (batch_size, window_size, 4) float32
            labels shape: (batch_size,) int32

    Raises:
        ValueError: If dataset_result contains no samples.
    """
    import tensorflow as tf

    n_samples = len(dataset_result.labels)
    if n_samples == 0:
        raise ValueError("Cannot build tf.data.Dataset from empty OnsetDatasetResult")

    dataset = tf.data.Dataset.from_tensor_slices(
        (dataset_result.windows, dataset_result.labels)
    )

    dataset = dataset.cache()

    if shuffle:
        buffer = min(shuffle_buffer, n_samples)
        dataset = dataset.shuffle(buffer_size=buffer)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def compute_class_weights(
    labels: np.ndarray,
) -> dict[int, float]:
    """Compute balanced class weights for imbalanced binary classification.

    Uses the formula: weight_c = n_samples / (n_classes * count_c),
    equivalent to sklearn's compute_class_weight(class_weight='balanced').
    The returned dict can be passed directly to model.fit(class_weight=...).

    Args:
        labels: 1-D int array of binary labels (0=healthy, 1=degraded).

    Returns:
        Dictionary mapping class label to weight, e.g. {0: 1.2, 1: 0.85}.

    Raises:
        ValueError: If labels is empty or contains values other than 0 and 1.
    """
    if len(labels) == 0:
        raise ValueError("Cannot compute class weights from empty labels array")

    unique = set(np.unique(labels))
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"Labels must contain only 0 and 1, got unique values: {sorted(unique)}"
        )

    n_samples = len(labels)
    n_classes = 2
    weights: dict[int, float] = {}

    for cls in (0, 1):
        count = int(np.sum(labels == cls))
        if count == 0:
            weights[cls] = 1.0
        else:
            weights[cls] = n_samples / (n_classes * count)

    return weights


@dataclass
class OnsetSplitResult:
    """Result from split_by_bearing.

    Attributes:
        train: OnsetDatasetResult for training bearings.
        val: OnsetDatasetResult for validation bearings.
        train_bearing_ids: Unique bearing IDs in the training set.
        val_bearing_ids: Unique bearing IDs in the validation set.
    """

    train: OnsetDatasetResult
    val: OnsetDatasetResult
    train_bearing_ids: list[str]
    val_bearing_ids: list[str]


def split_by_bearing(
    dataset_result: OnsetDatasetResult,
    val_bearing_ids: list[str],
) -> OnsetSplitResult:
    """Split an OnsetDatasetResult into train/val sets respecting bearing boundaries.

    Ensures no data leakage: all windows from a given bearing go entirely
    into either the training set or the validation set, never both.

    Args:
        dataset_result: Output from create_onset_dataset().
        val_bearing_ids: List of bearing IDs to use for validation.
            All other bearings go to training.

    Returns:
        OnsetSplitResult with train and val OnsetDatasetResult instances.

    Raises:
        ValueError: If val_bearing_ids is empty or contains IDs not present
            in the dataset.
    """
    if not val_bearing_ids:
        raise ValueError("val_bearing_ids must not be empty")

    val_set = set(val_bearing_ids)
    all_ids = set(dataset_result.bearing_ids)
    unknown = val_set - all_ids
    if unknown:
        raise ValueError(
            f"val_bearing_ids contains IDs not in dataset: {sorted(unknown)}"
        )

    bearing_arr = np.array(dataset_result.bearing_ids)
    val_mask = np.isin(bearing_arr, list(val_set))
    train_mask = ~val_mask

    train_ds = OnsetDatasetResult(
        windows=dataset_result.windows[train_mask],
        labels=dataset_result.labels[train_mask],
        bearing_ids=[b for b, m in zip(dataset_result.bearing_ids, train_mask) if m],
    )
    val_ds = OnsetDatasetResult(
        windows=dataset_result.windows[val_mask],
        labels=dataset_result.labels[val_mask],
        bearing_ids=[b for b, m in zip(dataset_result.bearing_ids, val_mask) if m],
    )

    return OnsetSplitResult(
        train=train_ds,
        val=val_ds,
        train_bearing_ids=sorted(all_ids - val_set),
        val_bearing_ids=sorted(val_set & all_ids),
    )
