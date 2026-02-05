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
N_FEATURES = len(FEATURE_COLUMNS)


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

        # Create sliding windows for this bearing
        for i in range(n_samples - window_size + 1):
            window = features[i : i + window_size]  # (window_size, 4)
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
