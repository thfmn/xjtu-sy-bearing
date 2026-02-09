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
"""Feature window dataset builder for sequence-based RUL prediction.

Creates sliding windows of extracted features for LSTM-based models.
Each window captures the degradation trajectory over consecutive samples,
enabling temporal pattern recognition for RUL prediction.

Input: DataFrame with 65 extracted features per sample
Output: Sliding windows of shape (window_size, n_features) with RUL labels

Functions:
    create_rul_feature_windows: Build sliding window dataset with per-bearing normalization
    split_feature_windows_by_bearing: Train/val split respecting bearing boundaries
    build_feature_window_tf_dataset: Build tf.data.Dataset for training
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


@dataclass
class FeatureWindowResult:
    """Result from create_rul_feature_windows.

    Attributes:
        windows: float32 array of shape (n_windows, window_size, n_features)
        rul_labels: float32 array of shape (n_windows,), continuous RUL values
        bearing_ids: list of bearing_id for each window (for split tracking)
    """

    windows: np.ndarray
    rul_labels: np.ndarray
    bearing_ids: list[str]


def create_rul_feature_windows(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    onset_labels: dict[str, OnsetLabelEntry] | None = None,
    window_size: int = 10,
    max_rul: float = 125.0,
    filter_pre_onset: bool = True,
    eval_mode: str = "post_onset",
) -> FeatureWindowResult:
    """Create sliding windows of features for RUL prediction.

    For each bearing:
    1. Sort by file_idx for temporal ordering
    2. Z-score normalize features using healthy baseline (pre-onset samples)
    3. Compute RUL labels (two-stage or full-life normalized)
    4. Create sliding windows of all features
    5. Optionally filter to only post-onset windows

    Windows do not cross bearing boundaries. Each window's RUL label is the
    RUL of the last sample in the window.

    Args:
        features_df: DataFrame with feature columns plus bearing_id, file_idx.
        feature_cols: List of feature column names to use.
        onset_labels: Dictionary mapping bearing_id to OnsetLabelEntry.
            Required when eval_mode="post_onset", optional for "full_life".
        window_size: Number of consecutive samples per window. Default 10.
        max_rul: Maximum RUL value for two-stage labeling. Default 125.
        filter_pre_onset: If True, only include windows where the last sample
            is post-onset. Default True. Forced to False in full_life mode.
        eval_mode: Evaluation mode. "post_onset" uses two-stage RUL with onset
            labels. "full_life" uses normalized [0, 1] RUL for all samples
            (no onset labels required). Default "post_onset".

    Returns:
        FeatureWindowResult with windows, rul_labels, and bearing_ids.

    Raises:
        ValueError: If required columns are missing or window_size < 1.
    """
    from src.data.rul_labels import compute_twostage_rul, generate_rul_for_bearing

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    required_cols = ["bearing_id", "file_idx"] + feature_cols
    missing_cols = [c for c in required_cols if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # In full_life mode, override filter_pre_onset since there's no onset
    if eval_mode == "full_life":
        filter_pre_onset = False

    all_windows: list[np.ndarray] = []
    all_rul_labels: list[float] = []
    all_bearing_ids: list[str] = []

    for bearing_id, group in features_df.groupby("bearing_id"):
        # In post_onset mode, skip bearings without onset labels
        if eval_mode != "full_life" and (onset_labels is None or bearing_id not in onset_labels):
            continue

        # Sort by file_idx to ensure temporal ordering
        group = group.sort_values("file_idx").reset_index(drop=True)

        # Extract feature matrix: (n_samples, n_features)
        features = group[feature_cols].values.astype(np.float32)
        file_indices = group["file_idx"].values
        num_files = len(features)

        if num_files < window_size:
            continue

        # Determine onset index for normalization baseline
        if eval_mode == "full_life":
            # No onset: use first 20% as pseudo-healthy baseline
            onset_idx = max(2, num_files // 5)
        else:
            onset_idx = onset_labels[bearing_id].onset_file_idx

        # Per-bearing z-score normalization using healthy baseline
        healthy_mask = file_indices < onset_idx
        n_healthy = int(np.sum(healthy_mask))
        if n_healthy >= 2:
            baseline_mean = features[healthy_mask].mean(axis=0)
            baseline_std = features[healthy_mask].std(axis=0)
        else:
            # Fallback: use first 20% as pseudo-baseline
            n_baseline = max(2, num_files // 5)
            baseline_mean = features[:n_baseline].mean(axis=0)
            baseline_std = features[:n_baseline].std(axis=0)

        # Avoid division by zero
        baseline_std = np.where(baseline_std < 1e-8, 1.0, baseline_std)
        features = (features - baseline_mean) / baseline_std

        # Compute RUL labels
        if eval_mode == "full_life":
            rul_values = generate_rul_for_bearing(
                num_files, strategy="linear", normalize=True
            )
        else:
            rul_values = compute_twostage_rul(num_files, onset_idx, max_rul=max_rul)

        # Create sliding windows
        for i in range(num_files - window_size + 1):
            # Label by the last sample in the window
            last_file_idx = file_indices[i + window_size - 1]

            # Optionally skip pre-onset windows (only in post_onset mode)
            if filter_pre_onset and last_file_idx < onset_idx:
                continue

            window = features[i : i + window_size]  # (window_size, n_features)
            rul_label = rul_values[i + window_size - 1]

            all_windows.append(window)
            all_rul_labels.append(float(rul_label))
            all_bearing_ids.append(bearing_id)

    if not all_windows:
        n_features = len(feature_cols)
        return FeatureWindowResult(
            windows=np.empty((0, window_size, n_features), dtype=np.float32),
            rul_labels=np.empty((0,), dtype=np.float32),
            bearing_ids=[],
        )

    return FeatureWindowResult(
        windows=np.stack(all_windows, axis=0),
        rul_labels=np.array(all_rul_labels, dtype=np.float32),
        bearing_ids=all_bearing_ids,
    )


@dataclass
class FeatureWindowSplitResult:
    """Result from split_feature_windows_by_bearing.

    Attributes:
        train: FeatureWindowResult for training bearings.
        val: FeatureWindowResult for validation bearings.
        train_bearing_ids: Unique bearing IDs in the training set.
        val_bearing_ids: Unique bearing IDs in the validation set.
    """

    train: FeatureWindowResult
    val: FeatureWindowResult
    train_bearing_ids: list[str]
    val_bearing_ids: list[str]


def split_feature_windows_by_bearing(
    result: FeatureWindowResult,
    val_bearing_ids: list[str],
) -> FeatureWindowSplitResult:
    """Split a FeatureWindowResult into train/val sets by bearing.

    All windows from a given bearing go entirely into either the training
    set or the validation set, ensuring no data leakage.

    Args:
        result: Output from create_rul_feature_windows().
        val_bearing_ids: List of bearing IDs to use for validation.

    Returns:
        FeatureWindowSplitResult with train and val FeatureWindowResult instances.

    Raises:
        ValueError: If val_bearing_ids is empty.
    """
    if not val_bearing_ids:
        raise ValueError("val_bearing_ids must not be empty")

    val_set = set(val_bearing_ids)
    bearing_arr = np.array(result.bearing_ids)
    val_mask = np.isin(bearing_arr, list(val_set))
    train_mask = ~val_mask

    all_ids = set(result.bearing_ids)

    train_result = FeatureWindowResult(
        windows=result.windows[train_mask],
        rul_labels=result.rul_labels[train_mask],
        bearing_ids=[b for b, m in zip(result.bearing_ids, train_mask) if m],
    )
    val_result = FeatureWindowResult(
        windows=result.windows[val_mask],
        rul_labels=result.rul_labels[val_mask],
        bearing_ids=[b for b, m in zip(result.bearing_ids, val_mask) if m],
    )

    return FeatureWindowSplitResult(
        train=train_result,
        val=val_result,
        train_bearing_ids=sorted(all_ids - val_set),
        val_bearing_ids=sorted(val_set & all_ids),
    )


def build_feature_window_tf_dataset(
    result: FeatureWindowResult,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from a FeatureWindowResult.

    Creates a tf.data.Dataset pipeline with optional shuffling, batching,
    and prefetching for training or evaluation.

    Args:
        result: Output from create_rul_feature_windows() or a split result.
        batch_size: Number of samples per batch. Default 32.
        shuffle: Whether to shuffle. True for training, False for eval.
        shuffle_buffer: Buffer size for shuffling. Default 1024.

    Returns:
        tf.data.Dataset yielding (windows, rul_labels) batches.
            windows shape: (batch_size, window_size, n_features) float32
            rul_labels shape: (batch_size,) float32

    Raises:
        ValueError: If result contains no samples.
    """
    import tensorflow as tf

    n_samples = len(result.rul_labels)
    if n_samples == 0:
        raise ValueError("Cannot build tf.data.Dataset from empty FeatureWindowResult")

    dataset = tf.data.Dataset.from_tensor_slices(
        (result.windows, result.rul_labels)
    )

    dataset = dataset.cache()

    if shuffle:
        buffer = min(shuffle_buffer, n_samples)
        dataset = dataset.shuffle(buffer_size=buffer)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
