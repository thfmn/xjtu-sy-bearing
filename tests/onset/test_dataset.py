"""Tests for src.onset.dataset - Onset classification dataset module.

Covers ONSET-14 acceptance criteria:
- Dataset yields (window_features, binary_label) tuples
- Class weights computed for imbalanced binary classification
- No data leakage: bearings in train set not in val set
- Window size is configurable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.onset.dataset import (
    OnsetDatasetResult,
    OnsetSplitResult,
    build_onset_tf_dataset,
    compute_class_weights,
    create_onset_dataset,
    split_by_bearing,
)
from src.onset.labels import OnsetLabelEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_label(bearing_id: str, onset: int, condition: str = "35Hz12kN") -> OnsetLabelEntry:
    return OnsetLabelEntry(
        bearing_id=bearing_id,
        condition=condition,
        onset_file_idx=onset,
        confidence="high",
        detection_method="kurtosis",
    )


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """3 bearings with known file counts and onset indices."""
    np.random.seed(42)
    rows: list[dict] = []
    for b_id, n_files in [("B1", 20), ("B2", 15), ("B3", 25)]:
        for i in range(n_files):
            rows.append(
                {
                    "bearing_id": b_id,
                    "file_idx": i,
                    "h_kurtosis": float(np.random.randn()),
                    "v_kurtosis": float(np.random.randn()),
                    "h_rms": float(np.random.randn() * 0.5),
                    "v_rms": float(np.random.randn() * 0.5),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def onset_labels() -> dict[str, OnsetLabelEntry]:
    return {
        "B1": _make_label("B1", onset=10),
        "B2": _make_label("B2", onset=8),
        "B3": _make_label("B3", onset=15),
    }


# ---------------------------------------------------------------------------
# ONSET-14 Acceptance: Dataset yields (window_features, binary_label) tuples
# ---------------------------------------------------------------------------


class TestDatasetYieldsTuples:
    """Verify that the tf.data.Dataset yields (window_features, binary_label) tuples."""

    def test_tf_dataset_yields_two_element_tuple(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        tf_ds = build_onset_tf_dataset(result, batch_size=4, shuffle=False)
        for batch in tf_ds.take(1):
            assert isinstance(batch, tuple)
            assert len(batch) == 2

    def test_window_features_shape_and_dtype(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        window_size = 5
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=window_size)
        tf_ds = build_onset_tf_dataset(result, batch_size=4, shuffle=False)
        for windows, labels in tf_ds.take(1):
            # window_features: (batch, window_size, n_features)
            assert windows.shape[1] == window_size
            assert windows.shape[2] == 4
            assert windows.dtype.name == "float32"

    def test_binary_label_shape_and_dtype(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        tf_ds = build_onset_tf_dataset(result, batch_size=4, shuffle=False)
        for windows, labels in tf_ds.take(1):
            # labels: (batch,) — 1D
            assert labels.ndim == 1
            assert labels.dtype.name == "int32"

    def test_labels_are_binary(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        tf_ds = build_onset_tf_dataset(result, batch_size=100, shuffle=False)
        all_labels = []
        for _, labels in tf_ds:
            all_labels.extend(labels.numpy().tolist())
        assert set(all_labels).issubset({0, 1})

    def test_numpy_result_yields_correct_types(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Even before building tf.data, the numpy arrays have correct types."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        assert result.windows.dtype == np.float32
        assert result.labels.dtype == np.int32
        assert result.windows.ndim == 3  # (n_windows, window_size, n_features)
        assert result.labels.ndim == 1  # (n_windows,)

    def test_total_samples_preserved_in_tf_dataset(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """All samples from numpy result appear in the tf.data.Dataset."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        tf_ds = build_onset_tf_dataset(result, batch_size=8, shuffle=False)
        total = sum(labels.shape[0] for _, labels in tf_ds)
        assert total == len(result.labels)
