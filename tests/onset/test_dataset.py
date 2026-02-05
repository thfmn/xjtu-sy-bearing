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


# ---------------------------------------------------------------------------
# ONSET-14 Acceptance: Class weights computed for imbalanced binary classification
# ---------------------------------------------------------------------------


class TestClassWeightsComputed:
    """Verify that compute_class_weights produces correct balanced weights."""

    def test_balanced_formula_matches_expected(self) -> None:
        """weight_c = n_samples / (n_classes * count_c)."""
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        weights = compute_class_weights(labels)
        # 10 samples, 2 classes: w0 = 10/(2*3) ≈ 1.667, w1 = 10/(2*7) ≈ 0.714
        assert weights[0] == pytest.approx(10.0 / 6.0, rel=1e-6)
        assert weights[1] == pytest.approx(10.0 / 14.0, rel=1e-6)

    def test_balanced_classes_give_equal_weights(self) -> None:
        """When classes are balanced, both weights should be 1.0."""
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        weights = compute_class_weights(labels)
        assert weights[0] == pytest.approx(1.0, rel=1e-6)
        assert weights[1] == pytest.approx(1.0, rel=1e-6)

    def test_minority_class_gets_higher_weight(self) -> None:
        """The minority class should always receive a higher weight."""
        # 90% class-1 (majority), 10% class-0 (minority)
        labels = np.array([0] * 10 + [1] * 90, dtype=np.int32)
        weights = compute_class_weights(labels)
        assert weights[0] > weights[1], (
            f"Minority weight {weights[0]} should be > majority weight {weights[1]}"
        )

    def test_weights_usable_with_dataset_result(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Class weights can be computed from a real dataset result's labels."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        weights = compute_class_weights(result.labels)
        assert 0 in weights and 1 in weights
        assert all(w > 0 for w in weights.values())

    def test_returns_dict_with_both_classes(self) -> None:
        """Even if only one class present, returns weights for both 0 and 1."""
        labels = np.array([0, 0, 0], dtype=np.int32)
        weights = compute_class_weights(labels)
        assert set(weights.keys()) == {0, 1}

    def test_empty_labels_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_class_weights(np.array([], dtype=np.int32))

    def test_non_binary_labels_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            compute_class_weights(np.array([0, 1, 2], dtype=np.int32))


# ---------------------------------------------------------------------------
# ONSET-14 Acceptance: No data leakage: bearings in train set not in val set
# ---------------------------------------------------------------------------


class TestNoDataLeakage:
    """Verify split_by_bearing enforces strict bearing-level separation."""

    def test_train_val_bearing_ids_are_disjoint(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """No bearing ID appears in both train and val sets."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        split = split_by_bearing(result, val_bearing_ids=["B2"])
        train_ids = set(split.train.bearing_ids)
        val_ids = set(split.val.bearing_ids)
        assert train_ids & val_ids == set(), (
            f"Leakage: bearing IDs in both sets: {train_ids & val_ids}"
        )

    def test_val_contains_only_specified_bearings(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Val set contains windows exclusively from the specified val bearing IDs."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        split = split_by_bearing(result, val_bearing_ids=["B3"])
        val_unique = set(split.val.bearing_ids)
        assert val_unique == {"B3"}, f"Expected only B3 in val, got {val_unique}"

    def test_train_excludes_val_bearings(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Train set contains no windows from the val bearing IDs."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        split = split_by_bearing(result, val_bearing_ids=["B1", "B3"])
        train_unique = set(split.train.bearing_ids)
        assert "B1" not in train_unique, "B1 leaked into train set"
        assert "B3" not in train_unique, "B3 leaked into train set"
        assert train_unique == {"B2"}, f"Expected only B2 in train, got {train_unique}"

    def test_all_windows_accounted_for(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Total windows = train windows + val windows (none lost or duplicated)."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        split = split_by_bearing(result, val_bearing_ids=["B2"])
        total = len(result.labels)
        train_count = len(split.train.labels)
        val_count = len(split.val.labels)
        assert train_count + val_count == total, (
            f"Window count mismatch: {train_count} + {val_count} != {total}"
        )

    def test_split_ids_metadata_matches_actual_data(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """split.train_bearing_ids and split.val_bearing_ids match actual data."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        split = split_by_bearing(result, val_bearing_ids=["B2"])
        assert set(split.val_bearing_ids) == set(split.val.bearing_ids)
        assert set(split.train_bearing_ids) == set(split.train.bearing_ids)

    def test_multiple_val_bearings_no_leakage(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """With 2 val bearings, train only has the remaining bearing."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        split = split_by_bearing(result, val_bearing_ids=["B1", "B2"])
        assert set(split.train.bearing_ids) == {"B3"}
        assert set(split.val.bearing_ids) == {"B1", "B2"}

    def test_windows_content_matches_original(
        self, synthetic_df: pd.DataFrame, onset_labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Windows in split subsets have identical content to original (no corruption)."""
        result = create_onset_dataset(synthetic_df, onset_labels, window_size=5)
        split = split_by_bearing(result, val_bearing_ids=["B2"])

        # Rebuild expected subsets from original data
        b2_mask = np.array([b == "B2" for b in result.bearing_ids])
        expected_val_windows = result.windows[b2_mask]
        expected_val_labels = result.labels[b2_mask]

        np.testing.assert_array_equal(split.val.windows, expected_val_windows)
        np.testing.assert_array_equal(split.val.labels, expected_val_labels)
