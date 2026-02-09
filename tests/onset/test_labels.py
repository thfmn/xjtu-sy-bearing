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

"""Tests for onset label loading and management (ONSET-11).

Tests cover:
- ONSET-9 Acceptance: Loader correctly parses YAML with all 15 bearings
- ONSET-9 Acceptance: Binary labels are consistent with file ordering
- ONSET-9 Acceptance: add_onset_column produces correct label distribution per bearing
- ONSET-9 Acceptance: Handles missing bearings gracefully (warning + skip)
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.onset.labels import (
    DEFAULT_ONSET_LABELS_PATH,
    OnsetLabelEntry,
    add_onset_column,
    get_onset_label,
    load_onset_labels,
)


class TestLoadOnsetLabels:
    """Tests for load_onset_labels function."""

    def test_loads_all_15_bearings(self) -> None:
        """ONSET-9 Acceptance: Loader correctly parses YAML with all 15 bearings."""
        labels = load_onset_labels()

        # Must have exactly 15 bearings
        assert len(labels) == 15, f"Expected 15 bearings, got {len(labels)}"

        # Verify all expected bearing IDs are present
        expected_bearing_ids = [
            "Bearing1_1",
            "Bearing1_2",
            "Bearing1_3",
            "Bearing1_4",
            "Bearing1_5",
            "Bearing2_1",
            "Bearing2_2",
            "Bearing2_3",
            "Bearing2_4",
            "Bearing2_5",
            "Bearing3_1",
            "Bearing3_2",
            "Bearing3_3",
            "Bearing3_4",
            "Bearing3_5",
        ]
        for bearing_id in expected_bearing_ids:
            assert bearing_id in labels, f"Missing bearing: {bearing_id}"

    def test_returns_onset_label_entry_dataclass(self) -> None:
        """Each entry should be an OnsetLabelEntry dataclass."""
        labels = load_onset_labels()

        for bearing_id, entry in labels.items():
            assert isinstance(entry, OnsetLabelEntry), f"{bearing_id} is not OnsetLabelEntry"
            # Verify all required fields are populated
            assert entry.bearing_id == bearing_id
            assert entry.condition in ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
            assert isinstance(entry.onset_file_idx, int)
            assert entry.onset_file_idx >= 0
            assert entry.confidence in ["high", "medium", "low"]
            assert entry.detection_method in ["kurtosis", "rms", "composite"]

    def test_parses_onset_range_correctly(self) -> None:
        """onset_range should be parsed as tuple or None."""
        labels = load_onset_labels()

        # Bearing2_4 has onset_range: [7, 15]
        assert labels["Bearing2_4"].onset_range is not None
        assert labels["Bearing2_4"].onset_range == (7, 15)

        # Bearing1_1 has no onset_range
        assert labels["Bearing1_1"].onset_range is None

    def test_parses_notes_correctly(self) -> None:
        """Notes field should be populated or empty string."""
        labels = load_onset_labels()

        for bearing_id, entry in labels.items():
            assert isinstance(entry.notes, str), f"{bearing_id}.notes is not string"

    def test_condition_distribution(self) -> None:
        """Verify correct number of bearings per condition."""
        labels = load_onset_labels()

        condition_counts = {}
        for entry in labels.values():
            condition_counts[entry.condition] = condition_counts.get(entry.condition, 0) + 1

        assert condition_counts["35Hz12kN"] == 5, "Condition 1 should have 5 bearings"
        assert condition_counts["37.5Hz11kN"] == 5, "Condition 2 should have 5 bearings"
        assert condition_counts["40Hz10kN"] == 5, "Condition 3 should have 5 bearings"

    def test_confidence_distribution(self) -> None:
        """Verify at least 10 high-confidence labels (ONSET-8 acceptance)."""
        labels = load_onset_labels()

        high_count = sum(1 for e in labels.values() if e.confidence == "high")
        assert high_count >= 10, f"Expected >=10 high confidence labels, got {high_count}"

    def test_known_onset_indices(self) -> None:
        """Verify specific known onset indices from YAML."""
        labels = load_onset_labels()

        # Spot-check several bearings
        assert labels["Bearing1_1"].onset_file_idx == 69
        assert labels["Bearing1_5"].onset_file_idx == 27
        assert labels["Bearing2_1"].onset_file_idx == 452
        assert labels["Bearing3_1"].onset_file_idx == 748

    def test_custom_yaml_path(self, tmp_path: Path) -> None:
        """load_onset_labels should accept custom YAML path."""
        # Create minimal valid YAML
        yaml_content = """
bearings:
  - bearing_id: TestBearing1
    condition: 35Hz12kN
    onset_file_idx: 50
    confidence: high
    detection_method: kurtosis
"""
        yaml_path = tmp_path / "test_labels.yaml"
        yaml_path.write_text(yaml_content)

        labels = load_onset_labels(yaml_path)
        assert len(labels) == 1
        assert "TestBearing1" in labels
        assert labels["TestBearing1"].onset_file_idx == 50

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing YAML file."""
        with pytest.raises(FileNotFoundError):
            load_onset_labels(tmp_path / "nonexistent.yaml")

    def test_missing_bearings_key_raises_error(self, tmp_path: Path) -> None:
        """Should raise KeyError if YAML lacks 'bearings' key."""
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("other_key: value")

        with pytest.raises(KeyError, match="bearings"):
            load_onset_labels(yaml_path)

    def test_missing_required_field_raises_error(self, tmp_path: Path) -> None:
        """Should raise KeyError if entry missing required field."""
        yaml_content = """
bearings:
  - bearing_id: Test
    condition: 35Hz12kN
    # Missing onset_file_idx, confidence, detection_method
"""
        yaml_path = tmp_path / "incomplete.yaml"
        yaml_path.write_text(yaml_content)

        with pytest.raises(KeyError, match="onset_file_idx"):
            load_onset_labels(yaml_path)

    def test_default_path_exists(self) -> None:
        """Default YAML path should exist."""
        assert DEFAULT_ONSET_LABELS_PATH.exists(), f"Default path missing: {DEFAULT_ONSET_LABELS_PATH}"


class TestBinaryLabelConsistencyWithFileOrdering:
    """ONSET-9 Acceptance: Binary labels are consistent with file ordering.

    Verifies that when files are processed in index order (0, 1, 2, ...),
    the binary labels form a monotonic non-decreasing sequence: all 0s
    followed by all 1s, with the transition occurring exactly at onset_file_idx.
    """

    @pytest.fixture
    def labels(self) -> dict[str, OnsetLabelEntry]:
        """Load labels once for tests."""
        return load_onset_labels()

    def test_labels_monotonic_nondecreasing_for_all_bearings(
        self, labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Binary labels must be monotonic (0 -> 1, never 1 -> 0) for all bearings."""
        for bearing_id, entry in labels.items():
            onset_idx = entry.onset_file_idx
            # Test sequence: a range that spans before and after onset
            test_range = list(range(max(0, onset_idx - 10), onset_idx + 20))

            previous_label = None
            for file_idx in test_range:
                label = get_onset_label(bearing_id, file_idx, labels)
                assert label in (0, 1), f"{bearing_id} idx {file_idx}: invalid label {label}"

                if previous_label is not None:
                    # Labels must be monotonic: can stay same or increase, never decrease
                    assert label >= previous_label, (
                        f"{bearing_id}: label decreased from {previous_label} to {label} "
                        f"at file_idx {file_idx} (onset={onset_idx})"
                    )
                previous_label = label

    def test_transition_occurs_exactly_at_onset_idx(
        self, labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """The 0->1 transition must occur exactly at onset_file_idx, not before or after."""
        for bearing_id, entry in labels.items():
            onset_idx = entry.onset_file_idx

            # One index before onset must be 0 (if onset > 0)
            if onset_idx > 0:
                label_before = get_onset_label(bearing_id, onset_idx - 1, labels)
                assert label_before == 0, (
                    f"{bearing_id}: expected label 0 at idx {onset_idx - 1} "
                    f"(one before onset {onset_idx}), got {label_before}"
                )

            # Exactly at onset must be 1
            label_at = get_onset_label(bearing_id, onset_idx, labels)
            assert label_at == 1, (
                f"{bearing_id}: expected label 1 at onset idx {onset_idx}, got {label_at}"
            )

    def test_all_indices_before_onset_are_healthy(
        self, labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Every file_idx < onset_file_idx must return label 0."""
        for bearing_id, entry in labels.items():
            onset_idx = entry.onset_file_idx
            # Test all indices from 0 to onset-1
            for file_idx in range(onset_idx):
                label = get_onset_label(bearing_id, file_idx, labels)
                assert label == 0, (
                    f"{bearing_id}: expected healthy (0) at idx {file_idx} "
                    f"(before onset {onset_idx}), got {label}"
                )

    def test_all_indices_at_and_after_onset_are_degraded(
        self, labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Every file_idx >= onset_file_idx must return label 1."""
        for bearing_id, entry in labels.items():
            onset_idx = entry.onset_file_idx
            # Test indices from onset to onset+50 (reasonable range)
            for file_idx in range(onset_idx, onset_idx + 50):
                label = get_onset_label(bearing_id, file_idx, labels)
                assert label == 1, (
                    f"{bearing_id}: expected degraded (1) at idx {file_idx} "
                    f"(at/after onset {onset_idx}), got {label}"
                )


class TestGetOnsetLabel:
    """Tests for get_onset_label function."""

    @pytest.fixture
    def labels(self) -> dict[str, OnsetLabelEntry]:
        """Load labels once for tests."""
        return load_onset_labels()

    def test_healthy_before_onset(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """ONSET-9 Acceptance: Binary labels consistent - healthy before onset."""
        # Bearing1_1 has onset at 69
        assert get_onset_label("Bearing1_1", 0, labels) == 0
        assert get_onset_label("Bearing1_1", 50, labels) == 0
        assert get_onset_label("Bearing1_1", 68, labels) == 0

    def test_degraded_at_and_after_onset(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """ONSET-9 Acceptance: Binary labels consistent - degraded at/after onset."""
        # Bearing1_1 has onset at 69
        assert get_onset_label("Bearing1_1", 69, labels) == 1
        assert get_onset_label("Bearing1_1", 70, labels) == 1
        assert get_onset_label("Bearing1_1", 100, labels) == 1

    def test_boundary_condition_exact_onset(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """Test exact boundary: file_idx == onset_file_idx should return 1."""
        # Bearing2_1 has onset at 452
        assert get_onset_label("Bearing2_1", 451, labels) == 0  # Just before
        assert get_onset_label("Bearing2_1", 452, labels) == 1  # Exact onset
        assert get_onset_label("Bearing2_1", 453, labels) == 1  # Just after

    def test_missing_bearing_raises_key_error(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """Should raise KeyError for unknown bearing."""
        with pytest.raises(KeyError, match="NonexistentBearing"):
            get_onset_label("NonexistentBearing", 50, labels)

    def test_all_bearings_have_valid_labels(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """Every bearing should return valid label (0 or 1)."""
        for bearing_id, entry in labels.items():
            # Test healthy region
            label = get_onset_label(bearing_id, 0, labels)
            assert label == 0, f"{bearing_id} should be healthy at idx 0"

            # Test degraded region (at onset)
            label = get_onset_label(bearing_id, entry.onset_file_idx, labels)
            assert label == 1, f"{bearing_id} should be degraded at onset idx {entry.onset_file_idx}"


class TestAddOnsetColumn:
    """Tests for add_onset_column function."""

    @pytest.fixture
    def labels(self) -> dict[str, OnsetLabelEntry]:
        """Load labels once for tests."""
        return load_onset_labels()

    def test_adds_is_degraded_column(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """Should add 'is_degraded' column to DataFrame."""
        df = pd.DataFrame({"bearing_id": ["Bearing1_1"], "file_idx": [50]})
        result = add_onset_column(df, labels)

        assert "is_degraded" in result.columns

    def test_does_not_modify_original(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """Should return copy, not modify original DataFrame."""
        df = pd.DataFrame({"bearing_id": ["Bearing1_1"], "file_idx": [50]})
        original_cols = list(df.columns)

        _ = add_onset_column(df, labels)

        assert list(df.columns) == original_cols
        assert "is_degraded" not in df.columns

    def test_correct_labels_for_bearing(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """ONSET-9 Acceptance: add_onset_column produces correct label distribution."""
        # Create test data for Bearing1_1 (onset at 69)
        df = pd.DataFrame(
            {
                "bearing_id": ["Bearing1_1"] * 5,
                "file_idx": [50, 68, 69, 70, 100],
            }
        )

        result = add_onset_column(df, labels)

        expected = [0, 0, 1, 1, 1]
        assert result["is_degraded"].tolist() == expected

    def test_multiple_bearings(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """Should handle multiple bearings in same DataFrame."""
        df = pd.DataFrame(
            {
                "bearing_id": ["Bearing1_1", "Bearing2_1", "Bearing1_1", "Bearing2_1"],
                "file_idx": [50, 451, 70, 452],
            }
        )

        result = add_onset_column(df, labels)

        # Bearing1_1: onset=69, Bearing2_1: onset=452
        expected = [0, 0, 1, 1]  # 50<69, 451<452, 70>=69, 452>=452
        assert result["is_degraded"].tolist() == expected

    def test_missing_bearing_warn_mode(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """ONSET-9 Acceptance: Handles missing bearings gracefully (warning + NaN)."""
        df = pd.DataFrame(
            {
                "bearing_id": ["Bearing1_1", "UnknownBearing"],
                "file_idx": [50, 10],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = add_onset_column(df, labels, missing_behavior="warn")

            # Should emit warning
            assert len(w) == 1
            assert "UnknownBearing" in str(w[0].message)

        # First row should have valid label, second should be NaN
        assert result["is_degraded"].iloc[0] == 0
        assert pd.isna(result["is_degraded"].iloc[1])

    def test_missing_bearing_skip_mode(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """skip mode should silently set NaN without warning."""
        df = pd.DataFrame(
            {
                "bearing_id": ["UnknownBearing"],
                "file_idx": [10],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = add_onset_column(df, labels, missing_behavior="skip")

            # Should NOT emit warning
            assert len(w) == 0

        assert pd.isna(result["is_degraded"].iloc[0])

    def test_missing_bearing_error_mode(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """error mode should raise KeyError for missing bearing."""
        df = pd.DataFrame(
            {
                "bearing_id": ["UnknownBearing"],
                "file_idx": [10],
            }
        )

        with pytest.raises(KeyError, match="UnknownBearing"):
            add_onset_column(df, labels, missing_behavior="error")

    def test_missing_required_columns_raises_error(
        self, labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """Should raise ValueError if required columns missing."""
        df = pd.DataFrame({"bearing_id": ["Bearing1_1"]})  # Missing file_idx

        with pytest.raises(ValueError, match="file_idx"):
            add_onset_column(df, labels)

    def test_full_bearing_distribution(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """Test label distribution across full range of a bearing."""
        # Bearing1_5 has onset at 27, total life ~52 files
        file_indices = list(range(52))
        df = pd.DataFrame(
            {
                "bearing_id": ["Bearing1_5"] * 52,
                "file_idx": file_indices,
            }
        )

        result = add_onset_column(df, labels)

        # Count healthy vs degraded
        healthy_count = (result["is_degraded"] == 0).sum()
        degraded_count = (result["is_degraded"] == 1).sum()

        # onset at 27 means 0-26 healthy (27 samples), 27-51 degraded (25 samples)
        assert healthy_count == 27
        assert degraded_count == 25

    def test_label_distribution_per_bearing_all_15(
        self, labels: dict[str, OnsetLabelEntry]
    ) -> None:
        """ONSET-9 Acceptance: add_onset_column produces correct label distribution per bearing.

        Verifies that for every bearing, the count of healthy (0) and degraded (1)
        labels matches the known onset index and total file count from the dataset.
        """
        # Ground truth: total files per bearing from XJTU-SY dataset
        total_files = {
            "Bearing1_1": 123,
            "Bearing1_2": 161,
            "Bearing1_3": 158,
            "Bearing1_4": 122,
            "Bearing1_5": 52,
            "Bearing2_1": 491,
            "Bearing2_2": 161,
            "Bearing2_3": 533,
            "Bearing2_4": 42,
            "Bearing2_5": 339,
            "Bearing3_1": 2538,
            "Bearing3_2": 2496,
            "Bearing3_3": 371,
            "Bearing3_4": 1515,
            "Bearing3_5": 114,
        }

        # Build a DataFrame with all file indices for all 15 bearings
        rows = []
        for bearing_id, n_files in total_files.items():
            for file_idx in range(n_files):
                rows.append({"bearing_id": bearing_id, "file_idx": file_idx})
        df = pd.DataFrame(rows)

        result = add_onset_column(df, labels)

        # Verify distribution per bearing
        for bearing_id, n_files in total_files.items():
            onset_idx = labels[bearing_id].onset_file_idx
            bearing_mask = result["bearing_id"] == bearing_id
            bearing_labels = result.loc[bearing_mask, "is_degraded"]

            healthy_count = (bearing_labels == 0).sum()
            degraded_count = (bearing_labels == 1).sum()

            assert healthy_count == onset_idx, (
                f"{bearing_id}: expected {onset_idx} healthy samples, got {healthy_count}"
            )
            assert degraded_count == n_files - onset_idx, (
                f"{bearing_id}: expected {n_files - onset_idx} degraded samples, "
                f"got {degraded_count}"
            )
            # No NaN values for known bearings
            assert bearing_labels.isna().sum() == 0, (
                f"{bearing_id}: unexpected NaN values in is_degraded column"
            )

    def test_preserves_other_columns(self, labels: dict[str, OnsetLabelEntry]) -> None:
        """Should preserve all existing columns in DataFrame."""
        df = pd.DataFrame(
            {
                "bearing_id": ["Bearing1_1"],
                "file_idx": [50],
                "extra_col1": ["value1"],
                "extra_col2": [123],
            }
        )

        result = add_onset_column(df, labels)

        assert "extra_col1" in result.columns
        assert "extra_col2" in result.columns
        assert result["extra_col1"].iloc[0] == "value1"
        assert result["extra_col2"].iloc[0] == 123
