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

"""Tests for cross-validation protocol splitters.

Validates the fixed-split CV protocols (Jin et al. 2025, Sun et al. 2024)
alongside existing LOBO CV for the benchmarking paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.cv import (
    fixed_split_jin,
    fixed_split_li,
    generate_cv_folds,
    leave_one_bearing_out,
    validate_no_leakage,
)


@pytest.fixture
def bearing_df():
    """Create a synthetic DataFrame mimicking the XJTU-SY dataset structure.

    15 bearings across 3 conditions, with varying sample counts per bearing
    (similar to actual dataset where bearings have 100-2500 files).
    """
    rows = []
    condition_map = {
        1: "35Hz12kN",
        2: "37.5Hz11kN",
        3: "40Hz10kN",
    }
    # Vary sample counts per bearing to mimic real dataset
    sample_counts = {
        "Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158,
        "Bearing1_4": 122, "Bearing1_5": 52,
        "Bearing2_1": 491, "Bearing2_2": 161, "Bearing2_3": 533,
        "Bearing2_4": 42, "Bearing2_5": 339,
        "Bearing3_1": 2538, "Bearing3_2": 1230, "Bearing3_3": 371,
        "Bearing3_4": 1515, "Bearing3_5": 114,
    }
    for cond_num, cond_name in condition_map.items():
        for i in range(1, 6):
            bearing_id = f"Bearing{cond_num}_{i}"
            n = sample_counts[bearing_id]
            for j in range(n):
                rows.append({
                    "condition": cond_name,
                    "bearing_id": bearing_id,
                    "file_idx": j,
                    "rul": float(n - 1 - j),
                })
    return pd.DataFrame(rows)


class TestFixedSplitJin:
    """Tests for Jin et al. 2025 fixed-split protocol."""

    def test_returns_single_fold(self, bearing_df):
        cv = fixed_split_jin(bearing_df)
        assert len(cv) == 1

    def test_strategy_name(self, bearing_df):
        cv = fixed_split_jin(bearing_df)
        assert cv.strategy == "jin_fixed"

    def test_train_bearings(self, bearing_df):
        cv = fixed_split_jin(bearing_df)
        fold = cv[0]
        assert sorted(fold.train_bearings) == ["Bearing1_4", "Bearing3_2"]

    def test_test_bearing_count(self, bearing_df):
        cv = fixed_split_jin(bearing_df)
        fold = cv[0]
        assert len(fold.val_bearings) == 13

    def test_all_conditions_in_test(self, bearing_df):
        """Jin-split test set should include bearings from all 3 conditions."""
        cv = fixed_split_jin(bearing_df)
        fold = cv[0]
        test_conditions = set()
        for b in fold.val_bearings:
            test_conditions.add(bearing_df[bearing_df["bearing_id"] == b]["condition"].iloc[0])
        assert len(test_conditions) == 3

    def test_no_data_leakage(self, bearing_df):
        cv = fixed_split_jin(bearing_df)
        assert validate_no_leakage(cv[0], bearing_df)

    def test_all_samples_covered(self, bearing_df):
        """Every sample should be in either train or test (not both, not neither)."""
        cv = fixed_split_jin(bearing_df)
        fold = cv[0]
        all_indices = set(range(len(bearing_df)))
        train_set = set(fold.train_indices.tolist())
        val_set = set(fold.val_indices.tolist())
        assert train_set | val_set == all_indices
        assert train_set & val_set == set()

    def test_train_sample_count(self, bearing_df):
        """Training set should contain only samples from Bearing1_4 and Bearing3_2."""
        cv = fixed_split_jin(bearing_df)
        fold = cv[0]
        train_bearings = bearing_df.iloc[fold.train_indices]["bearing_id"].unique()
        assert sorted(train_bearings) == ["Bearing1_4", "Bearing3_2"]

    def test_missing_bearing_raises(self):
        """Should raise ValueError if training bearings not in data."""
        df = pd.DataFrame({
            "condition": ["35Hz12kN"] * 10,
            "bearing_id": ["Bearing1_1"] * 10,
            "rul": list(range(10)),
        })
        with pytest.raises(ValueError, match="Training bearings not found"):
            fixed_split_jin(df)


class TestFixedSplitLi:
    """Tests for Sun et al. 2024 fixed-split protocol."""

    def test_returns_single_fold(self, bearing_df):
        cv = fixed_split_li(bearing_df)
        assert len(cv) == 1

    def test_strategy_name(self, bearing_df):
        cv = fixed_split_li(bearing_df)
        assert cv.strategy == "li_fixed"

    def test_train_bearings(self, bearing_df):
        cv = fixed_split_li(bearing_df)
        fold = cv[0]
        assert sorted(fold.train_bearings) == [
            "Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2",
        ]

    def test_test_bearings(self, bearing_df):
        cv = fixed_split_li(bearing_df)
        fold = cv[0]
        assert sorted(fold.val_bearings) == [
            "Bearing1_3", "Bearing1_4", "Bearing1_5",
            "Bearing2_3", "Bearing2_4", "Bearing2_5",
        ]

    def test_test_bearing_count(self, bearing_df):
        cv = fixed_split_li(bearing_df)
        fold = cv[0]
        assert len(fold.val_bearings) == 6

    def test_train_bearing_count(self, bearing_df):
        cv = fixed_split_li(bearing_df)
        fold = cv[0]
        assert len(fold.train_bearings) == 4

    def test_condition3_excluded(self, bearing_df):
        """Condition 3 bearings should not appear in train or test."""
        cv = fixed_split_li(bearing_df)
        fold = cv[0]
        cond3_bearings = {f"Bearing3_{i}" for i in range(1, 6)}

        train_bearings = set(bearing_df.iloc[fold.train_indices]["bearing_id"].unique())
        test_bearings = set(bearing_df.iloc[fold.val_indices]["bearing_id"].unique())

        assert train_bearings & cond3_bearings == set()
        assert test_bearings & cond3_bearings == set()

    def test_no_data_leakage(self, bearing_df):
        cv = fixed_split_li(bearing_df)
        assert validate_no_leakage(cv[0], bearing_df)

    def test_only_cond12_samples(self, bearing_df):
        """All indices should belong to Conditions 1-2 only."""
        cv = fixed_split_li(bearing_df)
        fold = cv[0]
        all_used = np.concatenate([fold.train_indices, fold.val_indices])
        conditions_used = bearing_df.iloc[all_used]["condition"].unique()
        assert "40Hz10kN" not in conditions_used
        assert len(conditions_used) == 2

    def test_n_samples_excludes_cond3(self, bearing_df):
        """CVSplit.n_samples should reflect only Conditions 1-2."""
        cv = fixed_split_li(bearing_df)
        cond3_count = bearing_df[bearing_df["bearing_id"].str.startswith("Bearing3")].shape[0]
        expected = len(bearing_df) - cond3_count
        assert cv.n_samples == expected

    def test_missing_bearing_raises(self):
        """Should raise ValueError if expected bearings not in data."""
        df = pd.DataFrame({
            "condition": ["35Hz12kN"] * 10,
            "bearing_id": ["Bearing1_1"] * 10,
            "rul": list(range(10)),
        })
        with pytest.raises(ValueError, match="Expected bearings not found"):
            fixed_split_li(df)


class TestGenerateCVFoldsDispatch:
    """Test that generate_cv_folds() correctly dispatches to new strategies."""

    def test_jin_fixed_dispatch(self, bearing_df):
        cv = generate_cv_folds(bearing_df, strategy="jin_fixed")
        assert cv.strategy == "jin_fixed"
        assert len(cv) == 1

    def test_li_fixed_dispatch(self, bearing_df):
        cv = generate_cv_folds(bearing_df, strategy="li_fixed")
        assert cv.strategy == "li_fixed"
        assert len(cv) == 1

    def test_lobo_still_works(self, bearing_df):
        cv = generate_cv_folds(bearing_df, strategy="leave_one_bearing_out")
        assert cv.strategy == "leave_one_bearing_out"
        assert len(cv) == 15

    def test_unknown_strategy_raises(self, bearing_df):
        with pytest.raises(ValueError, match="Unknown strategy"):
            generate_cv_folds(bearing_df, strategy="nonexistent")

    def test_no_leakage_validated_jin(self, bearing_df):
        """generate_cv_folds validates no-leakage internally."""
        cv = generate_cv_folds(bearing_df, strategy="jin_fixed")
        for fold in cv:
            assert validate_no_leakage(fold, bearing_df)

    def test_no_leakage_validated_li(self, bearing_df):
        cv = generate_cv_folds(bearing_df, strategy="li_fixed")
        for fold in cv:
            assert validate_no_leakage(fold, bearing_df)


class TestLOBORegression:
    """Ensure existing LOBO CV behavior is unchanged."""

    def test_lobo_fold_count(self, bearing_df):
        cv = leave_one_bearing_out(bearing_df)
        assert len(cv) == 15

    def test_lobo_each_fold_one_val_bearing(self, bearing_df):
        cv = leave_one_bearing_out(bearing_df)
        for fold in cv:
            assert len(fold.val_bearings) == 1

    def test_lobo_no_leakage(self, bearing_df):
        cv = leave_one_bearing_out(bearing_df)
        for fold in cv:
            assert validate_no_leakage(fold, bearing_df)
