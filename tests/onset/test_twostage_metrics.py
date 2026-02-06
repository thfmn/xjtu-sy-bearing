"""Tests for two-stage evaluation metrics (ONSET-23).

Verifies onset_detection_metrics, onset_timing_mae,
conditional_rul_metrics, and twostage_combined_score.
"""

import numpy as np
import pytest

from src.training.metrics import (
    conditional_rul_metrics,
    onset_detection_metrics,
    onset_timing_mae,
    twostage_combined_score,
)


class TestOnsetDetectionMetrics:
    """Tests for onset_detection_metrics()."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        m = onset_detection_metrics(y_true, y_pred)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["tp"] == 3
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["tn"] == 3

    def test_all_false_positives(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        m = onset_detection_metrics(y_true, y_pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0  # no true positives
        assert m["f1"] == 0.0
        assert m["fp"] == 4
        assert m["tp"] == 0

    def test_all_false_negatives(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        m = onset_detection_metrics(y_true, y_pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["fn"] == 4

    def test_mixed_predictions(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        m = onset_detection_metrics(y_true, y_pred)
        # tp=2, fp=1, fn=1, tn=1
        assert m["tp"] == 2
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["tn"] == 1
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)

    def test_empty_arrays(self):
        m = onset_detection_metrics(np.array([]), np.array([]))
        assert m["accuracy"] == 0.0
        assert m["f1"] == 0.0

    def test_returns_all_expected_keys(self):
        m = onset_detection_metrics(np.array([0, 1]), np.array([0, 1]))
        expected_keys = {"precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"}
        assert set(m.keys()) == expected_keys


class TestOnsetTimingMAE:
    """Tests for onset_timing_mae()."""

    def test_perfect_timing(self):
        true_idx = np.array([100, 200, 300])
        pred_idx = np.array([100, 200, 300])
        m = onset_timing_mae(true_idx, pred_idx)
        assert m["onset_timing_mae_samples"] == 0.0
        assert m["onset_timing_mae_minutes"] == 0.0
        assert m["n_valid"] == 3

    def test_known_errors(self):
        true_idx = np.array([100, 200])
        pred_idx = np.array([110, 190])
        m = onset_timing_mae(true_idx, pred_idx)
        assert m["onset_timing_mae_samples"] == pytest.approx(10.0)

    def test_samples_to_minutes_conversion(self):
        true_idx = np.array([0, 60])
        pred_idx = np.array([10, 50])
        # 10 samples error, 1 sample per minute
        m = onset_timing_mae(true_idx, pred_idx, samples_per_minute=2.0)
        assert m["onset_timing_mae_samples"] == pytest.approx(10.0)
        assert m["onset_timing_mae_minutes"] == pytest.approx(5.0)

    def test_nan_onset_excluded(self):
        true_idx = np.array([100, np.nan, 300])
        pred_idx = np.array([100, 200, 300])
        m = onset_timing_mae(true_idx, pred_idx)
        assert m["n_valid"] == 2
        assert m["onset_timing_mae_samples"] == pytest.approx(0.0)

    def test_negative_one_treated_as_no_onset(self):
        true_idx = np.array([100, -1])
        pred_idx = np.array([100, 200])
        m = onset_timing_mae(true_idx, pred_idx)
        assert m["n_valid"] == 1

    def test_all_invalid_returns_nan(self):
        m = onset_timing_mae(np.array([np.nan]), np.array([np.nan]))
        assert np.isnan(m["onset_timing_mae_samples"])
        assert m["n_valid"] == 0

    def test_consistent_units(self):
        """Verify MAE is computed in consistent units (samples and minutes)."""
        true_idx = np.array([50, 100, 150])
        pred_idx = np.array([55, 90, 160])
        # errors: 5, 10, 10 => MAE = 25/3
        m = onset_timing_mae(true_idx, pred_idx, samples_per_minute=0.5)
        expected_mae_samples = 25.0 / 3.0
        expected_mae_minutes = expected_mae_samples / 0.5
        assert m["onset_timing_mae_samples"] == pytest.approx(expected_mae_samples)
        assert m["onset_timing_mae_minutes"] == pytest.approx(expected_mae_minutes)


class TestConditionalRULMetrics:
    """Tests for conditional_rul_metrics()."""

    def test_post_onset_only(self):
        y_true = np.array([125, 125, 100, 50, 0])
        y_pred = np.array([125, 125, 90, 45, 5])
        onset_mask = np.array([False, False, True, True, True])
        m = conditional_rul_metrics(y_true, y_pred, onset_mask)
        # Post-onset: true=[100,50,0], pred=[90,45,5]
        # errors: 10, 5, 5 => MAE = 20/3
        assert m["post_onset_mae"] == pytest.approx(20.0 / 3.0)
        assert m["post_onset_n_samples"] == 3
        assert m["total_n_samples"] == 5

    def test_excludes_pre_onset_samples(self):
        """Verify pre-onset samples are excluded from metrics."""
        y_true = np.array([125, 125, 80, 40, 0])
        y_pred = np.array([0, 0, 80, 40, 0])  # terrible pre-onset, perfect post-onset
        onset_mask = np.array([0, 0, 1, 1, 1])
        m = conditional_rul_metrics(y_true, y_pred, onset_mask)
        # Only post-onset matters: perfect predictions
        assert m["post_onset_mae"] == pytest.approx(0.0)

    def test_no_post_onset_returns_nan(self):
        y_true = np.array([125, 125])
        y_pred = np.array([120, 120])
        onset_mask = np.array([False, False])
        m = conditional_rul_metrics(y_true, y_pred, onset_mask)
        assert np.isnan(m["post_onset_mae"])
        assert np.isnan(m["post_onset_rmse"])
        assert m["post_onset_n_samples"] == 0

    def test_all_post_onset(self):
        y_true = np.array([100, 50, 0])
        y_pred = np.array([100, 50, 0])
        onset_mask = np.array([True, True, True])
        m = conditional_rul_metrics(y_true, y_pred, onset_mask)
        assert m["post_onset_mae"] == pytest.approx(0.0)
        assert m["post_onset_rmse"] == pytest.approx(0.0)
        assert m["post_onset_n_samples"] == 3

    def test_returns_expected_keys(self):
        m = conditional_rul_metrics(
            np.array([100, 0]), np.array([90, 10]), np.array([True, True])
        )
        expected_keys = {
            "post_onset_mae",
            "post_onset_rmse",
            "post_onset_phm08",
            "post_onset_phm08_normalized",
            "post_onset_n_samples",
            "total_n_samples",
        }
        assert set(m.keys()) == expected_keys


class TestTwoStageCombinedScore:
    """Tests for twostage_combined_score()."""

    def test_perfect_score(self):
        onset_m = {"f1": 1.0}
        rul_m = {"post_onset_mae": 0.0}
        score = twostage_combined_score(onset_m, rul_m)
        assert score == pytest.approx(0.0)

    def test_worst_onset_detection(self):
        onset_m = {"f1": 0.0}
        rul_m = {"post_onset_mae": 0.0}
        # 0.3 * (1-0) * 125 + 0.7 * 0 = 37.5
        score = twostage_combined_score(onset_m, rul_m)
        assert score == pytest.approx(37.5)

    def test_provides_single_number(self):
        onset_m = {"f1": 0.8}
        rul_m = {"post_onset_mae": 10.0}
        score = twostage_combined_score(onset_m, rul_m)
        assert isinstance(score, float)
        # 0.7 * 10 + 0.3 * (1-0.8) * 125 = 7 + 7.5 = 14.5
        assert score == pytest.approx(14.5)

    def test_nan_mae_returns_nan(self):
        onset_m = {"f1": 0.9}
        rul_m = {"post_onset_mae": float("nan")}
        score = twostage_combined_score(onset_m, rul_m)
        assert np.isnan(score)

    def test_custom_weights(self):
        onset_m = {"f1": 0.5}
        rul_m = {"post_onset_mae": 20.0}
        score = twostage_combined_score(onset_m, rul_m, onset_weight=0.5, rul_weight=0.5)
        # 0.5 * 20 + 0.5 * (1-0.5) * 125 = 10 + 31.25 = 41.25
        assert score == pytest.approx(41.25)
