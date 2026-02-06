"""Integration tests for the two-stage pipeline (ONSET-25).

Tests the full end-to-end pipeline using real features_v2.csv data:
  1. Load data -> detect onset -> predict RUL -> compute metrics
  2. Real features_v2.csv data (subset for speed)
  3. Pipeline serialization: save and load complete pipeline
  4. End-to-end metrics match expected ranges

Acceptance criteria:
  - Integration test completes in <60 seconds
  - Pipeline produces valid predictions for all test bearings
  - Serialized pipeline produces identical results after reload
  - Test uses real data (not just mocks)
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.rul_labels import compute_twostage_rul
from src.onset.detectors import (
    CUSUMOnsetDetector,
    ThresholdOnsetDetector,
)
from src.onset.health_indicators import (
    compute_composite_hi,
    load_bearing_health_series,
)
from src.onset.labels import load_onset_labels
from src.onset.pipeline import TwoStagePipeline
from src.training.metrics import (
    conditional_rul_metrics,
    onset_detection_metrics,
    onset_timing_mae,
    twostage_combined_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES_CSV = Path("outputs/features/features_v2.csv")
ONSET_LABELS_YAML = Path("configs/onset_labels.yaml")

# Use a small subset of bearings for speed: one per condition, all high confidence
TEST_BEARINGS = ["Bearing1_3", "Bearing2_1", "Bearing3_3"]

# Skip the entire module if features CSV is not present (CI without data)
pytestmark = pytest.mark.skipif(
    not FEATURES_CSV.exists(),
    reason=f"Real data not found at {FEATURES_CSV}",
)


# ---------------------------------------------------------------------------
# Fixtures — real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    """Load the real features_v2.csv, filtered to test bearings only."""
    df = pd.read_csv(FEATURES_CSV)
    return df[df["bearing_id"].isin(TEST_BEARINGS)].copy()


@pytest.fixture(scope="module")
def onset_labels() -> dict:
    """Load the manual onset labels from YAML."""
    return load_onset_labels(ONSET_LABELS_YAML)


class MockRulModel:
    """Minimal RUL model that predicts linearly decreasing RUL.

    Mimics a Keras model's .predict() interface. Given N input samples,
    returns values linearly decreasing from ``start_rul`` to 0.
    """

    def __init__(self, start_rul: float = 100.0) -> None:
        self.start_rul = start_rul

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        n = x.shape[0]
        if n == 1:
            return np.array([[self.start_rul]])
        return np.linspace(self.start_rul, 0, n).reshape(-1, 1)


# ---------------------------------------------------------------------------
# ONSET-25 Task 2: Full pipeline — load data → detect onset → predict RUL → compute metrics
# ---------------------------------------------------------------------------


class TestFullPipelineEndToEnd:
    """Test the complete pipeline path with real feature data."""

    def test_load_data_detect_onset_predict_rul_compute_metrics(
        self, features_df, onset_labels
    ):
        """Full pipeline on one bearing: load → detect → predict → metrics."""
        bearing_id = "Bearing1_3"
        health = load_bearing_health_series(bearing_id, features_df)

        # Stage 1: detect onset with threshold detector
        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(health.composite[: max(3, len(health.composite) // 5)])

        rul_model = MockRulModel(start_rul=80.0)
        pipeline = TwoStagePipeline(
            onset_detector=detector,
            rul_model=rul_model,
        )

        # Run full pipeline
        rul_preds = pipeline.predict(health.composite)

        # Basic shape/type assertions
        n_samples = len(health.composite)
        assert rul_preds.shape == (n_samples,)
        assert rul_preds.dtype == np.float32

        # Generate ground-truth two-stage RUL
        true_onset_idx = onset_labels[bearing_id].onset_file_idx
        true_rul = compute_twostage_rul(n_samples, true_onset_idx)

        # Build binary onset labels (true & predicted)
        y_true_onset = np.zeros(n_samples, dtype=int)
        y_true_onset[true_onset_idx:] = 1

        # Predicted onset from pipeline
        onset_result = pipeline.detect_onset(health.composite)
        pred_onset_idx = onset_result.onset_idx
        y_pred_onset = np.zeros(n_samples, dtype=int)
        if pred_onset_idx is not None:
            y_pred_onset[pred_onset_idx:] = 1

        # Compute metrics — must not raise
        od_metrics = onset_detection_metrics(y_true_onset, y_pred_onset)
        assert "f1" in od_metrics
        assert 0.0 <= od_metrics["f1"] <= 1.0

        rul_metrics = conditional_rul_metrics(
            true_rul, rul_preds, onset_mask=(y_true_onset == 1)
        )
        assert "post_onset_mae" in rul_metrics

        combined = twostage_combined_score(od_metrics, rul_metrics)
        assert np.isfinite(combined)

    def test_pipeline_on_all_test_bearings(self, features_df, onset_labels):
        """Pipeline produces valid predictions for every test bearing."""
        rul_model = MockRulModel(start_rul=80.0)

        for bearing_id in TEST_BEARINGS:
            health = load_bearing_health_series(bearing_id, features_df)

            detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
            n_healthy = max(3, len(health.composite) // 5)
            detector.fit(health.composite[:n_healthy])

            pipeline = TwoStagePipeline(
                onset_detector=detector, rul_model=rul_model
            )
            rul_preds = pipeline.predict(health.composite)

            # Valid output for every bearing
            assert rul_preds.shape == (len(health.composite),)
            assert rul_preds.dtype == np.float32
            assert np.all(np.isfinite(rul_preds))
            assert rul_preds.min() >= 0.0

    def test_cusum_detector_pipeline(self, features_df, onset_labels):
        """Pipeline works with CUSUM detector on real data."""
        bearing_id = "Bearing2_1"
        health = load_bearing_health_series(bearing_id, features_df)

        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        n_healthy = max(3, len(health.composite) // 5)
        detector.fit(health.composite[:n_healthy])

        rul_model = MockRulModel(start_rul=80.0)
        pipeline = TwoStagePipeline(
            onset_detector=detector, rul_model=rul_model
        )
        rul_preds = pipeline.predict(health.composite)

        assert rul_preds.shape == (len(health.composite),)
        assert np.all(np.isfinite(rul_preds))


# ---------------------------------------------------------------------------
# ONSET-25 Task 3: Real features_v2.csv data (subset for speed)
# ---------------------------------------------------------------------------


class TestRealDataSubset:
    """Verify we're using real data and it's the right subset."""

    def test_features_have_expected_columns(self, features_df):
        """Features CSV has the columns needed for health indicators."""
        required = ["bearing_id", "file_idx", "h_kurtosis", "v_kurtosis", "h_rms", "v_rms"]
        for col in required:
            assert col in features_df.columns, f"Missing column: {col}"

    def test_test_bearings_present(self, features_df):
        """All test bearings are present in the loaded data."""
        present = set(features_df["bearing_id"].unique())
        for bid in TEST_BEARINGS:
            assert bid in present, f"Bearing {bid} not found in data"

    def test_bearings_have_multiple_samples(self, features_df):
        """Each test bearing has a meaningful number of samples."""
        for bid in TEST_BEARINGS:
            n = len(features_df[features_df["bearing_id"] == bid])
            assert n >= 50, f"Bearing {bid} has only {n} samples (need >=50)"

    def test_onset_labels_exist_for_test_bearings(self, onset_labels):
        """Manual onset labels available for all test bearings."""
        for bid in TEST_BEARINGS:
            assert bid in onset_labels
            entry = onset_labels[bid]
            assert entry.onset_file_idx >= 0
            assert entry.confidence in ("high", "medium", "low")

    def test_health_indicators_computed_from_real_data(self, features_df):
        """Health indicators are computed from real feature values, not zeros."""
        for bid in TEST_BEARINGS:
            health = load_bearing_health_series(bid, features_df)
            assert health.kurtosis_h.std() > 0, f"{bid} kurtosis_h is constant"
            assert health.rms_h.std() > 0, f"{bid} rms_h is constant"
            assert 0.0 <= health.composite.min()
            assert health.composite.max() <= 1.0


# ---------------------------------------------------------------------------
# ONSET-25 Task 4: Pipeline serialization — save and load
# ---------------------------------------------------------------------------


class TestPipelineSerialization:
    """Test save/load of complete pipeline produces identical results."""

    def test_pickle_round_trip_produces_identical_results(
        self, features_df, onset_labels, tmp_path
    ):
        """Pickled pipeline produces identical predictions after reload."""
        bearing_id = "Bearing1_3"
        health = load_bearing_health_series(bearing_id, features_df)

        # Build and run pipeline
        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(health.composite[: max(3, len(health.composite) // 5)])

        rul_model = MockRulModel(start_rul=80.0)
        pipeline = TwoStagePipeline(
            onset_detector=detector, rul_model=rul_model
        )

        original_preds = pipeline.predict(health.composite)

        # Serialize to disk
        pkl_path = tmp_path / "pipeline.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Reload
        with open(pkl_path, "rb") as f:
            loaded_pipeline = pickle.load(f)

        # Predict with loaded pipeline
        loaded_preds = loaded_pipeline.predict(health.composite)

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_detector_state_survives_serialization(
        self, features_df, tmp_path
    ):
        """Detector's fitted state (mean, std) is preserved through pickle."""
        health = load_bearing_health_series("Bearing1_3", features_df)
        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(health.composite[:20])

        pkl_path = tmp_path / "detector.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(detector, f)

        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        # Fitted state preserved
        assert loaded._mean == detector._mean
        assert loaded._std == detector._std
        assert loaded._n_samples == detector._n_samples

        # Detection results identical
        original_result = detector.detect(health.composite)
        loaded_result = loaded.detect(health.composite)
        assert original_result.onset_idx == loaded_result.onset_idx
        assert original_result.confidence == loaded_result.confidence

    def test_pipeline_components_json_metadata(
        self, features_df, tmp_path
    ):
        """Pipeline metadata (config, onset result) can be saved as JSON."""
        health = load_bearing_health_series("Bearing2_1", features_df)

        detector = CUSUMOnsetDetector(drift=0.5, threshold=5.0)
        detector.fit(health.composite[:50])

        rul_model = MockRulModel(start_rul=80.0)
        pipeline = TwoStagePipeline(
            onset_detector=detector, rul_model=rul_model
        )

        onset_result = pipeline.detect_onset(health.composite)

        # Save metadata as JSON (for reproducibility/logging)
        metadata = {
            "detector_type": type(pipeline.onset_detector).__name__,
            "onset_idx": onset_result.onset_idx,
            "onset_time": onset_result.onset_time,
            "confidence": onset_result.confidence,
            "max_rul": pipeline.max_rul,
        }
        json_path = tmp_path / "pipeline_meta.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Reload and verify
        with open(json_path) as f:
            loaded_meta = json.load(f)

        assert loaded_meta["detector_type"] == "CUSUMOnsetDetector"
        assert loaded_meta["max_rul"] == 125


# ---------------------------------------------------------------------------
# ONSET-25 Task 5: End-to-end metrics match expected ranges
# ---------------------------------------------------------------------------


class TestMetricsExpectedRanges:
    """Verify end-to-end metrics are in sensible ranges."""

    def test_onset_f1_within_expected_range(self, features_df, onset_labels):
        """Onset F1 score should be reasonable on high-confidence bearings."""
        all_true = []
        all_pred = []

        for bearing_id in TEST_BEARINGS:
            health = load_bearing_health_series(bearing_id, features_df)
            n = len(health.composite)

            # True onset labels
            true_idx = onset_labels[bearing_id].onset_file_idx
            y_true = np.zeros(n, dtype=int)
            y_true[true_idx:] = 1

            # Detect onset
            detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
            detector.fit(health.composite[: max(3, n // 5)])
            result = detector.detect(health.composite)

            y_pred = np.zeros(n, dtype=int)
            if result.onset_idx is not None:
                y_pred[result.onset_idx:] = 1

            all_true.append(y_true)
            all_pred.append(y_pred)

        combined_true = np.concatenate(all_true)
        combined_pred = np.concatenate(all_pred)

        metrics = onset_detection_metrics(combined_true, combined_pred)

        # F1 should be at least 0.5 for high-confidence bearings with threshold detector
        assert metrics["f1"] >= 0.5, (
            f"Onset F1={metrics['f1']:.3f} is too low for high-confidence bearings"
        )
        # Precision and recall both > 0
        assert metrics["precision"] > 0
        assert metrics["recall"] > 0

    def test_onset_timing_mae_within_tolerance(self, features_df, onset_labels):
        """Onset timing MAE should be bounded for high-confidence bearings."""
        true_indices = []
        pred_indices = []

        for bearing_id in TEST_BEARINGS:
            health = load_bearing_health_series(bearing_id, features_df)
            n = len(health.composite)

            detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
            detector.fit(health.composite[: max(3, n // 5)])
            result = detector.detect(health.composite)

            true_indices.append(onset_labels[bearing_id].onset_file_idx)
            pred_indices.append(result.onset_idx if result.onset_idx is not None else -1)

        timing = onset_timing_mae(
            np.array(true_indices, dtype=float),
            np.array(pred_indices, dtype=float),
        )

        # At least some bearings should have valid timing comparison
        assert timing["n_valid"] >= 1, "No bearings have valid onset timing comparison"

        # MAE should be finite
        assert np.isfinite(timing["onset_timing_mae_samples"])

    def test_conditional_rul_mae_is_finite(self, features_df, onset_labels):
        """Post-onset RUL MAE should be finite and positive."""
        bearing_id = "Bearing1_3"
        health = load_bearing_health_series(bearing_id, features_df)
        n = len(health.composite)

        true_idx = onset_labels[bearing_id].onset_file_idx
        true_rul = compute_twostage_rul(n, true_idx)

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(health.composite[: max(3, n // 5)])

        rul_model = MockRulModel(start_rul=80.0)
        pipeline = TwoStagePipeline(
            onset_detector=detector, rul_model=rul_model
        )

        rul_preds = pipeline.predict(health.composite)

        onset_mask = np.zeros(n, dtype=bool)
        onset_mask[true_idx:] = True

        rul_metrics = conditional_rul_metrics(true_rul, rul_preds, onset_mask)

        assert np.isfinite(rul_metrics["post_onset_mae"])
        assert rul_metrics["post_onset_mae"] >= 0
        assert rul_metrics["post_onset_n_samples"] > 0

    def test_combined_score_is_finite_and_positive(self, features_df, onset_labels):
        """Combined two-stage score should be a finite positive number."""
        bearing_id = "Bearing3_3"
        health = load_bearing_health_series(bearing_id, features_df)
        n = len(health.composite)

        true_idx = onset_labels[bearing_id].onset_file_idx
        true_rul = compute_twostage_rul(n, true_idx)

        detector = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
        detector.fit(health.composite[: max(3, n // 5)])

        rul_model = MockRulModel(start_rul=80.0)
        pipeline = TwoStagePipeline(
            onset_detector=detector, rul_model=rul_model
        )

        onset_result = pipeline.detect_onset(health.composite)
        rul_preds = pipeline.predict(health.composite)

        # Build binary labels
        y_true_onset = np.zeros(n, dtype=int)
        y_true_onset[true_idx:] = 1

        y_pred_onset = np.zeros(n, dtype=int)
        if onset_result.onset_idx is not None:
            y_pred_onset[onset_result.onset_idx:] = 1

        od_metrics = onset_detection_metrics(y_true_onset, y_pred_onset)

        onset_mask = y_true_onset.astype(bool)
        rul_metrics = conditional_rul_metrics(true_rul, rul_preds, onset_mask)

        combined = twostage_combined_score(od_metrics, rul_metrics)
        assert np.isfinite(combined)
        assert combined >= 0


# ---------------------------------------------------------------------------
# Performance: integration test should complete in <60 seconds
# ---------------------------------------------------------------------------


class TestPerformance:
    """Ensure the integration test suite runs within acceptable time."""

    def test_full_pipeline_under_60_seconds(self, features_df, onset_labels):
        """Running pipeline on all test bearings completes quickly."""
        start = time.monotonic()

        rul_model = MockRulModel(start_rul=80.0)

        for bearing_id in TEST_BEARINGS:
            health = load_bearing_health_series(bearing_id, features_df)
            n = len(health.composite)

            # Threshold detector
            det = ThresholdOnsetDetector(threshold_sigma=3.0, min_consecutive=3)
            det.fit(health.composite[: max(3, n // 5)])

            pipeline = TwoStagePipeline(onset_detector=det, rul_model=rul_model)

            # Full pipeline
            rul_preds = pipeline.predict(health.composite)
            assert rul_preds.shape == (n,)

            # Metrics
            true_idx = onset_labels[bearing_id].onset_file_idx
            true_rul = compute_twostage_rul(n, true_idx)
            onset_mask = np.zeros(n, dtype=bool)
            onset_mask[true_idx:] = True
            conditional_rul_metrics(true_rul, rul_preds, onset_mask)

        elapsed = time.monotonic() - start
        assert elapsed < 60.0, f"Pipeline took {elapsed:.1f}s (limit: 60s)"
