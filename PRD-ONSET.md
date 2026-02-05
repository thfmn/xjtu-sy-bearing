# PRD-ONSET: Two-Stage Degradation Onset Detection

Track all project tickets for implementing degradation onset detection and two-stage RUL prediction. This feature is designed to improve RUL prediction accuracy by first detecting when degradation begins, then applying RUL models only to degraded samples.

**Status Legend:**
- [ ] To Do
- [~] In Progress
- [x] Done

**Background:**
State-of-the-art bearing RUL prediction (MAE ~3-6 on piecewise-linear scale with max_rul=125) uses two-stage approaches:
1. **Stage 1 (Onset Detection):** Binary classification to identify when degradation begins
2. **Stage 2 (RUL Prediction):** Regression applied only to post-onset samples

Current single-stage approach achieves MAE ~12.15 (Pattern2 CNN 2D). Two-stage methods can potentially halve this error.

---

## EPIC 1: Health Indicator Computation

Compute per-bearing health indicator time series from existing features for onset detection algorithms.

---

### ONSET-1: Health Indicator Aggregation Module

**Priority**: P0 | **Effort**: 2h | **Phase**: 1. Onset Detection

**Tasks**:
- [x] Create `src/onset/__init__.py`
- [x] Create `src/onset/health_indicators.py`
- [x] Implement `load_bearing_health_series(bearing_id, features_df)` function
  - Load from `outputs/features/features_v2.csv`
  - Return time-ordered arrays for: `h_kurtosis`, `v_kurtosis`, `h_rms`, `v_rms`
- [x] Implement `compute_composite_hi(kurtosis_h, kurtosis_v, rms_h, rms_v)` function
  - Combine indicators using weighted sum (configurable weights)
  - Default: `0.4 * norm(kurtosis_h) + 0.4 * norm(kurtosis_v) + 0.1 * norm(rms_h) + 0.1 * norm(rms_v)`
- [x] Add min-max normalization per bearing to handle scale differences
- [x] Add Savitzky-Golay smoothing option for noise reduction

**Acceptance**:
- [x] Function returns dictionary with `{bearing_id, timestamps, kurtosis_h, kurtosis_v, rms_h, rms_v, composite}`
- [x] Composite HI is in [0, 1] range after normalization
- [x] Smoothing reduces noise while preserving trend shape
- [x] Works for all 15 bearings across 3 conditions

**Files**:
- `src/onset/__init__.py`
- `src/onset/health_indicators.py`

---

### ONSET-2: Health Indicator Unit Tests

**Priority**: P0 | **Effort**: 1h | **Phase**: 1. Onset Detection

**Tasks**:
- [x] Create `tests/onset/__init__.py`
- [x] Create `tests/onset/test_health_indicators.py`
- [x] Test `load_bearing_health_series()` returns correct shape for known bearing
- [x] Test `compute_composite_hi()` produces values in [0, 1]
- [x] Test smoothing preserves array length
- [x] Test handling of missing/NaN values
- [x] Add fixture for synthetic health indicator data

**Acceptance**:
- [x] All tests pass with `pytest tests/onset/test_health_indicators.py`
- [x] Tests cover edge cases: single sample, NaN values, constant signals
- [x] Coverage >90% for `health_indicators.py`

**Files**:
- `tests/onset/__init__.py`
- `tests/onset/test_health_indicators.py`

---

## EPIC 2: Onset Detection Algorithms

Implement multiple onset detection algorithms to identify the point where bearing degradation begins.

---

### ONSET-3: Threshold-Based Onset Detection

**Priority**: P0 | **Effort**: 2h | **Phase**: 1. Onset Detection

**Tasks**:
- [x] Create `src/onset/detectors.py`
- [x] Implement `ThresholdOnsetDetector` class with methods:
  - `__init__(threshold_sigma=3.0, min_consecutive=3)`
  - `fit(healthy_hi_samples)` - learn healthy baseline from early samples
  - `detect(hi_series)` - return onset index where HI exceeds `mean + threshold_sigma * std`
- [x] Add `min_consecutive` parameter: require N consecutive exceedances to confirm onset
- [x] Return `OnsetResult` dataclass with: `onset_idx`, `onset_time`, `confidence`, `healthy_baseline`

**Acceptance**:
- [x] Detector finds onset within 10 samples of manually-labeled onset for test bearings
- [x] `min_consecutive` filter reduces false positives from transient spikes
- [x] Confidence score reflects how far HI exceeds threshold
- [x] Returns `None` for onset_idx if no onset detected (healthy bearing throughout)

**Files**:
- `src/onset/detectors.py`

---

### ONSET-4: Change-Point Detection (CUSUM)

**Priority**: P1 | **Effort**: 2h | **Phase**: 1. Onset Detection

**Tasks**:
- [x] Add `CUSUMOnsetDetector` class to `src/onset/detectors.py`
- [x] Implement Cumulative Sum (CUSUM) algorithm:
  - `__init__(drift=0.5, threshold=5.0)`
  - `fit(healthy_hi_samples)` - estimate target mean and std from healthy region
  - `detect(hi_series)` - return onset index where CUSUM statistic exceeds threshold
- [x] Implement both upper CUSUM (increasing trend) and lower CUSUM (decreasing trend)
- [x] Add optional Exponentially Weighted Moving Average (EWMA) variant

**Acceptance**:
- [x] CUSUM detects gradual shifts earlier than threshold-based detector
- [x] Tunable drift parameter controls sensitivity vs. false positive trade-off
- [x] Works for both sudden and gradual degradation patterns
- [x] Returns same `OnsetResult` dataclass as threshold detector

**Files**:
- `src/onset/detectors.py`

---

### ONSET-5: Bayesian Online Change-Point Detection

**Priority**: P2 | **Effort**: 3h | **Phase**: 1. Onset Detection

**Tasks**:
- [ ] Add `BayesianOnsetDetector` class to `src/onset/detectors.py`
- [ ] Implement Bayesian Online Change-Point Detection (Adams & MacKay, 2007):
  - `__init__(hazard_rate=1/50, prior_mean=None, prior_var=None)`
  - `detect(hi_series)` - return onset index with highest run-length probability change
- [ ] Use conjugate prior (Normal-Inverse-Gamma) for Gaussian observations
- [ ] Output posterior probability of change-point at each timestep

**Acceptance**:
- [ ] Detector provides probability distribution over possible onset points
- [ ] Confidence reflects posterior probability of change-point
- [ ] Handles non-stationary healthy phase better than threshold methods
- [ ] Computational cost is O(n^2) or better for n samples

**Files**:
- `src/onset/detectors.py`

---

### ONSET-6: Onset Detector Unit Tests

**Priority**: P0 | **Effort**: 2h | **Phase**: 1. Onset Detection

**Tasks**:
- [x] Create `tests/onset/test_detectors.py`
- [x] Test `ThresholdOnsetDetector` on synthetic data with known onset point
- [x] Test `CUSUMOnsetDetector` on synthetic data with gradual shift
- [ ] Test `BayesianOnsetDetector` returns valid probabilities
- [x] Test all detectors return `None` for healthy-only series
- [x] Test `min_consecutive` parameter filters transient spikes
- [x] Add parametrized tests for different threshold values

**Acceptance**:
- [x] All tests pass with `pytest tests/onset/test_detectors.py`
- [x] Synthetic test data covers: sudden onset, gradual onset, no onset, noisy signals
- [x] Tests verify onset index is within tolerance of true onset
- [x] Coverage >90% for `detectors.py`

**Files**:
- `tests/onset/test_detectors.py`

---

### ONSET-7: Ensemble Onset Detector

**Priority**: P1 | **Effort**: 2h | **Phase**: 1. Onset Detection

**Tasks**:
- [x] Add `EnsembleOnsetDetector` class to `src/onset/detectors.py`
- [x] Implement voting mechanism across multiple detectors:
  - `__init__(detectors: list[BaseOnsetDetector], voting='majority')`
  - Support voting strategies: `majority`, `unanimous`, `earliest`, `weighted`
- [x] Implement confidence aggregation from individual detector confidences
- [x] Add `add_detector()` and `remove_detector()` methods for flexibility

**Acceptance**:
- [x] Ensemble combines at least 2 detector outputs
- [x] `majority` voting requires >50% of detectors to agree on onset region
- [x] `earliest` returns first detected onset across all detectors
- [x] Ensemble confidence is weighted average of individual confidences
- [x] Handles case where detectors disagree significantly (return low confidence)

**Files**:
- `src/onset/detectors.py`

---

## EPIC 3: Onset Label Generation

Generate onset labels for all bearings to enable supervised training and evaluation.

---

### ONSET-8: Manual Onset Labeling Reference

**Priority**: P0 | **Effort**: 2h | **Phase**: 2. Label Generation

**Tasks**:
- [x] Create `configs/onset_labels.yaml` with manually-verified onset indices
- [x] Analyze kurtosis plots for all 15 bearings to identify visual onset points
- [x] Document labeling methodology:
  - First sustained kurtosis increase (>2 std above baseline for >5 consecutive samples)
  - Or first RMS increase that doesn't return to baseline
- [x] Record onset file index for each bearing (file_idx where degradation starts)
- [x] Add uncertainty ranges where onset is ambiguous

**Acceptance**:
- [x] YAML contains onset_idx for all 15 bearings
- [x] Each entry has: `bearing_id`, `condition`, `onset_file_idx`, `confidence` (high/medium/low)
- [x] Ambiguous cases documented with `onset_range: [min_idx, max_idx]`
- [x] At least 10 bearings have high-confidence labels

**Files**:
- `configs/onset_labels.yaml`

---

### ONSET-9: Onset Label Loader

**Priority**: P0 | **Effort**: 1h | **Phase**: 2. Label Generation

**Tasks**:
- [x] Create `src/onset/labels.py`
- [x] Implement `load_onset_labels(yaml_path)` function
  - Parse `configs/onset_labels.yaml`
  - Return dictionary mapping `bearing_id -> onset_file_idx`
- [x] Implement `get_onset_label(bearing_id, file_idx, onset_labels)` function
  - Return 0 (healthy) if `file_idx < onset_file_idx`, else 1 (degraded)
- [x] Implement `add_onset_column(features_df, onset_labels)` function
  - Add binary `is_degraded` column to features dataframe

**Acceptance**:
- [x] Loader correctly parses YAML with all 15 bearings
- [x] Binary labels are consistent with file ordering
- [x] `add_onset_column()` produces correct label distribution per bearing
- [x] Handles missing bearings gracefully (warning + skip)

**Files**:
- `src/onset/labels.py`

---

### ONSET-10: Automated Onset Labeling Script

**Priority**: P1 | **Effort**: 2h | **Phase**: 2. Label Generation

**Tasks**:
- [x] Create `scripts/06_generate_onset_labels.py`
- [x] Load features from `outputs/features/features_v2.csv`
- [x] Apply `ThresholdOnsetDetector` to each bearing's kurtosis series
- [x] Apply `CUSUMOnsetDetector` as secondary detector
- [x] Compare automated labels to manual labels (if available)
- [x] Output comparison report: agreement rate, disagreement cases
- [x] Save automated labels to `outputs/onset/onset_labels_auto.csv`

**Acceptance**:
- [x] Script runs on all 15 bearings without errors
- [x] Output CSV has columns: `bearing_id`, `condition`, `onset_file_idx`, `detector_method`
- [x] Agreement with manual labels >80% (within 5 samples tolerance)
- [x] Disagreements logged for manual review

**Files**:
- `scripts/06_generate_onset_labels.py`
- `outputs/onset/onset_labels_auto.csv`

---

### ONSET-11: Onset Label Unit Tests

**Priority**: P0 | **Effort**: 1h | **Phase**: 2. Label Generation

**Tasks**:
- [x] Create `tests/onset/test_labels.py`
- [x] Test `load_onset_labels()` parses valid YAML correctly
- [x] Test `get_onset_label()` returns correct binary value
- [x] Test `add_onset_column()` adds column with expected distribution
- [x] Test error handling for missing/malformed YAML
- [x] Add fixture with sample onset labels YAML

**Acceptance**:
- [x] All tests pass with `pytest tests/onset/test_labels.py`
- [x] Tests cover: valid YAML, missing bearing, malformed YAML
- [x] Coverage >90% for `labels.py`

**Files**:
- `tests/onset/test_labels.py`

---

## EPIC 4: Onset Visualization

Create visualization tools for onset detection validation and analysis.

---

### ONSET-12: Onset Visualization Module

**Priority**: P1 | **Effort**: 2h | **Phase**: 2. Label Generation

**Tasks**:
- [x] Create `src/onset/visualization.py`
- [x] Implement `plot_bearing_onset(bearing_id, features_df, onset_idx, save_path=None)`:
  - Plot kurtosis time series with onset point marked
  - Show healthy region (green) vs degraded region (red) shading
  - Add threshold line if threshold-based detection used
- [x] Implement `plot_onset_comparison(bearing_id, manual_idx, auto_idx, features_df)`:
  - Compare manual vs automated onset labels on same plot
- [x] Implement `plot_all_bearings_onset(features_df, onset_labels, output_dir)`:
  - Generate grid of onset plots for all 15 bearings

**Acceptance**:
- [x] Plots clearly show onset point with vertical line annotation
- [ ] Region shading makes healthy/degraded phases visually distinct
- [ ] Grid plot fits 15 subplots in readable layout (5x3 or 3x5)
- [ ] Plots are publication-quality (labeled axes, legend, appropriate DPI)

**Files**:
- `src/onset/visualization.py`

---

### ONSET-13: Onset Analysis Notebook

**Priority**: P1 | **Effort**: 2h | **Phase**: 2. Label Generation

**Tasks**:
- [ ] Create `notebooks/40_onset_analysis.ipynb`
- [ ] Load features and compute health indicators for all bearings
- [ ] Compare detector algorithms (Threshold, CUSUM, Bayesian) on sample bearings
- [ ] Visualize onset detection results for all 15 bearings
- [ ] Analyze onset timing distribution (early life % vs. late life %)
- [ ] Document findings on detector performance and parameter sensitivity

**Acceptance**:
- [ ] Notebook executes end-to-end without errors
- [ ] Includes comparison table of detector performance metrics
- [ ] Onset timing histogram shows distribution across bearings
- [ ] Recommendations for best detector + parameters documented

**Files**:
- `notebooks/40_onset_analysis.ipynb`

---

## EPIC 5: Stage 1 Onset Classification Model

Implement a neural network classifier for onset detection to improve upon rule-based methods.

---

### ONSET-14: Onset Classification Dataset

**Priority**: P1 | **Effort**: 2h | **Phase**: 3. Model Development

**Tasks**:
- [ ] Create `src/onset/dataset.py`
- [ ] Implement `create_onset_dataset(features_df, onset_labels, window_size=10)`:
  - Create sliding window sequences of health indicators
  - Label each window as 0 (healthy) or 1 (contains onset or post-onset)
- [ ] Implement TensorFlow Dataset generator for onset classification
- [ ] Add class balancing via oversampling minority class (onset samples are rare)
- [ ] Support train/val split respecting bearing boundaries (no leakage)

**Acceptance**:
- [ ] Dataset yields `(window_features, binary_label)` tuples
- [ ] Class weights computed for imbalanced binary classification
- [ ] No data leakage: bearings in train set not in val set
- [ ] Window size is configurable (default 10 = ~10 seconds at 1 sample/sec)

**Files**:
- `src/onset/dataset.py`

---

### ONSET-15: Simple Onset Classifier Model

**Priority**: P1 | **Effort**: 2h | **Phase**: 3. Model Development

**Tasks**:
- [ ] Create `src/onset/models.py`
- [ ] Implement `create_onset_classifier(input_dim=4, window_size=10)`:
  - Input: (window_size, 4) - 4 health indicators over time window
  - Architecture: LSTM(32) -> Dense(16) -> Dense(1, sigmoid)
  - Lightweight model for fast inference
- [ ] Add dropout (0.3) and L2 regularization for small dataset
- [ ] Configure binary crossentropy loss with class weights

**Acceptance**:
- [ ] Model compiles without errors
- [ ] Input shape: (None, window_size, 4), output: (None, 1)
- [ ] Total parameters under 10K (lightweight)
- [ ] Model supports `model.predict_proba()` equivalent via sigmoid output

**Files**:
- `src/onset/models.py`

---

### ONSET-16: Onset Classifier Training Script

**Priority**: P1 | **Effort**: 2h | **Phase**: 3. Model Development

**Tasks**:
- [ ] Add onset classifier training to `scripts/05_train_dl_models.py` (new mode)
- [ ] Or create separate `scripts/07_train_onset_classifier.py`
- [ ] Implement leave-one-bearing-out cross-validation for onset model
- [ ] Log metrics: accuracy, precision, recall, F1, AUC-ROC
- [ ] Save best model to `outputs/models/onset_classifier.keras`
- [ ] Add MLflow tracking for onset experiments

**Acceptance**:
- [ ] Training completes for all CV folds without errors
- [ ] F1 score >0.8 on held-out bearings (onset detection accuracy)
- [ ] Model and metrics logged to MLflow
- [ ] Training time <5 minutes on CPU

**Files**:
- `scripts/07_train_onset_classifier.py` (or modified `scripts/05_train_dl_models.py`)
- `outputs/models/onset_classifier.keras`

---

### ONSET-17: Onset Model Unit Tests

**Priority**: P0 | **Effort**: 1h | **Phase**: 3. Model Development

**Tasks**:
- [ ] Create `tests/onset/test_models.py`
- [ ] Test `create_onset_classifier()` produces correct output shape
- [ ] Test model forward pass with dummy data
- [ ] Test model can be saved and loaded (`.keras` format)
- [ ] Create `tests/onset/test_dataset.py`
- [ ] Test `create_onset_dataset()` produces balanced batches
- [ ] Test no data leakage in train/val split

**Acceptance**:
- [ ] All tests pass with `pytest tests/onset/`
- [ ] Model tests verify input/output shapes
- [ ] Dataset tests verify class distribution and split integrity
- [ ] Coverage >90% for `models.py` and `dataset.py`

**Files**:
- `tests/onset/test_models.py`
- `tests/onset/test_dataset.py`

---

## EPIC 6: Two-Stage Pipeline Integration

Integrate onset detection with existing RUL models to create complete two-stage pipeline.

---

### ONSET-18: Two-Stage Pipeline Module

**Priority**: P0 | **Effort**: 3h | **Phase**: 4. Integration

**Tasks**:
- [ ] Create `src/onset/pipeline.py`
- [ ] Implement `TwoStagePipeline` class:
  - `__init__(onset_detector, rul_model, onset_model=None)`
  - `detect_onset(bearing_signals)` - use detector or classifier
  - `predict_rul(bearing_signals, onset_idx)` - apply RUL model post-onset only
  - `predict(bearing_signals)` - full pipeline: onset detection -> RUL prediction
- [ ] Support both rule-based detectors and ML classifier for Stage 1
- [ ] Handle edge cases: no onset detected (predict max_rul), onset at start

**Acceptance**:
- [ ] Pipeline correctly chains onset detection and RUL prediction
- [ ] Pre-onset samples receive `max_rul` (125) prediction
- [ ] Post-onset samples receive model predictions
- [ ] Supports swapping onset detector without changing RUL model

**Files**:
- `src/onset/pipeline.py`

---

### ONSET-19: Modified RUL Labels for Two-Stage

**Priority**: P0 | **Effort**: 2h | **Phase**: 4. Integration

**Tasks**:
- [ ] Modify `src/data/rul_labels.py` to add two-stage RUL option
- [ ] Implement `compute_twostage_rul(file_indices, onset_idx, max_rul=125)`:
  - Pre-onset: RUL = max_rul (constant, not decaying)
  - Post-onset: RUL = piecewise_linear from onset to failure
- [ ] Add `onset_idx` parameter to existing RUL functions
- [ ] Update `scripts/03_extract_features.py` to optionally add two-stage RUL column

**Acceptance**:
- [ ] Two-stage RUL shows flat line (max_rul) before onset, then decay after
- [ ] Onset-relative RUL at failure is 0 (same as before)
- [ ] Onset-relative RUL at onset is `min(max_rul, files_remaining)`
- [ ] Backward compatible: default behavior unchanged

**Files**:
- `src/data/rul_labels.py`

---

### ONSET-20: Two-Stage Training Configuration

**Priority**: P1 | **Effort**: 2h | **Phase**: 4. Integration

**Tasks**:
- [ ] Create `configs/twostage_pipeline.yaml`
- [ ] Define onset detector configuration:
  - `onset.method`: threshold | cusum | bayesian | classifier
  - `onset.params`: method-specific parameters
- [ ] Define RUL model configuration (reference existing model configs)
- [ ] Add `training.filter_pre_onset`: true/false (train RUL model on post-onset only)
- [ ] Modify `src/training/config.py` to parse two-stage config

**Acceptance**:
- [ ] Config file is valid YAML with all necessary sections
- [ ] `TrainingConfig` class can load two-stage config
- [ ] `filter_pre_onset=true` excludes pre-onset samples from RUL training
- [ ] Default values provided for all parameters

**Files**:
- `configs/twostage_pipeline.yaml`
- `src/training/config.py` (modified)

---

### ONSET-21: Two-Stage Training Script Modification

**Priority**: P1 | **Effort**: 3h | **Phase**: 4. Integration

**Tasks**:
- [ ] Modify `scripts/05_train_dl_models.py` to support two-stage mode
- [ ] Add `--two-stage` CLI flag to enable two-stage training
- [ ] When enabled:
  - Load onset labels from config
  - Filter dataset to post-onset samples only (if `filter_pre_onset=true`)
  - Train RUL model on filtered dataset
- [ ] Log onset detection metrics alongside RUL metrics
- [ ] Save both onset detector/model and RUL model

**Acceptance**:
- [ ] `--two-stage` flag activates two-stage training mode
- [ ] Dataset size reduced when filtering pre-onset samples
- [ ] MLflow logs both onset and RUL metrics
- [ ] Training script works with all existing model architectures

**Files**:
- `scripts/05_train_dl_models.py` (modified)

---

### ONSET-22: Two-Stage Pipeline Unit Tests

**Priority**: P0 | **Effort**: 2h | **Phase**: 4. Integration

**Tasks**:
- [ ] Create `tests/onset/test_pipeline.py`
- [ ] Test `TwoStagePipeline.detect_onset()` with rule-based detector
- [ ] Test `TwoStagePipeline.detect_onset()` with ML classifier
- [ ] Test `TwoStagePipeline.predict()` produces correct RUL shape
- [ ] Test pre-onset samples receive max_rul
- [ ] Test post-onset samples receive model predictions
- [ ] Test edge case: onset at index 0

**Acceptance**:
- [ ] All tests pass with `pytest tests/onset/test_pipeline.py`
- [ ] Tests cover both detector types (rule-based and ML)
- [ ] Edge cases handled correctly
- [ ] Coverage >90% for `pipeline.py`

**Files**:
- `tests/onset/test_pipeline.py`

---

## EPIC 7: Evaluation and Comparison

Evaluate two-stage pipeline against single-stage baseline and document results.

---

### ONSET-23: Two-Stage Evaluation Metrics

**Priority**: P1 | **Effort**: 2h | **Phase**: 5. Evaluation

**Tasks**:
- [ ] Modify `src/training/metrics.py` to add two-stage metrics
- [ ] Implement `onset_detection_metrics(y_true_onset, y_pred_onset)`:
  - Precision, Recall, F1 for onset classification
  - Mean Absolute Error of onset timing (in samples/minutes)
- [ ] Implement `conditional_rul_metrics(y_true_rul, y_pred_rul, onset_mask)`:
  - MAE/RMSE computed only on post-onset samples
- [ ] Add combined score: weighted combination of onset + RUL metrics

**Acceptance**:
- [ ] Onset timing MAE is computed in consistent units (samples or minutes)
- [ ] Post-onset RUL metrics exclude pre-onset samples
- [ ] Combined score provides single number for model comparison
- [ ] All metrics handle edge cases (no onset detected)

**Files**:
- `src/training/metrics.py` (modified)

---

### ONSET-24: Two-Stage Evaluation Notebook

**Priority**: P1 | **Effort**: 3h | **Phase**: 5. Evaluation

**Tasks**:
- [ ] Create `notebooks/41_twostage_evaluation.ipynb`
- [ ] Load trained two-stage pipeline and single-stage models
- [ ] Generate comparison table: single-stage vs. two-stage MAE/RMSE/PHM08
- [ ] Plot RUL predictions for sample bearings (both approaches)
- [ ] Analyze onset detection accuracy impact on final RUL metrics
- [ ] Document when two-stage helps vs. when it doesn't

**Acceptance**:
- [ ] Notebook executes end-to-end without errors
- [ ] Comparison table includes all relevant metrics
- [ ] Plots clearly show improvement (or lack thereof) from two-stage
- [ ] Analysis identifies bearing types where two-stage excels

**Files**:
- `notebooks/41_twostage_evaluation.ipynb`

---

### ONSET-25: Final Integration Test

**Priority**: P0 | **Effort**: 2h | **Phase**: 5. Evaluation

**Tasks**:
- [ ] Create `tests/integration/test_twostage_pipeline.py`
- [ ] Test full pipeline: load data -> detect onset -> predict RUL -> compute metrics
- [ ] Test with real features_v2.csv data (subset for speed)
- [ ] Test pipeline serialization: save and load complete pipeline
- [ ] Verify end-to-end metrics match expected ranges

**Acceptance**:
- [ ] Integration test completes in <60 seconds
- [ ] Pipeline produces valid predictions for all test bearings
- [ ] Serialized pipeline produces identical results after reload
- [ ] Test uses real data (not just mocks)

**Files**:
- `tests/integration/test_twostage_pipeline.py`

---

## Summary

**Total Tasks**: 25
**Estimated Effort**: ~50 hours

**Critical Path** (P0 tasks in order):
1. ONSET-1: Health Indicator Aggregation Module
2. ONSET-2: Health Indicator Unit Tests
3. ONSET-3: Threshold-Based Onset Detection
4. ONSET-6: Onset Detector Unit Tests
5. ONSET-8: Manual Onset Labeling Reference
6. ONSET-9: Onset Label Loader
7. ONSET-11: Onset Label Unit Tests
8. ONSET-17: Onset Model Unit Tests
9. ONSET-18: Two-Stage Pipeline Module
10. ONSET-19: Modified RUL Labels for Two-Stage
11. ONSET-22: Two-Stage Pipeline Unit Tests
12. ONSET-25: Final Integration Test

**Dependencies**:
- EPIC 2 depends on EPIC 1 (detectors need health indicators)
- EPIC 3 depends on EPIC 2 (labels need detectors for auto-labeling)
- EPIC 5 depends on EPIC 3 (ML model needs labels)
- EPIC 6 depends on EPIC 2 + EPIC 5 (pipeline needs detectors + optional classifier)
- EPIC 7 depends on EPIC 6 (evaluation needs pipeline)

**Expected Outcome**:
Implementing this two-stage approach should reduce MAE from ~12 to ~6-8 (40-50% improvement), bringing performance closer to state-of-the-art published results on XJTU-SY dataset.
