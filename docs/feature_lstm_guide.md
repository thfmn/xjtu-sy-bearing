# Feature LSTM Design Guide

A technical walkthrough of the Feature LSTM model for bearing Remaining Useful Life (RUL)
prediction on the XJTU-SY dataset.

---

## 1. Executive Summary

The Feature LSTM is a lightweight bidirectional LSTM model that predicts Remaining Useful Life
of rolling element bearings from sliding windows of 65 hand-crafted vibration features.
With only **11,041 trainable parameters**, it achieves a normalized RMSE of **0.160** under
15-fold Leave-One-Bearing-Out (LOBO) cross-validation -- the strictest evaluation protocol in
this benchmark, where the model must generalize to a completely unseen bearing's degradation
pattern. It outperforms all four deep learning baselines tested (1D CNN: 0.251, CNN2D: 0.289,
DTA-MLP: 0.402) and a LightGBM baseline (0.234).

The core insight is that on the XJTU-SY dataset (15 bearings, 3 operating conditions, ~9,216
samples), **domain-informed feature engineering combined with per-bearing normalization
provides a stronger inductive bias than end-to-end learning from raw signals**. The
dataset is too small for deep models to learn the physics of bearing degradation from scratch,
but large enough for a compact LSTM to learn degradation trajectories from well-chosen
features.

Under the Sun et al. 2024 evaluation protocol (Conditions 1-2 only, 4-train/6-test), the
Feature LSTM achieves RMSE **0.156**, matching the published MDSCT result (0.160) from the
original paper despite using 100x fewer parameters and requiring no GPU for training.

---

## 2. Problem Context

### The XJTU-SY Dataset

| Property           | Value                                       |
|--------------------|---------------------------------------------|
| Bearings           | 15 (run-to-failure)                         |
| Operating conditions | 3 (35 Hz/12 kN, 37.5 Hz/11 kN, 40 Hz/10 kN) |
| Sampling rate      | 25.6 kHz                                    |
| Channels           | 2 (horizontal + vertical accelerometer)     |
| Samples per file   | 32,768 (1.28 s recording)                   |
| Total files        | ~9,216                                      |
| Lifetimes          | 42 to 2,538 files (highly variable)         |

Each CSV file is a snapshot of vibration at one point in a bearing's life. Bearings range
from 42 files (Bearing2_4 -- about 42 minutes of recorded data) to 2,538 files (Bearing3_1).
This extreme variability in lifetime length makes simple approaches like fixed-length RUL
decay unsuitable.

### The Cross-Condition Challenge

The three operating conditions produce fundamentally different vibration signatures:

- **Condition 1** (35 Hz/12 kN): Kurtosis peaks up to 141.4 at failure (Bearing1_4)
- **Condition 2** (37.5 Hz/11 kN): Kurtosis peaks around 2.3 (Bearing2_2)
- **Condition 3** (40 Hz/10 kN): Some bearings show kurtosis *decrease* during degradation

A model trained on raw feature values from Condition 1 would see "kurtosis = 5" as healthy,
while the same value in Condition 2 would indicate catastrophic failure. This scale mismatch
is the central challenge for cross-condition generalization.

### Why Raw-Signal Deep Learning Struggles

With only 4 training bearings per LOBO fold (each with 42-2,538 files of 32,768 samples),
the effective training set is small. A 1D CNN processing raw signals has no prior knowledge
that kurtosis, RMS, or crest factor are relevant -- it must learn these features from scratch.
The data budget is simply too small for this to work reliably, leading to the 1D CNN's 0.251
RMSE vs. the Feature LSTM's 0.160.

---

## 3. Feature Engineering Design

### The 65 Base Features

Features are extracted per-file from the raw 32,768-sample dual-channel vibration signal.
The extraction pipeline is split across two modules:

**Time-domain features** (37 total) from `src/features/time_domain.py`:

Per channel (18 each for horizontal and vertical):
- Statistical: mean, std, variance
- Amplitude: RMS, peak, peak-to-peak
- Shape factors: crest factor, shape factor, impulse factor, clearance factor
- Distribution: kurtosis, skewness
- Other: line integral, zero crossing rate, Shannon entropy
- Percentiles: 5th, 50th (median), 95th

Plus 1 cross-channel feature: Pearson correlation coefficient between horizontal and vertical
channels.

**Frequency-domain features** (28 total) from `src/features/frequency_domain.py`:

Per channel (14 each):
- Spectral statistics: centroid, bandwidth, rolloff (85%), flatness
- Band powers: 0-1 kHz, 1-3 kHz, 3-6 kHz, 6-12 kHz
- Frequency measures: dominant frequency, mean frequency (Welch PSD)
- Bearing characteristic frequency band powers: BPFO, BPFI, BSF, FTF

The characteristic frequencies are computed from the LDK UER204 bearing geometry (8 balls,
7.92 mm diameter, 34.5 mm pitch diameter) and the shaft rotation frequency of each operating
condition.

### Per-Bearing Z-Score Normalization

This is the most important design decision in the pipeline. From
`src/data/feature_windows.py`, lines 145-159:

```python
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
```

Each bearing's features are normalized against its own healthy baseline (the pre-onset
period). This transforms raw values into "standard deviations from healthy," making
features comparable across operating conditions. A z-score of 3.0 means "3 standard
deviations from this bearing's healthy baseline" regardless of whether the absolute kurtosis
was 0.5 or 50.0.

This is a form of test-time adaptation: at inference, we compute z-scores using the first
~20% of the target bearing's data as a pseudo-healthy baseline. This is realistic in
production -- early sensor readings from a new bearing establish the healthy reference.

### The Bidirectional Kurtosis Problem

From the onset labels in `configs/onset_labels.yaml`:

- **Bearing2_4**: "Kurtosis shows high initial values then DECREASES during failure"
- **Bearing2_5**: "Kurtosis actually DECREASES toward failure (-0.51)"
- **Bearing3_2**: "Kurtosis never shows clear onset -- actually decreases toward failure"

This is a known phenomenon in bearing degradation: once a localized defect (high kurtosis)
spreads into distributed damage, the signal becomes less impulsive and kurtosis drops.

A naive model that learns "kurtosis goes up = degradation" would completely miss these
bearings. The onset classifier addresses this by augmenting features with their absolute
values. From `src/onset/dataset.py`, lines 150-155:

```python
# Append absolute z-scores as additional features. This helps the
# model detect degradation where kurtosis may decrease (e.g. Bearing3_2)
# instead of the typical increase.
features = np.concatenate([features, np.abs(features)], axis=1)
```

This gives the onset classifier 8 features per timestep: 4 signed z-scores (preserving
direction) + 4 absolute z-scores (capturing magnitude regardless of direction). The RUL
model uses all 65 raw features (before this augmentation) since the z-score normalization
already handles the scale issue.

---

## 4. Model Architecture

### Architecture Diagram

```
Input: (batch, 10, 65)
         |
    [BiLSTM(16)]         # 10,496 params
         |                # Output: (batch, 32) -- 16 forward + 16 backward
    [Dropout(0.2)]
         |
    [Dense(16, ReLU)]     #    528 params
         |
    [Dense(1, linear)]    #     17 params
         |
Output: (batch, 1)       # Total: 11,041 params
```

### Implementation

From `src/models/baselines/feature_lstm.py`:

```python
@dataclass
class FeatureLSTMConfig:
    window_size: int = 10
    n_features: int = 65
    lstm_units: int = 16
    bidirectional: bool = True
    dropout_rate: float = 0.2
    dense_units: int = 16

def build_feature_lstm_model(config, name="feature_lstm_rul"):
    inputs = layers.Input(shape=(config.window_size, config.n_features))
    lstm = layers.LSTM(config.lstm_units, name="lstm")
    if config.bidirectional:
        x = layers.Bidirectional(lstm, name="bilstm")(inputs)
    else:
        x = lstm(inputs)
    if config.dropout_rate > 0:
        x = layers.Dropout(config.dropout_rate, name="dropout")(x)
    x = layers.Dense(config.dense_units, activation="relu", name="dense")(x)
    outputs = layers.Dense(1, name="rul_output")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)
```

### Parameter Count Breakdown

| Layer                | Parameters | Computation                        |
|----------------------|-----------:|------------------------------------|
| BiLSTM (forward)     |      5,248 | 4 x (65 + 16 + 1) x 16            |
| BiLSTM (backward)    |      5,248 | 4 x (65 + 16 + 1) x 16            |
| Dense(16, ReLU)      |        528 | (32 + 1) x 16                      |
| Dense(1, linear)     |         17 | (16 + 1) x 1                       |
| **Total**            | **11,041** |                                    |

### Why This Size Matters

With only 4 training bearings per LOBO fold (typically 200-2,000 windows depending on
bearing lifetime), the ratio of training samples to parameters is roughly 20:1 to 200:1.
A 100K-parameter model would have a ratio as low as 2:1, virtually guaranteeing overfitting.
The compact 11K architecture balances expressiveness with the regularization constraint
imposed by the small dataset.

### Key Design Decisions

1. **Linear output activation**: The output layer uses no activation function (linear).
   This allows the model to predict any RUL value, including slightly negative ones that
   are clipped to 0 post-prediction. Using sigmoid would constrain outputs to [0, 1] but
   would create vanishing gradient issues when the pre-activation logits saturate.

2. **Bidirectional LSTM**: The bidirectional wrapper processes the 10-step window in both
   directions, allowing the model to capture both "features are getting worse" (forward)
   and "features were better before" (backward) patterns. This doubles the LSTM parameters
   but provides a richer temporal representation.

3. **Window size of 10**: Each window spans 10 consecutive files. At the XJTU-SY sampling
   interval, this represents roughly 10 minutes of operation. This is long enough to
   capture degradation trends but short enough to have sufficient windows even from the
   shortest-lived bearing (Bearing2_4: 42 files yields 33 windows).

4. **Huber loss (delta=10.0 for two-stage, 0.08 for full-life)**: Huber loss is less
   sensitive to outlier RUL values than MSE, which is important because some degradation
   patterns are inherently noisy.

---

## 5. Training and Evaluation

### Training Configuration

From `configs/twostage_feature_lstm.yaml`:

| Parameter          | Value    | Rationale                              |
|--------------------|----------|----------------------------------------|
| Optimizer          | AdamW    | Weight decay for implicit regularization |
| Learning rate      | 0.001    | Standard starting point                |
| Weight decay       | 0.0001   | Mild regularization                    |
| Batch size         | 32       |                                        |
| Max epochs         | 100      | Usually stops at 20-40 via early stopping |
| Early stop patience| 7        | Monitors val_loss                      |
| LR reduce patience | 5        | Factor 0.5, min 1e-6                   |
| Loss               | Huber    | delta=10.0 (two-stage), delta=0.08 (full-life) |
| Seed               | 9        | Single seed; variance not reported     |

### Three Evaluation Protocols

The benchmark evaluates every model under three protocols, each reflecting a different
published methodology:

**1. LOBO (Leave-One-Bearing-Out) -- 15-fold CV**

- For each operating condition, train on 4 bearings, test on the held-out 5th
- 5 bearings x 3 conditions = 15 folds
- Strictest protocol: model must generalize to an unseen degradation pattern
- The model trains from scratch for each fold (no transfer learning)

**2. Jin et al. 2025 Fixed Split**

- Train on Bearing1_4 + Bearing3_2 (2 bearings)
- Test on the remaining 13 bearings (all 3 conditions)
- Inferred from Jin et al. 2025 results tables
- Tests extreme generalization: only 2 training bearings

**3. Sun et al. 2024 Fixed Split**

- Train on Bearing1_1, Bearing1_2, Bearing2_1, Bearing2_2 (4 bearings)
- Test on Bearing1_3-1_5, Bearing2_3-2_5 (6 bearings)
- Conditions 1-2 only (Condition 3 excluded entirely)
- Standard XJTU-SY convention

### RUL Normalization

All RUL values are normalized to the [0, 1] range. Each bearing's RUL decays linearly from
1.0 (start of life or onset) to 0.0 (failure). This makes RMSE comparable across bearings
with vastly different lifetimes (42 vs. 2,538 files).

In two-stage mode, pre-onset samples receive a constant RUL of max_rul (125), and
post-onset samples decay linearly to 0. In full-life mode, RUL decays from 1.0 over the
entire lifetime.

### Results

#### Aggregate Results (Normalized RMSE, lower is better)

| Model           | LOBO (15-fold) | Jin (fixed) | Sun (fixed) |
|-----------------|---------------:|------------:|------------:|
| **Feature LSTM**|      **0.160** |       0.302 |   **0.156** |
| LightGBM        |          0.234 |       0.284 |       0.227 |
| 1D CNN           |          0.251 |       0.280 |       0.199 |
| CNN2D            |          0.289 |   **0.262** |       0.229 |
| DTA-MLP          |          0.402 |       0.445 |       0.353 |

#### Per-Bearing LOBO Results (Normalized RMSE)

| Bearing     | Cond | Files | Feature LSTM | LightGBM | 1D CNN | CNN2D | DTA-MLP |
|-------------|------|------:|-------------:|---------:|-------:|------:|--------:|
| Bearing1_1  | 1    |   123 |        0.127 |    0.166 |  0.141 | 0.482 |   0.440 |
| Bearing1_2  | 1    |   161 |        0.219 |    0.231 |  0.204 | 0.367 |   0.477 |
| Bearing1_3  | 1    |   158 |        0.090 |    0.201 |  0.207 | 0.337 |   0.542 |
| Bearing1_4  | 1    |   122 |        0.149 |    0.294 |  0.272 | 0.303 |   0.298 |
| Bearing1_5  | 1    |    52 |        0.120 |    0.215 |  0.219 | 0.271 |   0.303 |
| Bearing2_1  | 2    |   491 |        0.188 |    0.272 |  0.280 | 0.258 |   0.289 |
| Bearing2_2  | 2    |   161 |        0.227 |    0.149 |  0.210 | 0.149 |   0.417 |
| Bearing2_3  | 2    |   533 |        0.189 |    0.094 |  0.068 | 0.129 |   0.515 |
| Bearing2_4  | 2    |    42 |        0.067 |    0.183 |  0.175 | 0.194 |   0.355 |
| Bearing2_5  | 2    |   339 |        0.143 |    0.291 |  0.194 | 0.138 |   0.342 |
| Bearing3_1  | 3    | 2,538 |        0.175 |    0.274 |  0.289 | 0.312 |   0.370 |
| Bearing3_2  | 3    | 2,496 |        0.162 |    0.271 |  0.290 | 0.347 |   0.535 |
| Bearing3_3  | 3    |   371 |        0.219 |    0.284 |  0.502 | 0.310 |   0.352 |
| Bearing3_4  | 3    | 1,515 |        0.184 |    0.288 |  0.224 | 0.244 |   0.281 |
| Bearing3_5  | 3    |   114 |        0.141 |    0.292 |  0.491 | 0.502 |   0.520 |
| **Mean**    |      |       |    **0.160** |**0.234** |**0.251**|**0.289**|**0.402**|

The Feature LSTM achieves the best RMSE on 9 out of 15 bearings. Its performance is
notably consistent: the worst-case bearing (Bearing2_2: 0.227) is still better than
most other models' averages.

### Metrics Suite

From `src/training/metrics.py`, every evaluation computes:

- **RMSE**: Primary comparison metric (normalized [0, 1] scale)
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **PHM08 Score**: Asymmetric scoring function that penalizes late predictions (predicting
  failure won't happen when it will) more heavily than early predictions. Uses
  exp(d/10) - 1 for late predictions vs. exp(-d/13) - 1 for early predictions.

---

## 6. Two-Stage Pipeline

The project implements a two-stage degradation detection + RUL prediction pipeline:

### Stage 1: Onset Detection

An LSTM classifier (5,793 parameters) predicts whether a bearing is healthy (0) or
degraded (1) based on sliding windows of 4 health indicators with z-score augmentation
(8 features total: kurtosis_h, kurtosis_v, rms_h, rms_v, plus their absolute values).

The onset labels are algorithmically derived for all 15 bearings with confidence levels
(13 high, 2 medium, 0 low). Detection methods vary by bearing: 4 use kurtosis-based
onset, 8 use RMS-based onset, and 3 use composite indicators.

Onset classifier performance (15-fold LOBO CV): F1 = 0.844 +/- 0.243.

### Stage 2: RUL Prediction

Once onset is detected, the Feature LSTM predicts RUL only in the degraded region.
Pre-onset samples receive a constant max_rul = 125. This focuses the model's capacity
on the part of the degradation curve where RUL is actually changing.

### When Two-Stage Helps

The two-stage approach helps most for bearings with a clear, late onset (e.g., Bearing2_1
with onset at file 452 of 491 -- 92% healthy life). Without onset detection, the model
must also learn to predict a flat RUL during the long healthy period, wasting capacity.

It helps less for bearings with gradual, ambiguous onset (e.g., Bearing3_2 with onset
range spanning files 1020-1441 of 2,496). The full-life configuration is an alternative
that skips onset detection and normalizes RUL to [0, 1] over the entire lifetime.

---

## 7. Why Feature LSTM Wins

### 1. Domain Knowledge as Inductive Bias

The 65 features encode decades of vibration analysis knowledge: kurtosis captures
impulsive defects (spalling, cage cracks), RMS tracks overall vibration energy, crest
factor detects localized peaks, and bearing characteristic frequencies (BPFO, BPFI, BSF,
FTF) target specific defect locations. A deep network would need to rediscover all of
this from raw waveforms.

### 2. Per-Bearing Normalization Solves the Scale Problem

Z-score normalization against each bearing's healthy baseline transforms the problem from
"predict RUL from absolute vibration levels" (condition-dependent) to "predict RUL from
deviations relative to this bearing's normal" (condition-invariant). This is the single
most impactful design decision -- without it, cross-condition evaluation would require
either condition-specific models or complex domain adaptation.

### 3. Small Model = Appropriate Regularization

With 4 training bearings per LOBO fold, each producing ~100-2,500 windows, a model with
11K parameters is near the sweet spot. Larger models (CNN2D: ~500K+ params, DTA-MLP:
~1M+ params) overfit to the training bearings' specific degradation patterns and fail
to generalize. The Feature LSTM's size constraint forces it to learn generic degradation
patterns.

### 4. Temporal Modeling Without Complexity

The sliding window of 10 timesteps captures degradation trends (is kurtosis increasing?
is RMS accelerating?) that a single-snapshot model like LightGBM cannot see. This
explains the gap between Feature LSTM (0.160) and LightGBM (0.234): the temporal context
provides information about degradation *rate* that improves RUL estimation.

---

## 8. Limitations and Future Work

### Honest Limitations

1. **Bearing3_1 remains difficult** (2,538 files, onset at 748). The degradation in this
   bearing is extremely gradual -- most "degraded" samples look nearly identical to
   healthy ones at the individual sample level. The onset classifier achieves AUC-ROC
   ~0.4 on this bearing. The Feature LSTM achieves 0.175 RMSE, which is close to the
   model mean, but this likely benefits from the z-score normalization masking the true
   difficulty.

2. **Test-time adaptation assumption**: The z-score normalization requires early data
   from the target bearing as a healthy baseline. In a true cold-start scenario (new
   bearing, no history), the model would need a different normalization strategy.

3. **Single seed**: All results use seed=9. Variance across random seeds is not reported.
   The LOBO protocol provides 15 folds of robustness, but the training initialization
   is fixed.

4. **Jin protocol weakness** (0.302 RMSE): Training on only 2 bearings (Bearing1_4 +
   Bearing3_2) from different conditions is insufficient for the Feature LSTM. The
   CNN2D outperforms it here (0.262), likely because the 2D CNN's architecture provides
   additional regularization through convolutional weight sharing.

5. **Degradation regime sensitivity**: The absolute z-score augmentation helps with
   bearings where kurtosis decreases during failure, but the model still has lower
   confidence on these atypical degradation patterns (Bearings 2_4, 2_5, 3_2).

### What I Would Try Next

- **Multi-task learning**: Joint onset detection + RUL prediction in a single model,
  sharing the LSTM encoder.
- **Attention over the feature window**: Replace the LSTM with a small Transformer
  (2-layer, 4-head) to see if self-attention over the 10-step window captures
  degradation dynamics more effectively.
- **Ensemble with LightGBM**: LightGBM excels on Bearing2_3 (0.094 vs. LSTM's 0.189)
  and Bearing2_2 (0.149 vs. 0.227). A simple average could capture both models' strengths.
- **Condition-aware normalization**: Instead of per-bearing z-scores, learn a
  condition-specific normalization layer that could enable cold-start predictions.

---

*Copyright 2026 Tobias Hoffmann. MIT License.*
