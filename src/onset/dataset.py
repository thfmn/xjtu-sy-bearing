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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import tensorflow as tf

    from src.onset.labels import OnsetLabelEntry

# Health indicator columns used as input features for onset classification
FEATURE_COLUMNS = ["h_kurtosis", "v_kurtosis", "h_rms", "v_rms"]
N_FEATURES = len(FEATURE_COLUMNS)
