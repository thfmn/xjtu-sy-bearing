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

"""Signal augmentation for vibration data in tf.data pipelines.

Provides augmentation transforms for raw vibration signals (32768, 2)
to improve generalization of deep learning models on small datasets.
Applied as a .map() step in the training tf.data.Dataset.
"""

from __future__ import annotations

import tensorflow as tf


def _add_gaussian_noise(signal: tf.Tensor) -> tf.Tensor:
    """Add Gaussian noise scaled to 1-5% of per-channel standard deviation."""
    std = tf.math.reduce_std(signal, axis=0, keepdims=True)  # (1, 2)
    noise_scale = tf.random.uniform((), minval=0.01, maxval=0.05)
    noise = tf.random.normal(tf.shape(signal)) * std * noise_scale
    return signal + noise


def _amplitude_scaling(signal: tf.Tensor) -> tf.Tensor:
    """Randomly scale amplitude by 0.8x–1.2x (same factor for both channels)."""
    scale = tf.random.uniform((), minval=0.8, maxval=1.2)
    return signal * scale


def _circular_time_shift(signal: tf.Tensor) -> tf.Tensor:
    """Circular-shift signal by up to 25% of its length."""
    max_shift = tf.shape(signal)[0] // 4  # 8192 samples max
    shift = tf.random.uniform((), minval=0, maxval=max_shift, dtype=tf.int32)
    return tf.roll(signal, shift=shift, axis=0)


def _channel_dropout(signal: tf.Tensor) -> tf.Tensor:
    """Zero out one randomly chosen channel (h or v)."""
    channel = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
    mask = 1.0 - tf.one_hot(channel, depth=2)  # e.g. [1, 0] or [0, 1]
    return signal * mask


def augment_signal(signal: tf.Tensor, rul: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentations to a raw vibration signal.

    Each transform is applied independently with its own probability,
    so a sample might get 0, 1, 2, or all transforms applied.

    Args:
        signal: Float32 tensor of shape (32768, 2) — horizontal and vertical
            vibration channels sampled at 25.6 kHz.
        rul: Float32 scalar — remaining useful life label (unchanged by augmentation).

    Returns:
        Tuple of (augmented_signal, rul). The signal shape remains (32768, 2).
    """
    # 1. Gaussian noise — 50% probability
    signal = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: _add_gaussian_noise(signal),
        lambda: signal,
    )

    # 2. Amplitude scaling — 50% probability
    signal = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: _amplitude_scaling(signal),
        lambda: signal,
    )

    # 3. Circular time shift — 50% probability
    signal = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: _circular_time_shift(signal),
        lambda: signal,
    )

    # 4. Channel dropout — 10% probability (aggressive, use sparingly)
    signal = tf.cond(
        tf.random.uniform(()) < 0.1,
        lambda: _channel_dropout(signal),
        lambda: signal,
    )

    return signal, rul
