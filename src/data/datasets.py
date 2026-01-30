"""tf.data Dataset builders for the XJTU-SY bearing training pipeline.

Provides dataset builder functions that take row indices from features_v2.csv
and return tf.data.Dataset objects for training DL models. Two input types
are supported:

- Raw signals: loads (32768, 2) CSV files on-the-fly
- Spectrograms: loads (128, 128, 2) .npy files on-the-fly (DATA-1, future)

These builders are designed to work with the CV framework (src/training/cv.py)
which provides train/val index arrays into a metadata DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def _load_csv_signal(path_bytes: tf.Tensor) -> tf.Tensor:
    """Load a raw signal CSV file and return a (32768, 2) float32 tensor.

    This is a Python function wrapped by tf.py_function for use in
    tf.data pipelines. The CSV files have a header row and 32768 data rows
    with 2 columns (horizontal and vertical vibration).

    Args:
        path_bytes: Scalar string tensor containing the file path.

    Returns:
        Float32 tensor of shape (32768, 2).
    """
    path_str = path_bytes.numpy().decode("utf-8")
    df = pd.read_csv(path_str)
    return tf.constant(df.values, dtype=tf.float32)


def build_raw_signal_dataset(
    metadata_df: pd.DataFrame,
    data_root: str | Path,
    indices: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 5000,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of raw vibration signals for given row indices.

    Loads CSV files on-the-fly using tf.py_function, constructs file paths
    from metadata columns (condition, bearing_id, filename), and pairs each
    signal with its RUL label.

    Args:
        metadata_df: DataFrame with columns: condition, bearing_id, filename, rul.
            Typically loaded from outputs/features/features_v2.csv.
        data_root: Root directory of the raw bearing CSV files.
            E.g., "assets/Data/XJTU-SY_Bearing_Datasets".
        indices: Array of row indices into metadata_df selecting the subset
            to include (e.g., train or val indices from CVFold).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer_size: Buffer size for tf.data.Dataset.shuffle().

    Returns:
        tf.data.Dataset yielding (signal, rul) batches.
        signal shape: (batch_size, 32768, 2) float32
        rul shape: (batch_size,) float32
    """
    data_root = Path(data_root)
    subset = metadata_df.iloc[indices]

    # Build file paths: {data_root}/{condition}/{bearing_id}/{filename}
    paths = [
        str(data_root / row.condition / row.bearing_id / row.filename)
        for _, row in subset.iterrows()
    ]
    rul_values = subset["rul"].values.astype(np.float32)

    # Create dataset from path strings and RUL labels
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    rul_ds = tf.data.Dataset.from_tensor_slices(rul_values)
    ds = tf.data.Dataset.zip((path_ds, rul_ds))

    # Map: load CSV signal for each path
    def _load_sample(path_tensor: tf.Tensor, rul: tf.Tensor):
        signal = tf.py_function(_load_csv_signal, [path_tensor], tf.float32)
        signal.set_shape((32768, 2))
        return signal, rul

    ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
