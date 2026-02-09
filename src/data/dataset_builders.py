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

"""tf.data Dataset builders for the XJTU-SY bearing training pipeline.

Provides dataset builder functions that take row indices from features_v2.csv
and return tf.data.Dataset objects for training DL models. Two input types
are supported:

- Raw signals: loads (32768, 2) CSV files on-the-fly
- Spectrograms: loads (128, 128, 2) .npy files on-the-fly

These builders are designed to work with the CV framework (src/training/cv.py)
which provides train/val index arrays into a metadata DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.augmentation import augment_signal
from src.models.registry import get_model_info


SPECTROGRAM_DIR = "outputs/spectrograms/stft"
CWT_DIR = "outputs/spectrograms/cwt"


def _load_npy(path_bytes: tf.Tensor) -> tf.Tensor:
    """Load a .npy spectrogram file and return a (128, 128, 2) float32 tensor.

    This is a Python function wrapped by tf.py_function for use in
    tf.data pipelines. Each .npy file contains a (128, 128, 2) array
    of dual-channel STFT spectrogram data.

    Args:
        path_bytes: Scalar string tensor containing the .npy file path.

    Returns:
        Float32 tensor of shape (128, 128, 2).
    """
    path_str = path_bytes.numpy().decode("utf-8")
    arr = np.load(path_str)
    return tf.constant(arr, dtype=tf.float32)


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
    augment: bool = False,
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
        augment: Whether to apply signal augmentation (training only).

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

    # Apply augmentation after loading, before batching (training only)
    if augment:
        ds = ds.map(augment_signal, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_spectrogram_dataset(
    metadata_df: pd.DataFrame,
    spectrogram_dir: str | Path,
    indices: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 5000,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of STFT spectrograms for given row indices.

    Loads .npy files on-the-fly using tf.py_function, constructs file paths
    from metadata columns (condition, bearing_id, filename), and pairs each
    spectrogram with its RUL label.

    Path formula: {spectrogram_dir}/condition={condition}/bearing_id={bearing_id}/{stem}.npy
    where stem = filename without the .csv extension.

    Args:
        metadata_df: DataFrame with columns: condition, bearing_id, filename, rul.
            Typically loaded from outputs/features/features_v2.csv.
        spectrogram_dir: Root directory of the .npy spectrogram files.
            E.g., "outputs/spectrograms/stft".
        indices: Array of row indices into metadata_df selecting the subset
            to include (e.g., train or val indices from CVFold).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer_size: Buffer size for tf.data.Dataset.shuffle().

    Returns:
        tf.data.Dataset yielding (spectrogram, rul) batches.
        spectrogram shape: (batch_size, 128, 128, 2) float32
        rul shape: (batch_size,) float32
    """
    spectrogram_dir = Path(spectrogram_dir)
    subset = metadata_df.iloc[indices]

    # Build file paths: {spectrogram_dir}/condition={cond}/bearing_id={id}/{stem}.npy
    paths = [
        str(
            spectrogram_dir
            / f"condition={row.condition}"
            / f"bearing_id={row.bearing_id}"
            / f"{Path(row.filename).stem}.npy"
        )
        for _, row in subset.iterrows()
    ]
    rul_values = subset["rul"].values.astype(np.float32)

    # Create dataset from path strings and RUL labels
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    rul_ds = tf.data.Dataset.from_tensor_slices(rul_values)
    ds = tf.data.Dataset.zip((path_ds, rul_ds))

    # Map: load .npy spectrogram for each path
    def _load_sample(path_tensor: tf.Tensor, rul: tf.Tensor):
        spectrogram = tf.py_function(_load_npy, [path_tensor], tf.float32)
        spectrogram.set_shape((128, 128, 2))
        return spectrogram, rul

    ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_cwt_dataset(
    metadata_df: pd.DataFrame,
    cwt_dir: str | Path,
    indices: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 5000,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of CWT scaleograms for given row indices.

    Loads .npy files on-the-fly using tf.py_function, constructs file paths
    from metadata columns (condition, bearing_id, filename), and pairs each
    scaleogram with its RUL label.

    Path formula: {cwt_dir}/condition={condition}/bearing_id={bearing_id}/{stem}.npy

    Args:
        metadata_df: DataFrame with columns: condition, bearing_id, filename, rul.
        cwt_dir: Root directory of the .npy CWT scaleogram files.
            E.g., "outputs/spectrograms/cwt".
        indices: Array of row indices into metadata_df selecting the subset.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer_size: Buffer size for tf.data.Dataset.shuffle().

    Returns:
        tf.data.Dataset yielding (scaleogram, rul) batches.
        scaleogram shape: (batch_size, 64, 128, 2) float32
        rul shape: (batch_size,) float32
    """
    cwt_dir = Path(cwt_dir)
    subset = metadata_df.iloc[indices]

    # Build file paths: {cwt_dir}/condition={cond}/bearing_id={id}/{stem}.npy
    paths = [
        str(
            cwt_dir
            / f"condition={row.condition}"
            / f"bearing_id={row.bearing_id}"
            / f"{Path(row.filename).stem}.npy"
        )
        for _, row in subset.iterrows()
    ]
    rul_values = subset["rul"].values.astype(np.float32)

    # Create dataset from path strings and RUL labels
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    rul_ds = tf.data.Dataset.from_tensor_slices(rul_values)
    ds = tf.data.Dataset.zip((path_ds, rul_ds))

    # Map: load .npy scaleogram for each path
    def _load_sample(path_tensor: tf.Tensor, rul: tf.Tensor):
        scaleogram = tf.py_function(_load_npy, [path_tensor], tf.float32)
        scaleogram.set_shape((64, 128, 2))
        return scaleogram, rul

    ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_dataset_for_model(
    model_name: str,
    metadata_df: pd.DataFrame,
    indices: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    data_root: str | Path = "assets/Data/XJTU-SY_Bearing_Datasets",
    spectrogram_dir: str | Path = "outputs/spectrograms/stft",
    cwt_dir: str | Path = "outputs/spectrograms/cwt",
    augment: bool = False,
) -> tf.data.Dataset:
    """Build the appropriate tf.data.Dataset for a model based on its registry input type.

    Looks up the model in the registry and dispatches to the correct dataset
    builder (raw signal, spectrogram, or CWT scaleogram).

    Args:
        model_name: Registered model name (e.g., "cnn1d_baseline", "cnn2d_simple").
        metadata_df: DataFrame with columns: condition, bearing_id, filename, rul.
        indices: Array of row indices into metadata_df.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        data_root: Root directory of the raw bearing CSV files.
        spectrogram_dir: Root directory of the .npy STFT spectrogram files.
        cwt_dir: Root directory of the .npy CWT scaleogram files.
        augment: Whether to apply data augmentation (training only).

    Returns:
        tf.data.Dataset yielding (input, rul) batches appropriate for the model.

    Raises:
        KeyError: If model_name is not found in the registry.
        ValueError: If the model's input_type is not supported.
    """
    info = get_model_info(model_name)

    if info.input_type == "raw_signal":
        return build_raw_signal_dataset(
            metadata_df=metadata_df,
            data_root=data_root,
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
            augment=augment,
        )
    elif info.input_type == "spectrogram":
        return build_spectrogram_dataset(
            metadata_df=metadata_df,
            spectrogram_dir=spectrogram_dir,
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    elif info.input_type == "cwt_scaleogram":
        return build_cwt_dataset(
            metadata_df=metadata_df,
            cwt_dir=cwt_dir,
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        raise ValueError(
            f"Unsupported input_type '{info.input_type}' for model '{model_name}'. "
            f"Expected 'raw_signal', 'spectrogram', or 'cwt_scaleogram'."
        )
