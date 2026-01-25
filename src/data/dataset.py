"""TensorFlow Dataset generators for XJTU-SY bearing data.

This module provides tf.data.Dataset generators for loading raw vibration signals
with windowing support, optimized for batch training of deep learning models.

Features:
    - Multiple window sizes (2048, 4096, 8192, 32768)
    - Configurable overlap for sliding window
    - Automatic RUL label generation
    - Prefetching and parallel loading for performance
    - Integration with existing data loader and RUL utilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tensorflow as tf

from src.data.loader import (
    BEARINGS_PER_CONDITION,
    CONDITIONS,
    SAMPLES_PER_FILE,
    XJTUBearingLoader,
)
from src.data.rul_labels import RULStrategy, generate_rul_labels
from src.data.windowing import (
    DEFAULT_WINDOW_SIZE,
    WINDOW_SIZES,
    WindowConfig,
    calculate_num_windows_per_file,
    extract_windows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class DatasetConfig:
    """Configuration for dataset generation.

    Attributes:
        window_size: Number of samples per window. Must be one of WINDOW_SIZES.
        overlap: Fraction of overlap between consecutive windows (0.0 to 0.9).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer_size: Buffer size for shuffling (if shuffle=True).
        rul_strategy: Strategy for generating RUL labels.
        max_rul: Maximum RUL value for piecewise_linear strategy.
        normalize_signals: Whether to normalize signals to zero mean, unit variance.
        prefetch_buffer: Number of batches to prefetch (tf.data.AUTOTUNE if -1).
        num_parallel_calls: Number of parallel calls for mapping (tf.data.AUTOTUNE if -1).
        cache: Whether to cache the dataset in memory.
    """

    window_size: int = DEFAULT_WINDOW_SIZE
    overlap: float = 0.0
    batch_size: int = 32
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    rul_strategy: RULStrategy = "piecewise_linear"
    max_rul: float = 125.0
    normalize_signals: bool = False
    prefetch_buffer: int = -1  # AUTOTUNE
    num_parallel_calls: int = -1  # AUTOTUNE
    cache: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.window_size not in WINDOW_SIZES:
            raise ValueError(
                f"window_size must be one of {WINDOW_SIZES}, got {self.window_size}"
            )
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    @property
    def window_config(self) -> WindowConfig:
        """Get WindowConfig from dataset config."""
        return WindowConfig(window_size=self.window_size, overlap=self.overlap)


@dataclass
class BearingDataset:
    """TensorFlow Dataset wrapper for bearing vibration data.

    Provides a clean interface for creating tf.data.Dataset objects
    from the XJTU-SY bearing dataset with windowing and RUL labels.

    Attributes:
        data_root: Path to the dataset root directory.
        config: DatasetConfig object with all configuration options.
    """

    data_root: str | Path = "assets/Data/XJTU-SY_Bearing_Datasets"
    config: DatasetConfig = field(default_factory=DatasetConfig)
    _loader: XJTUBearingLoader | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the data loader."""
        self._loader = XJTUBearingLoader(self.data_root)

    @property
    def loader(self) -> XJTUBearingLoader:
        """Get the data loader instance."""
        if self._loader is None:
            self._loader = XJTUBearingLoader(self.data_root)
        return self._loader

    def _get_bearing_data(
        self,
        condition: str,
        bearing_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load all windows and labels for a single bearing.

        Args:
            condition: Operating condition (e.g., "35Hz12kN").
            bearing_id: Bearing identifier (e.g., "Bearing1_1").

        Returns:
            Tuple of (windows, labels) arrays.
        """
        # Load all signals for the bearing
        signals, filenames = self.loader.load_bearing(condition, bearing_id)
        num_files = len(filenames)

        # Generate RUL labels for each file
        rul_per_file = generate_rul_labels(
            num_files,
            strategy=self.config.rul_strategy,
            max_rul=self.config.max_rul,
        )

        # Extract windows from each file
        all_windows = []
        all_labels = []

        window_config = self.config.window_config

        for file_idx, signal in enumerate(signals):
            windows = extract_windows(signal, config=window_config)
            num_windows = len(windows)

            # All windows from the same file share the same RUL
            labels = np.full(num_windows, rul_per_file[file_idx], dtype=np.float32)

            all_windows.append(windows)
            all_labels.append(labels)

        windows = np.concatenate(all_windows, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return windows, labels

    def _get_bearings_data(
        self,
        bearings: Sequence[tuple[str, str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load all windows and labels for multiple bearings.

        Args:
            bearings: List of (condition, bearing_id) tuples.

        Returns:
            Tuple of (windows, labels) arrays concatenated from all bearings.
        """
        all_windows = []
        all_labels = []

        for condition, bearing_id in bearings:
            windows, labels = self._get_bearing_data(condition, bearing_id)
            all_windows.append(windows)
            all_labels.append(labels)

        return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)

    def create_dataset(
        self,
        conditions: Sequence[str] | None = None,
        bearings: Sequence[tuple[str, str]] | None = None,
        exclude_bearings: Sequence[tuple[str, str]] | None = None,
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset from specified bearings.

        Args:
            conditions: List of conditions to include. If None, uses all conditions.
            bearings: Explicit list of (condition, bearing_id) tuples. Overrides conditions.
            exclude_bearings: List of (condition, bearing_id) to exclude.

        Returns:
            tf.data.Dataset yielding (windows, labels) batches.
            Windows shape: (batch_size, window_size, 2)
            Labels shape: (batch_size,)
        """
        # Determine which bearings to include
        if bearings is not None:
            selected_bearings = list(bearings)
        else:
            if conditions is None:
                conditions = list(CONDITIONS.keys())

            selected_bearings = []
            for condition in conditions:
                for bearing_id in BEARINGS_PER_CONDITION[condition]:
                    selected_bearings.append((condition, bearing_id))

        # Exclude specified bearings
        if exclude_bearings:
            exclude_set = set(exclude_bearings)
            selected_bearings = [b for b in selected_bearings if b not in exclude_set]

        if not selected_bearings:
            raise ValueError("No bearings selected for dataset")

        # Load all data
        windows, labels = self._get_bearings_data(selected_bearings)

        # Optional normalization
        if self.config.normalize_signals:
            windows = self._normalize(windows)

        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((windows, labels))

        # Apply optimizations
        if self.config.cache:
            dataset = dataset.cache()

        if self.config.shuffle:
            dataset = dataset.shuffle(
                buffer_size=min(self.config.shuffle_buffer_size, len(labels))
            )

        dataset = dataset.batch(self.config.batch_size)

        # Prefetch
        prefetch_buffer = (
            tf.data.AUTOTUNE
            if self.config.prefetch_buffer == -1
            else self.config.prefetch_buffer
        )
        dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    def create_generator_dataset(
        self,
        conditions: Sequence[str] | None = None,
        bearings: Sequence[tuple[str, str]] | None = None,
        exclude_bearings: Sequence[tuple[str, str]] | None = None,
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset using generator for memory efficiency.

        This version loads data on-the-fly instead of loading all at once,
        suitable for very large datasets that don't fit in memory.

        Args:
            conditions: List of conditions to include. If None, uses all conditions.
            bearings: Explicit list of (condition, bearing_id) tuples.
            exclude_bearings: List of (condition, bearing_id) to exclude.

        Returns:
            tf.data.Dataset yielding (windows, labels) batches.
        """
        # Determine which bearings to include
        if bearings is not None:
            selected_bearings = list(bearings)
        else:
            if conditions is None:
                conditions = list(CONDITIONS.keys())

            selected_bearings = []
            for condition in conditions:
                for bearing_id in BEARINGS_PER_CONDITION[condition]:
                    selected_bearings.append((condition, bearing_id))

        if exclude_bearings:
            exclude_set = set(exclude_bearings)
            selected_bearings = [b for b in selected_bearings if b not in exclude_set]

        if not selected_bearings:
            raise ValueError("No bearings selected for dataset")

        # Pre-compute file paths and RUL labels
        file_info = []
        for condition, bearing_id in selected_bearings:
            bearing_path = Path(self.data_root) / condition / bearing_id
            csv_files = sorted(
                bearing_path.glob("*.csv"),
                key=lambda p: int(p.stem),
            )
            num_files = len(csv_files)
            rul_values = generate_rul_labels(
                num_files,
                strategy=self.config.rul_strategy,
                max_rul=self.config.max_rul,
            )
            for idx, csv_file in enumerate(csv_files):
                file_info.append((str(csv_file), rul_values[idx]))

        window_config = self.config.window_config
        normalize = self.config.normalize_signals

        def generator():
            for file_path, rul in file_info:
                # Load signal
                signal = self.loader.load_file(file_path)

                # Extract windows
                windows = extract_windows(signal, config=window_config)

                if normalize:
                    windows = (windows - windows.mean()) / (windows.std() + 1e-8)

                # Yield each window with its RUL
                for window in windows:
                    yield window.astype(np.float32), np.float32(rul)

        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=(self.config.window_size, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
        )

        # Apply optimizations
        if self.config.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.config.shuffle_buffer_size
            )

        dataset = dataset.batch(self.config.batch_size)

        prefetch_buffer = (
            tf.data.AUTOTUNE
            if self.config.prefetch_buffer == -1
            else self.config.prefetch_buffer
        )
        dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    def _normalize(self, windows: np.ndarray) -> np.ndarray:
        """Normalize windows to zero mean, unit variance.

        Normalization is applied per-window across all samples and channels.

        Args:
            windows: Array of shape (num_windows, window_size, channels).

        Returns:
            Normalized array with same shape.
        """
        # Compute mean and std per window
        mean = windows.mean(axis=(1, 2), keepdims=True)
        std = windows.std(axis=(1, 2), keepdims=True) + 1e-8
        return (windows - mean) / std

    def get_dataset_size(
        self,
        conditions: Sequence[str] | None = None,
        bearings: Sequence[tuple[str, str]] | None = None,
        exclude_bearings: Sequence[tuple[str, str]] | None = None,
    ) -> int:
        """Calculate the total number of windows in the dataset.

        Args:
            conditions: List of conditions to include.
            bearings: Explicit list of (condition, bearing_id) tuples.
            exclude_bearings: List of (condition, bearing_id) to exclude.

        Returns:
            Total number of windows (samples) in the dataset.
        """
        # Determine which bearings to include
        if bearings is not None:
            selected_bearings = list(bearings)
        else:
            if conditions is None:
                conditions = list(CONDITIONS.keys())

            selected_bearings = []
            for condition in conditions:
                for bearing_id in BEARINGS_PER_CONDITION[condition]:
                    selected_bearings.append((condition, bearing_id))

        if exclude_bearings:
            exclude_set = set(exclude_bearings)
            selected_bearings = [b for b in selected_bearings if b not in exclude_set]

        # Calculate windows per file
        windows_per_file = calculate_num_windows_per_file(
            window_size=self.config.window_size,
            overlap=self.config.overlap,
        )

        # Count total files
        metadata = self.loader.get_metadata()
        total_files = 0
        for condition, bearing_id in selected_bearings:
            mask = (metadata["condition"] == condition) & (metadata["bearing_id"] == bearing_id)
            total_files += mask.sum()

        return total_files * windows_per_file

    def create_train_val_datasets(
        self,
        val_bearing: tuple[str, str],
        conditions: Sequence[str] | None = None,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create train and validation datasets with leave-one-bearing-out split.

        Args:
            val_bearing: (condition, bearing_id) tuple for validation.
            conditions: List of conditions to use. If None, uses all conditions.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        # Training: all bearings except val_bearing
        train_ds = self.create_dataset(
            conditions=conditions,
            exclude_bearings=[val_bearing],
        )

        # Validation: only val_bearing (no shuffle)
        val_config = DatasetConfig(
            window_size=self.config.window_size,
            overlap=self.config.overlap,
            batch_size=self.config.batch_size,
            shuffle=False,  # No shuffle for validation
            rul_strategy=self.config.rul_strategy,
            max_rul=self.config.max_rul,
            normalize_signals=self.config.normalize_signals,
            prefetch_buffer=self.config.prefetch_buffer,
            cache=self.config.cache,
        )

        val_dataset = BearingDataset(self.data_root, val_config)
        val_ds = val_dataset.create_dataset(bearings=[val_bearing])

        return train_ds, val_ds


def create_bearing_dataset(
    data_root: str | Path = "assets/Data/XJTU-SY_Bearing_Datasets",
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: float = 0.0,
    batch_size: int = 32,
    shuffle: bool = True,
    rul_strategy: RULStrategy = "piecewise_linear",
    max_rul: float = 125.0,
    normalize: bool = False,
    conditions: Sequence[str] | None = None,
    bearings: Sequence[tuple[str, str]] | None = None,
    exclude_bearings: Sequence[tuple[str, str]] | None = None,
) -> tf.data.Dataset:
    """Convenience function to create a bearing dataset.

    Args:
        data_root: Path to the dataset root directory.
        window_size: Number of samples per window.
        overlap: Fraction of overlap between windows.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        rul_strategy: Strategy for RUL label generation.
        max_rul: Maximum RUL for piecewise_linear strategy.
        normalize: Whether to normalize signals.
        conditions: List of conditions to include.
        bearings: Explicit list of (condition, bearing_id) tuples.
        exclude_bearings: List of (condition, bearing_id) to exclude.

    Returns:
        tf.data.Dataset yielding (windows, labels) batches.

    Example:
        >>> ds = create_bearing_dataset(
        ...     window_size=8192,
        ...     overlap=0.5,
        ...     batch_size=32,
        ...     conditions=["35Hz12kN"],
        ... )
        >>> for windows, labels in ds.take(1):
        ...     print(windows.shape, labels.shape)
        (32, 8192, 2) (32,)
    """
    config = DatasetConfig(
        window_size=window_size,
        overlap=overlap,
        batch_size=batch_size,
        shuffle=shuffle,
        rul_strategy=rul_strategy,
        max_rul=max_rul,
        normalize_signals=normalize,
    )

    dataset = BearingDataset(data_root, config)
    return dataset.create_dataset(
        conditions=conditions,
        bearings=bearings,
        exclude_bearings=exclude_bearings,
    )


def get_all_bearings() -> list[tuple[str, str]]:
    """Get a list of all (condition, bearing_id) tuples in the dataset.

    Returns:
        List of 15 (condition, bearing_id) tuples.
    """
    bearings = []
    for condition, bearing_ids in BEARINGS_PER_CONDITION.items():
        for bearing_id in bearing_ids:
            bearings.append((condition, bearing_id))
    return bearings
