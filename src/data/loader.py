"""XJTU-SY Bearing Dataset Loader.

This module provides utilities for loading the XJTU-SY bearing dataset,
which contains run-to-failure vibration data from 15 bearings across
3 operating conditions.

Dataset structure:
    - 3 conditions: 35Hz12kN, 37.5Hz11kN, 40Hz10kN
    - 5 bearings per condition (15 total)
    - Each file: 32,768 samples at 25.6kHz (horizontal + vertical channels)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


# Dataset constants
SAMPLING_RATE = 25600  # 25.6 kHz
SAMPLES_PER_FILE = 32768
NUM_CHANNELS = 2
EXPECTED_SHAPE = (SAMPLES_PER_FILE, NUM_CHANNELS)

# Condition mappings
CONDITIONS = {
    "35Hz12kN": {"rpm": 2100, "load_kn": 12.0},
    "37.5Hz11kN": {"rpm": 2250, "load_kn": 11.0},
    "40Hz10kN": {"rpm": 2400, "load_kn": 10.0},
}

# Bearing IDs per condition
BEARINGS_PER_CONDITION = {
    "35Hz12kN": ["Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
    "37.5Hz11kN": ["Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4", "Bearing2_5"],
    "40Hz10kN": ["Bearing3_1", "Bearing3_2", "Bearing3_3", "Bearing3_4", "Bearing3_5"],
}


class XJTUBearingLoader:
    """Loader for the XJTU-SY bearing vibration dataset.

    Attributes:
        data_root: Root directory containing the dataset.
        cache_size: Maximum number of files to keep in memory cache.
    """

    def __init__(
        self,
        data_root: str | Path = "assets/Data/XJTU-SY_Bearing_Datasets",
        cache_size: int = 100,
    ) -> None:
        """Initialize the loader.

        Args:
            data_root: Path to the dataset root directory.
            cache_size: Maximum number of files to cache in memory.
        """
        self.data_root = Path(data_root)
        self._cache_size = cache_size
        self._validate_data_root()
        self._metadata: pd.DataFrame | None = None

        # Create cached load function
        self._cached_load = functools.lru_cache(maxsize=cache_size)(
            self._load_file_uncached
        )

    def _validate_data_root(self) -> None:
        """Validate that the data root exists and has expected structure."""
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        for condition in CONDITIONS:
            condition_path = self.data_root / condition
            if not condition_path.exists():
                raise FileNotFoundError(
                    f"Condition directory not found: {condition_path}"
                )

    def _load_file_uncached(self, path_str: str) -> np.ndarray:
        """Load a single CSV file (internal, for caching).

        Args:
            path_str: String path to the CSV file.

        Returns:
            Array of shape (32768, 2) with horizontal and vertical channels.
        """
        path = Path(path_str)
        df = pd.read_csv(path)
        data = df.values.astype(np.float32)

        if data.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Unexpected shape {data.shape} for {path}, expected {EXPECTED_SHAPE}"
            )

        return data

    def load_file(self, path: str | Path) -> np.ndarray:
        """Load a single CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            Array of shape (32768, 2) with horizontal and vertical channels.
            Column 0: horizontal vibration
            Column 1: vertical vibration
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return self._cached_load(str(path.resolve()))

    def load_bearing(
        self,
        condition: str,
        bearing_id: str,
    ) -> tuple[np.ndarray, list[str]]:
        """Load all files for a specific bearing.

        Args:
            condition: Operating condition (e.g., "35Hz12kN").
            bearing_id: Bearing identifier (e.g., "Bearing1_1").

        Returns:
            Tuple of:
                - Array of shape (num_files, 32768, 2) containing all signals
                - List of filenames in order
        """
        if condition not in CONDITIONS:
            raise ValueError(
                f"Unknown condition: {condition}. Must be one of {list(CONDITIONS.keys())}"
            )

        if bearing_id not in BEARINGS_PER_CONDITION[condition]:
            raise ValueError(
                f"Unknown bearing {bearing_id} for condition {condition}. "
                f"Must be one of {BEARINGS_PER_CONDITION[condition]}"
            )

        bearing_path = self.data_root / condition / bearing_id
        if not bearing_path.exists():
            raise FileNotFoundError(f"Bearing directory not found: {bearing_path}")

        # Get sorted file list (numerical order)
        csv_files = sorted(
            bearing_path.glob("*.csv"),
            key=lambda p: int(p.stem),
        )

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {bearing_path}")

        # Load all files
        signals = []
        filenames = []
        for csv_file in csv_files:
            signals.append(self.load_file(csv_file))
            filenames.append(csv_file.name)

        return np.stack(signals), filenames

    def iter_bearing(
        self,
        condition: str,
        bearing_id: str,
    ) -> Iterator[tuple[np.ndarray, str, int]]:
        """Iterate over files for a bearing (memory-efficient).

        Args:
            condition: Operating condition.
            bearing_id: Bearing identifier.

        Yields:
            Tuple of (signal array, filename, file index).
        """
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")

        bearing_path = self.data_root / condition / bearing_id
        csv_files = sorted(
            bearing_path.glob("*.csv"),
            key=lambda p: int(p.stem),
        )

        for idx, csv_file in enumerate(csv_files):
            yield self.load_file(csv_file), csv_file.name, idx

    def get_metadata(self) -> pd.DataFrame:
        """Get metadata for all files in the dataset.

        Returns:
            DataFrame with columns: condition, bearing_id, filename, file_idx,
            file_path, num_files_in_bearing.
        """
        if self._metadata is not None:
            return self._metadata

        records = []
        for condition, bearings in BEARINGS_PER_CONDITION.items():
            for bearing_id in bearings:
                bearing_path = self.data_root / condition / bearing_id
                if not bearing_path.exists():
                    continue

                csv_files = sorted(
                    bearing_path.glob("*.csv"),
                    key=lambda p: int(p.stem),
                )
                num_files = len(csv_files)

                for idx, csv_file in enumerate(csv_files):
                    records.append({
                        "condition": condition,
                        "bearing_id": bearing_id,
                        "filename": csv_file.name,
                        "file_idx": idx,
                        "file_path": str(csv_file),
                        "num_files_in_bearing": num_files,
                    })

        self._metadata = pd.DataFrame(records)
        return self._metadata

    def get_bearing_file_counts(self) -> dict[str, dict[str, int]]:
        """Get the number of files per bearing.

        Returns:
            Nested dict mapping condition -> bearing_id -> file_count.
        """
        counts: dict[str, dict[str, int]] = {}
        for condition, bearings in BEARINGS_PER_CONDITION.items():
            counts[condition] = {}
            for bearing_id in bearings:
                bearing_path = self.data_root / condition / bearing_id
                if bearing_path.exists():
                    counts[condition][bearing_id] = len(list(bearing_path.glob("*.csv")))
                else:
                    counts[condition][bearing_id] = 0
        return counts

    def clear_cache(self) -> None:
        """Clear the file cache."""
        self._cached_load.cache_clear()

    @property
    def cache_info(self) -> functools._CacheInfo:
        """Get cache statistics."""
        return self._cached_load.cache_info()
