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

"""Onset label loading and management for two-stage RUL prediction.

This module provides functions to load curated onset labels from YAML
configuration and apply them to features dataframes for training and evaluation.

Functions:
    load_onset_labels: Parse onset_labels.yaml into a dictionary
    get_onset_label: Get binary healthy/degraded label for a specific sample
    add_onset_column: Add is_degraded column to features DataFrame
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Default path to onset labels YAML
DEFAULT_ONSET_LABELS_PATH = Path(__file__).parent.parent.parent / "configs" / "onset_labels.yaml"


@dataclass
class OnsetLabelEntry:
    """Single bearing onset label entry.

    Attributes:
        bearing_id: Unique identifier (e.g., 'Bearing1_1')
        condition: Operating condition (e.g., '35Hz12kN')
        onset_file_idx: File index where degradation begins (0-indexed)
        confidence: Labeling confidence ('high', 'medium', 'low')
        detection_method: Health indicator used ('kurtosis', 'rms', 'composite')
        failure_mode: Documented failure mode (e.g., 'Outer race', 'Cage')
        onset_range: Optional [min_idx, max_idx] for ambiguous cases
        notes: Additional observations
    """

    bearing_id: str
    condition: str
    onset_file_idx: int
    confidence: str
    detection_method: str
    failure_mode: str = ""
    onset_range: tuple[int, int] | None = None
    notes: str = ""


def load_onset_labels(
    yaml_path: str | Path | None = None,
) -> dict[str, OnsetLabelEntry]:
    """Load onset labels from YAML configuration file.

    Parses the onset_labels.yaml file containing curated onset indices
    for all bearings. Returns a dictionary mapping bearing_id to label entry.

    Args:
        yaml_path: Path to YAML file. If None, uses default configs/onset_labels.yaml.

    Returns:
        Dictionary mapping bearing_id (str) to OnsetLabelEntry dataclass.

    Raises:
        FileNotFoundError: If YAML file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
        KeyError: If required fields are missing from YAML entries.

    Example:
        >>> labels = load_onset_labels()
        >>> labels['Bearing1_1'].onset_file_idx
        69
        >>> labels['Bearing1_1'].confidence
        'high'
    """
    if yaml_path is None:
        yaml_path = DEFAULT_ONSET_LABELS_PATH
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Onset labels file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if data is None or "bearings" not in data:
        raise KeyError("YAML file must contain 'bearings' key with list of entries")

    labels: dict[str, OnsetLabelEntry] = {}
    for entry in data["bearings"]:
        # Validate required fields
        required_fields = ["bearing_id", "condition", "onset_file_idx", "confidence", "detection_method"]
        for field in required_fields:
            if field not in entry:
                raise KeyError(f"Missing required field '{field}' in entry: {entry}")

        # Parse onset_range if present
        onset_range = None
        if "onset_range" in entry and entry["onset_range"] is not None:
            onset_range = tuple(entry["onset_range"])

        labels[entry["bearing_id"]] = OnsetLabelEntry(
            bearing_id=entry["bearing_id"],
            condition=entry["condition"],
            onset_file_idx=entry["onset_file_idx"],
            confidence=entry["confidence"],
            detection_method=entry["detection_method"],
            failure_mode=entry.get("failure_mode", ""),
            onset_range=onset_range,
            notes=entry.get("notes", ""),
        )

    return labels


def get_onset_label(
    bearing_id: str,
    file_idx: int,
    onset_labels: dict[str, OnsetLabelEntry],
) -> int:
    """Get binary healthy/degraded label for a specific sample.

    Args:
        bearing_id: Bearing identifier (e.g., 'Bearing1_1')
        file_idx: File index within bearing (0-indexed)
        onset_labels: Dictionary of onset labels from load_onset_labels()

    Returns:
        0 if file_idx < onset_file_idx (healthy)
        1 if file_idx >= onset_file_idx (degraded)

    Raises:
        KeyError: If bearing_id not found in onset_labels.

    Example:
        >>> labels = load_onset_labels()
        >>> get_onset_label('Bearing1_1', 50, labels)  # Before onset at 65
        0
        >>> get_onset_label('Bearing1_1', 70, labels)  # After onset at 65
        1
    """
    if bearing_id not in onset_labels:
        raise KeyError(f"Bearing '{bearing_id}' not found in onset labels")

    onset_idx = onset_labels[bearing_id].onset_file_idx
    return 0 if file_idx < onset_idx else 1


def add_onset_column(
    features_df: pd.DataFrame,
    onset_labels: dict[str, OnsetLabelEntry],
    missing_behavior: str = "warn",
) -> pd.DataFrame:
    """Add binary is_degraded column to features DataFrame.

    Creates a new column 'is_degraded' where:
    - 0 = healthy (file_idx < onset_file_idx)
    - 1 = degraded (file_idx >= onset_file_idx)

    Args:
        features_df: DataFrame with 'bearing_id' and 'file_idx' columns.
            The file_idx column should contain 0-indexed file indices.
        onset_labels: Dictionary of onset labels from load_onset_labels().
        missing_behavior: How to handle bearings not in onset_labels:
            - 'warn': Log warning and set is_degraded to NaN (default)
            - 'skip': Silently set is_degraded to NaN
            - 'error': Raise KeyError

    Returns:
        Copy of features_df with new 'is_degraded' column added.

    Raises:
        KeyError: If missing_behavior='error' and bearing not found.
        ValueError: If required columns missing from features_df.

    Example:
        >>> labels = load_onset_labels()
        >>> df = pd.DataFrame({
        ...     'bearing_id': ['Bearing1_1', 'Bearing1_1'],
        ...     'file_idx': [50, 70]
        ... })
        >>> result = add_onset_column(df, labels)
        >>> result['is_degraded'].tolist()
        [0, 1]
    """
    # Validate input DataFrame
    required_cols = ["bearing_id", "file_idx"]
    missing_cols = [c for c in required_cols if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create copy to avoid modifying original
    df = features_df.copy()

    # Initialize is_degraded column
    df["is_degraded"] = pd.NA

    # Track missing bearings for warning
    missing_bearings: set[str] = set()

    # Process each unique bearing
    for bearing_id in df["bearing_id"].unique():
        mask = df["bearing_id"] == bearing_id

        if bearing_id not in onset_labels:
            missing_bearings.add(bearing_id)
            if missing_behavior == "error":
                raise KeyError(f"Bearing '{bearing_id}' not found in onset labels")
            # Leave as NaN for 'warn' and 'skip'
            continue

        onset_idx = onset_labels[bearing_id].onset_file_idx
        # Vectorized comparison: 0 if file_idx < onset_idx, else 1
        df.loc[mask, "is_degraded"] = (df.loc[mask, "file_idx"] >= onset_idx).astype(int)

    # Warn about missing bearings if applicable
    if missing_bearings and missing_behavior == "warn":
        import warnings

        warnings.warn(
            f"Bearings not found in onset labels (set to NaN): {sorted(missing_bearings)}",
            UserWarning,
            stacklevel=2,
        )

    return df
