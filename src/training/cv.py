"""Cross-validation utilities for bearing RUL prediction.

Provides cross-validation strategies specific to the XJTU-SY bearing dataset:
- Leave-one-bearing-out CV (within each condition)
- Leave-one-condition-out CV (for generalization testing)
- Stratified splitting utilities

The dataset has 3 operating conditions with 5 bearings each (15 total):
- 35Hz12kN: Bearing1_1 to Bearing1_5
- 37.5Hz11kN: Bearing2_1 to Bearing2_5
- 40Hz10kN: Bearing3_1 to Bearing3_5
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

# Dataset structure constants
CONDITIONS = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
BEARINGS_PER_CONDITION = 5
BEARING_IDS = {
    "35Hz12kN": [f"Bearing1_{i}" for i in range(1, 6)],
    "37.5Hz11kN": [f"Bearing2_{i}" for i in range(1, 6)],
    "40Hz10kN": [f"Bearing3_{i}" for i in range(1, 6)],
}


@dataclass
class CVFold:
    """Represents a single cross-validation fold.

    Attributes:
        fold_id: Unique identifier for this fold
        train_indices: Indices of training samples
        val_indices: Indices of validation samples
        train_bearings: Bearing IDs in training set
        val_bearings: Bearing IDs in validation set
        condition: Operating condition (for leave-one-bearing-out)
        description: Human-readable description of the fold
    """
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_bearings: list[str] = field(default_factory=list)
    val_bearings: list[str] = field(default_factory=list)
    condition: str | None = None
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"CVFold(id={self.fold_id}, "
            f"train={len(self.train_indices)}, "
            f"val={len(self.val_indices)}, "
            f"val_bearings={self.val_bearings})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "fold_id": self.fold_id,
            "train_indices": self.train_indices.tolist(),
            "val_indices": self.val_indices.tolist(),
            "train_bearings": self.train_bearings,
            "val_bearings": self.val_bearings,
            "condition": self.condition,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CVFold:
        """Create from dictionary."""
        return cls(
            fold_id=d["fold_id"],
            train_indices=np.array(d["train_indices"]),
            val_indices=np.array(d["val_indices"]),
            train_bearings=d["train_bearings"],
            val_bearings=d["val_bearings"],
            condition=d.get("condition"),
            description=d.get("description", ""),
        )


@dataclass
class CVSplit:
    """Collection of CV folds with metadata.

    Attributes:
        folds: List of CVFold objects
        strategy: CV strategy name (e.g., 'leave_one_bearing_out')
        random_seed: Random seed used for reproducibility
        n_samples: Total number of samples
        description: Description of the CV split
    """
    folds: list[CVFold]
    strategy: str
    random_seed: int | None = None
    n_samples: int = 0
    description: str = ""

    def __len__(self) -> int:
        return len(self.folds)

    def __iter__(self) -> Iterator[CVFold]:
        return iter(self.folds)

    def __getitem__(self, idx: int) -> CVFold:
        return self.folds[idx]

    def save(self, path: str | Path) -> None:
        """Save CV split to JSON file for reproducibility."""
        path = Path(path)
        data = {
            "strategy": self.strategy,
            "random_seed": self.random_seed,
            "n_samples": self.n_samples,
            "description": self.description,
            "n_folds": len(self.folds),
            "folds": [f.to_dict() for f in self.folds],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CVSplit:
        """Load CV split from JSON file."""
        with open(path) as f:
            data = json.load(f)
        folds = [CVFold.from_dict(f) for f in data["folds"]]
        return cls(
            folds=folds,
            strategy=data["strategy"],
            random_seed=data.get("random_seed"),
            n_samples=data.get("n_samples", 0),
            description=data.get("description", ""),
        )

    def summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of all folds."""
        rows = []
        for fold in self.folds:
            rows.append({
                "fold_id": fold.fold_id,
                "n_train": len(fold.train_indices),
                "n_val": len(fold.val_indices),
                "val_bearings": ", ".join(fold.val_bearings),
                "condition": fold.condition or "all",
            })
        return pd.DataFrame(rows)


def leave_one_bearing_out(
    df: pd.DataFrame,
    condition_col: str = "condition",
    bearing_col: str = "bearing_id",
) -> CVSplit:
    """Generate leave-one-bearing-out cross-validation folds.

    Creates folds where one bearing is held out for validation at a time,
    within each operating condition. This results in 5 folds per condition
    (15 total folds for the full XJTU-SY dataset).

    Args:
        df: DataFrame with bearing data (must have condition and bearing_id columns)
        condition_col: Name of the condition column
        bearing_col: Name of the bearing ID column

    Returns:
        CVSplit with 15 folds (5 per condition)

    Example:
        >>> features_df = pd.read_csv("outputs/features/features_v2.csv")
        >>> cv = leave_one_bearing_out(features_df)
        >>> for fold in cv:
        ...     X_train = features_df.iloc[fold.train_indices]
        ...     X_val = features_df.iloc[fold.val_indices]
    """
    folds = []
    fold_id = 0

    for condition in df[condition_col].unique():
        condition_mask = df[condition_col] == condition
        condition_indices = df.index[condition_mask].values

        bearings = df.loc[condition_mask, bearing_col].unique()

        for val_bearing in bearings:
            # Validation: the held-out bearing
            val_mask = (df[condition_col] == condition) & (df[bearing_col] == val_bearing)
            val_indices = df.index[val_mask].values

            # Training: all other bearings in this condition
            train_mask = (df[condition_col] == condition) & (df[bearing_col] != val_bearing)
            train_indices = df.index[train_mask].values

            train_bearings = [b for b in bearings if b != val_bearing]

            fold = CVFold(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                train_bearings=sorted(train_bearings),
                val_bearings=[val_bearing],
                condition=condition,
                description=f"Leave-out {val_bearing} from {condition}",
            )
            folds.append(fold)
            fold_id += 1

    return CVSplit(
        folds=folds,
        strategy="leave_one_bearing_out",
        n_samples=len(df),
        description="Leave-one-bearing-out CV within each condition (15 folds total)",
    )


def leave_one_condition_out(
    df: pd.DataFrame,
    condition_col: str = "condition",
    bearing_col: str = "bearing_id",
) -> CVSplit:
    """Generate leave-one-condition-out cross-validation folds.

    Creates 3 folds where each operating condition is held out for validation.
    This tests generalization across different operating conditions.

    Args:
        df: DataFrame with bearing data
        condition_col: Name of the condition column
        bearing_col: Name of the bearing ID column

    Returns:
        CVSplit with 3 folds (one per condition)

    Example:
        >>> features_df = pd.read_csv("outputs/features/features_v2.csv")
        >>> cv = leave_one_condition_out(features_df)
        >>> # Test how well model generalizes to unseen operating conditions
    """
    folds = []
    conditions = df[condition_col].unique()

    for fold_id, val_condition in enumerate(conditions):
        # Validation: all bearings in the held-out condition
        val_mask = df[condition_col] == val_condition
        val_indices = df.index[val_mask].values
        val_bearings = df.loc[val_mask, bearing_col].unique().tolist()

        # Training: all bearings in other conditions
        train_mask = df[condition_col] != val_condition
        train_indices = df.index[train_mask].values
        train_bearings = df.loc[train_mask, bearing_col].unique().tolist()

        fold = CVFold(
            fold_id=fold_id,
            train_indices=train_indices,
            val_indices=val_indices,
            train_bearings=sorted(train_bearings),
            val_bearings=sorted(val_bearings),
            condition=val_condition,
            description=f"Leave-out condition {val_condition}",
        )
        folds.append(fold)

    return CVSplit(
        folds=folds,
        strategy="leave_one_condition_out",
        n_samples=len(df),
        description="Leave-one-condition-out CV for generalization testing (3 folds)",
    )


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
    condition_col: str = "condition",
    bearing_col: str = "bearing_id",
) -> CVSplit:
    """Create a stratified train/test split.

    Splits data so that each condition is represented proportionally in both
    train and test sets. Within each condition, bearings are randomly assigned
    to train or test (not split within a bearing to avoid data leakage).

    Args:
        df: DataFrame with bearing data
        test_size: Fraction of bearings to use for testing (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        condition_col: Name of the condition column
        bearing_col: Name of the bearing ID column

    Returns:
        CVSplit with a single fold containing train/test indices
    """
    rng = np.random.RandomState(random_seed)

    train_indices = []
    val_indices = []
    train_bearings = []
    val_bearings = []

    for condition in df[condition_col].unique():
        condition_mask = df[condition_col] == condition
        bearings = df.loc[condition_mask, bearing_col].unique()

        # Randomly select bearings for test set
        n_test = max(1, int(len(bearings) * test_size))
        test_bearing_ids = rng.choice(bearings, size=n_test, replace=False)

        for bearing in bearings:
            bearing_mask = condition_mask & (df[bearing_col] == bearing)
            bearing_indices = df.index[bearing_mask].values

            if bearing in test_bearing_ids:
                val_indices.extend(bearing_indices)
                val_bearings.append(bearing)
            else:
                train_indices.extend(bearing_indices)
                train_bearings.append(bearing)

    fold = CVFold(
        fold_id=0,
        train_indices=np.array(train_indices),
        val_indices=np.array(val_indices),
        train_bearings=sorted(train_bearings),
        val_bearings=sorted(val_bearings),
        description=f"Stratified split with test_size={test_size}",
    )

    return CVSplit(
        folds=[fold],
        strategy="stratified_split",
        random_seed=random_seed,
        n_samples=len(df),
        description=f"Stratified train/test split (test_size={test_size}, seed={random_seed})",
    )


def time_series_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    condition_col: str = "condition",
    bearing_col: str = "bearing_id",
    file_idx_col: str = "file_idx",
) -> CVSplit:
    """Create time-aware cross-validation splits.

    For each bearing, splits the lifecycle into n_splits segments.
    Early portions are used for training, later portions for validation.
    This respects the temporal nature of RUL prediction (can't use future
    data to predict past).

    Args:
        df: DataFrame with bearing data
        n_splits: Number of time-based splits per bearing
        condition_col: Name of the condition column
        bearing_col: Name of the bearing ID column
        file_idx_col: Name of the file index column (for temporal ordering)

    Returns:
        CVSplit with n_splits folds
    """
    folds = []

    for fold_id in range(n_splits):
        train_indices = []
        val_indices = []
        train_bearings = []
        val_bearings = []

        for condition in df[condition_col].unique():
            for bearing in df[df[condition_col] == condition][bearing_col].unique():
                bearing_mask = (df[condition_col] == condition) & (df[bearing_col] == bearing)
                bearing_df = df[bearing_mask].sort_values(file_idx_col)
                bearing_indices = bearing_df.index.values
                n_samples = len(bearing_indices)

                # Calculate split point for this fold
                # Fold 0: first 20% train, rest val
                # Fold 4: first 80% train, last 20% val
                train_ratio = (fold_id + 1) / (n_splits + 1)
                split_idx = int(n_samples * train_ratio)

                train_indices.extend(bearing_indices[:split_idx])
                val_indices.extend(bearing_indices[split_idx:])

                if bearing not in train_bearings:
                    train_bearings.append(bearing)
                if bearing not in val_bearings:
                    val_bearings.append(bearing)

        fold = CVFold(
            fold_id=fold_id,
            train_indices=np.array(train_indices),
            val_indices=np.array(val_indices),
            train_bearings=sorted(train_bearings),
            val_bearings=sorted(val_bearings),
            description=f"Time-series split {fold_id + 1}/{n_splits}",
        )
        folds.append(fold)

    return CVSplit(
        folds=folds,
        strategy="time_series_split",
        n_samples=len(df),
        description=f"Time-aware CV split ({n_splits} folds, expanding window)",
    )


def fixed_split_jin(
    df: pd.DataFrame,
    condition_col: str = "condition",
    bearing_col: str = "bearing_id",
) -> CVSplit:
    """Generate a fixed train/test split reproducing the Jin et al. 2025 protocol.

    Train on Bearing1_4 + Bearing3_2, test on the remaining 13 bearings.
    This is the evaluation protocol used in "Enhanced bearing RUL prediction
    based on dynamic temporal attention and mixed MLP" (Jin et al., 2025).

    Note: The paper does not explicitly state its split. This reconstruction is
    inferred from results tables that exclude Bearing1_4 and Bearing3_2.

    Args:
        df: DataFrame with bearing data (must have condition and bearing_id columns)
        condition_col: Name of the condition column
        bearing_col: Name of the bearing ID column

    Returns:
        CVSplit with 1 fold: 2 training bearings, 13 test bearings
    """
    train_bearing_ids = {"Bearing1_4", "Bearing3_2"}
    all_bearings = set(df[bearing_col].unique())

    missing = train_bearing_ids - all_bearings
    if missing:
        raise ValueError(f"Training bearings not found in data: {missing}")

    train_mask = df[bearing_col].isin(train_bearing_ids)
    val_mask = ~train_mask

    train_indices = df.index[train_mask].values
    val_indices = df.index[val_mask].values
    val_bearings = sorted(df.loc[val_mask, bearing_col].unique().tolist())

    fold = CVFold(
        fold_id=0,
        train_indices=train_indices,
        val_indices=val_indices,
        train_bearings=sorted(train_bearing_ids),
        val_bearings=val_bearings,
        description="Jin et al. 2025 fixed split: train={Bearing1_4, Bearing3_2}, test=13 remaining",
    )

    return CVSplit(
        folds=[fold],
        strategy="jin_fixed",
        n_samples=len(df),
        description=(
            "Jin et al. 2025 fixed split: train on Bearing1_4 + Bearing3_2, "
            "test on 13 remaining bearings across all 3 conditions"
        ),
    )


def fixed_split_li(
    df: pd.DataFrame,
    condition_col: str = "condition",
    bearing_col: str = "bearing_id",
) -> CVSplit:
    """Generate a fixed train/test split reproducing the Sun et al. 2024 protocol.

    Uses only Conditions 1 and 2 (10 bearings total). Trains on Bearing1_1,
    Bearing1_2, Bearing2_1, Bearing2_2. Tests on Bearing1_3, Bearing1_4,
    Bearing1_5, Bearing2_3, Bearing2_4, Bearing2_5.

    Note: Sun et al. do not describe their train/test split. This follows the
    standard XJTU-SY convention of 2-train/3-test per condition. This
    assumption is documented in the paper.

    Condition 3 (40Hz/10kN) bearings are excluded entirely: their indices do
    not appear in either train or test sets.

    Args:
        df: DataFrame with bearing data (must have condition and bearing_id columns)
        condition_col: Name of the condition column
        bearing_col: Name of the bearing ID column

    Returns:
        CVSplit with 1 fold: 4 training bearings, 6 test bearings (Conditions 1-2 only)
    """
    train_bearing_ids = {"Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2"}
    test_bearing_ids = {
        "Bearing1_3", "Bearing1_4", "Bearing1_5",
        "Bearing2_3", "Bearing2_4", "Bearing2_5",
    }

    # Filter to Conditions 1-2 only
    cond3_bearings = {f"Bearing3_{i}" for i in range(1, 6)}
    relevant_mask = ~df[bearing_col].isin(cond3_bearings)
    relevant_df = df[relevant_mask]

    all_bearings = set(relevant_df[bearing_col].unique())
    expected = train_bearing_ids | test_bearing_ids
    missing = expected - all_bearings
    if missing:
        raise ValueError(f"Expected bearings not found in data: {missing}")

    train_mask = relevant_df[bearing_col].isin(train_bearing_ids)
    val_mask = relevant_df[bearing_col].isin(test_bearing_ids)

    train_indices = relevant_df.index[train_mask].values
    val_indices = relevant_df.index[val_mask].values

    fold = CVFold(
        fold_id=0,
        train_indices=train_indices,
        val_indices=val_indices,
        train_bearings=sorted(train_bearing_ids),
        val_bearings=sorted(test_bearing_ids),
        description=(
            "Sun et al. 2024 fixed split: train={B1_1, B1_2, B2_1, B2_2}, "
            "test={B1_3-B1_5, B2_3-B2_5}, Conditions 1-2 only"
        ),
    )

    return CVSplit(
        folds=[fold],
        strategy="li_fixed",
        n_samples=len(relevant_df),
        description=(
            "Sun et al. 2024 fixed split: 2-train/3-test per condition, "
            "Conditions 1-2 only (10 bearings, Condition 3 excluded)"
        ),
    )


def get_bearing_groups(
    df: pd.DataFrame,
    condition_col: str = "condition",
    bearing_col: str = "bearing_id",
) -> dict[str, list[str]]:
    """Get bearings grouped by condition.

    Args:
        df: DataFrame with bearing data
        condition_col: Name of the condition column
        bearing_col: Name of the bearing ID column

    Returns:
        Dictionary mapping conditions to lists of bearing IDs
    """
    groups = {}
    for condition in df[condition_col].unique():
        bearings = df[df[condition_col] == condition][bearing_col].unique()
        groups[condition] = sorted(bearings.tolist())
    return groups


def validate_no_leakage(fold: CVFold, df: pd.DataFrame, bearing_col: str = "bearing_id") -> bool:
    """Validate that there's no data leakage between train and val sets.

    Checks that no bearing appears in both training and validation sets.

    Args:
        fold: CVFold to validate
        df: DataFrame with bearing data
        bearing_col: Name of the bearing ID column

    Returns:
        True if no leakage detected, False otherwise
    """
    train_bearings = set(df.iloc[fold.train_indices][bearing_col].unique())
    val_bearings = set(df.iloc[fold.val_indices][bearing_col].unique())

    overlap = train_bearings & val_bearings
    return len(overlap) == 0


def generate_cv_folds(
    df: pd.DataFrame,
    strategy: str = "leave_one_bearing_out",
    output_path: str | Path | None = None,
    **kwargs,
) -> CVSplit:
    """Generate cross-validation folds using specified strategy.

    Args:
        df: DataFrame with bearing data
        strategy: One of 'leave_one_bearing_out', 'leave_one_condition_out',
                 'stratified', 'time_series'
        output_path: Optional path to save fold indices for reproducibility
        **kwargs: Additional arguments passed to the strategy function

    Returns:
        CVSplit object with generated folds

    Raises:
        ValueError: If unknown strategy specified
    """
    strategies = {
        "leave_one_bearing_out": leave_one_bearing_out,
        "leave_one_condition_out": leave_one_condition_out,
        "stratified": stratified_split,
        "time_series": time_series_split,
        "jin_fixed": fixed_split_jin,
        "li_fixed": fixed_split_li,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(strategies.keys())}")

    cv_split = strategies[strategy](df, **kwargs)

    # Validate no data leakage
    for fold in cv_split:
        if not validate_no_leakage(fold, df):
            raise RuntimeError(f"Data leakage detected in fold {fold.fold_id}")

    if output_path:
        cv_split.save(output_path)

    return cv_split
