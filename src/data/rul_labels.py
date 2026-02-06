"""RUL (Remaining Useful Life) Label Generation for XJTU-SY Bearing Dataset.

This module provides multiple RUL labeling strategies for bearing degradation:
- Piecewise linear: Caps early-life RUL at max_rul, then linear decay to 0
- Linear decay: Simple linear decay from num_files to 0
- Exponential decay: Non-linear exponential curve to 0

Reference: PHM literature commonly uses piecewise linear with max_rul=125.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes


RULStrategy = Literal["piecewise_linear", "linear", "exponential", "twostage"]


def piecewise_linear_rul(
    num_files: int,
    max_rul: float = 125.0,
) -> np.ndarray:
    """Generate piecewise linear RUL labels.

    Early life is capped at max_rul, then linearly decays to 0 at failure.
    This is the standard approach from PHM literature.

    Args:
        num_files: Total number of files (samples) for the bearing.
        max_rul: Maximum RUL value to cap early-life samples at (default: 125).

    Returns:
        Array of shape (num_files,) with RUL values.
        Values are monotonically decreasing, ending at 0.

    Example:
        >>> rul = piecewise_linear_rul(200, max_rul=125)
        >>> rul[0]  # First sample capped at 125
        125.0
        >>> rul[-1]  # Last sample (failure)
        0.0
    """
    if num_files <= 0:
        raise ValueError(f"num_files must be positive, got {num_files}")

    if num_files == 1:
        return np.array([0.0])

    # Linear decay from (num_files - 1) to 0
    linear_rul = np.arange(num_files - 1, -1, -1, dtype=np.float32)

    # Cap at max_rul
    return np.minimum(linear_rul, max_rul)


def linear_rul(num_files: int) -> np.ndarray:
    """Generate linear decay RUL labels.

    Simple linear decay from num_files-1 to 0.
    Useful as a baseline or for short-lived bearings.

    Args:
        num_files: Total number of files (samples) for the bearing.

    Returns:
        Array of shape (num_files,) with RUL values.
        Values decrease linearly from num_files-1 to 0.

    Example:
        >>> rul = linear_rul(100)
        >>> rul[0]
        99.0
        >>> rul[-1]
        0.0
    """
    if num_files <= 0:
        raise ValueError(f"num_files must be positive, got {num_files}")

    if num_files == 1:
        return np.array([0.0])

    return np.arange(num_files - 1, -1, -1, dtype=np.float32)


def exponential_rul(
    num_files: int,
    decay_rate: float = 3.0,
) -> np.ndarray:
    """Generate exponential decay RUL labels.

    Uses exponential decay: RUL = max_rul * exp(-decay_rate * t / T)
    where t is time from start and T is total lifetime.

    This produces a smooth non-linear curve that decays faster near end-of-life.

    Args:
        num_files: Total number of files (samples) for the bearing.
        decay_rate: Controls steepness of decay. Higher = steeper at end.

    Returns:
        Array of shape (num_files,) with RUL values.
        Values are monotonically decreasing with exponential profile.

    Example:
        >>> rul = exponential_rul(100, decay_rate=3.0)
        >>> rul[0] > rul[50] > rul[-1]
        True
        >>> rul[-1]  # Approaches but not exactly 0
        0.049...
    """
    if num_files <= 0:
        raise ValueError(f"num_files must be positive, got {num_files}")

    if num_files == 1:
        return np.array([0.0])

    # Normalized time from 0 to 1
    t_normalized = np.linspace(0, 1, num_files)

    # Exponential decay scaled to start at num_files-1
    max_rul = num_files - 1
    rul = max_rul * np.exp(-decay_rate * t_normalized)

    return rul.astype(np.float32)


def compute_twostage_rul(
    num_files: int,
    onset_idx: int | None,
    max_rul: float = 125.0,
) -> np.ndarray:
    """Generate two-stage RUL labels using onset detection.

    Pre-onset samples receive a constant max_rul value (bearing is healthy,
    RUL is not decaying yet). Post-onset samples receive piecewise-linear
    decay from onset to failure.

    Args:
        num_files: Total number of files (samples) for the bearing.
        onset_idx: File index where degradation begins. If None, all samples
            receive max_rul (no onset detected).
        max_rul: Maximum RUL value (default: 125).

    Returns:
        Array of shape (num_files,) with RUL values.
        Pre-onset: constant max_rul.
        Post-onset: linear decay from min(max_rul, files_remaining) to 0.

    Example:
        >>> rul = compute_twostage_rul(200, onset_idx=100, max_rul=125)
        >>> rul[0]  # Pre-onset: constant max_rul
        125.0
        >>> rul[99]  # Just before onset: still max_rul
        125.0
        >>> rul[100]  # At onset: min(125, 100) = 100
        100.0
        >>> rul[-1]  # Failure
        0.0
    """
    if num_files <= 0:
        raise ValueError(f"num_files must be positive, got {num_files}")

    if onset_idx is None:
        # No onset detected: all samples get max_rul
        return np.full(num_files, max_rul, dtype=np.float32)

    if onset_idx < 0:
        raise ValueError(f"onset_idx must be non-negative, got {onset_idx}")

    rul = np.empty(num_files, dtype=np.float32)

    # Pre-onset: constant max_rul
    rul[:onset_idx] = max_rul

    # Post-onset: linear decay from (num_files - 1 - onset_idx) down to 0
    post_onset_count = num_files - onset_idx
    if post_onset_count > 0:
        post_onset_rul = np.arange(
            post_onset_count - 1, -1, -1, dtype=np.float32
        )
        # Cap at max_rul (in case post_onset_count - 1 > max_rul)
        np.minimum(post_onset_rul, max_rul, out=post_onset_rul)
        rul[onset_idx:] = post_onset_rul

    return rul


def generate_rul_labels(
    num_files: int,
    strategy: RULStrategy = "piecewise_linear",
    max_rul: float = 125.0,
    decay_rate: float = 3.0,
    onset_idx: int | None = None,
) -> np.ndarray:
    """Generate RUL labels using specified strategy.

    Unified interface for all RUL labeling strategies.

    Args:
        num_files: Total number of files for the bearing.
        strategy: One of "piecewise_linear", "linear", "exponential", "twostage".
        max_rul: Maximum RUL for piecewise linear and twostage strategies.
        decay_rate: Decay rate for exponential (ignored for other strategies).
        onset_idx: File index where degradation begins. Required for "twostage"
            strategy; ignored for other strategies.

    Returns:
        Array of shape (num_files,) with RUL values.

    Raises:
        ValueError: If strategy is unknown or num_files is invalid.
    """
    if strategy == "piecewise_linear":
        return piecewise_linear_rul(num_files, max_rul=max_rul)
    elif strategy == "linear":
        return linear_rul(num_files)
    elif strategy == "exponential":
        return exponential_rul(num_files, decay_rate=decay_rate)
    elif strategy == "twostage":
        return compute_twostage_rul(num_files, onset_idx=onset_idx, max_rul=max_rul)
    else:
        raise ValueError(
            f"Unknown RUL strategy: {strategy}. "
            f"Must be one of: piecewise_linear, linear, exponential, twostage"
        )


def generate_rul_for_bearing(
    bearing_file_count: int,
    strategy: RULStrategy = "piecewise_linear",
    max_rul: float = 125.0,
    normalize: bool = False,
    onset_idx: int | None = None,
) -> np.ndarray:
    """Generate RUL labels for a complete bearing lifecycle.

    Convenience function that handles edge cases and optional normalization.

    Args:
        bearing_file_count: Number of files in the bearing's lifecycle.
        strategy: RUL labeling strategy to use.
        max_rul: Maximum RUL value (for piecewise linear and twostage).
        normalize: If True, normalize RUL to [0, 1] range.
        onset_idx: File index where degradation begins. Only used when
            strategy="twostage". Defaults to None.

    Returns:
        Array of RUL values for each file in the bearing lifecycle.
    """
    if bearing_file_count <= 0:
        raise ValueError(f"bearing_file_count must be positive, got {bearing_file_count}")

    rul = generate_rul_labels(
        bearing_file_count, strategy=strategy, max_rul=max_rul, onset_idx=onset_idx
    )

    if normalize:
        rul_max = rul.max()
        if rul_max > 0:
            rul = rul / rul_max

    return rul


def visualize_rul_curves(
    num_files: int = 200,
    max_rul: float = 125.0,
    ax: Axes | None = None,
) -> Axes:
    """Visualize all RUL strategies for comparison.

    Creates a plot comparing piecewise linear, linear, and exponential
    RUL labeling strategies.

    Args:
        num_files: Number of files to simulate.
        max_rul: Maximum RUL for piecewise linear strategy.
        ax: Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        Matplotlib Axes object with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    file_indices = np.arange(num_files)

    # Generate all strategies
    rul_piecewise = piecewise_linear_rul(num_files, max_rul=max_rul)
    rul_linear = linear_rul(num_files)
    rul_exp = exponential_rul(num_files)

    # Plot
    ax.plot(file_indices, rul_piecewise, label=f"Piecewise Linear (max={max_rul})", linewidth=2)
    ax.plot(file_indices, rul_linear, label="Linear", linewidth=2, linestyle="--")
    ax.plot(file_indices, rul_exp, label="Exponential", linewidth=2, linestyle=":")

    ax.set_xlabel("File Index (Time)")
    ax.set_ylabel("RUL (Remaining Useful Life)")
    ax.set_title("RUL Labeling Strategies Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_files - 1)
    ax.set_ylim(0, max(num_files - 1, max_rul) * 1.05)

    return ax


def visualize_bearing_rul(
    bearing_id: str,
    num_files: int,
    strategy: RULStrategy = "piecewise_linear",
    max_rul: float = 125.0,
    ax: Axes | None = None,
) -> Axes:
    """Visualize RUL progression for a specific bearing.

    Args:
        bearing_id: Identifier for the bearing (for plot title).
        num_files: Number of files in the bearing lifecycle.
        strategy: RUL strategy to visualize.
        max_rul: Maximum RUL for piecewise linear.
        ax: Matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib Axes object with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    file_indices = np.arange(num_files)
    rul = generate_rul_labels(num_files, strategy=strategy, max_rul=max_rul)

    ax.plot(file_indices, rul, linewidth=2, color="steelblue")
    ax.fill_between(file_indices, rul, alpha=0.3, color="steelblue")

    ax.set_xlabel("File Index (Time)")
    ax.set_ylabel("RUL")
    ax.set_title(f"RUL Progression: {bearing_id} ({strategy})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_files - 1)
    ax.set_ylim(0, rul.max() * 1.05)

    # Add annotation for key points
    if len(rul) > 1:
        ax.axhline(y=max_rul, color="red", linestyle="--", alpha=0.5, label=f"max_rul={max_rul}")
        ax.legend()

    return ax
