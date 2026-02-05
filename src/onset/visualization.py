"""Onset detection visualization tools.

This module provides plotting functions for validating and analyzing onset
detection results. All functions produce matplotlib figures with clear
annotations showing healthy vs degraded regions and onset points.

Functions:
    plot_bearing_onset: Plot single bearing with onset point marked
    plot_onset_comparison: Compare manual vs automated onset labels
    plot_all_bearings_onset: Grid of onset plots for all bearings
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from src.onset.health_indicators import BearingHealthSeries, load_bearing_health_series


def plot_bearing_onset(
    bearing_id: str,
    features_df: pd.DataFrame,
    onset_idx: int | None,
    save_path: str | Path | None = None,
    threshold: float | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Plot kurtosis time series for a bearing with onset point marked.

    Shows the health indicator (average kurtosis) over time with:
    - Green shading for the healthy region (before onset)
    - Red shading for the degraded region (after onset)
    - Vertical dashed line at the onset point
    - Optional horizontal threshold line

    Args:
        bearing_id: Bearing identifier (e.g., "Bearing1_1").
        features_df: DataFrame with columns: bearing_id, file_idx,
            h_kurtosis, v_kurtosis, h_rms, v_rms.
        onset_idx: File index where degradation begins. None if no onset.
        save_path: If provided, save figure to this path.
        threshold: If provided, draw horizontal threshold line.
        figsize: Figure size in inches (width, height).

    Returns:
        The matplotlib Figure object.
    """
    health = load_bearing_health_series(bearing_id, features_df)
    kurtosis_avg = (health.kurtosis_h + health.kurtosis_v) / 2
    file_indices = health.file_indices

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # Plot kurtosis time series
    ax.plot(file_indices, kurtosis_avg, linewidth=1.0, color="#2c3e50", label="Kurtosis (avg)")

    # Shade healthy vs degraded regions
    if onset_idx is not None:
        ax.axvspan(file_indices[0], onset_idx, alpha=0.15, color="green", label="Healthy")
        ax.axvspan(onset_idx, file_indices[-1], alpha=0.15, color="red", label="Degraded")
        ax.axvline(onset_idx, color="red", linestyle="--", linewidth=1.5, label=f"Onset (idx={onset_idx})")
    else:
        ax.axvspan(file_indices[0], file_indices[-1], alpha=0.15, color="green", label="Healthy (no onset)")

    # Optional threshold line
    if threshold is not None:
        ax.axhline(threshold, color="orange", linestyle=":", linewidth=1.0, label=f"Threshold={threshold:.2f}")

    ax.set_xlabel("File Index")
    ax.set_ylabel("Kurtosis (avg H+V)")
    ax.set_title(f"{bearing_id} — {health.condition}")
    ax.legend(loc="upper left", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_onset_comparison(
    bearing_id: str,
    manual_idx: int | None,
    auto_idx: int | None,
    features_df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Compare manual vs automated onset labels on the same plot.

    Draws two vertical lines — one for the manual label and one for the
    automated label — so the user can visually assess agreement.

    Args:
        bearing_id: Bearing identifier.
        manual_idx: Manual onset file index (None if no manual label).
        auto_idx: Automated onset file index (None if not detected).
        features_df: Features DataFrame.
        save_path: If provided, save figure to this path.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure object.
    """
    health = load_bearing_health_series(bearing_id, features_df)
    kurtosis_avg = (health.kurtosis_h + health.kurtosis_v) / 2
    file_indices = health.file_indices

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.plot(file_indices, kurtosis_avg, linewidth=1.0, color="#2c3e50", label="Kurtosis (avg)")

    if manual_idx is not None:
        ax.axvline(manual_idx, color="blue", linestyle="--", linewidth=1.5, label=f"Manual (idx={manual_idx})")
    if auto_idx is not None:
        ax.axvline(auto_idx, color="red", linestyle="-.", linewidth=1.5, label=f"Auto (idx={auto_idx})")

    # Annotate difference if both are available
    if manual_idx is not None and auto_idx is not None:
        diff = auto_idx - manual_idx
        sign = "+" if diff >= 0 else ""
        ax.set_title(f"{bearing_id} — Manual vs Auto (diff={sign}{diff})")
    else:
        ax.set_title(f"{bearing_id} — Onset Comparison")

    ax.set_xlabel("File Index")
    ax.set_ylabel("Kurtosis (avg H+V)")
    ax.legend(loc="upper left", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_all_bearings_onset(
    features_df: pd.DataFrame,
    onset_labels: dict[str, int | None],
    output_dir: str | Path | None = None,
    figsize: tuple[float, float] = (18, 12),
) -> plt.Figure:
    """Generate grid of onset plots for all bearings.

    Creates a 5x3 grid (5 rows, 3 columns — one column per operating
    condition) with each subplot showing a bearing's kurtosis series and
    its onset point.

    Args:
        features_df: Features DataFrame for all bearings.
        onset_labels: Dictionary mapping bearing_id to onset file index.
            Value is None if no onset for that bearing.
        output_dir: If provided, save the grid figure to this directory
            as ``onset_grid.png``.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure object.
    """
    # Group bearings by condition (columns) in fixed order
    conditions = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
    cond_bearings: dict[str, list[str]] = {c: [] for c in conditions}

    for bid in sorted(onset_labels.keys()):
        # Extract condition number from bearing_id (e.g., "Bearing1_1" -> condition 1)
        cond_num = int(bid.replace("Bearing", "")[0])
        cond_bearings[conditions[cond_num - 1]].append(bid)

    n_rows = max(len(v) for v in cond_bearings.values())
    n_cols = len(conditions)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, layout="constrained")

    # Handle edge case: single row
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col_idx, cond in enumerate(conditions):
        bearings = cond_bearings[cond]
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]

            if row_idx >= len(bearings):
                ax.set_visible(False)
                continue

            bid = bearings[row_idx]
            onset_idx = onset_labels.get(bid)

            health = load_bearing_health_series(bid, features_df)
            kurtosis_avg = (health.kurtosis_h + health.kurtosis_v) / 2
            fi = health.file_indices

            ax.plot(fi, kurtosis_avg, linewidth=0.8, color="#2c3e50")

            if onset_idx is not None:
                ax.axvspan(fi[0], onset_idx, alpha=0.15, color="green")
                ax.axvspan(onset_idx, fi[-1], alpha=0.15, color="red")
                ax.axvline(onset_idx, color="red", linestyle="--", linewidth=1.0)

            ax.set_title(bid, fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)

            if row_idx == n_rows - 1 or row_idx == len(bearings) - 1:
                ax.set_xlabel("File Index", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Kurtosis", fontsize=8)

        # Column header
        axes[0, col_idx].annotate(
            cond,
            xy=(0.5, 1.15),
            xycoords="axes fraction",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("Onset Detection — All Bearings", fontsize=14, fontweight="bold")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "onset_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
