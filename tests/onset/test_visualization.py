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

"""Tests for onset detection visualization module.

Tests all public functions in src/onset/visualization.py:
- plot_bearing_onset: Single bearing onset plot
- plot_onset_comparison: Curated vs automated onset comparison
- plot_all_bearings_onset: Grid of onset plots for all bearings

Tests cover:
- Return type (matplotlib Figure objects)
- Edge cases (no onset, single data point, empty data)
- Save-to-file functionality
- Figure generation without exceptions (headless-compatible)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.onset.visualization import (
    plot_all_bearings_onset,
    plot_bearing_onset,
    plot_onset_comparison,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """Create synthetic features DataFrame for visualization testing.

    Creates data for 3 bearings (one per condition) with 50 samples each,
    matching the structure expected by load_bearing_health_series.
    """
    np.random.seed(42)
    data = []

    # Bearing1_1: gradual degradation (condition 1)
    for i in range(50):
        base_kurtosis = 3.0 + (i / 50) * 10
        data.append({
            "bearing_id": "Bearing1_1",
            "condition": "35Hz12kN",
            "file_idx": i,
            "h_kurtosis": base_kurtosis + np.random.randn() * 0.5,
            "v_kurtosis": base_kurtosis * 0.9 + np.random.randn() * 0.5,
            "h_rms": 0.1 + (i / 50) * 0.3 + np.random.randn() * 0.01,
            "v_rms": 0.08 + (i / 50) * 0.25 + np.random.randn() * 0.01,
        })

    # Bearing2_1: sudden degradation at sample 30 (condition 2)
    for i in range(50):
        if i < 30:
            base_kurtosis = 3.0 + np.random.randn() * 0.3
            base_rms = 0.1 + np.random.randn() * 0.01
        else:
            base_kurtosis = 15.0 + np.random.randn() * 2.0
            base_rms = 0.5 + np.random.randn() * 0.05
        data.append({
            "bearing_id": "Bearing2_1",
            "condition": "37.5Hz11kN",
            "file_idx": i,
            "h_kurtosis": base_kurtosis,
            "v_kurtosis": base_kurtosis * 0.95,
            "h_rms": base_rms,
            "v_rms": base_rms * 0.9,
        })

    # Bearing3_1: stable healthy (condition 3)
    for i in range(50):
        data.append({
            "bearing_id": "Bearing3_1",
            "condition": "40Hz10kN",
            "file_idx": i,
            "h_kurtosis": 3.0 + np.random.randn() * 0.2,
            "v_kurtosis": 2.8 + np.random.randn() * 0.2,
            "h_rms": 0.1 + np.random.randn() * 0.005,
            "v_rms": 0.09 + np.random.randn() * 0.005,
        })

    return pd.DataFrame(data)


@pytest.fixture
def single_sample_df() -> pd.DataFrame:
    """Create DataFrame with a single sample for edge case testing."""
    return pd.DataFrame([{
        "bearing_id": "Bearing1_1",
        "condition": "35Hz12kN",
        "file_idx": 0,
        "h_kurtosis": 5.0,
        "v_kurtosis": 4.5,
        "h_rms": 0.15,
        "v_rms": 0.12,
    }])


@pytest.fixture
def onset_labels() -> dict[str, int | None]:
    """Onset labels for the 3 synthetic bearings."""
    return {
        "Bearing1_1": 20,
        "Bearing2_1": 30,
        "Bearing3_1": None,
    }


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


# ============================================================================
# Tests for plot_bearing_onset()
# ============================================================================


class TestPlotBearingOnset:
    """Tests for plot_bearing_onset function."""

    def test_returns_figure(self, synthetic_features_df: pd.DataFrame):
        """Test that function returns a matplotlib Figure."""
        fig = plot_bearing_onset("Bearing1_1", synthetic_features_df, onset_idx=20)
        assert isinstance(fig, plt.Figure)

    def test_with_onset(self, synthetic_features_df: pd.DataFrame):
        """Test plot with an onset point marked."""
        fig = plot_bearing_onset("Bearing1_1", synthetic_features_df, onset_idx=25)
        ax = fig.axes[0]
        # Should have at least one vertical line (onset marker)
        assert len(ax.lines) >= 1
        assert isinstance(fig, plt.Figure)

    def test_without_onset(self, synthetic_features_df: pd.DataFrame):
        """Test plot when onset_idx is None (no onset detected)."""
        fig = plot_bearing_onset("Bearing1_1", synthetic_features_df, onset_idx=None)
        assert isinstance(fig, plt.Figure)

    def test_with_threshold(self, synthetic_features_df: pd.DataFrame):
        """Test plot with horizontal threshold line."""
        fig = plot_bearing_onset(
            "Bearing1_1", synthetic_features_df, onset_idx=20, threshold=5.0,
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_figsize(self, synthetic_features_df: pd.DataFrame):
        """Test plot with custom figure size."""
        fig = plot_bearing_onset(
            "Bearing1_1", synthetic_features_df, onset_idx=20,
            figsize=(12, 6),
        )
        assert isinstance(fig, plt.Figure)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12.0)
        assert h == pytest.approx(6.0)

    def test_save_to_file(self, synthetic_features_df: pd.DataFrame, tmp_path: Path):
        """Test saving figure to disk."""
        save_path = tmp_path / "test_onset.png"
        fig = plot_bearing_onset(
            "Bearing1_1", synthetic_features_df, onset_idx=20,
            save_path=save_path,
        )
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_single_sample(self, single_sample_df: pd.DataFrame):
        """Test plot with a single data point."""
        fig = plot_bearing_onset("Bearing1_1", single_sample_df, onset_idx=None)
        assert isinstance(fig, plt.Figure)

    def test_onset_at_boundaries(self, synthetic_features_df: pd.DataFrame):
        """Test onset at first and last file index."""
        # Onset at very beginning
        fig_start = plot_bearing_onset(
            "Bearing1_1", synthetic_features_df, onset_idx=0,
        )
        assert isinstance(fig_start, plt.Figure)

        # Onset at very end
        fig_end = plot_bearing_onset(
            "Bearing1_1", synthetic_features_df, onset_idx=49,
        )
        assert isinstance(fig_end, plt.Figure)


# ============================================================================
# Tests for plot_onset_comparison()
# ============================================================================


class TestPlotOnsetComparison:
    """Tests for plot_onset_comparison function."""

    def test_returns_figure(self, synthetic_features_df: pd.DataFrame):
        """Test that function returns a matplotlib Figure."""
        fig = plot_onset_comparison(
            "Bearing1_1", curated_idx=20, auto_idx=22,
            features_df=synthetic_features_df,
        )
        assert isinstance(fig, plt.Figure)

    def test_both_labels(self, synthetic_features_df: pd.DataFrame):
        """Test comparison with both curated and auto labels."""
        fig = plot_onset_comparison(
            "Bearing2_1", curated_idx=28, auto_idx=30,
            features_df=synthetic_features_df,
        )
        ax = fig.axes[0]
        # Title should contain the difference
        assert "diff=" in ax.get_title()

    def test_curated_only(self, synthetic_features_df: pd.DataFrame):
        """Test comparison with only curated label (auto=None)."""
        fig = plot_onset_comparison(
            "Bearing1_1", curated_idx=20, auto_idx=None,
            features_df=synthetic_features_df,
        )
        assert isinstance(fig, plt.Figure)

    def test_auto_only(self, synthetic_features_df: pd.DataFrame):
        """Test comparison with only auto label (curated=None)."""
        fig = plot_onset_comparison(
            "Bearing1_1", curated_idx=None, auto_idx=25,
            features_df=synthetic_features_df,
        )
        assert isinstance(fig, plt.Figure)

    def test_neither_label(self, synthetic_features_df: pd.DataFrame):
        """Test comparison with no labels at all."""
        fig = plot_onset_comparison(
            "Bearing1_1", curated_idx=None, auto_idx=None,
            features_df=synthetic_features_df,
        )
        assert isinstance(fig, plt.Figure)

    def test_save_to_file(self, synthetic_features_df: pd.DataFrame, tmp_path: Path):
        """Test saving comparison figure to disk."""
        save_path = tmp_path / "test_comparison.png"
        fig = plot_onset_comparison(
            "Bearing1_1", curated_idx=20, auto_idx=22,
            features_df=synthetic_features_df, save_path=save_path,
        )
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_identical_labels(self, synthetic_features_df: pd.DataFrame):
        """Test when curated and auto labels are the same."""
        fig = plot_onset_comparison(
            "Bearing1_1", curated_idx=20, auto_idx=20,
            features_df=synthetic_features_df,
        )
        assert "diff=+0" in fig.axes[0].get_title()

    def test_negative_difference(self, synthetic_features_df: pd.DataFrame):
        """Test when auto onset is before curated (negative diff)."""
        fig = plot_onset_comparison(
            "Bearing1_1", curated_idx=30, auto_idx=20,
            features_df=synthetic_features_df,
        )
        assert "diff=-10" in fig.axes[0].get_title()


# ============================================================================
# Tests for plot_all_bearings_onset()
# ============================================================================


class TestPlotAllBearingsOnset:
    """Tests for plot_all_bearings_onset function."""

    def test_returns_figure(
        self, synthetic_features_df: pd.DataFrame,
        onset_labels: dict[str, int | None],
    ):
        """Test that function returns a matplotlib Figure."""
        fig = plot_all_bearings_onset(synthetic_features_df, onset_labels)
        assert isinstance(fig, plt.Figure)

    def test_grid_dimensions(
        self, synthetic_features_df: pd.DataFrame,
        onset_labels: dict[str, int | None],
    ):
        """Test that grid has correct number of subplots."""
        fig = plot_all_bearings_onset(synthetic_features_df, onset_labels)
        # 3 conditions x 1 bearing each = 3 visible axes
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) >= 3

    def test_save_to_directory(
        self, synthetic_features_df: pd.DataFrame,
        onset_labels: dict[str, int | None],
        tmp_path: Path,
    ):
        """Test saving grid figure to output directory."""
        fig = plot_all_bearings_onset(
            synthetic_features_df, onset_labels, output_dir=tmp_path,
        )
        assert isinstance(fig, plt.Figure)
        expected_file = tmp_path / "onset_grid.png"
        assert expected_file.exists()
        assert expected_file.stat().st_size > 0

    def test_all_bearings_none_onset(self, synthetic_features_df: pd.DataFrame):
        """Test grid when no bearings have onset."""
        labels = {
            "Bearing1_1": None,
            "Bearing2_1": None,
            "Bearing3_1": None,
        }
        fig = plot_all_bearings_onset(synthetic_features_df, labels)
        assert isinstance(fig, plt.Figure)

    def test_custom_figsize(
        self, synthetic_features_df: pd.DataFrame,
        onset_labels: dict[str, int | None],
    ):
        """Test grid with custom figure size."""
        fig = plot_all_bearings_onset(
            synthetic_features_df, onset_labels, figsize=(20, 14),
        )
        assert isinstance(fig, plt.Figure)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(14.0)

    def test_single_bearing(self, synthetic_features_df: pd.DataFrame):
        """Test grid with only one bearing."""
        # Filter to single bearing
        single_df = synthetic_features_df[
            synthetic_features_df["bearing_id"] == "Bearing1_1"
        ].copy()
        labels = {"Bearing1_1": 20}
        fig = plot_all_bearings_onset(single_df, labels)
        assert isinstance(fig, plt.Figure)
