"""Tests for data loading functionality.

Tests the XJTUBearingLoader class from src/data/loader.py including:
- Single file loading with correct shape (32768, 2)
- Both channels present
- Metadata generation
- Bearing loading
- Caching functionality
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import (
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLES_PER_FILE,
    EXPECTED_SHAPE,
    SAMPLING_RATE,
)


class TestFileLoading:
    """Test single file loading."""

    def test_load_file_shape(self, bearing_loader, sample_signal):
        """Load sample file and verify shape (32768, 2)."""
        assert sample_signal.shape == EXPECTED_SHAPE, (
            f"Expected shape {EXPECTED_SHAPE}, got {sample_signal.shape}"
        )

    def test_both_channels_present(self, sample_signal):
        """Verify both horizontal and vertical channels are present."""
        assert sample_signal.shape[1] == EXPECTED_CHANNELS
        # Both channels should have non-zero data
        h_channel = sample_signal[:, 0]
        v_channel = sample_signal[:, 1]
        assert np.std(h_channel) > 0, "Horizontal channel appears empty"
        assert np.std(v_channel) > 0, "Vertical channel appears empty"

    def test_data_type(self, sample_signal):
        """Verify data is float32."""
        assert sample_signal.dtype == np.float32

    def test_no_nan_values(self, sample_signal):
        """Verify no NaN values in loaded data."""
        assert not np.isnan(sample_signal).any(), "NaN values found in signal"

    def test_no_inf_values(self, sample_signal):
        """Verify no infinite values in loaded data."""
        assert not np.isinf(sample_signal).any(), "Inf values found in signal"

    def test_reasonable_amplitude(self, sample_signal):
        """Verify signal amplitudes are in reasonable range for vibration data."""
        # Typical bearing vibration: -100 to 100 g or mm/s^2
        max_abs = np.abs(sample_signal).max()
        assert max_abs < 1000, f"Unexpectedly large amplitude: {max_abs}"
        assert max_abs > 0.001, f"Unexpectedly small amplitude: {max_abs}"


class TestBearingLoading:
    """Test bearing-level loading."""

    def test_load_bearing(self, bearing_loader):
        """Load all files for a bearing."""
        signals, filenames = bearing_loader.load_bearing("35Hz12kN", "Bearing1_1")

        # Should have multiple files
        assert len(filenames) > 100, f"Expected >100 files, got {len(filenames)}"
        assert signals.shape[0] == len(filenames)
        assert signals.shape[1:] == EXPECTED_SHAPE

    def test_load_bearing_filenames_sorted(self, bearing_loader):
        """Verify filenames are in numerical order."""
        _, filenames = bearing_loader.load_bearing("35Hz12kN", "Bearing1_1")

        # Extract numbers and verify sorted
        numbers = [int(Path(f).stem) for f in filenames]
        assert numbers == sorted(numbers), "Filenames not in numerical order"

    def test_iter_bearing(self, bearing_loader):
        """Test memory-efficient bearing iteration."""
        count = 0
        for signal, filename, idx in bearing_loader.iter_bearing("35Hz12kN", "Bearing1_1"):
            assert signal.shape == EXPECTED_SHAPE
            assert isinstance(filename, str)
            assert idx == count
            count += 1
            if count >= 3:  # Only check first 3 for speed
                break
        assert count == 3

    def test_invalid_condition(self, bearing_loader):
        """Test error handling for invalid condition."""
        with pytest.raises(ValueError, match="Unknown condition"):
            bearing_loader.load_bearing("invalid_condition", "Bearing1_1")

    def test_invalid_bearing(self, bearing_loader):
        """Test error handling for invalid bearing."""
        with pytest.raises(ValueError, match="Unknown bearing"):
            bearing_loader.load_bearing("35Hz12kN", "Bearing1_99")


class TestMetadata:
    """Test metadata generation."""

    def test_get_metadata_structure(self, bearing_loader):
        """Verify metadata DataFrame structure."""
        metadata = bearing_loader.get_metadata()

        # Check required columns
        required_cols = [
            "condition",
            "bearing_id",
            "filename",
            "file_idx",
            "file_path",
            "num_files_in_bearing",
        ]
        for col in required_cols:
            assert col in metadata.columns, f"Missing column: {col}"

    def test_metadata_conditions(self, bearing_loader):
        """Verify all 3 conditions are present."""
        metadata = bearing_loader.get_metadata()
        conditions = set(metadata["condition"].unique())
        expected = {"35Hz12kN", "37.5Hz11kN", "40Hz10kN"}
        assert conditions == expected, f"Expected {expected}, got {conditions}"

    def test_metadata_bearings(self, bearing_loader):
        """Verify all 15 bearings are present."""
        metadata = bearing_loader.get_metadata()
        bearings = set(metadata["bearing_id"].unique())
        assert len(bearings) == 15, f"Expected 15 bearings, got {len(bearings)}"

    def test_metadata_total_files(self, bearing_loader):
        """Verify total file count is approximately 9216."""
        metadata = bearing_loader.get_metadata()
        total_files = len(metadata)
        # Allow some tolerance in case dataset changes slightly
        assert 9000 < total_files < 10000, (
            f"Expected ~9216 files, got {total_files}"
        )


class TestCaching:
    """Test caching functionality."""

    def test_cache_hit(self, bearing_loader, data_root):
        """Verify cache improves performance on repeated loads."""
        file_path = data_root / "35Hz12kN" / "Bearing1_1" / "1.csv"

        # Clear cache first
        bearing_loader.clear_cache()

        # First load (cache miss)
        bearing_loader.load_file(file_path)
        info_after_first = bearing_loader.cache_info

        # Second load (should be cache hit)
        bearing_loader.load_file(file_path)
        info_after_second = bearing_loader.cache_info

        assert info_after_second.hits > info_after_first.hits, "Cache not working"

    def test_cache_clear(self, bearing_loader, data_root):
        """Verify cache can be cleared."""
        file_path = data_root / "35Hz12kN" / "Bearing1_1" / "1.csv"

        # Load to populate cache
        bearing_loader.load_file(file_path)

        # Clear cache
        bearing_loader.clear_cache()
        info = bearing_loader.cache_info

        assert info.hits == 0, "Cache not properly cleared"
        assert info.misses == 0, "Cache not properly cleared"


class TestConstants:
    """Test dataset constants."""

    def test_sampling_rate(self):
        """Verify sampling rate constant."""
        from src.data.loader import SAMPLING_RATE
        assert SAMPLING_RATE == 25600

    def test_samples_per_file(self):
        """Verify samples per file constant."""
        from src.data.loader import SAMPLES_PER_FILE
        assert SAMPLES_PER_FILE == 32768

    def test_num_channels(self):
        """Verify number of channels constant."""
        from src.data.loader import NUM_CHANNELS
        assert NUM_CHANNELS == 2

    def test_conditions_dict(self):
        """Verify conditions dictionary."""
        from src.data.loader import CONDITIONS
        assert len(CONDITIONS) == 3
        assert "35Hz12kN" in CONDITIONS
        assert CONDITIONS["35Hz12kN"]["rpm"] == 2100

    def test_bearings_per_condition(self):
        """Verify bearings per condition."""
        from src.data.loader import BEARINGS_PER_CONDITION
        for condition, bearings in BEARINGS_PER_CONDITION.items():
            assert len(bearings) == 5, f"Expected 5 bearings for {condition}"
