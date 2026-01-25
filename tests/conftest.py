"""Pytest fixtures for XJTU-SY bearing tests.

Provides common fixtures for data loading, sample signals, and test configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Project root and data paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "assets" / "Data" / "XJTU-SY_Bearing_Datasets"
FEATURES_PATH = PROJECT_ROOT / "outputs" / "features" / "features_v2.csv"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_root() -> Path:
    """Get data root directory."""
    return DATA_ROOT


@pytest.fixture(scope="session")
def data_available() -> bool:
    """Check if dataset is available."""
    return DATA_ROOT.exists()


@pytest.fixture(scope="session")
def features_path() -> Path:
    """Get path to extracted features CSV."""
    return FEATURES_PATH


@pytest.fixture(scope="session")
def features_available() -> bool:
    """Check if features CSV is available."""
    return FEATURES_PATH.exists()


@pytest.fixture(scope="session")
def bearing_loader(data_available):
    """Create XJTUBearingLoader instance."""
    if not data_available:
        pytest.skip("Dataset not available")
    from src.data.loader import XJTUBearingLoader
    return XJTUBearingLoader(DATA_ROOT)


@pytest.fixture(scope="session")
def sample_signal(bearing_loader) -> np.ndarray:
    """Load a sample signal from the dataset.

    Returns:
        Array of shape (32768, 2) with horizontal and vertical channels.
    """
    # Use first file from first bearing (35Hz12kN/Bearing1_1/1.csv)
    first_file = DATA_ROOT / "35Hz12kN" / "Bearing1_1" / "1.csv"
    return bearing_loader.load_file(first_file)


@pytest.fixture(scope="session")
def sample_signal_batch(bearing_loader) -> np.ndarray:
    """Load a batch of sample signals.

    Returns:
        Array of shape (3, 32768, 2) with 3 samples.
    """
    bearing_path = DATA_ROOT / "35Hz12kN" / "Bearing1_1"
    signals = []
    for i in [1, 60, 120]:  # Healthy, mid-life, near-failure
        file_path = bearing_path / f"{i}.csv"
        if file_path.exists():
            signals.append(bearing_loader.load_file(file_path))
    return np.stack(signals)


@pytest.fixture
def synthetic_signal() -> np.ndarray:
    """Create a synthetic test signal.

    Returns:
        Array of shape (32768, 2) with synthetic vibration-like data.
    """
    np.random.seed(42)
    n_samples = 32768
    t = np.linspace(0, n_samples / 25600, n_samples)

    # Simulate bearing vibration: base frequency + harmonics + noise
    h_channel = (
        0.5 * np.sin(2 * np.pi * 100 * t) +  # Shaft frequency
        0.3 * np.sin(2 * np.pi * 250 * t) +  # BPFO-like
        0.1 * np.random.randn(n_samples)     # Noise
    )
    v_channel = (
        0.4 * np.sin(2 * np.pi * 100 * t) +
        0.35 * np.sin(2 * np.pi * 250 * t) +
        0.12 * np.random.randn(n_samples)
    )
    return np.column_stack([h_channel, v_channel]).astype(np.float32)


@pytest.fixture
def synthetic_signal_batch() -> np.ndarray:
    """Create a batch of synthetic signals.

    Returns:
        Array of shape (4, 32768, 2).
    """
    np.random.seed(42)
    batch_size = 4
    n_samples = 32768
    signals = []

    for i in range(batch_size):
        t = np.linspace(0, n_samples / 25600, n_samples)
        # Vary amplitude to simulate different degradation levels
        amp_factor = 0.5 + 0.5 * (i / batch_size)
        h = amp_factor * np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(n_samples)
        v = amp_factor * np.sin(2 * np.pi * 100 * t) + 0.12 * np.random.randn(n_samples)
        signals.append(np.column_stack([h, v]))

    return np.stack(signals).astype(np.float32)


@pytest.fixture(scope="session")
def features_df(features_available, features_path):
    """Load features DataFrame."""
    if not features_available:
        pytest.skip("Features CSV not available")
    import pandas as pd
    return pd.read_csv(features_path)


@pytest.fixture
def small_features_df(features_df):
    """Get a small subset of features for fast testing."""
    # One bearing per condition
    bearings = ["Bearing1_1", "Bearing2_1", "Bearing3_1"]
    return features_df[features_df["bearing_id"].isin(bearings)].copy()


# Constants for test assertions
EXPECTED_SAMPLES_PER_FILE = 32768
EXPECTED_CHANNELS = 2
EXPECTED_SHAPE = (EXPECTED_SAMPLES_PER_FILE, EXPECTED_CHANNELS)
SAMPLING_RATE = 25600
NUM_TIME_FEATURES = 37
NUM_FREQ_FEATURES = 28
NUM_ALL_FEATURES = 65
