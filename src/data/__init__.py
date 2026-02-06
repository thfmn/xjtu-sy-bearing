"""Data loading and processing utilities."""

from src.data.loader import (
    BEARINGS_PER_CONDITION,
    CONDITIONS,
    EXPECTED_SHAPE,
    NUM_CHANNELS,
    SAMPLES_PER_FILE,
    SAMPLING_RATE,
    XJTUBearingLoader,
)
from src.data.rul_labels import (
    RULStrategy,
    compute_twostage_rul,
    exponential_rul,
    generate_rul_for_bearing,
    generate_rul_labels,
    linear_rul,
    piecewise_linear_rul,
    visualize_bearing_rul,
    visualize_rul_curves,
)
from src.data.windowing import (
    DEFAULT_WINDOW_SIZE,
    WINDOW_SIZES,
    WindowConfig,
    calculate_num_windows_per_file,
    extract_windows,
    get_window_duration_ms,
    iter_windows,
    window_with_labels,
)
from src.data.dataset import (
    BearingDataset,
    DatasetConfig,
    create_bearing_dataset,
    get_all_bearings,
)

__all__ = [
    # Loader constants and class
    "BEARINGS_PER_CONDITION",
    "CONDITIONS",
    "EXPECTED_SHAPE",
    "NUM_CHANNELS",
    "SAMPLES_PER_FILE",
    "SAMPLING_RATE",
    "XJTUBearingLoader",
    # RUL utilities
    "RULStrategy",
    "compute_twostage_rul",
    "exponential_rul",
    "generate_rul_for_bearing",
    "generate_rul_labels",
    "linear_rul",
    "piecewise_linear_rul",
    "visualize_bearing_rul",
    "visualize_rul_curves",
    # Windowing utilities
    "DEFAULT_WINDOW_SIZE",
    "WINDOW_SIZES",
    "WindowConfig",
    "calculate_num_windows_per_file",
    "extract_windows",
    "get_window_duration_ms",
    "iter_windows",
    "window_with_labels",
    # TensorFlow Dataset
    "BearingDataset",
    "DatasetConfig",
    "create_bearing_dataset",
    "get_all_bearings",
]
