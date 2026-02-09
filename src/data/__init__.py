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
