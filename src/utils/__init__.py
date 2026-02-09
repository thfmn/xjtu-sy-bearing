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

"""General utilities."""

from src.utils.audio import (
    # Core conversion functions
    resample_signal,
    normalize_audio,
    signal_to_wav,
    wav_to_signal,
    convert_vibration_to_audio,
    # Configuration
    AudioConfig,
    # Constants
    SOURCE_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    # Utility functions
    get_resampled_duration_ms,
    get_resampled_num_samples,
)
from src.utils.tracking import (
    # Core classes
    ExperimentTracker,
    RunInfo,
    RunContext,
    TrackingBackend,
    # Backend implementations
    MLflowBackend,
    VertexBackend,
    # Keras integration
    KerasTrackingCallback,
    UnifiedTrackingCallback,
    # Comparison utilities
    compare_runs,
    plot_metric_comparison,
    export_comparison_report,
    # Convenience functions
    create_mlflow_tracker,
    create_vertex_tracker,
)

__all__ = [
    # Audio conversion
    "resample_signal",
    "normalize_audio",
    "signal_to_wav",
    "wav_to_signal",
    "convert_vibration_to_audio",
    "AudioConfig",
    "SOURCE_SAMPLE_RATE",
    "TARGET_SAMPLE_RATE",
    "get_resampled_duration_ms",
    "get_resampled_num_samples",
    # Core classes
    "ExperimentTracker",
    "RunInfo",
    "RunContext",
    "TrackingBackend",
    # Backend implementations
    "MLflowBackend",
    "VertexBackend",
    # Keras integration
    "KerasTrackingCallback",
    "UnifiedTrackingCallback",
    # Comparison utilities
    "compare_runs",
    "plot_metric_comparison",
    "export_comparison_report",
    # Convenience functions
    "create_mlflow_tracker",
    "create_vertex_tracker",
]
