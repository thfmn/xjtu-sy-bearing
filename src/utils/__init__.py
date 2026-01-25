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
