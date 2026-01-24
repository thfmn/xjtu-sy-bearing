"""Feature extraction modules."""

from src.features import frequency_domain, time_domain
from src.features.time_domain import (
    CHANNEL_FEATURES as TIME_DOMAIN_FEATURES,
    NUM_FEATURES_PER_CHANNEL as NUM_TIME_FEATURES_PER_CHANNEL,
    extract_channel_features as extract_time_channel_features,
    extract_cross_correlation,
    extract_features as extract_time_features,
    extract_features_dict as extract_time_features_dict,
    get_feature_names as get_time_feature_names,
)
from src.features.frequency_domain import (
    SPECTRAL_FEATURES,
    CHARACTERISTIC_FEATURES,
    NUM_SPECTRAL_FEATURES,
    NUM_CHARACTERISTIC_FEATURES,
    calculate_bearing_frequencies,
    get_characteristic_frequencies_for_condition,
    extract_features as extract_freq_features,
    extract_features_dict as extract_freq_features_dict,
    get_feature_names as get_freq_feature_names,
)

__all__ = [
    # Submodules
    "time_domain",
    "frequency_domain",
    # Time-domain exports
    "TIME_DOMAIN_FEATURES",
    "NUM_TIME_FEATURES_PER_CHANNEL",
    "extract_time_channel_features",
    "extract_cross_correlation",
    "extract_time_features",
    "extract_time_features_dict",
    "get_time_feature_names",
    # Frequency-domain exports
    "SPECTRAL_FEATURES",
    "CHARACTERISTIC_FEATURES",
    "NUM_SPECTRAL_FEATURES",
    "NUM_CHARACTERISTIC_FEATURES",
    "calculate_bearing_frequencies",
    "get_characteristic_frequencies_for_condition",
    "extract_freq_features",
    "extract_freq_features_dict",
    "get_freq_feature_names",
]
