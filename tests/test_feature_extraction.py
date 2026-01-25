"""Tests for feature extraction functionality.

Tests the feature extraction modules including:
- Time-domain features (37 features)
- Frequency-domain features (28 features)
- Unified FeatureExtractor (65 features)
- STFT spectrogram generation
- CWT scalogram generation
- Envelope analysis
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import (
    NUM_ALL_FEATURES,
    NUM_FREQ_FEATURES,
    NUM_TIME_FEATURES,
    SAMPLING_RATE,
)


class TestTimeDomainFeatures:
    """Test time-domain feature extraction."""

    def test_feature_count(self, synthetic_signal):
        """Verify 37 time-domain features are extracted."""
        from src.features.time_domain import extract_features, get_feature_names

        features = extract_features(synthetic_signal)
        names = get_feature_names()

        assert features.shape == (NUM_TIME_FEATURES,), (
            f"Expected {NUM_TIME_FEATURES} features, got {features.shape[0]}"
        )
        assert len(names) == NUM_TIME_FEATURES

    def test_batch_extraction(self, synthetic_signal_batch):
        """Test batch feature extraction."""
        from src.features.time_domain import extract_features

        features = extract_features(synthetic_signal_batch)
        assert features.shape == (4, NUM_TIME_FEATURES)

    def test_no_nan_values(self, synthetic_signal):
        """Verify no NaN values in extracted features."""
        from src.features.time_domain import extract_features

        features = extract_features(synthetic_signal)
        assert not np.isnan(features).any(), "NaN values in time-domain features"

    def test_no_inf_values(self, synthetic_signal):
        """Verify no infinite values in extracted features."""
        from src.features.time_domain import extract_features

        features = extract_features(synthetic_signal)
        assert not np.isinf(features).any(), "Inf values in time-domain features"

    def test_real_data(self, sample_signal):
        """Test extraction on real bearing data."""
        from src.features.time_domain import extract_features

        features = extract_features(sample_signal)
        assert features.shape == (NUM_TIME_FEATURES,)
        assert not np.isnan(features).any()

    def test_feature_names_match(self, synthetic_signal):
        """Verify feature names match extracted values."""
        from src.features.time_domain import extract_features, get_feature_names

        features = extract_features(synthetic_signal)
        names = get_feature_names()

        assert len(features) == len(names)
        # Check some expected names
        assert "h_rms" in names
        assert "v_kurtosis" in names
        assert "cross_correlation" in names


class TestFrequencyDomainFeatures:
    """Test frequency-domain feature extraction."""

    def test_feature_count(self, synthetic_signal):
        """Verify 28 frequency-domain features are extracted."""
        from src.features.frequency_domain import extract_features, get_feature_names

        features = extract_features(synthetic_signal)
        names = get_feature_names()

        assert features.shape == (NUM_FREQ_FEATURES,), (
            f"Expected {NUM_FREQ_FEATURES} features, got {features.shape[0]}"
        )
        assert len(names) == NUM_FREQ_FEATURES

    def test_batch_extraction(self, synthetic_signal_batch):
        """Test batch feature extraction."""
        from src.features.frequency_domain import extract_features

        features = extract_features(synthetic_signal_batch)
        assert features.shape == (4, NUM_FREQ_FEATURES)

    def test_no_nan_values(self, synthetic_signal):
        """Verify no NaN values in extracted features."""
        from src.features.frequency_domain import extract_features

        features = extract_features(synthetic_signal)
        assert not np.isnan(features).any(), "NaN values in frequency-domain features"

    def test_no_inf_values(self, synthetic_signal):
        """Verify no infinite values in extracted features."""
        from src.features.frequency_domain import extract_features

        features = extract_features(synthetic_signal)
        assert not np.isinf(features).any(), "Inf values in frequency-domain features"

    def test_with_condition(self, synthetic_signal):
        """Test extraction with operating condition specified."""
        from src.features.frequency_domain import extract_features

        features = extract_features(
            synthetic_signal, condition="35Hz12kN", sampling_rate=SAMPLING_RATE
        )
        assert features.shape == (NUM_FREQ_FEATURES,)
        assert not np.isnan(features).any()

    def test_real_data(self, sample_signal):
        """Test extraction on real bearing data."""
        from src.features.frequency_domain import extract_features

        features = extract_features(sample_signal, condition="35Hz12kN")
        assert features.shape == (NUM_FREQ_FEATURES,)
        assert not np.isnan(features).any()


class TestUnifiedFeatureExtractor:
    """Test unified FeatureExtractor class."""

    def test_all_features_count(self, synthetic_signal):
        """Verify 65 total features in 'all' mode."""
        from src.features.fusion import FeatureExtractor

        extractor = FeatureExtractor(mode="all")
        features = extractor.extract(synthetic_signal)

        assert features.shape == (NUM_ALL_FEATURES,), (
            f"Expected {NUM_ALL_FEATURES} features, got {features.shape[0]}"
        )
        assert extractor.n_features == NUM_ALL_FEATURES

    def test_time_only_mode(self, synthetic_signal):
        """Test time-domain only extraction mode."""
        from src.features.fusion import FeatureExtractor

        extractor = FeatureExtractor(mode="time")
        features = extractor.extract(synthetic_signal)

        assert features.shape == (NUM_TIME_FEATURES,)

    def test_freq_only_mode(self, synthetic_signal):
        """Test frequency-domain only extraction mode."""
        from src.features.fusion import FeatureExtractor

        extractor = FeatureExtractor(mode="freq")
        features = extractor.extract(synthetic_signal)

        assert features.shape == (NUM_FREQ_FEATURES,)

    def test_extract_dict(self, synthetic_signal):
        """Test dictionary output format."""
        from src.features.fusion import FeatureExtractor

        extractor = FeatureExtractor(mode="all")
        feature_dict = extractor.extract_dict(synthetic_signal)

        assert isinstance(feature_dict, dict)
        assert len(feature_dict) == NUM_ALL_FEATURES
        assert "h_rms" in feature_dict
        assert "v_spectral_centroid" in feature_dict

    def test_extract_dataframe(self, synthetic_signal):
        """Test DataFrame output format."""
        from src.features.fusion import FeatureExtractor

        extractor = FeatureExtractor(mode="all")
        df = extractor.extract_dataframe(synthetic_signal)

        assert len(df) == 1
        assert len(df.columns) == NUM_ALL_FEATURES

    def test_batch_extraction(self, synthetic_signal_batch):
        """Test batch extraction."""
        from src.features.fusion import FeatureExtractor

        extractor = FeatureExtractor(mode="all")
        features = extractor.extract(synthetic_signal_batch)

        assert features.shape == (4, NUM_ALL_FEATURES)

    def test_deterministic(self, synthetic_signal):
        """Verify extraction is deterministic."""
        from src.features.fusion import FeatureExtractor

        extractor = FeatureExtractor(mode="all")
        features1 = extractor.extract(synthetic_signal)
        features2 = extractor.extract(synthetic_signal)

        np.testing.assert_array_equal(features1, features2)

    def test_feature_names_property(self):
        """Test feature_names property."""
        from src.features.fusion import FeatureExtractor

        extractor = FeatureExtractor(mode="all")
        names = extractor.feature_names

        assert len(names) == NUM_ALL_FEATURES
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_get_info(self):
        """Test get_info method."""
        from src.features.fusion import ExtractionMode, FeatureExtractor

        extractor = FeatureExtractor(mode="all")
        info = extractor.get_info()

        assert info.n_features == NUM_ALL_FEATURES
        assert info.mode == ExtractionMode.ALL
        assert len(info.names) == NUM_ALL_FEATURES


class TestSTFTSpectrogram:
    """Test STFT spectrogram generation."""

    def test_output_shape(self, synthetic_signal):
        """Verify output shape is (128, 128, 2)."""
        from src.features.stft import extract_spectrogram

        spectrogram = extract_spectrogram(synthetic_signal)
        assert spectrogram.shape == (128, 128, 2), (
            f"Expected (128, 128, 2), got {spectrogram.shape}"
        )

    def test_no_nan_values(self, synthetic_signal):
        """Verify no NaN values in spectrogram."""
        from src.features.stft import extract_spectrogram

        spectrogram = extract_spectrogram(synthetic_signal)
        assert not np.isnan(spectrogram).any(), "NaN values in spectrogram"

    def test_no_inf_values(self, synthetic_signal):
        """Verify no infinite values in spectrogram."""
        from src.features.stft import extract_spectrogram

        spectrogram = extract_spectrogram(synthetic_signal)
        assert not np.isinf(spectrogram).any(), "Inf values in spectrogram"

    def test_batch_generation(self, synthetic_signal_batch):
        """Test batch spectrogram generation."""
        from src.features.stft import extract_spectrogram

        # Process batch by iterating
        spectrograms = np.stack([extract_spectrogram(s) for s in synthetic_signal_batch])
        assert spectrograms.shape == (4, 128, 128, 2)

    def test_real_data(self, sample_signal):
        """Test spectrogram generation on real data."""
        from src.features.stft import extract_spectrogram

        spectrogram = extract_spectrogram(sample_signal)
        assert spectrogram.shape == (128, 128, 2)
        assert not np.isnan(spectrogram).any()


class TestCWTScalogram:
    """Test CWT scalogram generation."""

    def test_output_shape(self, synthetic_signal):
        """Verify output shape is (64, 128, 2)."""
        from src.features.cwt import extract_scalogram

        scalogram = extract_scalogram(synthetic_signal)
        assert scalogram.shape == (64, 128, 2), (
            f"Expected (64, 128, 2), got {scalogram.shape}"
        )

    def test_no_nan_values(self, synthetic_signal):
        """Verify no NaN values in scalogram."""
        from src.features.cwt import extract_scalogram

        scalogram = extract_scalogram(synthetic_signal)
        assert not np.isnan(scalogram).any(), "NaN values in scalogram"

    def test_no_inf_values(self, synthetic_signal):
        """Verify no infinite values in scalogram."""
        from src.features.cwt import extract_scalogram

        scalogram = extract_scalogram(synthetic_signal)
        assert not np.isinf(scalogram).any(), "Inf values in scalogram"

    def test_batch_generation(self, synthetic_signal_batch):
        """Test batch scalogram generation."""
        from src.features.cwt import generate_scalograms_batched

        # Only test first 2 for speed (CWT is slow)
        scalograms = generate_scalograms_batched(synthetic_signal_batch[:2])
        assert scalograms.shape == (2, 64, 128, 2)


class TestEnvelopeAnalysis:
    """Test envelope analysis features."""

    def test_envelope_extraction(self, synthetic_signal):
        """Test basic envelope extraction."""
        from src.features.envelope import compute_envelope

        h_channel = synthetic_signal[:, 0]
        envelope = compute_envelope(h_channel)

        assert envelope.shape == h_channel.shape
        assert not np.isnan(envelope).any()
        # Envelope should be non-negative
        assert (envelope >= 0).all()

    def test_dual_channel_features(self, synthetic_signal):
        """Test dual-channel envelope feature extraction."""
        from src.features.envelope import extract_envelope_features_dual_channel

        features = extract_envelope_features_dual_channel(synthetic_signal)
        # Returns numpy array (not dict)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        # Should have features from both channels (42 total)
        assert len(features) >= 40

    def test_no_nan_values(self, synthetic_signal):
        """Verify no NaN values in envelope features."""
        from src.features.envelope import extract_envelope_features_dual_channel

        features = extract_envelope_features_dual_channel(synthetic_signal)
        assert not np.isnan(features).any(), "NaN values in envelope features"


class TestFeatureGroups:
    """Test feature grouping for interpretability."""

    def test_get_feature_groups(self):
        """Test feature group retrieval."""
        from src.features.fusion import get_feature_groups

        groups = get_feature_groups()
        assert isinstance(groups, dict)
        assert "time_horizontal" in groups
        assert "time_vertical" in groups
        assert "freq_horizontal_spectral" in groups

    def test_all_features_covered(self):
        """Verify all features are in some group."""
        from src.features.fusion import get_all_feature_names, get_feature_groups

        all_names = set(get_all_feature_names())
        groups = get_feature_groups()
        grouped_names = set()
        for group_names in groups.values():
            grouped_names.update(group_names)

        assert all_names == grouped_names, (
            f"Features not in groups: {all_names - grouped_names}"
        )


class TestPerformance:
    """Test feature extraction performance."""

    def test_batch_processing_speed(self, synthetic_signal_batch):
        """Verify batch processing handles 100 samples in <1 second."""
        import time

        from src.features.fusion import FeatureExtractor

        # Create larger batch
        batch = np.tile(synthetic_signal_batch, (25, 1, 1))  # 100 samples
        assert len(batch) == 100

        extractor = FeatureExtractor(mode="all")

        start = time.time()
        features = extractor.extract(batch)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Batch processing too slow: {elapsed:.2f}s"
        assert features.shape == (100, NUM_ALL_FEATURES)
