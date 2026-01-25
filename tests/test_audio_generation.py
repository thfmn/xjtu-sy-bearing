"""Tests for audio generation functionality.

Tests WAV file generation from bearing vibration signals.
Note: The audio module (src/utils/audio.py) is not yet implemented.
These tests serve as placeholders and will be enabled when the module exists.

Expected functionality:
- Convert 25.6kHz vibration signals to playable WAV files
- Resample to 44.1kHz for standard audio playback
- Generate audio for different lifecycle stages (healthy, degrading, failed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import DATA_ROOT, SAMPLING_RATE


def audio_module_available() -> bool:
    """Check if audio module is implemented."""
    try:
        from src.utils import audio
        return True
    except ImportError:
        return False


# Mark all tests to skip if audio module not available
pytestmark = pytest.mark.skipif(
    not audio_module_available(),
    reason="Audio module (src/utils/audio.py) not yet implemented"
)


class TestWAVGeneration:
    """Test WAV file generation from vibration signals."""

    def test_wav_generation_basic(self, synthetic_signal, tmp_path):
        """Generate WAV file from synthetic signal."""
        from src.utils.audio import signal_to_wav

        output_path = tmp_path / "test.wav"
        signal_to_wav(
            synthetic_signal[:, 0],  # Use horizontal channel
            output_path,
            sample_rate=SAMPLING_RATE,
        )

        assert output_path.exists(), "WAV file not created"
        assert output_path.stat().st_size > 0, "WAV file is empty"

    def test_wav_playable(self, synthetic_signal, tmp_path):
        """Verify generated WAV is a valid audio file."""
        from scipy.io import wavfile

        from src.utils.audio import signal_to_wav

        output_path = tmp_path / "test.wav"
        signal_to_wav(synthetic_signal[:, 0], output_path, sample_rate=SAMPLING_RATE)

        # Read back and verify
        sample_rate, data = wavfile.read(output_path)
        assert sample_rate > 0
        assert len(data) > 0


class TestResampling:
    """Test resampling from 25.6kHz to 44.1kHz."""

    def test_resample_preserves_length_ratio(self, synthetic_signal, tmp_path):
        """Verify resampling produces correct output length."""
        from src.utils.audio import resample_signal

        original_rate = 25600
        target_rate = 44100
        h_channel = synthetic_signal[:, 0]

        resampled = resample_signal(h_channel, original_rate, target_rate)

        expected_length = int(len(h_channel) * target_rate / original_rate)
        assert abs(len(resampled) - expected_length) <= 1

    def test_resample_preserves_frequency_content(self, tmp_path):
        """Verify resampling preserves signal frequency content (no aliasing)."""
        from src.utils.audio import resample_signal

        # Create signal with known frequency
        original_rate = 25600
        target_rate = 44100
        duration = 1.0
        freq = 100  # Hz

        t = np.linspace(0, duration, int(original_rate * duration))
        signal = np.sin(2 * np.pi * freq * t)

        resampled = resample_signal(signal, original_rate, target_rate)

        # Check that resampled signal still has the same dominant frequency
        from scipy import signal as scipy_signal
        freqs, psd = scipy_signal.welch(resampled, fs=target_rate, nperseg=4096)
        dominant_freq = freqs[np.argmax(psd)]

        assert abs(dominant_freq - freq) < 5, (
            f"Frequency content not preserved: expected ~{freq}Hz, got {dominant_freq}Hz"
        )


class TestNormalization:
    """Test amplitude normalization."""

    def test_normalize_to_unit_range(self, synthetic_signal):
        """Verify normalization produces [-1, 1] range."""
        from src.utils.audio import normalize_amplitude

        h_channel = synthetic_signal[:, 0]
        normalized = normalize_amplitude(h_channel)

        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0
        assert abs(max(abs(normalized.min()), normalized.max()) - 1.0) < 0.01

    def test_normalize_prevents_clipping(self, synthetic_signal):
        """Verify normalization prevents audio clipping."""
        from src.utils.audio import normalize_amplitude

        # Create signal with large amplitude
        large_signal = synthetic_signal[:, 0] * 1000
        normalized = normalize_amplitude(large_signal)

        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0


class TestBearingAudioGeneration:
    """Test generating audio for specific bearing lifecycle stages."""

    def test_generate_bearing_audio(self, bearing_loader, tmp_path):
        """Generate audio for Bearing1_1 at lifecycle stages."""
        from src.utils.audio import generate_bearing_audio

        output_dir = tmp_path / "bearing_audio"
        output_dir.mkdir()

        # Generate for early, mid, and late lifecycle
        stages = [1, 60, 120]  # File indices
        paths = generate_bearing_audio(
            "35Hz12kN",
            "Bearing1_1",
            stages,
            output_dir,
            data_loader=bearing_loader,
        )

        assert len(paths) == len(stages)
        for path in paths:
            assert Path(path).exists(), f"Audio file not created: {path}"

    def test_audio_files_playable(self, bearing_loader, tmp_path):
        """Verify all generated audio files are valid WAV files."""
        from scipy.io import wavfile

        from src.utils.audio import generate_bearing_audio

        output_dir = tmp_path / "bearing_audio"
        output_dir.mkdir()

        paths = generate_bearing_audio(
            "35Hz12kN",
            "Bearing1_1",
            [1, 60, 120],
            output_dir,
            data_loader=bearing_loader,
        )

        for path in paths:
            sample_rate, data = wavfile.read(path)
            assert sample_rate in [25600, 44100], f"Unexpected sample rate: {sample_rate}"
            assert len(data) > 0, f"Empty audio file: {path}"

    def test_audible_differences(self, bearing_loader, tmp_path):
        """Verify audible differences between healthy and failed states."""
        from src.utils.audio import generate_bearing_audio

        output_dir = tmp_path / "bearing_audio"
        output_dir.mkdir()

        # Get file count to determine end-of-life
        metadata = bearing_loader.get_metadata()
        bearing_files = metadata[
            (metadata["condition"] == "35Hz12kN") &
            (metadata["bearing_id"] == "Bearing1_1")
        ]
        last_file_idx = bearing_files["file_idx"].max()

        # Compare early vs late
        stages = [1, last_file_idx]
        paths = generate_bearing_audio(
            "35Hz12kN",
            "Bearing1_1",
            stages,
            output_dir,
            data_loader=bearing_loader,
        )

        from scipy.io import wavfile

        _, healthy = wavfile.read(paths[0])
        _, failed = wavfile.read(paths[1])

        # Failed bearing should have higher RMS (more vibration)
        healthy_rms = np.sqrt(np.mean(healthy.astype(float) ** 2))
        failed_rms = np.sqrt(np.mean(failed.astype(float) ** 2))

        assert failed_rms > healthy_rms, (
            "Expected failed bearing to have higher RMS vibration"
        )


class TestCLIInterface:
    """Test CLI interface for batch audio generation."""

    def test_cli_basic(self, tmp_path):
        """Test basic CLI invocation."""
        # This would test scripts/05_create_audio.py
        # Placeholder for when script is implemented
        pytest.skip("CLI script not yet implemented")

    def test_cli_bearing_selection(self, tmp_path):
        """Test CLI bearing selection options."""
        pytest.skip("CLI script not yet implemented")

    def test_cli_lifecycle_percentages(self, tmp_path):
        """Test CLI lifecycle percentage options."""
        pytest.skip("CLI script not yet implemented")
