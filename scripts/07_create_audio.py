#!/usr/bin/env python3

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

"""Audio Generation Script for XJTU-SY Bearing Dataset (FEAT-12).

This script converts vibration signals to WAV audio files for auditory analysis
of bearing degradation patterns.

Features:
    - Converts 25.6kHz vibration signals to 44.1kHz WAV files
    - Supports selecting specific bearings and lifecycle stages
    - Amplitude normalization to prevent clipping
    - Batch generation via CLI arguments

Usage:
    python scripts/07_create_audio.py
    python scripts/07_create_audio.py --bearing 35Hz12kN/Bearing1_1
    python scripts/07_create_audio.py --bearing 35Hz12kN/Bearing1_1 --stages 0 50 100
    python scripts/07_create_audio.py --all-bearings --stages 0 50 100

Output:
    WAV files in outputs/audio/ organized by bearing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import XJTUBearingLoader, BEARINGS_PER_CONDITION


# Configuration
DEFAULT_OUTPUT = "outputs/audio"
LOG_FILE = "outputs/audio/generation.log"

# Audio parameters
SOURCE_SAMPLE_RATE = 25600  # Hz - XJTU-SY dataset sampling rate
TARGET_SAMPLE_RATE = 44100  # Hz - Standard audio sample rate


def setup_logging(log_file: str | Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("audio_generation")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def resample_signal(
    data: np.ndarray,
    source_rate: int = SOURCE_SAMPLE_RATE,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample signal from source to target sample rate.

    Uses scipy.signal.resample_poly for efficient polyphase resampling.

    Args:
        data: Input signal array (samples,) or (samples, channels).
        source_rate: Original sampling rate in Hz.
        target_rate: Target sampling rate in Hz.

    Returns:
        Resampled signal array.
    """
    # Calculate resampling ratio using GCD for rational approximation
    from math import gcd
    g = gcd(source_rate, target_rate)
    up = target_rate // g
    down = source_rate // g

    # Handle multi-channel
    if data.ndim == 1:
        return signal.resample_poly(data, up, down)
    else:
        # Resample each channel
        resampled_channels = []
        for ch in range(data.shape[1]):
            resampled = signal.resample_poly(data[:, ch], up, down)
            resampled_channels.append(resampled)
        return np.column_stack(resampled_channels)


def normalize_audio(
    data: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    """Normalize audio amplitude to prevent clipping.

    Args:
        data: Input signal array.
        target_peak: Target peak amplitude (0-1). Default 0.95 for headroom.

    Returns:
        Normalized signal in range [-target_peak, target_peak].
    """
    max_val = np.abs(data).max()
    if max_val > 0:
        return data * (target_peak / max_val)
    return data


def signal_to_wav(
    data: np.ndarray,
    output_path: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
) -> None:
    """Convert signal array to WAV file.

    Args:
        data: Signal array (samples,) or (samples, channels).
        output_path: Output WAV file path.
        sample_rate: Sample rate for the WAV file.
        normalize: Whether to normalize amplitude.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        data = normalize_audio(data)

    # Convert to 16-bit PCM
    # Scale from [-1, 1] to int16 range
    data_int16 = (data * 32767).astype(np.int16)

    wavfile.write(str(output_path), sample_rate, data_int16)


def get_lifecycle_indices(
    total_files: int,
    percentages: list[int],
) -> list[int]:
    """Convert lifecycle percentages to file indices.

    Args:
        total_files: Total number of files for the bearing.
        percentages: List of lifecycle percentages (0-100).

    Returns:
        List of file indices corresponding to percentages.
    """
    indices = []
    for pct in percentages:
        # Clamp to valid range
        pct = max(0, min(100, pct))
        idx = int((pct / 100) * (total_files - 1))
        indices.append(idx)
    return indices


def generate_audio_for_bearing(
    loader: XJTUBearingLoader,
    condition: str,
    bearing_id: str,
    output_dir: Path,
    lifecycle_percentages: list[int] | None = None,
    channel: str = "horizontal",
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Generate WAV files for a bearing at specified lifecycle stages.

    Args:
        loader: Data loader instance.
        condition: Operating condition (e.g., '35Hz12kN').
        bearing_id: Bearing identifier (e.g., 'Bearing1_1').
        output_dir: Output directory for WAV files.
        lifecycle_percentages: List of lifecycle percentages (default: [0, 50, 100]).
        channel: Channel to use ('horizontal', 'vertical', or 'both').
        logger: Logger instance.

    Returns:
        List of generated WAV file paths.
    """
    if lifecycle_percentages is None:
        lifecycle_percentages = [0, 50, 100]

    # Load bearing data
    try:
        data_list, filenames = loader.load_bearing(condition, bearing_id)
    except Exception as e:
        if logger:
            logger.error(f"Failed to load {condition}/{bearing_id}: {e}")
        return []

    total_files = len(data_list)
    if total_files == 0:
        if logger:
            logger.warning(f"No files found for {condition}/{bearing_id}")
        return []

    # Get file indices for lifecycle stages
    indices = get_lifecycle_indices(total_files, lifecycle_percentages)

    # Create output directory for this bearing
    bearing_output_dir = output_dir / condition / bearing_id
    bearing_output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for pct, idx in zip(lifecycle_percentages, indices):
        # Get signal data
        signal_data = data_list[idx]
        filename = filenames[idx]

        # Select channel(s)
        if channel == "horizontal":
            audio_data = signal_data[:, 0]
            ch_suffix = "_h"
        elif channel == "vertical":
            audio_data = signal_data[:, 1]
            ch_suffix = "_v"
        else:  # both
            audio_data = signal_data  # Keep both channels
            ch_suffix = "_stereo"

        # Resample to 44.1kHz
        resampled = resample_signal(audio_data)

        # Determine lifecycle stage name
        if pct <= 10:
            stage = "healthy"
        elif pct >= 90:
            stage = "failed"
        else:
            stage = "degrading"

        # Generate output filename
        wav_filename = f"{bearing_id}_{stage}_{pct}pct{ch_suffix}.wav"
        wav_path = bearing_output_dir / wav_filename

        # Write WAV file
        signal_to_wav(resampled, wav_path)
        generated_files.append(wav_path)

        if logger:
            logger.info(f"Generated: {wav_path.relative_to(output_dir)}")

    return generated_files


def parse_bearing_spec(spec: str) -> tuple[str, str]:
    """Parse bearing specification string.

    Args:
        spec: Bearing spec like '35Hz12kN/Bearing1_1' or 'Bearing1_1'.

    Returns:
        Tuple of (condition, bearing_id).
    """
    if "/" in spec:
        parts = spec.split("/")
        return parts[0], parts[1]
    else:
        # Assume it's just bearing_id, need to find condition
        # Check all conditions
        for condition, bearings in BEARINGS_PER_CONDITION.items():
            if spec in bearings:
                return condition, spec
        raise ValueError(f"Bearing '{spec}' not found in any condition")


def main():
    parser = argparse.ArgumentParser(
        description="Generate WAV audio files from bearing vibration signals"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--bearing", "-b",
        type=str,
        help="Specific bearing to process (e.g., '35Hz12kN/Bearing1_1')"
    )
    parser.add_argument(
        "--all-bearings",
        action="store_true",
        help="Process all bearings"
    )
    parser.add_argument(
        "--stages", "-s",
        type=int,
        nargs="+",
        default=[0, 50, 100],
        help="Lifecycle percentages to generate (default: 0 50 100)"
    )
    parser.add_argument(
        "--channel", "-c",
        type=str,
        choices=["horizontal", "vertical", "both"],
        default="horizontal",
        help="Channel to convert (default: horizontal)"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_FILE)

    logger.info("=" * 60)
    logger.info("Audio Generation Script for XJTU-SY Bearing Dataset")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Lifecycle stages: {args.stages}%")
    logger.info(f"Channel: {args.channel}")

    # Initialize loader
    loader = XJTUBearingLoader()

    # Determine bearings to process
    bearings_to_process = []

    if args.bearing:
        condition, bearing_id = parse_bearing_spec(args.bearing)
        bearings_to_process.append((condition, bearing_id))
    elif args.all_bearings:
        for condition, bearings in BEARINGS_PER_CONDITION.items():
            for bearing_id in bearings:
                bearings_to_process.append((condition, bearing_id))
    else:
        # Default: process one sample bearing from each condition
        for condition in BEARINGS_PER_CONDITION:
            first_bearing = BEARINGS_PER_CONDITION[condition][0]
            bearings_to_process.append((condition, first_bearing))

    logger.info(f"Bearings to process: {len(bearings_to_process)}")

    # Process bearings
    all_generated = []

    for condition, bearing_id in tqdm(bearings_to_process, desc="Processing bearings"):
        logger.info(f"\nProcessing {condition}/{bearing_id}...")

        generated = generate_audio_for_bearing(
            loader=loader,
            condition=condition,
            bearing_id=bearing_id,
            output_dir=output_dir,
            lifecycle_percentages=args.stages,
            channel=args.channel,
            logger=logger,
        )
        all_generated.extend(generated)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Total WAV files generated: {len(all_generated)}")
    logger.info(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
