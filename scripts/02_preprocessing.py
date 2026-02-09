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

"""GCS Preprocessing Script for XJTU-SY Bearing Dataset.

This script processes bearing vibration data from GCS and generates:
- Spectrograms (256x256 or 512x256 dual-channel)
- Statistical features (both horizontal and vertical channels)

Features:
- Dual-channel processing (horizontal and vertical vibration)
- Configurable spectrogram output (single/dual channel)
- Statistical features: mean, std, kurtosis per channel
- Cross-channel correlation feature
- Backward compatible with original single-channel output

Usage:
    python scripts/02_preprocessing.py
    python scripts/02_preprocessing.py --dual-channel
    python scripts/02_preprocessing.py --local-test  # Test without GCS

Output:
    GCS: gs://{BUCKET}/processed/spectrograms/{bearing_id}_{filename}.png
    GCS: gs://{BUCKET}/processed/stats/features.csv
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO, StringIO

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats


# CONFIG
BUCKET_NAME = os.environ.get("BUCKET_NAME", "xjtu-bearing-failure-dev-data")
INPUT_PREFIX = "xjtu_data"
OUTPUT_IMG_PREFIX = "processed/spectrograms"
OUTPUT_STATS_PREFIX = "processed/stats"
SAMPLING_RATE = 25600  # Hz


def generate_spectrogram_single(signal_data: np.ndarray, fs: int = SAMPLING_RATE) -> BytesIO:
    """Generate spectrogram for single channel (backward compatible).

    Args:
        signal_data: 1D array of vibration signal
        fs: Sampling frequency in Hz

    Returns:
        BytesIO buffer containing PNG image (256x256)
    """
    f, t, Sxx = scipy.signal.spectrogram(signal_data, fs)

    # Handle zero values for log scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(2.56, 2.56), dpi=100)  # 256x256 image
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno')
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf


def generate_spectrogram_dual(
    signal_h: np.ndarray,
    signal_v: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> BytesIO:
    """Generate dual-channel spectrogram (side-by-side).

    Args:
        signal_h: Horizontal vibration signal
        signal_v: Vertical vibration signal
        fs: Sampling frequency in Hz

    Returns:
        BytesIO buffer containing PNG image (512x256)
    """
    fig, axes = plt.subplots(1, 2, figsize=(5.12, 2.56), dpi=100)  # 512x256 image

    for ax, signal, title in [
        (axes[0], signal_h, 'Horizontal'),
        (axes[1], signal_v, 'Vertical'),
    ]:
        f, t, Sxx = scipy.signal.spectrogram(signal, fs)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno')
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel('Time (s)', fontsize=6)
        ax.set_ylabel('Freq (Hz)', fontsize=6)
        ax.tick_params(labelsize=5)

    plt.tight_layout(pad=0.5)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    return buf


def extract_features(
    signal_h: np.ndarray,
    signal_v: np.ndarray | None = None,
) -> dict:
    """Extract statistical features from vibration signals.

    Args:
        signal_h: Horizontal vibration signal
        signal_v: Vertical vibration signal (optional for backward compat)

    Returns:
        Dictionary of features
    """
    features = {}

    # Horizontal channel features (with original names for backward compat)
    features['mean'] = float(np.mean(signal_h))
    features['std'] = float(np.std(signal_h))
    features['kurtosis'] = float(scipy.stats.kurtosis(signal_h))

    # New prefixed horizontal features
    features['h_mean'] = features['mean']
    features['h_std'] = features['std']
    features['h_kurtosis'] = features['kurtosis']
    features['h_rms'] = float(np.sqrt(np.mean(signal_h ** 2)))
    features['h_peak'] = float(np.max(np.abs(signal_h)))
    features['h_skewness'] = float(scipy.stats.skew(signal_h))

    # Vertical channel features (if provided)
    if signal_v is not None:
        features['v_mean'] = float(np.mean(signal_v))
        features['v_std'] = float(np.std(signal_v))
        features['v_kurtosis'] = float(scipy.stats.kurtosis(signal_v))
        features['v_rms'] = float(np.sqrt(np.mean(signal_v ** 2)))
        features['v_peak'] = float(np.max(np.abs(signal_v)))
        features['v_skewness'] = float(scipy.stats.skew(signal_v))

        # Cross-channel correlation
        features['cross_correlation'] = float(np.corrcoef(signal_h, signal_v)[0, 1])
    else:
        # Fill with NaN for backward compat if vertical not available
        features['v_mean'] = np.nan
        features['v_std'] = np.nan
        features['v_kurtosis'] = np.nan
        features['v_rms'] = np.nan
        features['v_peak'] = np.nan
        features['v_skewness'] = np.nan
        features['cross_correlation'] = np.nan

    return features


def process_file(blob, bucket, dual_channel: bool = True):
    """Process a single file from GCS.

    Args:
        blob: GCS blob object
        bucket: GCS bucket object
        dual_channel: Whether to process both channels

    Returns:
        Dictionary with metadata and features
    """
    # 1. Download and parse CSV
    content = blob.download_as_string()
    df = pd.read_csv(StringIO(content.decode('utf-8')))

    # Extract Metadata from Hive Path
    # Path: xjtu_data/condition=X/bearing_id=Y/filename.csv
    parts = blob.name.split('/')
    condition = parts[1].split('=')[1]
    bearing_id = parts[2].split('=')[1]
    filename = parts[3]

    # 2. Extract signals
    signal_h = df['Horizontal_vibration_signals'].values
    signal_v = df['Vertical_vibration_signals'].values if dual_channel else None

    # 3. Extract features
    features = extract_features(signal_h, signal_v)

    # Add metadata
    stats = {
        'bearing_id': bearing_id,
        'condition': condition,
        'filename': filename,
        **features,
    }

    # 4. Generate spectrogram
    if dual_channel and signal_v is not None:
        img_buf = generate_spectrogram_dual(signal_h, signal_v)
        img_suffix = '_dual'
    else:
        img_buf = generate_spectrogram_single(signal_h)
        img_suffix = ''

    # 5. Set output path and upload
    img_path = f"{OUTPUT_IMG_PREFIX}/{bearing_id}_{filename.replace('.csv', '')}{img_suffix}.png"
    stats['gcs_image_uri'] = f"gs://{BUCKET_NAME}/{img_path}"

    img_blob = bucket.blob(img_path)
    img_blob.upload_from_file(img_buf, content_type='image/png')

    return stats


def process_file_local(
    file_path: str,
    dual_channel: bool = True,
    output_dir: str = 'outputs/gcs_preview',
) -> dict:
    """Process a single local file (for testing without GCS).

    Args:
        file_path: Path to local CSV file
        dual_channel: Whether to process both channels
        output_dir: Directory for output files

    Returns:
        Dictionary with metadata and features
    """
    import os
    from pathlib import Path

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(file_path)

    # Extract metadata from path (assuming standard structure)
    path = Path(file_path)
    filename = path.name
    bearing_id = path.parent.name
    condition = path.parent.parent.name

    # Extract signals
    signal_h = df['Horizontal_vibration_signals'].values
    signal_v = df['Vertical_vibration_signals'].values if dual_channel else None

    # Extract features
    features = extract_features(signal_h, signal_v)

    # Add metadata
    stats = {
        'bearing_id': bearing_id,
        'condition': condition,
        'filename': filename,
        **features,
    }

    # Generate and save spectrogram
    if dual_channel and signal_v is not None:
        img_buf = generate_spectrogram_dual(signal_h, signal_v)
        img_suffix = '_dual'
    else:
        img_buf = generate_spectrogram_single(signal_h)
        img_suffix = ''

    img_filename = f"{bearing_id}_{filename.replace('.csv', '')}{img_suffix}.png"
    img_path = os.path.join(output_dir, img_filename)

    with open(img_path, 'wb') as f:
        f.write(img_buf.getvalue())

    stats['local_image_path'] = img_path
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess XJTU-SY bearing data from GCS'
    )
    parser.add_argument(
        '--dual-channel',
        action='store_true',
        default=True,
        help='Process both horizontal and vertical channels (default: True)',
    )
    parser.add_argument(
        '--single-channel',
        action='store_true',
        help='Process only horizontal channel (backward compatible mode)',
    )
    parser.add_argument(
        '--local-test',
        type=str,
        default=None,
        help='Test with local file instead of GCS (provide path to CSV)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/gcs_preview',
        help='Output directory for local test (default: outputs/gcs_preview)',
    )
    args = parser.parse_args()

    dual_channel = not args.single_channel

    # Local test mode
    if args.local_test:
        print(f"Testing locally with: {args.local_test}")
        print(f"Dual channel: {dual_channel}")

        stats = process_file_local(
            args.local_test,
            dual_channel=dual_channel,
            output_dir=args.output_dir,
        )

        print("\nExtracted features:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return

    # GCS mode
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    all_stats = []
    blobs = client.list_blobs(BUCKET_NAME, prefix=INPUT_PREFIX)

    print("Starting Processing...")
    print(f"Dual channel mode: {dual_channel}")

    for blob in blobs:
        if blob.name.endswith('.csv'):
            try:
                stats = process_file(blob, bucket, dual_channel=dual_channel)
                all_stats.append(stats)
                print(f"Processed: {blob.name}")
            except Exception as e:
                print(f"Error {blob.name}: {e}")

    # Save Stats to CSV (for BigQuery loading)
    stats_df = pd.DataFrame(all_stats)

    # Define column order for consistent output
    base_cols = ['bearing_id', 'condition', 'filename']
    feature_cols = [
        # Original (backward compat)
        'mean', 'std', 'kurtosis',
        # Horizontal
        'h_mean', 'h_std', 'h_kurtosis', 'h_rms', 'h_peak', 'h_skewness',
        # Vertical
        'v_mean', 'v_std', 'v_kurtosis', 'v_rms', 'v_peak', 'v_skewness',
        # Cross-channel
        'cross_correlation',
    ]
    end_cols = ['gcs_image_uri']

    # Reorder columns (only include columns that exist)
    ordered_cols = base_cols + [c for c in feature_cols if c in stats_df.columns] + end_cols
    stats_df = stats_df[[c for c in ordered_cols if c in stats_df.columns]]

    stats_df.to_csv('features.csv', index=False)

    # Upload features CSV
    bucket.blob(f"{OUTPUT_STATS_PREFIX}/features.csv").upload_from_filename('features.csv')
    print("Job Complete. Features uploaded.")
    print(f"Total files processed: {len(all_stats)}")
    print(f"Features CSV columns: {list(stats_df.columns)}")


if __name__ == "__main__":
    main()
