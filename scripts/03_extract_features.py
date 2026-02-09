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

"""Batch Feature Extraction Script for XJTU-SY Bearing Dataset (FEAT-10).

This script processes all bearing vibration files and extracts time-domain
and frequency-domain features using the unified FeatureExtractor.

Features:
    - Processes all 9,216 files across 15 bearings and 3 conditions
    - Extracts 65 features per sample (37 time + 28 frequency)
    - Supports checkpoint-based resumable processing
    - Progress tracking with estimated time remaining
    - Error logging and extraction statistics

Usage:
    python scripts/03_extract_features.py
    python scripts/03_extract_features.py --resume  # Resume from checkpoint
    python scripts/03_extract_features.py --output outputs/features/custom.csv

Output:
    CSV file with columns: condition, bearing_id, filename, file_idx, rul,
    and 65 feature columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import XJTUBearingLoader, BEARINGS_PER_CONDITION
from src.data.rul_labels import compute_twostage_rul, piecewise_linear_rul
from src.features.fusion import FeatureExtractor
from src.onset.labels import load_onset_labels


# Configuration
DEFAULT_OUTPUT = "outputs/features/features_v2.csv"
CHECKPOINT_FILE = "outputs/features/.checkpoint.json"
LOG_FILE = "outputs/features/extraction.log"


def setup_logging(log_file: str | Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("feature_extraction")
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


def load_checkpoint(checkpoint_file: str | Path) -> set[str]:
    """Load checkpoint of processed files.

    Returns:
        Set of processed file identifiers (condition/bearing_id/filename).
    """
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
            return set(data.get("processed", []))
    return set()


def save_checkpoint(
    checkpoint_file: str | Path,
    processed: set[str],
    stats: dict,
) -> None:
    """Save checkpoint with processed files and statistics."""
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "w") as f:
        json.dump({
            "processed": list(processed),
            "stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)


def extract_features_for_bearing(
    loader: XJTUBearingLoader,
    extractor: FeatureExtractor,
    condition: str,
    bearing_id: str,
    processed: set[str],
    logger: logging.Logger,
    onset_labels: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """Extract features for all files in a bearing.

    Args:
        loader: Data loader instance.
        extractor: Feature extractor instance.
        condition: Operating condition.
        bearing_id: Bearing identifier.
        processed: Set of already processed file IDs.
        logger: Logger instance.
        onset_labels: Optional onset labels dict for two-stage RUL computation.

    Returns:
        Tuple of (list of feature records, list of newly processed IDs).
    """
    records = []
    newly_processed = []
    errors = []

    # Get total files for RUL calculation
    bearing_path = loader.data_root / condition / bearing_id
    total_files = len(list(bearing_path.glob("*.csv")))

    # Pre-compute RUL arrays (once per bearing, not per file)
    rul_array = piecewise_linear_rul(total_files, max_rul=125.0)

    twostage_rul_array = None
    if onset_labels is not None:
        onset_idx = onset_labels[bearing_id].onset_file_idx if bearing_id in onset_labels else None
        if onset_idx is not None:
            twostage_rul_array = compute_twostage_rul(total_files, onset_idx=onset_idx, max_rul=125.0)
        else:
            logger.warning(f"No onset label for {bearing_id}, skipping rul_twostage")

    for signal, filename, file_idx in loader.iter_bearing(condition, bearing_id):
        file_id = f"{condition}/{bearing_id}/{filename}"

        # Skip if already processed
        if file_id in processed:
            continue

        try:
            # Extract features
            features = extractor.extract(signal)
            feature_dict = dict(zip(extractor.feature_names, features))

            # Calculate RUL - get the value at file_idx from the RUL array
            rul = float(rul_array[file_idx])

            # Build record with metadata
            record = {
                "condition": condition,
                "bearing_id": bearing_id,
                "filename": filename,
                "file_idx": file_idx,
                "total_files": total_files,
                "rul": rul,
                **feature_dict,
            }

            # Add two-stage RUL if available
            if twostage_rul_array is not None:
                record["rul_twostage"] = float(twostage_rul_array[file_idx])

            records.append(record)
            newly_processed.append(file_id)

        except Exception as e:
            error_msg = f"Error processing {file_id}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return records, newly_processed


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from XJTU-SY bearing dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="assets/Data/XJTU-SY_Bearing_Datasets",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Save checkpoint every N files (default: 500)",
    )
    parser.add_argument(
        "--twostage-rul",
        action="store_true",
        help="Add rul_twostage column using onset labels from YAML",
    )
    parser.add_argument(
        "--onset-labels",
        type=str,
        default=None,
        help="Path to onset_labels.yaml (default: configs/onset_labels.yaml)",
    )
    args = parser.parse_args()

    # Setup
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_FILE)

    logger.info("=" * 60)
    logger.info("XJTU-SY Bearing Feature Extraction (FEAT-10)")
    logger.info("=" * 60)

    # Initialize loader and extractor
    try:
        loader = XJTUBearingLoader(args.data_root)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        sys.exit(1)

    # Load onset labels if two-stage RUL requested
    onset_labels = None
    if args.twostage_rul:
        onset_labels = load_onset_labels(args.onset_labels)
        logger.info(f"Two-stage RUL enabled: loaded onset labels for {len(onset_labels)} bearings")

    # Get total file count
    metadata = loader.get_metadata()
    total_files = len(metadata)
    logger.info(f"Dataset: {total_files} files across 15 bearings")

    # Load checkpoint if resuming
    processed: set[str] = set()
    if args.resume:
        processed = load_checkpoint(CHECKPOINT_FILE)
        if processed:
            logger.info(f"Resuming: {len(processed)} files already processed")
        else:
            logger.info("No checkpoint found, starting fresh")

    # Collect all records
    all_records: list[dict] = []
    stats = {
        "total_files": total_files,
        "processed": 0,
        "errors": 0,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Process each condition/bearing
    start_time = time.time()
    files_since_checkpoint = 0

    # Create progress bar for bearings
    bearings = [
        (cond, bid)
        for cond, bids in BEARINGS_PER_CONDITION.items()
        for bid in bids
    ]

    with tqdm(bearings, desc="Processing bearings", unit="bearing") as pbar:
        for condition, bearing_id in pbar:
            pbar.set_postfix({"bearing": bearing_id})

            # Create extractor with condition-specific frequencies
            extractor = FeatureExtractor(
                mode="all",
                condition=condition,
                sampling_rate=25600.0,
            )

            records, newly_processed = extract_features_for_bearing(
                loader=loader,
                extractor=extractor,
                condition=condition,
                bearing_id=bearing_id,
                processed=processed,
                logger=logger,
                onset_labels=onset_labels,
            )

            all_records.extend(records)
            processed.update(newly_processed)
            files_since_checkpoint += len(newly_processed)
            stats["processed"] = len(processed)

            # Periodic checkpoint
            if files_since_checkpoint >= args.checkpoint_interval:
                save_checkpoint(CHECKPOINT_FILE, processed, stats)
                files_since_checkpoint = 0
                logger.debug(f"Checkpoint saved: {len(processed)} files")

    # Final statistics
    elapsed = time.time() - start_time
    stats["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["files_per_second"] = round(len(all_records) / elapsed, 2) if elapsed > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Extraction Complete")
    logger.info("=" * 60)
    logger.info(f"Files processed: {len(all_records)}")
    logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Throughput: {stats['files_per_second']:.1f} files/second")

    # Create DataFrame and save
    if all_records:
        df = pd.DataFrame(all_records)

        # Sort by condition, bearing, file index
        df = df.sort_values(["condition", "bearing_id", "file_idx"])

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Output saved: {output_path}")
        logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Log feature summary
        feature_cols = [c for c in df.columns if c not in
                       ["condition", "bearing_id", "filename", "file_idx", "total_files", "rul"]]
        logger.info(f"Features: {len(feature_cols)} columns")

        # Final checkpoint with completion flag
        stats["completed"] = True
        save_checkpoint(CHECKPOINT_FILE, processed, stats)

    else:
        logger.warning("No records to save!")
        if args.resume and len(processed) == total_files:
            logger.info("All files were already processed. Use without --resume to reprocess.")

    # Clean up checkpoint on full completion
    checkpoint_path = Path(CHECKPOINT_FILE)
    if checkpoint_path.exists() and len(processed) >= total_files:
        logger.info("Full dataset processed. Checkpoint retained for reference.")

    logger.info("Done!")


if __name__ == "__main__":
    main()
