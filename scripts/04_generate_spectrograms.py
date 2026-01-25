#!/usr/bin/env python3
"""Batch Spectrogram Generation Script for XJTU-SY Bearing Dataset (FEAT-11).

This script generates STFT spectrograms (and optionally CWT scalograms) for all
bearing vibration files using the time-frequency modules.

Features:
    - Generates STFT spectrograms for all 9,216 files
    - Optional CWT scalogram generation (--include-cwt flag)
    - Configurable output format: PNG (visualization) or NPY (training)
    - Hive-style partitioning: condition={cond}/bearing_id={id}/
    - Parallel processing for speedup
    - Checkpoint-based resumable processing
    - Progress tracking with estimated time remaining

Usage:
    python scripts/04_generate_spectrograms.py
    python scripts/04_generate_spectrograms.py --include-cwt
    python scripts/04_generate_spectrograms.py --format png --workers 4
    python scripts/04_generate_spectrograms.py --resume

Output Structure:
    outputs/spectrograms/
    ├── stft/
    │   └── condition={cond}/
    │       └── bearing_id={id}/
    │           ├── 1.npy (or 1.png)
    │           ├── 2.npy
    │           └── ...
    └── cwt/  (if --include-cwt)
        └── condition={cond}/
            └── bearing_id={id}/
                └── ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    pass


# Configuration
DEFAULT_OUTPUT_DIR = "outputs/spectrograms"


@dataclass
class GenerationConfig:
    """Configuration for spectrogram generation."""

    output_dir: Path
    format: str  # "npy" or "png"
    include_cwt: bool
    workers: int
    resume: bool
    data_root: Path
    checkpoint_interval: int


def setup_logging(log_file: str | Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("spectrogram_generation")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def load_checkpoint(checkpoint_file: str | Path) -> dict:
    """Load checkpoint of processed files.

    Returns:
        Dict with processed file sets for stft and cwt.
    """
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
            return {
                "stft": set(data.get("stft_processed", [])),
                "cwt": set(data.get("cwt_processed", [])),
            }
    return {"stft": set(), "cwt": set()}


def save_checkpoint(
    checkpoint_file: str | Path,
    stft_processed: set[str],
    cwt_processed: set[str],
    stats: dict,
) -> None:
    """Save checkpoint with processed files and statistics."""
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "w") as f:
        json.dump(
            {
                "stft_processed": list(stft_processed),
                "cwt_processed": list(cwt_processed),
                "stats": stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )


def save_spectrogram_npy(
    spectrogram: np.ndarray,
    output_path: Path,
) -> None:
    """Save spectrogram as NPY file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, spectrogram)


def save_spectrogram_png(
    spectrogram: np.ndarray,
    output_path: Path,
) -> None:
    """Save spectrogram as PNG image.

    For dual-channel spectrograms, creates a side-by-side visualization.
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if spectrogram.ndim == 3 and spectrogram.shape[-1] == 2:
        # Dual channel: create side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Horizontal channel
        im0 = axes[0].imshow(
            spectrogram[:, :, 0],
            aspect="auto",
            origin="lower",
            cmap="inferno",
        )
        axes[0].set_title("Horizontal")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Frequency")
        plt.colorbar(im0, ax=axes[0])

        # Vertical channel
        im1 = axes[1].imshow(
            spectrogram[:, :, 1],
            aspect="auto",
            origin="lower",
            cmap="inferno",
        )
        axes[1].set_title("Vertical")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Frequency")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
    else:
        # Single channel
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            cmap="inferno",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax)

    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def process_single_file(args: tuple) -> dict:
    """Process a single file (for parallel execution).

    Args:
        args: Tuple of (file_path, condition, bearing_id, filename, output_dir,
              format, include_cwt)

    Returns:
        Dict with processing results.
    """
    # Import inside function for multiprocessing
    from src.data.loader import XJTUBearingLoader
    from src.features.stft import extract_spectrogram as extract_stft
    from src.features.cwt import extract_scalogram as extract_cwt

    file_path, condition, bearing_id, filename, output_dir, fmt, include_cwt = args
    file_id = f"{condition}/{bearing_id}/{filename}"
    results = {"file_id": file_id, "stft_success": False, "cwt_success": False, "error": None}

    try:
        # Load signal
        loader = XJTUBearingLoader()
        signal = loader.load_file(file_path)

        # Determine base filename (without extension)
        base_name = Path(filename).stem

        # Generate STFT spectrogram
        stft_spec = extract_stft(signal)
        stft_output_dir = output_dir / "stft" / f"condition={condition}" / f"bearing_id={bearing_id}"

        if fmt == "npy":
            stft_path = stft_output_dir / f"{base_name}.npy"
            save_spectrogram_npy(stft_spec, stft_path)
        else:
            stft_path = stft_output_dir / f"{base_name}.png"
            save_spectrogram_png(stft_spec, stft_path)

        results["stft_success"] = True

        # Generate CWT scalogram if requested
        if include_cwt:
            cwt_spec = extract_cwt(signal)
            cwt_output_dir = output_dir / "cwt" / f"condition={condition}" / f"bearing_id={bearing_id}"

            if fmt == "npy":
                cwt_path = cwt_output_dir / f"{base_name}.npy"
                save_spectrogram_npy(cwt_spec, cwt_path)
            else:
                cwt_path = cwt_output_dir / f"{base_name}.png"
                save_spectrogram_png(cwt_spec, cwt_path)

            results["cwt_success"] = True

    except Exception as e:
        results["error"] = str(e)

    return results


def get_all_files(data_root: Path) -> list[dict]:
    """Get list of all files to process.

    Returns:
        List of dicts with file info: condition, bearing_id, filename, file_path.
    """
    from src.data.loader import BEARINGS_PER_CONDITION

    files = []
    for condition, bearings in BEARINGS_PER_CONDITION.items():
        for bearing_id in bearings:
            bearing_path = data_root / condition / bearing_id
            if bearing_path.exists():
                csv_files = sorted(
                    bearing_path.glob("*.csv"),
                    key=lambda p: int(p.stem),
                )
                for csv_file in csv_files:
                    files.append({
                        "condition": condition,
                        "bearing_id": bearing_id,
                        "filename": csv_file.name,
                        "file_path": str(csv_file),
                    })
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectrograms from XJTU-SY bearing dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npy", "png"],
        default="npy",
        help="Output format: npy for training, png for visualization (default: npy)",
    )
    parser.add_argument(
        "--include-cwt",
        action="store_true",
        help="Also generate CWT scalograms (slower)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
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
        default=100,
        help="Save checkpoint every N files (default: 100)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially instead of parallel (for debugging)",
    )
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    # Paths relative to output directory
    checkpoint_file = output_dir / ".checkpoint.json"
    log_file = output_dir / "generation.log"

    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("XJTU-SY Bearing Spectrogram Generation (FEAT-11)")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output format: {args.format.upper()}")
    logger.info(f"Include CWT: {args.include_cwt}")
    logger.info(f"Workers: {args.workers}")

    # Validate data root
    if not data_root.exists():
        logger.error(f"Dataset not found: {data_root}")
        sys.exit(1)

    # Get all files
    all_files = get_all_files(data_root)
    total_files = len(all_files)
    logger.info(f"Dataset: {total_files} files across 15 bearings")

    # Load checkpoint if resuming
    checkpoint_data = {"stft": set(), "cwt": set()}
    if args.resume:
        checkpoint_data = load_checkpoint(checkpoint_file)
        stft_done = len(checkpoint_data["stft"])
        cwt_done = len(checkpoint_data["cwt"])
        if stft_done or cwt_done:
            logger.info(f"Resuming: {stft_done} STFT, {cwt_done} CWT already processed")
        else:
            logger.info("No checkpoint found, starting fresh")

    # Filter out already processed files
    stft_processed = checkpoint_data["stft"]
    cwt_processed = checkpoint_data["cwt"]

    files_to_process = []
    for f in all_files:
        file_id = f"{f['condition']}/{f['bearing_id']}/{f['filename']}"
        needs_stft = file_id not in stft_processed
        needs_cwt = args.include_cwt and file_id not in cwt_processed
        if needs_stft or needs_cwt:
            files_to_process.append(f)

    if not files_to_process:
        logger.info("All files already processed!")
        return

    logger.info(f"Files to process: {len(files_to_process)}")

    # Statistics
    stats = {
        "total_files": total_files,
        "stft_processed": len(stft_processed),
        "cwt_processed": len(cwt_processed),
        "errors": 0,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    start_time = time.time()
    files_since_checkpoint = 0

    # Prepare arguments for parallel processing
    process_args = [
        (
            f["file_path"],
            f["condition"],
            f["bearing_id"],
            f["filename"],
            output_dir,
            args.format,
            args.include_cwt,
        )
        for f in files_to_process
    ]

    if args.sequential:
        # Sequential processing (for debugging)
        logger.info("Running in sequential mode...")
        for proc_args in tqdm(process_args, desc="Generating spectrograms"):
            result = process_single_file(proc_args)
            file_id = result["file_id"]

            if result["stft_success"]:
                stft_processed.add(file_id)
                stats["stft_processed"] = len(stft_processed)

            if result["cwt_success"]:
                cwt_processed.add(file_id)
                stats["cwt_processed"] = len(cwt_processed)

            if result["error"]:
                stats["errors"] += 1
                logger.error(f"Error processing {file_id}: {result['error']}")

            files_since_checkpoint += 1
            if files_since_checkpoint >= args.checkpoint_interval:
                save_checkpoint(checkpoint_file, stft_processed, cwt_processed, stats)
                files_since_checkpoint = 0

    else:
        # Parallel processing
        logger.info(f"Running with {args.workers} parallel workers...")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_file, proc_args): proc_args
                for proc_args in process_args
            }

            # Process results as they complete
            with tqdm(total=len(futures), desc="Generating spectrograms") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        file_id = result["file_id"]

                        if result["stft_success"]:
                            stft_processed.add(file_id)
                            stats["stft_processed"] = len(stft_processed)

                        if result["cwt_success"]:
                            cwt_processed.add(file_id)
                            stats["cwt_processed"] = len(cwt_processed)

                        if result["error"]:
                            stats["errors"] += 1
                            logger.error(f"Error processing {file_id}: {result['error']}")

                        files_since_checkpoint += 1
                        if files_since_checkpoint >= args.checkpoint_interval:
                            save_checkpoint(
                                checkpoint_file, stft_processed, cwt_processed, stats
                            )
                            files_since_checkpoint = 0

                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Task failed: {e}")

                    pbar.update(1)
                    pbar.set_postfix({
                        "STFT": len(stft_processed),
                        "CWT": len(cwt_processed) if args.include_cwt else "N/A",
                        "Errors": stats["errors"],
                    })

    # Final statistics
    elapsed = time.time() - start_time
    stats["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["elapsed_seconds"] = round(elapsed, 2)
    files_generated = len(stft_processed) - (len(checkpoint_data["stft"]) if args.resume else 0)
    stats["files_per_second"] = round(files_generated / elapsed, 2) if elapsed > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Generation Complete")
    logger.info("=" * 60)
    logger.info(f"STFT spectrograms: {len(stft_processed)}")
    if args.include_cwt:
        logger.info(f"CWT scalograms: {len(cwt_processed)}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Throughput: {stats['files_per_second']:.1f} files/second")

    # Final checkpoint
    stats["completed"] = len(stft_processed) >= total_files
    save_checkpoint(checkpoint_file, stft_processed, cwt_processed, stats)

    # Summary of output structure
    logger.info("")
    logger.info("Output structure:")
    logger.info(f"  {output_dir}/stft/condition={{cond}}/bearing_id={{id}}/*.{args.format}")
    if args.include_cwt:
        logger.info(f"  {output_dir}/cwt/condition={{cond}}/bearing_id={{id}}/*.{args.format}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
