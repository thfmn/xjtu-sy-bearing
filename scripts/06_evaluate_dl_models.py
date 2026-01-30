#!/usr/bin/env python3
"""Evaluate and aggregate DL model results for XJTU-SY Bearing Dataset.

Reads per-fold DL results from dl_model_results.csv, computes per-model
summary statistics, merges with existing model_comparison.csv, and
generates per-bearing breakdown CSVs.

Usage:
    python scripts/06_evaluate_dl_models.py
    python scripts/06_evaluate_dl_models.py --results-dir outputs/evaluation --output-dir outputs/evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and aggregate DL model training results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory containing dl_model_results.csv and predictions/ (default: outputs/evaluation)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory to write summary and comparison CSVs (default: outputs/evaluation)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for DL model evaluation."""
    args = parse_args()
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir:  {args.output_dir}")


if __name__ == "__main__":
    main()
