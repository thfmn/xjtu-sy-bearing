#!/usr/bin/env python3
"""Upload features_v2.csv to GCS for Vertex AI training jobs.

This is a one-time setup script that uploads the local features metadata file
to GCS so that Vertex AI training jobs can access it via the /gcs/ fuse mount.

Usage:
    python scripts/upload_features_to_gcs.py

The script uploads:
    outputs/features/features_v2.csv
    -> gs://xjtu-bearing-failure-dev-data/outputs/features/features_v2.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ID = "xjtu-bearing-failure"
BUCKET_NAME = "xjtu-bearing-failure-dev-data"
LOCAL_FEATURES_CSV = Path("outputs/features/features_v2.csv")
GCS_FEATURES_PATH = "outputs/features/features_v2.csv"


def main() -> None:
    """Upload features_v2.csv to GCS."""
    from google.cloud import storage

    print("=" * 60)
    print("Upload Features CSV to GCS")
    print("=" * 60)

    # Verify local file exists
    if not LOCAL_FEATURES_CSV.exists():
        print(f"ERROR: Local file not found: {LOCAL_FEATURES_CSV}")
        print("\nRun the feature extraction script first:")
        print("  python scripts/03_extract_features.py")
        sys.exit(1)

    # Get file size
    file_size_mb = LOCAL_FEATURES_CSV.stat().st_size / (1024 * 1024)
    print(f"\nLocal file:  {LOCAL_FEATURES_CSV}")
    print(f"File size:   {file_size_mb:.2f} MB")

    # Upload to GCS
    print(f"\nUploading to gs://{BUCKET_NAME}/{GCS_FEATURES_PATH}")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(GCS_FEATURES_PATH)

    # Check if already exists
    if blob.exists():
        print("\nWARNING: File already exists in GCS.")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    blob.upload_from_filename(str(LOCAL_FEATURES_CSV))

    # Verify upload
    blob.reload()
    gcs_size_mb = blob.size / (1024 * 1024)

    print(f"\nUpload complete!")
    print(f"  GCS URI:   gs://{BUCKET_NAME}/{GCS_FEATURES_PATH}")
    print(f"  Size:      {gcs_size_mb:.2f} MB")
    print(f"  Updated:   {blob.updated}")

    print("\n" + "=" * 60)
    print("Done. The features CSV is now available for Vertex AI jobs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
