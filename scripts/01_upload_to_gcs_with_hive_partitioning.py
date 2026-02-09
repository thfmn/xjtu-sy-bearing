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

import os
from pathlib import Path

from google.cloud import storage
from dotenv import load_dotenv
import sys

load_dotenv()

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("BUCKET_NAME")
LOCAL_ROOT = str(Path(__file__).resolve().parent.parent / "assets" / "Data" / "XJTU-SY_Bearing_Datasets")
GCS_PREFIX = "xjtu_data"
DRY_RUN = False  # Set to True to simulate upload without actually uploading
# ---------------------

def upload_hive_style():
    print("=== XJTU-SY Bearing Data Upload to GCS ===")
    print(f"Local data directory: {LOCAL_ROOT}")
    print(f"GCS Bucket: {BUCKET_NAME}")
    print(f"GCS Prefix: {GCS_PREFIX}")
    print(f"Dry Run Mode: {DRY_RUN}")
    print()

    # Verify environment
    if not os.path.exists(LOCAL_ROOT):
        print(f"ERROR: Data directory '{LOCAL_ROOT}' does not exist.")
        sys.exit(1)

    if not BUCKET_NAME:
        print("ERROR: BUCKET_NAME environment variable not set.")
        sys.exit(1)

    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        # Test bucket access
        bucket.exists()
        print(f"✓ Connected to GCS bucket: {BUCKET_NAME}")
    except Exception as e:
        print(f"ERROR: Could not connect to GCS bucket '{BUCKET_NAME}': {e}")
        sys.exit(1)

    # Scan for CSV files
    csv_files = []
    for root, dirs, files in os.walk(LOCAL_ROOT):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found in the data directory.")
        sys.exit(0)

    print(f"Found {len(csv_files)} CSV files to process.")
    print()

    # Confirm upload
    if not DRY_RUN:
        response = input(f"This will upload {len(csv_files)} files to GCS. Continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Upload cancelled.")
            sys.exit(0)

    print("Starting upload..." if not DRY_RUN else "Starting dry run...")
    print()

    uploaded = 0
    errors = 0

    for i, local_path in enumerate(csv_files, 1):
        try:
            # Parse path for hive partitioning
            norm_path = os.path.normpath(local_path)
            parts = norm_path.split(os.sep)

            if len(parts) < 3:
                print(f"SKIP: Path too short: {norm_path}")
                continue

            condition = parts[-3]
            bearing_id = parts[-2]
            filename = parts[-1]

            # GCS path
            blob_name = f"{GCS_PREFIX}/condition={condition}/bearing_id={bearing_id}/{filename}"

            print(f"[{i}/{len(csv_files)}] Uploading {filename} -> {blob_name}")

            if not DRY_RUN:
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_path)
                uploaded += 1
            else:
                uploaded += 1  # In dry run, count as uploaded

        except Exception as e:
            print(f"ERROR uploading {local_path}: {e}")
            errors += 1

    print()
    print("=== Upload Complete ===")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Successfully uploaded: {uploaded}")
    if errors > 0:
        print(f"Errors: {errors}")
    print(f"Mode: {'Dry Run' if DRY_RUN else 'Live Upload'}")

if __name__ == "__main__":
    upload_hive_style()