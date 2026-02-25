#!/usr/bin/env python3
"""Download test TIFF files from a public S3 bucket to tests/data/.

This is the primary way to get test data for running the test suite.
All test files (GDAL and external) are mirrored in S3.

To populate S3 from original sources, use sync_gdal_tiffs.py and
sync_external_tiffs.py followed by upload_test_data.py.

Usage:
    # Download new files
    uv run scripts/download_test_data.py

    # Dry-run: show files that would be downloaded
    uv run scripts/download_test_data.py --dry-run

    # Force re-download all files
    uv run scripts/download_test_data.py --force
"""

import argparse
import sys
from pathlib import Path

import obstore as obs
from obstore.store import S3Store

BUCKET = "us-west-2.opendata.source.coop"
PREFIX = "pangeo/example-tiff"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"


def get_store():
    return S3Store(
        bucket=BUCKET,
        config={"AWS_SKIP_SIGNATURE": "true", "AWS_REGION": "us-west-2"},
    )


def list_remote_files(store):
    """List all files under the prefix in S3."""
    result = obs.list(store, prefix=PREFIX)
    files = []
    for batch in result:
        for meta in batch:
            path = meta["path"]
            if path.endswith(".tif"):
                # Strip the prefix to get the relative path
                rel = path[len(PREFIX) + 1 :]  # +1 for trailing /
                if rel:
                    files.append(rel)
    return sorted(files)


def download_file(store, rel_path, force=False):
    """Download a single file from S3 to the local data directory."""
    local_path = DATA_DIR / rel_path
    if local_path.exists() and not force:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_key = f"{PREFIX}/{rel_path}"
    data = obs.get(store, remote_key)
    content = b"".join(data)
    local_path.write_bytes(content)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    args = parser.parse_args()

    store = get_store()
    print(f"Listing files in s3://{BUCKET}/{PREFIX}/ ...")
    files = list_remote_files(store)
    if not files:
        print("No .tif files found in S3 bucket. Is the bucket populated?")
        sys.exit(1)

    print(f"Found {len(files)} .tif files")

    if args.force:
        to_download = files
    else:
        to_download = [f for f in files if not (DATA_DIR / f).exists()]

    if not to_download:
        print("Nothing to download.")
        return

    if args.dry_run:
        print(f"\nWould download {len(to_download)} files:")
        for f in to_download:
            print(f"  {f}")
        return

    downloaded = 0
    for i, rel_path in enumerate(to_download, 1):
        download_file(store, rel_path, force=args.force)
        downloaded += 1
        print(f"  [{i}/{len(to_download)}] Downloaded {rel_path}")

    print(f"\nDone: {downloaded} downloaded")


if __name__ == "__main__":
    main()
