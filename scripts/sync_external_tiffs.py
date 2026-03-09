#!/usr/bin/env python3
"""Sync external test TIFF files into tests/data/github/.

Downloads files listed in scripts/github_sources.json from their original
URLs. This populates the local test data directory so it can then be uploaded
to S3 with upload_test_data.py.

Usage:
    # Download new files
    uv run scripts/sync_external_tiffs.py

    # Dry-run: show files that would be downloaded
    uv run scripts/sync_external_tiffs.py --dry-run

    # Force re-download all files
    uv run scripts/sync_external_tiffs.py --force
"""

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
GITHUB_DIR = DATA_DIR / "github"
SOURCES_FILE = REPO_ROOT / "scripts" / "github_sources.json"


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

    sources = json.loads(SOURCES_FILE.read_text())
    print(f"Found {len(sources)} entries in {SOURCES_FILE.name}")

    new_files = []
    existing = 0
    for filename in sources:
        local_path = GITHUB_DIR / filename
        if local_path.exists() and not args.force:
            existing += 1
        else:
            new_files.append(filename)

    print(f"  {existing} already in tests/data/github/")
    print(f"  {len(new_files)} to download")

    if not new_files:
        print("\nNothing to download.")
        return

    if args.dry_run:
        print(f"\nWould download {len(new_files)} files:")
        for f in new_files:
            print(f"  {f}")
        return

    GITHUB_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0
    for i, filename in enumerate(new_files, 1):
        url = sources[filename]
        local_path = GITHUB_DIR / filename
        try:
            print(f"  [{i}/{len(new_files)}] Downloading {filename}...")
            urlretrieve(url, local_path)
            downloaded += 1
        except Exception as e:
            print(f"    Failed: {e}")
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {failed} failed")


if __name__ == "__main__":
    main()
