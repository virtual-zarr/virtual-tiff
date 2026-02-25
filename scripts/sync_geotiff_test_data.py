#!/usr/bin/env python3
"""Sync TIFF test files from the geotiff-test-data repository into tests/data/geotiff-test-data/.

Downloads a tarball of the main branch from GitHub and copies .tif files
into tests/data/geotiff-test-data/ preserving the directory structure.
Tracks the last-synced commit SHA in tests/data/geotiff-test-data/.revision
to avoid redundant downloads.

Example:
    rasterio_generated/fixtures/uint8_rgb_deflate_block64_cog.tif
      -> tests/data/geotiff-test-data/rasterio_generated/fixtures/uint8_rgb_deflate_block64_cog.tif

Usage:
    # Sync new files
    uv run scripts/sync_geotiff_test_data.py

    # Dry-run: show new files that would be copied
    uv run scripts/sync_geotiff_test_data.py --dry-run

    # Sync new files and update data-sources.md
    uv run scripts/sync_geotiff_test_data.py --update-sources

    # Force re-download even if already on latest revision
    uv run scripts/sync_geotiff_test_data.py --force
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

GITHUB_API_TARBALL = "https://api.github.com/repos/developmentseed/geotiff-test-data/tarball/max/zstd_level1"
GITHUB_API_COMMITS = "https://api.github.com/repos/developmentseed/geotiff-test-data/commits/max/zstd_level1"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
GEOTIFF_DIR = DATA_DIR / "geotiff-test-data"
REVISION_FILE = GEOTIFF_DIR / ".revision"
SOURCES_FILE = DATA_DIR / "data-sources.md"


def get_latest_revision() -> str:
    """Return the SHA of the latest commit on main."""
    req = Request(GITHUB_API_COMMITS, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["sha"]


def get_local_revision() -> str | None:
    """Return the locally stored revision SHA, or None."""
    if REVISION_FILE.exists():
        return REVISION_FILE.read_text().strip()
    return None


def save_local_revision(sha: str):
    """Store the revision SHA locally."""
    REVISION_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVISION_FILE.write_text(sha + "\n")


def extract_tiffs(tarball_path: Path, dest: Path) -> list[str]:
    """Extract only .tif files from the tarball.

    GitHub tarballs extract into a directory like
    developmentseed-geotiff-test-data-abc1234/ so we detect the prefix
    dynamically.
    """
    tiff_paths = []
    with tarfile.open(tarball_path, "r:gz") as tar:
        # Detect top-level directory name
        top_dirs = {m.name.split("/")[0] for m in tar.getmembers() if "/" in m.name}
        if len(top_dirs) != 1:
            raise RuntimeError(f"Expected 1 top-level dir in tarball, got: {top_dirs}")
        top_dir = top_dirs.pop()
        prefix = f"{top_dir}/"

        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".tif"):
                rel = member.name[len(prefix) :]
                if rel:
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src:
                        target.write_bytes(src.read())
                    tiff_paths.append(rel)
    return sorted(tiff_paths)


def find_local_tiffs() -> set[str]:
    """Find all .tif files already in tests/data/geotiff-test-data/."""
    if not GEOTIFF_DIR.exists():
        return set()
    return {str(tif.relative_to(GEOTIFF_DIR)) for tif in GEOTIFF_DIR.rglob("*.tif")}


def update_sources(new_files: list[str]):
    """Append new files to data-sources.md."""
    if not SOURCES_FILE.exists():
        SOURCES_FILE.write_text(
            "# Data Sources\n\n| File | Source |\n|------|--------|\n"
        )
    with open(SOURCES_FILE, "a") as f:
        for rel_path in new_files:
            f.write(
                f"| geotiff-test-data/{rel_path} "
                f"| https://github.com/developmentseed/geotiff-test-data |\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if on latest revision"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying",
    )
    parser.add_argument(
        "--update-sources",
        action="store_true",
        help="Append new entries to data-sources.md",
    )
    args = parser.parse_args()

    print("Checking latest geotiff-test-data revision...")
    sha = get_latest_revision()
    local_sha = get_local_revision()
    print(f"  Latest revision: {sha[:12]}")
    print(f"  Local revision:  {local_sha[:12] if local_sha else '(none)'}")

    if sha == local_sha and not args.force:
        print(f"\nAlready synced to {sha[:12]}. Use --force to re-download.")
        return

    local_tiffs = find_local_tiffs()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tarball_path = tmpdir / "geotiff-test-data.tar.gz"
        extracted_dir = tmpdir / "extracted_tiffs"

        print(f"\nDownloading geotiff-test-data archive ({sha[:12]})...")
        urlretrieve(GITHUB_API_TARBALL, tarball_path)

        print("Extracting .tif files...")
        repo_tiffs = extract_tiffs(tarball_path, extracted_dir)
        print(f"Found {len(repo_tiffs)} .tif files in geotiff-test-data")

        new_files = [f for f in repo_tiffs if f not in local_tiffs]
        print(f"  {len(local_tiffs)} already in tests/data/geotiff-test-data/")
        print(f"  {len(new_files)} new files")

        if not new_files:
            print("\nNo new files to sync.")
            save_local_revision(sha)
            print(f"Updated revision marker to {sha[:12]}")
            return

        if args.dry_run:
            print(f"\nWould copy {len(new_files)} files:")
            for f in new_files:
                print(f"  {f}")
            return

        print(f"\nCopying {len(new_files)} new files...")
        GEOTIFF_DIR.mkdir(parents=True, exist_ok=True)
        for rel in new_files:
            src = extracted_dir / rel
            dst = GEOTIFF_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        save_local_revision(sha)
        print(f"Done. Revision marker updated to {sha[:12]}")

        if args.update_sources:
            update_sources(new_files)
            print(f"Updated {SOURCES_FILE}")


if __name__ == "__main__":
    main()
