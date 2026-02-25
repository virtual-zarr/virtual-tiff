#!/usr/bin/env python3
"""Sync TIFF test files from the latest GDAL release into tests/data/gdal/.

Downloads a tarball of the latest GDAL release via the GitHub API and copies
new .tif files from autotest/ into tests/data/gdal/ preserving the directory
structure. Tracks the last-synced release tag in tests/data/gdal/.release to
avoid redundant downloads.

Example:
    autotest/gcore/data/byte.tif  ->  tests/data/gdal/gcore/data/byte.tif

Usage:
    # Sync new files
    uv run scripts/sync_gdal_tiffs.py

    # Dry-run: show new files that would be copied
    uv run scripts/sync_gdal_tiffs.py --dry-run

    # Sync new files and update data-sources.md
    uv run scripts/sync_gdal_tiffs.py --update-sources

    # Force re-download even if already on latest release
    uv run scripts/sync_gdal_tiffs.py --force
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

GITHUB_API_LATEST = "https://api.github.com/repos/OSGeo/gdal/releases/latest"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
GDAL_DIR = DATA_DIR / "gdal"
RELEASE_FILE = GDAL_DIR / ".release"
SOURCES_FILE = DATA_DIR / "data-sources.md"


def get_latest_release() -> tuple[str, str]:
    """Return (tag_name, tarball_url) for the latest GDAL release."""
    req = Request(GITHUB_API_LATEST, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["tag_name"], data["tarball_url"]


def get_local_release() -> str | None:
    """Return the locally stored release tag, or None."""
    if RELEASE_FILE.exists():
        return RELEASE_FILE.read_text().strip()
    return None


def save_local_release(tag: str):
    """Store the release tag locally."""
    RELEASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RELEASE_FILE.write_text(tag + "\n")


def extract_autotest_tiffs(tarball_path: Path, dest: Path) -> list[str]:
    """Extract only .tif files from autotest/ in the tarball.

    GitHub release tarballs extract into a directory like gdal-vX.Y.Z/
    so we detect the prefix dynamically.
    """
    tiff_paths = []
    with tarfile.open(tarball_path, "r:gz") as tar:
        # Detect top-level directory name (e.g., "OSGeo-gdal-abc1234/" or "gdal-vX.Y.Z/")
        top_dirs = {m.name.split("/")[0] for m in tar.getmembers() if "/" in m.name}
        if len(top_dirs) != 1:
            raise RuntimeError(f"Expected 1 top-level dir in tarball, got: {top_dirs}")
        top_dir = top_dirs.pop()
        autotest_prefix = f"{top_dir}/autotest/"

        for member in tar.getmembers():
            if (
                member.isfile()
                and member.name.startswith(autotest_prefix)
                and member.name.endswith(".tif")
            ):
                rel = member.name[len(autotest_prefix) :]
                if rel:
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src:
                        target.write_bytes(src.read())
                    tiff_paths.append(rel)
    return sorted(tiff_paths)


def find_local_tiffs() -> set[str]:
    """Find all .tif files already in tests/data/gdal/."""
    if not GDAL_DIR.exists():
        return set()
    return {str(tif.relative_to(GDAL_DIR)) for tif in GDAL_DIR.rglob("*.tif")}


def update_sources(new_files: list[str]):
    """Append new files to data-sources.md."""
    if not SOURCES_FILE.exists():
        SOURCES_FILE.write_text(
            "# Data Sources\n\n| File | Source |\n|------|--------|\n"
        )
    with open(SOURCES_FILE, "a") as f:
        for rel_path in new_files:
            f.write(
                f"| {rel_path} | https://github.com/OSGeo/gdal/tree/master/autotest |\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if on latest release"
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

    print("Checking latest GDAL release...")
    tag, tarball_url = get_latest_release()
    local_tag = get_local_release()
    print(f"  Latest release: {tag}")
    print(f"  Local release:  {local_tag or '(none)'}")

    if tag == local_tag and not args.force:
        print(f"\nAlready synced to {tag}. Use --force to re-download.")
        return

    local_tiffs = find_local_tiffs()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tarball_path = tmpdir / "gdal.tar.gz"
        extracted_dir = tmpdir / "autotest_tiffs"

        print(f"\nDownloading {tag} tarball (this may take a moment)...")
        urlretrieve(tarball_url, tarball_path)

        print("Extracting .tif files from autotest/...")
        gdal_tiffs = extract_autotest_tiffs(tarball_path, extracted_dir)
        print(f"Found {len(gdal_tiffs)} .tif files in GDAL autotest/")

        new_files = [f for f in gdal_tiffs if f not in local_tiffs]
        print(f"  {len(local_tiffs)} already in tests/data/gdal/")
        print(f"  {len(new_files)} new files")

        if not new_files:
            print("\nNo new files to sync.")
            save_local_release(tag)
            print(f"Updated release marker to {tag}")
            return

        if args.dry_run:
            print(f"\nWould copy {len(new_files)} files:")
            for f in new_files:
                print(f"  {f}")
            return

        print(f"\nCopying {len(new_files)} new files...")
        GDAL_DIR.mkdir(parents=True, exist_ok=True)
        for rel in new_files:
            src = extracted_dir / rel
            dst = GDAL_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        save_local_release(tag)
        print(f"Done. Release marker updated to {tag}")

        if args.update_sources:
            update_sources(new_files)
            print(f"Updated {SOURCES_FILE}")


if __name__ == "__main__":
    main()
