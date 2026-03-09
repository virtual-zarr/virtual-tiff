#!/usr/bin/env python3
"""Upload test TIFF files from tests/data/ to S3 bucket.

Uploads .tif files and a generated README.md (with per-file source URLs
and license information) to s3://us-west-2.opendata.source.coop/pangeo/example-tiff/.
Only new files are uploaded unless --force is given.

Usage:
    # Upload new files and README
    uv run scripts/upload_test_data.py

    # Dry-run: show files that would be uploaded and preview README
    uv run scripts/upload_test_data.py --dry-run

    # Force re-upload all files
    uv run scripts/upload_test_data.py --force

    # Upload without regenerating README
    uv run scripts/upload_test_data.py --skip-readme
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import obstore as obs
from obstore.store import S3Store

BUCKET = "us-west-2.opendata.source.coop"
PREFIX = "pangeo/example-tiff"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
SOURCES_FILE = REPO_ROOT / "scripts" / "github_sources.json"


def get_store():
    return S3Store(
        bucket=BUCKET,
        config={"AWS_REGION": "us-west-2"},
    )


def list_local_files():
    """List all .tif files under the local data directory."""
    files = []
    for tif in DATA_DIR.rglob("*.tif"):
        rel = tif.relative_to(DATA_DIR)
        files.append(str(rel))
    return sorted(files)


def list_remote_files(store):
    """List all .tif files under the prefix in S3."""
    result = obs.list(store, prefix=PREFIX)
    files = set()
    for batch in result:
        for meta in batch:
            path = meta["path"]
            if path.endswith(".tif"):
                rel = path[len(PREFIX) + 1 :]
                if rel:
                    files.add(rel)
    return files


def upload_file(store, rel_path):
    """Upload a single file to S3."""
    local_path = DATA_DIR / rel_path
    remote_key = f"{PREFIX}/{rel_path}"
    data = local_path.read_bytes()
    obs.put(store, remote_key, data)


def build_readme(all_files):
    """Build a README.md with metadata and license information."""
    # Load external source URLs
    external_sources = {}
    if SOURCES_FILE.exists():
        external_sources = json.loads(SOURCES_FILE.read_text())

    # Classify files
    gdal_files = sorted(f for f in all_files if f.startswith("gdal/"))
    github_files = sorted(f for f in all_files if f.startswith("github/"))
    geotiff_test_data_files = sorted(
        f for f in all_files if f.startswith("geotiff-test-data/")
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "# TIFF test data",
        "",
        "A public collection of TIFF files covering a wide range of "
        "compressions, tiling schemes, data types, and internal layouts. "
        "The files are useful for testing any TIFF reading or parsing "
        "library and are maintained as part of the "
        "[virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) project.",
        "",
        f"Last updated: {now}",
        "",
        "## Sources and licenses",
        "",
    ]

    # GDAL section
    if gdal_files:
        lines += [
            "### GDAL autotest (`gdal/`)",
            "",
            f"{len(gdal_files)} files extracted from the GDAL "
            "[autotest](https://github.com/OSGeo/gdal/tree/master/autotest) "
            "directory.",
            "",
            "- **Source:** <https://github.com/OSGeo/gdal>",
            "- **License:** MIT/X — "
            "see [GDAL LICENSE.TXT]"
            "(https://github.com/OSGeo/gdal/blob/master/LICENSE.TXT)",
            "- **Copyright:** Frank Warmerdam, Even Rouault, and contributors",
            "",
        ]

    # External files section
    if github_files:
        lines += [
            "### External files (`github/`)",
            "",
            "Files downloaded from third-party sources for testing a "
            "variety of TIFF encodings.",
            "",
            "| File | Source URL | License |",
            "|------|-----------|---------|",
        ]

        license_map = {
            "test_reference.tif": "Unknown",
            "SIG0_20250404T234439__VH_A091_E048N063T3_SA020M_V1M1R2_S1AIWGRDH_TUWIEN.tif": (
                "CC BY 4.0 (Copernicus Sentinel)"
            ),
            "rema_mosaic_1km_v2.0_filled_cop30_dem.tif": (
                "Public domain (REMA / Byrd Polar)"
            ),
            "TCI.tif": "CC BY-SA 3.0 IGO (Copernicus Sentinel-2)",
            "20250331090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif": (
                "Public domain (NASA JPL)"
            ),
            "IBCSO_v2_ice-surface_cog.tif": "CC BY 4.0 (AWI / IBCSO)",
            "50N_120W.tif": ("CC BY 4.0 (Hansen / UMD / Google / USGS / NASA)"),
            "40613.tif": "Public domain (UCSD / NCMIR)",
            "B04.tif": "CC BY-SA 3.0 IGO (Copernicus Sentinel-2)",
        }

        for f in github_files:
            filename = Path(f).name
            url = external_sources.get(filename, "")
            license_info = license_map.get(filename, "See source")
            lines.append(f"| `{filename}` | {url} | {license_info} |")

        lines.append("")

    # geotiff-test-data section
    if geotiff_test_data_files:
        lines += [
            "### geotiff-test-data (`geotiff-test-data/`)",
            "",
            f"{len(geotiff_test_data_files)} files from the "
            "[geotiff-test-data](https://github.com/developmentseed/geotiff-test-data) "
            "repository.",
            "",
            "- **Source:** <https://github.com/developmentseed/geotiff-test-data>",
            "- **License:** MIT (synthetic fixtures); see individual README files "
            "in real_data/ subdirectories for real-world data licenses",
            "- **Copyright:** Development Seed and contributors",
            "",
        ]

    lines += [
        "## Usage",
        "",
        "These files are publicly hosted and can be used by any project. "
        "To download them with the virtual-tiff helper script:",
        "",
        "```bash",
        "uv run scripts/download_test_data.py",
        "```",
        "",
        "## Disclaimer",
        "",
        "These files are redistributed solely for automated testing. "
        "Refer to each source for authoritative license terms.",
        "",
    ]

    return "\n".join(lines)


def upload_readme(store, all_files, dry_run=False):
    """Generate and upload README.md to the S3 prefix."""
    readme_content = build_readme(all_files)
    remote_key = f"{PREFIX}/README.md"

    if dry_run:
        print(f"\nWould upload README.md to s3://{BUCKET}/{remote_key}")
        print("--- README.md preview ---")
        print(readme_content)
        print("--- end preview ---")
        return

    obs.put(store, remote_key, readme_content.encode("utf-8"))
    print(f"  Uploaded README.md to s3://{BUCKET}/{remote_key}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-upload files that already exist"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip generating and uploading README.md",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Run download_test_data.py first or populate tests/data/ manually.")
        return

    store = get_store()
    local_files = list_local_files()
    print(f"Found {len(local_files)} local .tif files")

    if args.force:
        to_upload = local_files
    else:
        print(f"Listing existing files in s3://{BUCKET}/{PREFIX}/ ...")
        remote_files = list_remote_files(store)
        to_upload = [f for f in local_files if f not in remote_files]
        print(f"  {len(remote_files)} already in S3, {len(to_upload)} new")

    if args.dry_run:
        if to_upload:
            print(f"\nWould upload {len(to_upload)} files:")
            for f in to_upload:
                print(f"  {f}")
        else:
            print("No new .tif files to upload.")
        if not args.skip_readme:
            upload_readme(store, local_files, dry_run=True)
        return

    for i, rel_path in enumerate(to_upload, 1):
        upload_file(store, rel_path)
        print(f"  [{i}/{len(to_upload)}] Uploaded {rel_path}")

    if not args.skip_readme:
        upload_readme(store, local_files)

    print(f"\nDone: {len(to_upload)} .tif files uploaded")


if __name__ == "__main__":
    main()
