# Virtual TIFF

**Turn TIFF and COG archives into Zarr stores without copying any data.**

Virtual TIFF reads the metadata of TIFF/COG files and emits a [VirtualiZarr](https://virtualizarr.readthedocs.io/)-compatible
manifest that points back at the original byte ranges. The result is a real,
spec-compliant Zarr v3 store you can persist with [Icechunk](https://icechunk.io/),
share with consumers in any language, and read with any Zarr client — while
the data stay where they are.

## What this is for

Virtual TIFF is for **publishing TIFF/COG archives as Zarr datacubes**.

If you have many GeoTIFFs in object storage and you want to expose them as a
single, queryable, versionable, Zarr dataset — not just inside a Python
session, but as a durable artifact other tools can consume — Virtual TIFF is
the parser that builds the manifest.

The TIFFs themselves never move. The output is small (a manifest of byte
ranges and codec metadata) and lives alongside the data in object storage,
typically inside an Icechunk repository.

## When to use Virtual TIFF

- You're building a **datacube product** over a TIFF/COG archive that should
  outlive any single Python session.
- You need **non-Python clients** (zarrs, zarrita.js, zarr-layer) to read
  the archive without knowing it's TIFF underneath.
- You want **Icechunk-versioned** access to the archive: snapshots,
  transactions, time-travel as new acquisitions land.
- The archive is queried **many times**, and amortizing per-file IFD discovery
  across all those queries actually matters.
- You want to expose **overviews** as a native Zarr multiscale group, so
  downstream tools (visualization, fast analytics) can use them directly.

## When *not* to use Virtual TIFF

If your workflow is "open a STAC search, get an xarray DataArray, do
analysis," you probably don't need a virtual store. Reach for one of:

- [**lazycogs**](https://developmentseed.org/lazycogs) — STAC + async-geotiff
  with on-the-fly reprojection, for dynamic queries and heterogeneous-CRS data.
- [**stackstac**](https://stackstac.readthedocs.io/) /
  [**odc-stac**](https://odc-stac.readthedocs.io/) — established
  STAC-to-DataArray loaders for analyst workflows.
- [**async-tiff**](https://developmentseed.org/async-tiff/) /
  [**async-geotiff**](https://github.com/developmentseed/async-geotiff)
  directly — when you just want a fast async TIFF reader and don't need a
  Zarr surface at all.

Virtual TIFF and these tools share the same underlying I/O layer
(async-tiff). They differ in what they produce: a runtime DataArray versus a
publishable manifest. Pick the one that matches your output.

## How it fits

The point of Virtual TIFF is that it's **not in the read path**. It runs once,
when the manifest is built. After that, every consumer goes straight from
their Zarr client to the manifest to the TIFF byte ranges.

**Build-time (once, by the data publisher)**

```
   TIFFs / COGs in S3, GCS, Azure, …
              │
              │  byte-range GETs for IFD metadata
              ▼
   async-tiff + obstore
              │
              ▼
   Virtual TIFF  ── VirtualiZarr parser, run once
              │
              ▼
   manifest committed to an Icechunk repo
```

**Read-time (every time, in any session)**

```
   Zarr v3 client  +  Icechunk store driver
   (e.g. zarr-python + icechunk-python,
         zarrs + icechunk-rs, …)
              │
              │  Zarr reads issued through the Icechunk Store
              ▼
   Icechunk repo  ── snapshot + manifest
              │
              │  Icechunk resolves chunk keys to
              │  (file_url, offset, length) per chunk
              ▼
   TIFFs / COGs in S3, GCS, Azure, …
              │
              │  parallel byte-range GETs
              ▼
   decoded chunks via the Zarr codec pipeline
```

Virtual TIFF and async-tiff don't appear in the read-time path. The two pieces
the consumer always needs are a Zarr v3 reader and an Icechunk store driver
in their language — Icechunk has a Rust core with Python bindings today, and
other-language bindings track the Rust core. Once those are in place, no TIFF
parsing, no Python-specific code, and nothing from this library is on the read
path.

## Quick start

```bash
python -m pip install virtual-tiff
```

### Open a single TIFF as a Zarr-backed xarray dataset

```python
import obstore
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from virtual_tiff import VirtualTIFF

bucket_url = "s3://e84-earth-search-sentinel-data/"
file_url = f"{bucket_url}sentinel-2-c1-l2a/10/T/FR/2023/12/S2B_T10TFR_20231223T190950_L2A/B04.tif"

s3_store = obstore.store.from_url(bucket_url, region="us-west-2", skip_signature=True)
registry = ObjectStoreRegistry({bucket_url: s3_store})

parser = VirtualTIFF(ifd=0)
manifest_store = parser(url=file_url, registry=registry)
ds = xr.open_zarr(manifest_store, zarr_format=3, consolidated=False)
```

Works equally for GCS, Azure, or any obstore-supported backend — swap the
store factory.

### Build a virtual dataset for use with VirtualiZarr

```python
from virtualizarr import open_virtual_dataset
from virtual_tiff import VirtualTIFF

ds = open_virtual_dataset(
    url=file_url,
    registry=registry,
    parser=VirtualTIFF(ifd=0),
)
```

Combine many of these into a datacube and serialize to Icechunk to publish
the result. See the [VirtualiZarr scaling
guide](https://virtualizarr.readthedocs.io/en/stable/scaling/) for the
map-reduce pattern.

## What's supported

| TIFF feature | Supported | Notes |
|---|:---:|---|
| Strips | ✅ | Image height must be evenly divisible by rows-per-strip |
| Tiles | ✅ | |
| Multiple IFDs | ✅ | |
| Nested pages / IFDs | ❌ | |
| Compressions: Uncompressed, PackBits, Zlib, LZMA, Lerc, PNG, Deflate, LZW, JPEG, JPEGXL, JPEG8, WebP | ✅ | |
| JPEG with quantization tables | ❌ | |
| CMYK | ✅ | |
| YCbCr / CIE L\*a\*b\* / Palette-color | ❌ | |
| Grayscale, RGB | ✅ | |
| PlanarConfiguration (chunky and planar) | ✅ | |
| Both byte orders (II & MM) | ✅ | |
| BigTIFF (64-bit offsets) | ✅ | |

## Motivation

Why virtualize at all, given that async-tiff and lazycogs already give you
fast async access to COGs?

1. **The manifest is the product.** A virtual Zarr store is a publishable
   artifact. You hand a colleague a URL; they open it with their preferred
   Zarr reader. They don't need Python, they don't need to know it was TIFF,
   and they don't have to run your code.
2. **Per-file IFD discovery is paid once.** Reading a TIFF requires walking
   the IFD chain — cheap on warm reads, but a meaningful round trip on cold
   ones. Bake those byte ranges into a manifest and every future read in
   every session, in every language, skips that step.
3. **Internally tiled non-COG TIFFs become accessible.** TIFFs that have
   internal tiles but aren't quite COG-compliant (wrong IFD order, missing
   overviews) can still be cracked open and exposed efficiently — without
   touching the source files.
4. **Versioning via Icechunk.** Append new acquisitions, branch the archive,
   read historical snapshots — all transactions on the manifest, none of which
   touch the underlying TIFFs.
5. **Reproducibility.** Manifests can record etags so a reader knows whether
   the underlying file is the one that was virtualized.
6. **Overviews as multiscale.** Expose the TIFF's overview pyramids as a Zarr
   multiscale group so visualization and analytics tools can use them
   natively.

## Contributing

1. `git clone https://github.com/virtual-zarr/virtual-tiff.git`
2. `pixi run -e test download-test-images` (downloads ~1.4 GB of test TIFFs)
3. `pixi run -e test run-tests` — note: some tests are expected to fail while
   the implementation is in progress.
4. `pixi run -e test zsh` for a dev shell.

Test data is populated from three upstream sources via sync scripts:

- `uv run scripts/sync_gdal_tiffs.py` — GDAL autotest TIFFs
- `uv run scripts/sync_external_tiffs.py` — external TIFFs from various URLs
- `uv run scripts/sync_geotiff_test_data.py` — fixtures from
  [geotiff-test-data](https://github.com/developmentseed/geotiff-test-data)

## License

`virtual-tiff` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.
