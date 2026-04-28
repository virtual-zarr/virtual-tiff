# Virtual TIFF

**Turn TIFF and COG archives into Zarr stores without copying any data.**

Virtual TIFF allows you to create a Zarr store that you can persist with [Icechunk](https://icechunk.io/).
There are four main benefits to this approach:

- You can curate views to your TIFF collection. Users get a coherent datacube without even knowing that the data is split across hundreds, thousands, or even millions of files. You can also chose which bands, overviews, etc. are exposed in your Zarr store.
- You can use the virtual Zarr store in any language or application that understands Zarr and Icechunk.
- You can provide fast access to non-cloud optimized TIFFs. The process of virtualization adapts regular GeoTIFFs into cloud-optimized Zarrs without data duplication.
- The versioning functionality in Icechunk will warn users if the TIFFs have changed since you produced the virtual Zarr store and run your analysis, improving reproducibility.

## When to use Virtual TIFF

- You're building a **datacube product** over a TIFF/COG archive that should outlive any single Python session.
- You need **non-Python clients** (zarrs, zarrita.js, zarr-layer) to read the archive without knowing it's TIFF underneath.
- You want **Icechunk-versioned** access to the archive: snapshots, transactions, time-travel as new acquisitions land.
- The archive is queried **many times**, and amortizing per-file data discovery across all those queries actually matters.
- You want to expose **overviews** as a native Zarr multiscale group, so downstream tools (visualization, fast analytics) can use them directly.

## When *not* to use Virtual TIFF

If your workflow is "open a STAC search, get an xarray DataArray, do
analysis," you probably don't need a virtual store. Reach for one of:

- [**lazycogs**](https://developmentseed.org/lazycogs) — STAC + async-geotiff with on-the-fly reprojection, for dynamic queries and heterogeneous-CRS data.
- [**stackstac**](https://stackstac.readthedocs.io/) / [**odc-stac**](https://odc-stac.readthedocs.io/) — established STAC-to-DataArray loaders for analyst workflows.
- [**async-tiff**](https://developmentseed.org/async-tiff/) / [**async-geotiff**](https://github.com/developmentseed/async-geotiff)
  directly — when you just want a fast async TIFF reader and don't need a Zarr surface at all.

Virtual TIFF and these tools share the same underlying I/O layer (async-tiff). They differ in what they produce: a runtime DataArray versus a publishable virtual Zarr store. Pick the one that matches your output.

## How it fits

The point of Virtual TIFF is that it's **not in the read path**. It runs once, when the manifest is built. After that, every consumer goes straight from their Zarr client to the manifest to the TIFF byte ranges.

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
