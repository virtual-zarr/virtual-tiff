# Virtual TIFF

## Background

First, some thoughts on why we should virtualize GeoTIFFs and/or COGS:

1. Provide faster access to non-cloud-optimized GeoTIFFS that contain some form of internal tiling without any data duplication [see notebook #1](demos/01_faster_loading_3.0.ipynb).
2. Provide fully async I/O for both GeoTIFFs and COGs using Zarr-Python
3. Allow loading a stack of GeoTIFFS/COGS into a data cube while minimizing the number of GET requests relative to using stackstac/odc-stac, thereby decreasing cost and increasing performance
4. Provide users access to a lazily loaded DataTree providing both the data and the overviews, allowing scientists to use the overviews not only for tile-based visualization but also quickly iterating on analytics
5. Include etags in the virtualized datasets to support reproducibility
6. A motivation that's less clear to me, but maybe possible, is using the virtualization layer to access COGs with disparate CRSs as a single dataset (https://github.com/zarr-developers/geozarr-spec/issues/53)

## Getting started

The library can be installed from PyPI:

```bash
python -m pip install virtual-tiff
```

You can use Virtual TIFF to load data directly:

```python
import obstore
from virtualizarr.registry import ObjectStoreRegistry
from virtual_tiff import VirtualTIFF
import xarray as xr

# Configuration
bucket_url = "s3://e84-earth-search-sentinel-data/"
file_url = f"{bucket_url}sentinel-2-c1-l2a/10/T/FR/2023/12/S2B_T10TFR_20231223T190950_L2A/B04.tif"

# Setup and open dataset
s3_store = obstore.store.from_url(bucket_url, region="us-west-2", skip_signature=True)
registry = ObjectStoreRegistry({bucket_url: s3_store})

parser = VirtualTIFF(ifd=0)
manifest_store = parser(url=file_url, registry=registry)
ds = xr.open_zarr(manifest_store, zarr_format=3, consolidated=False)
ds.load()
```

or create a virtual dataset:

```python
import obstore
from virtualizarr import open_virtual_dataset
from virtualizarr.registry import ObjectStoreRegistry
from virtual_tiff import VirtualTIFF

# Configuration
bucket_url = "s3://e84-earth-search-sentinel-data/"
file_url = f"{bucket_url}sentinel-2-c1-l2a/10/T/FR/2023/12/S2B_T10TFR_20231223T190950_L2A/B04.tif"

# Setup and open dataset
s3_store = obstore.store.from_url(bucket_url, region="us-west-2", skip_signature=True)
registry = ObjectStoreRegistry({bucket_url: s3_store})

ds = open_virtual_dataset(
    url=file_url,
    registry=registry,
    parser=VirtualTIFF(ifd=0)
)
```

## Contributing

1. Clone the repository: `git clone https://github.com/virtual-zarr/virtual-tiff.git`.
2. Pull baseline image data from dvc remote `pixi run -e test download-test-images` WARNING: This will download ~1.4GB of TIFFs for testing to your machine.
3. Run the test suite using `pixi run -e test run-tests` WARNING: Some tests will fail due to incomplete status of the implementation.
4. Start a shell if needed in the development environment using `pixi run -e test zsh`.

## TIFF structure support

| TIFF Structure/Feature         | Supported | Notes |
|-------------------------------|:----------------------------------:|-------|
| Strips | ✅ | Only supported if the image height is evenly divisible by the rows per strip) |
| Tiles                         | ✅                                 | |
| Multiple IFDS                 | ✅                                 | |
| Nested pages/IFDS             | ❌                                 | |
| Uncompressed                  | ✅                                 | |
| PackBits Compression          | ✅                                 | |
| Zlib Compression              | ✅                                 | |
| LZMA Compression              | ✅                                 | |
| Lerc Compression              | ✅                                 | |
| PNG Compression               | ✅                                 | |
| Deflate Compression           | ✅                                 | |
| LZW Compression               | ✅                                 | |
| JPEG Compression              | ✅                                 | |
| JPEGXL Compression            | ✅                                 | |
| JPEG8 Compression             | ✅                                 | |
| JPEG Compression with quantization tables | ❌                     | |
| Webp Compression              | ✅                                 | |
| CMYK Images                   | ✅                                 | |
| YCbCr Images                  | ❌                                 | |
| CIE L*a*b* Images             | ❌                                 | |
| Palette-color Images          | ❌                                 | |
| Grayscale Images              | ✅                                 | |
| RGB Images                    | ✅                                 | |
| PlanarConfiguration (chunky)  | ✅                                 | |
| PlanarConfiguration (planar)  | ✅                                 | |
| Both Byte Orders (II & MM)    | ✅                                 | |
| BigTIFF (64-bit offsets)      | ✅                                 | |

## License

`virtual-tiff` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
