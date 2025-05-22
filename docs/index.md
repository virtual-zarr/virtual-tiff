# Virtual TIFF

## Status

Experimental, proof-of-concept.

## Background

This repository started as a simple set of demonstrations to prompt discussions over whether and how we should approach Virtualizing GeoTIFFs and COGs.

First, some thoughts on why we should virtualize GeoTIFFs and/or COGS:

1. Provide faster access to non-cloud-optimized GeoTIFFS that contain some form of internal tiling without any data duplication.
2. Provide fully async I/O for both GeoTIFFs and COGs using Zarr-Python
3. Allow loading a stack of GeoTIFFS/COGS into a data cube while minimizing the number of GET requests relative to using stackstac/xstac, thereby decreasing cost and increasing performance
4. Provide users access to a lazily loaded DataTree providing both the data and the overviews, allowing scientists to use the overviews not only for tile-based visualization but also quickly iterating on analytics
5. Include etags in the virtualized datasets to support reproducibility
6. A motivation that's less clear to me, but maybe possible, is using the virtualization layer to access COGs with disparate CRSs as a single dataset (https://github.com/zarr-developers/geozarr-spec/issues/53)

## Getting started

1. Clone the repository: `git clone https://github.com/maxrjones/virtual-tiff.git`.
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
