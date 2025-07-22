# Virtual TIFF

A Parser for creating Virtual Zarr stores from TIFF files using [VirtualiZarr 2.0](https://virtualizarr.readthedocs.io/en/stable/index.html) and [async-tiff](https://developmentseed.org/async-tiff/latest/).

## Background

First, some thoughts on why we should virtualize GeoTIFFs and/or COGS:

1. Provide faster access to non-cloud-optimized GeoTIFFS that contain some form of internal tiling without any data duplication [see notebook #1](demos/01_faster_loading_3.0.ipynb).
2. Provide fully async I/O for both GeoTIFFs and COGs using Zarr-Python
3. Allow loading a stack of GeoTIFFS/COGS into a data cube while minimizing the number of GET requests relative to using stackstac/xstac, thereby decreasing cost and increasing performance
4. Provide users access to a lazily loaded DataTree providing both the data and the overviews, allowing scientists to use the overviews not only for tile-based visualization but also quickly iterating on analytics
5. Include etags in the virtualized datasets to support reproducibility
6. A motivation that's less clear to me, but maybe possible, is using the virtualization layer to access COGs with disparate CRSs as a single dataset (https://github.com/zarr-developers/geozarr-spec/issues/53)

## Getting started

1. Clone the repository: `git clone https://github.com/virtual-zarr/virtual-tiff.git`.
2. Pull baseline image data from dvc remote `pixi run -e test download-test-images` WARNING: This will download ~1.4GB of TIFFs for testing to your machine.
3. Run the test suite using `pixi run -e test run-tests` WARNING: Some tests will fail due to incomplete status of the implementation.
4. Start a shell if needed in the development environment using `pixi run -e test zsh`.

## License

`virtual-tiff` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
