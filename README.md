# Why Virtualize GeoTIFFs or COGs?

This repository is a simple set of demonstrations to prompt discussions over whether and how we should approach Virtualizing GeoTIFFs and COGs.

First, some thoughts on why we should virtualize GeoTIFFs and/or COGS:

1. Provide faster access to non-cloud-optimized GeoTIFFS that contain some form of internal tiling without any data duplication [see notebook #1](01_faster_loading.ipynb).
2. Provide fully async I/O for both GeoTIFFs and COGs using Zarr-Python
3. Allow loading a stack of GeoTIFFS/COGS into a data cube while minimizing the number of GET requests relative to using stackstac/xstac, thereby decreasing cost and increasing performance
4. Provide users access to a lazily loaded DataTree providing both the data and the overviews, allowing scientists to use the overviews not only for tile-based visualization but also quickly iterating on analytics
5. Include etags in the virtualized datasets to support reproducibility
6. A motivation that's less clear to me, but maybe possible, is using the virtualization layer to access COGs with disparate CRSs as a single dataset (https://github.com/zarr-developers/geozarr-spec/issues/53)

## License

`why-virtualize-geotiff` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
