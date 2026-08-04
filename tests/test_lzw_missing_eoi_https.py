"""Real-file coverage for LZW tiles written without an End-Of-Information code.

GLAD/UMD's annual class maps are 256x256-tiled LZW BigTIFFs in which a handful of
tile streams omit the EOI code TIFF 6.0 requires. This is the only test that runs
the fix against real streams: the offline ones are built from 9-bit literal codes,
so nothing there exercises a grown dictionary or TIFF's early-change code widths.
"""

import math
from urllib.parse import urlparse

import numpy as np
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore
from zarr.core.sync import sync

from virtual_tiff import VirtualTIFF
from virtual_tiff.parser import _open_tiff

from .conftest import requires_network

URL = "https://glad.umd.edu/projects/AnnualClassMapsV1/SouthAmerica_Soybean_2020.tif"

# Tiles whose LZW streams carry no EOI code: 322502 used to make the size pre-scan
# over-emit by 4 bytes, 123609 to make it raise IMCD_LZW_CORRUPT.
TILES_MISSING_EOI = (322502, 123609)

# The geometry those tile indices were recorded against, checked against the file.
EXPECTED_TILE = 256
EXPECTED_TILES_ACROSS = 751  # ceil(192004 / 256)


@pytest.fixture(scope="module")
def registry() -> ObjectStoreRegistry:
    parsed = urlparse(URL)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return ObjectStoreRegistry({base: HTTPStore.from_url(base)})


@pytest.fixture(scope="module")
def geometry(registry: ObjectStoreRegistry) -> tuple[int, int]:
    """Tile size and tile-grid width, read from the file rather than assumed.

    A tile index only identifies a window under the geometry it was recorded
    against, so if the source file is ever republished at a different width or
    tile size these tests must fail rather than compare an unaffected window.
    """
    store, path = registry.resolve(URL)
    ifd = sync(_open_tiff(store=store, path=path)).ifds[0]
    assert ifd.tile_width == ifd.tile_height == EXPECTED_TILE
    assert ifd.samples_per_pixel == 1 and tuple(ifd.bits_per_sample) == (8,)
    tiles_across = math.ceil(ifd.image_width / ifd.tile_width)
    assert tiles_across == EXPECTED_TILES_ACROSS
    return ifd.tile_width, tiles_across


@requires_network
@pytest.mark.parametrize("tile_index", TILES_MISSING_EOI)
def test_lzw_tile_without_eoi_matches_gdal(registry, geometry, tile_index):
    """Reading an EOI-less tile through the full parser -> zarr -> xarray stack
    returns the same pixels GDAL reads from the same window."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.windows import Window

    tile, tiles_across = geometry
    ds = xr.open_zarr(
        VirtualTIFF(ifd=0)(URL, registry=registry),
        zarr_format=3,
        consolidated=False,
        chunks=None,
        mask_and_scale=False,
    )

    row, col = divmod(tile_index, tiles_across)  # TIFF mandates row-major tiles
    y0, x0 = row * tile, col * tile
    actual = np.asarray(ds["0"][y0 : y0 + tile, x0 : x0 + tile])

    with rasterio.open(URL) as src:
        expected = src.read(1, window=Window(x0, y0, tile, tile))

    np.testing.assert_array_equal(actual, expected)
