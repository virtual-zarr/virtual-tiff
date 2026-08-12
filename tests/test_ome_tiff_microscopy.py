"""OME-TIFFs as microscopy platforms write them.

Three published files this parser could not read:

* a BigTIFF whose tags use the 64-bit IFD8 value type, which `async_tiff` refused;
* a pyramid written into SubIFDs rather than the top-level IFD chain;
* a 17.7 GB image whose first IFD sits in its last kilobytes, which a metadata cache filling
  from offset 0 turned into a whole-object request.

Nothing here downloads the images (1.2 to 17.7 GB): a parser reads only the directory.
"""

from __future__ import annotations

import pytest
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore

from virtual_tiff import VirtualTIFF

from .conftest import requires_network

TENX = "https://cf.10xgenomics.com/"
XENIUM_LUNG = TENX + "samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/"
AWS = "https://s3-us-west-2.amazonaws.com/"
ATERA = (AWS + "10x.files/samples/atera/dev/WTA_Preview_FFPE_Cervical_Cancer/"
         "WTA_Preview_FFPE_Cervical_Cancer_")


def parse(host: str, url: str, **kwargs):
    registry = ObjectStoreRegistry({host: HTTPStore(host)})
    return VirtualTIFF(**kwargs)(url, registry)


@requires_network
def test_bigtiff_with_ifd8_tag_values_reads():
    """Xenium morphology: a BigTIFF whose tag 330 is typed IFD8, with JPEG 2000 tiles.

    The eleven top-level IFDs are focal planes; the levels are their SubIFDs.
    """
    store = parse(TENX, XENIUM_LUNG + "Xenium_V1_humanLung_Cancer_FFPE_morphology.ome.tif")
    arrays = store._group.arrays
    assert len(arrays) == 11
    first = arrays["0"]
    assert tuple(first.shape) == (17098, 51187)
    assert "Jpeg2KCodec" in [type(codec).__name__ for codec in first.metadata.codecs]


@requires_network
def test_subifd_levels_are_read_when_asked_for():
    """Xenium H&E: five reduced levels, each half the last, outside the top-level chain."""
    url = XENIUM_LUNG + "Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif"
    arrays = parse(TENX, url, ifd=0, subifds=True)._group.arrays
    assert list(arrays) == ["0", "0.0", "0.1", "0.2", "0.3", "0.4"]
    assert [tuple(a.shape) for a in arrays.values()] == [
        (3, 45087, 11580),
        (3, 22544, 5790),
        (3, 11272, 2895),
        (3, 5636, 1448),
        (3, 2818, 724),
        (3, 1409, 362),
    ]
    assert sum(len(list(a.manifest.values())) for a in arrays.values()) == 731

    with pytest.warns(UserWarning, match="SubIFD"):
        alone = parse(TENX, url, ifd=0)._group.arrays
    assert list(alone) == ["0"]


@requires_network
def test_the_directory_of_a_17gb_object_is_reachable():
    """Atera H&E: uncompressed, tiled, planar, ten levels, first IFD at byte 17,731,963,126."""
    array = parse(AWS, ATERA + "he_image.ome.tif", ifd=0)._group.arrays["0"]
    assert tuple(array.shape) == (3, 47337, 90368)
    assert len(list(array.manifest.values())) == 12_549


@requires_network
def test_the_pixels_are_the_files_own():
    """That it parses is not that it is right: compare a window against tifffile's own read."""
    tifffile = pytest.importorskip("tifffile")
    fsspec = pytest.importorskip("fsspec")
    import numpy as np
    import zarr

    url = XENIUM_LUNG + "Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif"
    array = zarr.open_array(parse(TENX, url, ifd=0, subifds=True), path="0", mode="r")

    rows, cols = slice(20000, 20064), slice(4000, 4064)
    with fsspec.filesystem("http", block_size=1 << 22).open(url, "rb") as handle:
        page = tifffile.TiffFile(handle).series[0].levels[0]
        # tifffile hands back (y, x, samples) for a chunky page; the manifest is (c, y, x).
        expected = np.moveaxis(page.asarray()[rows, cols], -1, 0)
    assert np.array_equal(np.asarray(array[:, rows, cols]), expected)
