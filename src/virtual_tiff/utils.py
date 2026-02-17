from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

from async_tiff.store import AzureStore, GCSStore, HTTPStore, LocalStore, S3Store
from obstore.store import ObjectStore
from zarr.core.chunk_grids import (
    ChunkGrid,
)

if TYPE_CHECKING:
    from async_tiff.store import ObjectStore as AsyncTiffObjectStore
    from obstore.store import (
        ObjectStore,
    )

store_matching = {
    "LocalStore": LocalStore,
    "AzureStore": AzureStore,
    "S3Store": S3Store,
    "GCSStore": GCSStore,
    "HTTPStore": HTTPStore,
}


def gdal_metadata_to_dict(xml_string: str) -> dict[str, str]:
    """
    Convert GDAL metadata XML to a dictionary.
    """
    root = ET.fromstring(xml_string)
    metadata_dict = {}
    for item in root.findall("Item"):
        name = item.get("name")
        if name:
            value = item.text.strip() if item.text else ""
            metadata_dict[name] = value
    return metadata_dict


def convert_obstore_to_async_tiff_store(store: ObjectStore) -> AsyncTiffObjectStore:
    """
    We need to use an async_tiff ObjectStore instance rather than an ObjectStore instance for opening and parsing the TIFF file,
    so that the store isn't passed through Python.
    """

    newargs = store.__getnewargs_ex__()
    name = store.__class__.__name__
    return store_matching[name](*newargs[0], **newargs[1])


def check_no_partial_strips(image_height: int, rows_per_strip: int):
    """Check that there are no partial chunks based on the image height and rows per strip"""
    if image_height % rows_per_strip > 0:
        raise ValueError(
            "Zarr's default chunk grid expects all chunks to be equal size, but this TIFF has an image height of "
            f"{image_height} which isn't evenly divisible by its rows per strip {rows_per_strip}. "
            "See https://github.com/developmentseed/virtual-tiff/issues/24 for more details."
        )


def _is_nested_sequence(chunks: Any) -> bool:
    """
    Check if chunks is a nested sequence (tuple of tuples/lists).

    Returns True for inputs like [[10, 20], [5, 5]] or [(10, 20), (5, 5)].
    Returns False for flat sequences like (10, 10) or [10, 10].

    Vendored from https://github.com/zarr-developers/zarr-python/pull/3534
    """
    # Not a sequence if it's a string, int, tuple of basic types, or ChunkGrid
    if isinstance(chunks, str | int | ChunkGrid):
        return False

    # Check if it's iterable
    if not hasattr(chunks, "__iter__"):
        return False

    # Check if first element is a sequence (but not string/bytes/int)
    try:
        first_elem = next(iter(chunks), None)
        if first_elem is None:
            return False
        return hasattr(first_elem, "__iter__") and not isinstance(
            first_elem, str | bytes | int
        )
    except (TypeError, StopIteration):
        return False
