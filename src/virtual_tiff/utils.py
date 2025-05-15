from __future__ import annotations
from async_tiff.store import (
    AzureStore,
    GCSStore,
    LocalStore,
    S3Store,
)
from obstore.store import ObjectStore
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

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
