from __future__ import annotations
from async_tiff.store import (
    AzureStore,
    GCSStore,
    HTTPStore,
    LocalStore,
    S3Store,
)
from obstore.store import ObjectStore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from async_tiff.store import ObjectStore as AsyncTiffObjectStore
    from obstore.store import (
        ObjectStore,
    )

store_matching = {
    "LocalStore": LocalStore,
    "HTTPStore": HTTPStore,
    "AzureStore": AzureStore,
    "S3Store": S3Store,
    "GCSStore": GCSStore,
}


def convert_obstore_to_async_tiff_store(store: ObjectStore) -> AsyncTiffObjectStore:
    """
    We need to use an async_tiff ObjectStore instance rather than an ObjectStore instance for opening and parsing the TIFF file,
    so that the store isn't passed through Python.
    """

    newargs = store.__getnewargs_ex__()
    name = store.__class__.__name__
    return store_matching[name](*newargs[0], **newargs[1])
