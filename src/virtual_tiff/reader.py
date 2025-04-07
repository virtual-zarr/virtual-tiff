from __future__ import annotations

from virtual_tiff.constants import SAMPLE_DTYPES
import math
from typing import TYPE_CHECKING, Any, Iterable

from zarr.core.sync import sync

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata

if TYPE_CHECKING:
    from async_tiff import TIFF, ImageFileDirectory
    from async_tiff.store import ObjectStore as AsyncTiffObjectStore
    from obstore.store import AzureStore, GCSStore, HTTPStore, LocalStore, S3Store
    from zarr.core.abc.store import Store

import numpy as np


def _get_compression(ifd, compression):
    if compression in (2, 3, 4):
        raise NotImplementedError("CCITT compression is not yet supported")
    elif compression == 5:
        return dict(name="imagecodecs_lzw")
    elif compression in (6, 7):  # 6 is old style, 7 in new style
        raise NotImplementedError("JPEG compression is not yet supported")
    elif compression == 8:  # Deflate (zlib), Adobe variant
        return dict(name="imagecodecs_deflate")
    elif compression == 32773:
        return NotImplementedError("Packbits compression is not yet supported")
    elif compression == 50000:
        # Based on https://github.com/OSGeo/gdal/blob/ecd914511ba70b4278cc233b97caca1afc9a6e05/frmts/gtiff/gtiff.h#L106-L112
        level = ifd.other_tags.get("65564", 9)
        return dict(name="imagecodecs_zstd", level=level)
    else:
        raise ValueError(f"Compression {compression} not recognized")


def _construct_chunk_manifest(
    *,
    path: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    offsets: Iterable[int],
    byte_counts: Iterable[int],
) -> ChunkManifest:
    chunk_manifest_shape = tuple(math.ceil(a / b) for a, b in zip(shape, chunks))
    offsets = np.array(offsets, dtype=np.uint64).reshape(chunk_manifest_shape)
    byte_counts = np.array(byte_counts, dtype=np.uint64).reshape(chunk_manifest_shape)
    # See https://web.archive.org/web/20240329145228/https://www.awaresystems.be/imaging/tiff/tifftags/tileoffsets.html for ordering of offsets.
    paths = np.full_like(offsets, path, dtype=np.dtypes.StringDType)
    if np.all(offsets == 0) or np.all(byte_counts == 0):
        raise NotImplementedError(
            "TIFFs without byte counts and offsets aren't supported"
        )
    return ChunkManifest.from_arrays(
        paths=paths,
        offsets=offsets,
        lengths=byte_counts,
    )


async def _open_tiff(
    *, path: str, store: AzureStore | GCSStore | HTTPStore | S3Store | LocalStore
) -> TIFF:
    from async_tiff import TIFF

    return await TIFF.open(path, store=store)


def _construct_manifest_array(*, ifd: ImageFileDirectory, path: str) -> ManifestArray:
    shape = (ifd.image_height, ifd.image_width)

    try:
        dtype = np.dtype(
            SAMPLE_DTYPES[(int(ifd.sample_format[0]), int(ifd.bits_per_sample[0]))]
        )
    except KeyError as e:
        raise ValueError(
            f"Unrecognized datatype, got sample_format = {ifd.sample_format[0]} and bits_per_sample = {ifd.bits_per_sample[0]}"
        ) from e
    if ifd.samples_per_pixel > 1:
        raise NotImplementedError(
            f"Only one sample per pixel is currently supported, got {ifd.samples_per_pixel}"
        )
    if ifd.tile_height and ifd.tile_width:
        chunks = (ifd.tile_height, ifd.tile_width)
        offsets = ifd.tile_offsets
        byte_counts = ifd.tile_byte_counts
    elif ifd.rows_per_strip:
        chunks = (ifd.rows_per_strip, ifd.image_width)
        offsets = ifd.strip_offsets
        byte_counts = ifd.strip_byte_counts
    chunk_manifest = _construct_chunk_manifest(
        path=path, shape=shape, chunks=chunks, offsets=offsets, byte_counts=byte_counts
    )
    codecs = []
    if ifd.predictor == 2:
        codec = dict(name="imagecodecs_delta", dtype=dtype.str)
        codecs.append(codec)
    elif ifd.predictor == 3:
        codec = dict(name="imagecodecs_floatpred", dtype=dtype.str, shape=chunks)
        codecs.append(codec)
    compression = ifd.compression
    if compression > 1:
        codecs.append(_get_compression(ifd, compression))
    # # Use CF style fill value for GDAL fill value
    # gdal_fill_value = ifd.other_tags.get(42113, None)
    # if gdal_fill_value:
    #     attributes["_FillValue"] = FillValueCoder.encode(gdal_fill_value, dtype)
    dimension_names = ("y", "x")  # Following rioxarray's behavior

    metadata = create_v3_array_metadata(
        shape=shape,
        data_type=dtype,
        chunk_shape=chunks,
        fill_value=None,
        codecs=codecs,
        dimension_names=dimension_names,
        attributes=None,
    )
    return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)


def _construct_manifest_group(
    store: AzureStore | GCSStore | HTTPStore | S3Store | LocalStore,
    path: str,
    *,
    group: str | None = None,
) -> ManifestGroup:
    """
    Construct a virtual Group from a tiff file.
    """
    # TODO: Make an async approach
    tiff = sync(_open_tiff(store=store, path=path))
    attrs: dict[str, Any] = {}
    manifest_arrays = {}
    if group:
        manifest_arrays[group] = _construct_manifest_array(
            ifd=tiff.ifds[int(group)], path=path
        )
    else:
        for ind, ifd in enumerate(tiff.ifds):
            manifest_arrays[str(ind)] = _construct_manifest_array(ifd=ifd, path=path)
    return ManifestGroup(arrays=manifest_arrays, attributes=attrs)


def _convert_obstore_to_async_tiff_store(store: LocalStore) -> AsyncTiffObjectStore:
    """
    We need to use an async_tiff ObjectStore instance rather than an ObjectStore instance for opening and parsing the TIFF file,
    so that the store isn't passed through Python.
    """
    # TODO: Support all ObjectStore instance types
    from async_tiff.store import LocalStore as AsyncTiffLocalStore

    newargs = store.__getnewargs_ex__()
    return AsyncTiffLocalStore(*newargs[0], **newargs[1])


def create_manifest_store(
    filepath: str,
    group: str,
    file_id: str,
    store: LocalStore,
) -> Store:
    async_tiff_store = _convert_obstore_to_async_tiff_store(store)
    # Create a group containing dataset level metadata and all the manifest arrays
    manifest_group = _construct_manifest_group(
        store=async_tiff_store, path=filepath, group=group
    )
    # Convert to a manifest store
    return ManifestStore(stores={file_id: store}, group=manifest_group)
