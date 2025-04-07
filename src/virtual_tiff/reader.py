from __future__ import annotations

from virtual_tiff.constants import SAMPLE_DTYPES
import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import numcodecs.registry as registry
from zarr.core.sync import sync

from virtualizarr.codecs import numcodec_config_to_configurable
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
        raise NotImplementedError("LZW compression is not yet supported")
    elif compression in (6, 7):  # 6 is old style, 7 in new style
        raise NotImplementedError("JPEG compression is not yet supported")
    elif compression == 8:  # Dellate (zlib), Adobe fariant
        return registry.get_codec(dict(id="zlib", level=6))
    elif compression == 32773:
        return NotImplementedError("Packbits compression is not yet supported")
    elif compression == 50000:
        # Based on https://github.com/OSGeo/gdal/blob/ecd914511ba70b4278cc233b97caca1afc9a6e05/frmts/gtiff/gtiff.h#L106-L112
        level = ifd.other_tags.get("65564", 9)
        return registry.get_codec(dict(id="zstd", level=level))
    else:
        raise ValueError(f"Compression {compression} not recognized")


def _construct_chunk_manifest(
    ifd: ImageFileDirectory,
    *,
    path: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> ChunkManifest:
    tile_shape = tuple(math.ceil(a / b) for a, b in zip(shape, chunks))
    # See https://web.archive.org/web/20240329145228/https://www.awaresystems.be/imaging/tiff/tifftags/tileoffsets.html for ordering of offsets.
    tile_offsets = np.array(ifd.tile_offsets, dtype=np.uint64).reshape(tile_shape)
    tile_counts = np.array(ifd.tile_byte_counts, dtype=np.uint64).reshape(tile_shape)
    paths = np.full_like(tile_offsets, path, dtype=np.dtypes.StringDType)
    return ChunkManifest.from_arrays(
        paths=paths,
        offsets=tile_offsets,
        lengths=tile_counts,
    )


async def _open_tiff(
    *, path: str, store: AzureStore | GCSStore | HTTPStore | S3Store | LocalStore
) -> TIFF:
    from async_tiff import TIFF

    return await TIFF.open(path, store=store)


def _construct_manifest_array(*, ifd: ImageFileDirectory, path: str) -> ManifestArray:
    if not ifd.tile_height or not ifd.tile_width:
        raise NotImplementedError(
            f"TIFF reader currently only supports tiled TIFFs, but {path} has no internal tiling."
        )
    chunks = (ifd.tile_height, ifd.tile_height)
    shape = (ifd.image_height, ifd.image_width)
    dtype = np.dtype(
        SAMPLE_DTYPES[(int(ifd.sample_format[0]), int(ifd.bits_per_sample[0]))]
    )
    chunk_manifest = _construct_chunk_manifest(
        ifd, path=path, shape=shape, chunks=chunks
    )
    codecs = []
    if ifd.predictor == 2:
        codecs.append(registry.get_codec(dict(id="imagecodecs_delta", dtype=dtype.str)))
    compression = ifd.compression
    if compression > 1:
        codecs.append(_get_compression(ifd, compression))

    codec_configs = [
        numcodec_config_to_configurable(codec.get_config()) for codec in codecs
    ]
    dimension_names = ("y", "x")  # Following rioxarray's behavior

    metadata = create_v3_array_metadata(
        shape=shape,
        data_type=dtype,
        chunk_shape=chunks,
        fill_value=None,  # TODO: Fix fill value
        codecs=codec_configs,
        dimension_names=dimension_names,
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
