from __future__ import annotations

from virtual_tiff.constants import SAMPLE_DTYPES
import dataclasses
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
    from obstore.store import AzureStore, GCSStore, HTTPStore, LocalStore, S3Store
    from zarr.core.abc.store import Store


import numpy as np


@dataclasses.dataclass
class ZlibProperties:
    level: int


def _get_codecs(compression):
    if compression == 8:  # Adobe DEFLATE
        zlib_props = ZlibProperties(level=6)  # type: ignore
        conf = dataclasses.asdict(zlib_props)
        conf["id"] = "zlib"
    else:
        raise NotImplementedError
    codec = registry.get_codec(conf)
    return codec


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
    chunk_manifest = _construct_chunk_manifest(
        ifd, path=path, shape=shape, chunks=chunks
    )
    codecs = [_get_codecs(ifd.compression)]
    codec_configs = [
        numcodec_config_to_configurable(codec.get_config()) for codec in codecs
    ]
    dimension_names = ("y", "x")  # Following rioxarray's behavior
    dtype = np.dtype(
        SAMPLE_DTYPES[(int(ifd.sample_format[0]), int(ifd.bits_per_sample[0]))]
    )

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


def create_manifest_store(
    filepath: str,
    group: str,
    file_id: str,
    object_store: AzureStore | GCSStore | HTTPStore | S3Store | LocalStore,
) -> Store:
    # TODO: Make this less sketchy, but it's better to use an AsyncTIFF store rather than an obstore store
    from async_tiff.store import LocalStore as ATStore

    newargs = object_store.__getnewargs_ex__()
    at_store = ATStore(*newargs[0], **newargs[1])

    # Create a group containing dataset level metadata and all the manifest arrays
    manifest_group = _construct_manifest_group(
        store=at_store, path=filepath, group=group
    )
    # Convert to a manifest store
    return ManifestStore(stores={file_id: object_store}, group=manifest_group)
