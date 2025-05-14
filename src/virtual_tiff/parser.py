from __future__ import annotations

from virtual_tiff.utils import (
    convert_obstore_to_async_tiff_store,
    check_no_partial_strips,
)
from virtual_tiff.constants import SAMPLE_DTYPES, COMPRESSORS
from virtual_tiff.imagecodecs import ZstdCodec
import math
from typing import TYPE_CHECKING, Any, Iterable, Tuple
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.abc.codec import BaseCodec
import numpy as np
from urllib.parse import urlparse

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.store import ObjectStoreRegistry, default_object_store
from zarr.codecs import BytesCodec

if TYPE_CHECKING:
    from async_tiff import TIFF, ImageFileDirectory
    from obstore.store import (
        AzureStore,
        GCSStore,
        HTTPStore,
        LocalStore,
        S3Store,
        ObjectStore,
    )
    from zarr.core.abc.store import Store


def _get_compression(ifd: ImageFileDirectory, compression: int):
    codec = COMPRESSORS.get(compression)
    if not codec:
        raise ValueError(
            f"TIFF has compressor tag {compression}, which is not recognized. Please raise an issue for support."
        )
    if hasattr(ifd, "jpeg_tables") and ifd.jpeg_tables:
        raise NotImplementedError(
            "JPEG compression with quantization tables is not yet supported."
        )
    if codec.codec_name == "imagecodecs_zstd":
        # Based on https://github.com/OSGeo/gdal/blob/ecd914511ba70b4278cc233b97caca1afc9a6e05/frmts/gtiff/gtiff.h#L106-L112
        return ZstdCodec(level=ifd.other_tags.get("65564", 9))
    else:
        return codec()


def _get_dtype(
    sample_format: tuple[int, ...], bits_per_sample: tuple[int, ...]
) -> np.dtype:
    if not all(x == sample_format[0] for x in sample_format):
        raise ValueError(
            f"The Zarr specification does not allow multiple data types in a single array, but the TIFF had multiple sample formats in a single IFD: {sample_format}"
        )
    if not all(x == bits_per_sample[0] for x in bits_per_sample):
        raise ValueError(
            f"The Zarr specification does not allow multiple data types in a single array, but the TIFF had multiple bits per sample in a single IFD: {bits_per_sample}"
        )
    try:
        return np.dtype(SAMPLE_DTYPES[(int(sample_format[0]), int(bits_per_sample[0]))])
    except KeyError as e:
        raise ValueError(
            f"Unrecognized datatype, got sample_format = {sample_format} and bits_per_sample = {bits_per_sample}"
        ) from e


def _get_chunks_from_tiles(
    ifd: ImageFileDirectory,
) -> tuple[tuple[int, ...], list[int], list[int]]:
    chunks = (ifd.tile_height, ifd.tile_width)
    offsets = ifd.tile_offsets
    byte_counts = ifd.tile_byte_counts
    return chunks, offsets, byte_counts


def _get_chunks_from_strips(
    ifd: ImageFileDirectory, image_height: int
) -> tuple[tuple[int, ...], list[int], list[int]]:
    rows_per_strip = ifd.rows_per_strip
    if rows_per_strip > image_height:
        chunks = (image_height, ifd.image_width)
    else:
        chunks = (rows_per_strip, ifd.image_width)
    check_no_partial_strips(image_height=image_height, rows_per_strip=chunks[0])
    offsets = ifd.strip_offsets
    byte_counts = ifd.strip_byte_counts
    return chunks, offsets, byte_counts


def _get_codecs(ifd: ImageFileDirectory, *, shape, chunks, dtype) -> list[BaseCodec]:
    codecs = []
    if ifd.predictor == 2:
        from virtual_tiff.codecs import DeltaCodec

        codecs.append(DeltaCodec())
    elif ifd.predictor == 3:
        from virtual_tiff.imagecodecs import FloatPredCodec

        codec = FloatPredCodec(dtype=dtype.str, shape=chunks)
        codecs.append(codec)
    compression = ifd.compression
    if ifd.planar_configuration == 1 and ifd.samples_per_pixel > 1:
        from zarr.codecs import TransposeCodec
        from virtual_tiff.codecs import ChunkyCodec

        codecs.append(TransposeCodec(order=(0, *tuple(range(1, len(shape)))[::-1])))
        codecs.append(ChunkyCodec())
    else:
        codecs.append(BytesCodec())
    if compression > 1:
        codecs.append(_get_compression(ifd, compression))
    return codecs


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
    if ifd.other_tags.get(330):
        raise NotImplementedError("TIFFs with Sub-IFDs are not yet supported.")
    shape: Tuple[int, ...] = (ifd.image_height, ifd.image_width)
    dtype = _get_dtype(
        sample_format=ifd.sample_format, bits_per_sample=ifd.bits_per_sample
    )
    dimension_names: Tuple[str, ...] = ("y", "x")  # Following rioxarray's behavior
    if ifd.tile_height:
        chunks, offsets, byte_counts = _get_chunks_from_tiles(ifd)
    elif ifd.rows_per_strip:
        chunks, offsets, byte_counts = _get_chunks_from_strips(
            ifd, image_height=shape[0]
        )
    else:
        raise NotImplementedError(
            "TIFFs without byte counts and offsets aren't supported"
        )
    if ifd.photometric_interpretation in [6, 8]:
        raise NotImplementedError(
            f"{ifd.photometric_interpretation._name_} PhotometricInterpretation is not yet supported."
        )
    if ifd.samples_per_pixel > 1:
        shape = (int(ifd.samples_per_pixel),) + shape
        if ifd.photometric_interpretation == 2 and ifd.planar_configuration == 2:
            # For PlanarConfiguration = 2, the StripOffsets for the component planes are stored
            # in the indicated order: first the Red component plane StripOffsets, then the Green plane
            # StripOffsets, then the Blue plane StripOffsets.
            chunks = (1,) + chunks
        else:
            chunks = (int(ifd.samples_per_pixel),) + chunks
        dimension_names = ("band",) + dimension_names
    chunk_manifest = _construct_chunk_manifest(
        path=path, shape=shape, chunks=chunks, offsets=offsets, byte_counts=byte_counts
    )
    codecs = _get_codecs(ifd, shape=shape, chunks=chunks, dtype=dtype)
    # # Use CF style fill value for GDAL fill value
    # gdal_fill_value = ifd.other_tags.get(42113, None)
    # if gdal_fill_value:
    #     attributes["_FillValue"] = FillValueCoder.encode(gdal_fill_value, dtype)

    metadata = ArrayV3Metadata(
        shape=shape,
        data_type=dtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": chunks},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=None,
        codecs=codecs,
        attributes=None,
        dimension_names=dimension_names,
        storage_transformers=None,
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
    urlpath = urlparse(path).path
    tiff = sync(_open_tiff(store=store, path=urlpath))
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
    store: ObjectStore | None = None,
) -> Store:
    if not store:
        store = default_object_store(filepath)
    urlpath = urlparse(filepath).path
    endianness = store.get_range(urlpath, start=0, end=2)
    if endianness == b"MM":
        raise NotImplementedError("Big endian TIFFs are not yet supported.")
    async_tiff_store = convert_obstore_to_async_tiff_store(store)
    # Create a group containing dataset level metadata and all the manifest arrays
    manifest_group = _construct_manifest_group(
        store=async_tiff_store, path=filepath, group=group
    )
    registry = ObjectStoreRegistry({filepath: store})
    # Convert to a manifest store
    return ManifestStore(store_registry=registry, group=manifest_group)
