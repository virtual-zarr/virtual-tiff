from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Tuple
from urllib.parse import urlparse

import numpy as np
from async_tiff import TIFF
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.registry import ObjectStoreRegistry
from zarr.abc.codec import BaseCodec
from zarr.codecs import BytesCodec, TransposeCodec
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.dtype import parse_data_type

from virtual_tiff.codecs import ChunkyCodec, HorizontalDeltaCodec
from virtual_tiff.constants import COMPRESSORS, ENDIAN, GEO_KEYS, SAMPLE_DTYPES
from virtual_tiff.imagecodecs import FloatPredCodec, ZstdCodec
from virtual_tiff.utils import (
    check_no_partial_strips,
    convert_obstore_to_async_tiff_store,
    gdal_metadata_to_dict,
)

if TYPE_CHECKING:
    from async_tiff import TIFF, GeoKeyDirectory, ImageFileDirectory
    from obstore.store import (
        ObjectStore,
    )


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
        dtype = SAMPLE_DTYPES[(int(sample_format[0]), int(bits_per_sample[0]))]
        if dtype in ["q", "Q"]:
            raise NotImplementedError(
                "Requires upstream fix; see https://github.com/virtual-zarr/virtual-tiff/issues/42."
            )
        return np.dtype(dtype)
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


def _get_codecs(
    ifd: ImageFileDirectory,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    endian: str,
) -> list[BaseCodec]:
    codecs = []
    if ifd.predictor == 2:
        codecs.append(HorizontalDeltaCodec())
    elif ifd.predictor == 3:
        codec = FloatPredCodec(dtype=dtype.str, shape=chunks)
        codecs.append(codec)
    compression = ifd.compression
    if ifd.planar_configuration == 1 and ifd.samples_per_pixel > 1:
        codecs.append(TransposeCodec(order=(0, *tuple(range(1, len(shape)))[::-1])))
        codecs.append(ChunkyCodec(endian=str(endian)))
    else:
        codecs.append(BytesCodec(endian=str(endian)))
    if compression > 1:
        codecs.append(_get_compression(ifd, compression))
    return codecs


def _parse_geo_key_directory(geo_key_directory: GeoKeyDirectory) -> dict[str, Any]:
    attrs = {}
    for key in GEO_KEYS:
        if value := getattr(geo_key_directory, key):
            attrs[key] = value
    return attrs


def _get_attributes(ifd: ImageFileDirectory) -> dict[str, Any]:
    attrs = {}
    if ifd.geo_key_directory:
        attrs = _parse_geo_key_directory(ifd.geo_key_directory)
    else:
        attrs = {}
    extra_keys = ["model_pixel_scale", "model_tiepoint", "photometric_interpretation"]
    for key in extra_keys:
        if value := getattr(ifd, key):
            attrs[key] = value
    if ifd.other_tags:
        if gdal_xml := ifd.other_tags.get(42112):
            attrs = {**attrs, **gdal_metadata_to_dict(gdal_xml)}
        if fill_value := ifd.other_tags.get(42113):
            attrs["gdal_no_data"] = fill_value
    return attrs


def _add_dim_for_samples_per_pixel(
    ifd: ImageFileDirectory, shape: tuple[int, ...], chunks: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    sample_dim_length = int(ifd.samples_per_pixel)
    shape = (sample_dim_length,) + shape
    if ifd.photometric_interpretation == 2 and ifd.planar_configuration == 2:
        # For PlanarConfiguration = 2, the StripOffsets for the component planes are stored
        # in the indicated order: first the Red component plane StripOffsets, then the Green plane
        # StripOffsets, then the Blue plane StripOffsets.
        chunks = (1,) + chunks
    else:
        chunks = (sample_dim_length,) + chunks
    return shape, chunks


def _construct_chunk_manifest(
    *,
    path: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    offsets: Iterable[int],
    byte_counts: Iterable[int],
) -> ChunkManifest:
    chunk_manifest_shape = tuple(math.ceil(a / b) for a, b in zip(shape, chunks))
    offsets = np.array(offsets, dtype=np.uint64)
    byte_counts = np.array(byte_counts, dtype=np.uint64)

    if np.all(offsets == 0) or np.all(byte_counts == 0):
        raise NotImplementedError(
            "TIFFs without byte counts and offsets aren't supported"
        )
    offsets = offsets.reshape(chunk_manifest_shape)
    byte_counts = byte_counts.reshape(chunk_manifest_shape)
    paths = np.full_like(offsets, path, dtype=np.dtypes.StringDType)
    return ChunkManifest.from_arrays(
        paths=paths,
        offsets=offsets,
        lengths=byte_counts,
    )


async def _open_tiff(*, path: str, store: ObjectStore) -> TIFF:
    return await TIFF.open(path, store=store)


def _construct_manifest_array(
    *, ifd: ImageFileDirectory, path: str, endian: str
) -> ManifestArray:
    if ifd.other_tags.get(330):
        raise NotImplementedError("TIFFs with Sub-IFDs are not yet supported.")
    shape: Tuple[int, ...] = (ifd.image_height, ifd.image_width)
    dtype = _get_dtype(
        sample_format=ifd.sample_format, bits_per_sample=ifd.bits_per_sample
    )
    dimension_names: Tuple[str, ...] = ("y", "x")  # Following rioxarray's behavior
    attributes = _get_attributes(ifd)
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
        shape, chunks = _add_dim_for_samples_per_pixel(
            ifd=ifd, shape=shape, chunks=chunks
        )
        dimension_names = ("band",) + dimension_names
    chunk_manifest = _construct_chunk_manifest(
        path=path, shape=shape, chunks=chunks, offsets=offsets, byte_counts=byte_counts
    )
    codecs = _get_codecs(ifd, shape=shape, chunks=chunks, dtype=dtype, endian=endian)
    attributes = _get_attributes(ifd)
    if nested := attributes.get("number_of_nested_grids"):
        raise NotImplementedError(
            f"Nested grids are not supported, but file has {nested} nested grid based on GDAL metadata."
        )
    zdtype = parse_data_type(dtype, zarr_format=3)
    metadata = ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": chunks},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=zdtype.default_scalar(),
        codecs=codecs,
        attributes=attributes,
        dimension_names=dimension_names,
        storage_transformers=None,
    )
    return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)


def _construct_manifest_group(
    store: ObjectStore,
    path: str,
    *,
    endian: str,
    ifd: int | None = None,
) -> ManifestGroup:
    # TODO: Make an async approach
    urlpath = urlparse(path).path
    tiff = sync(_open_tiff(store=store, path=urlpath))
    attrs: dict[str, Any] = {}
    manifest_arrays = {}
    if ifd is not None:
        manifest_arrays[str(ifd)] = _construct_manifest_array(
            ifd=tiff.ifds[ifd], path=path, endian=endian
        )
    else:
        for ind, ifd in enumerate(tiff.ifds):
            manifest_arrays[str(ind)] = _construct_manifest_array(
                ifd=ifd, path=path, endian=endian
            )
    return ManifestGroup(arrays=manifest_arrays, attributes=attrs)


class VirtualTIFF:
    _ifd: int | None

    def __init__(
        self,
        ifd: int | None = None,
    ) -> None:
        """Configure VirtualTIFF parser.

        Args:
            ifd (int | None, optional): IFD within the TIFF file to virtualize. Defaults to None, meaning that all IFDs will be virtualized as Zarr groups.
        """
        self._ifd = ifd

    def __call__(self, url: str, registry: ObjectStoreRegistry) -> ManifestStore:
        """Produce a ManifestStore from a file path and object store instance.

        Args:
            url (str): URL to the TIFF.
            registry (ObjectStoreRegistry): ObjectStoreRegistry to use for reading the TIFF.

        Returns:
            ManifestStore: ManifestStore containing ChunkManifests and Array metadata for the specified IFDs, along with an ObjectStore instance for loading any data.
        """
        parsed = urlparse(url)
        urlpath = parsed.path
        store, path_in_store = registry.resolve(url)
        endian = ENDIAN[store.get_range(urlpath, start=0, end=2).to_bytes()]
        async_tiff_store = convert_obstore_to_async_tiff_store(store)
        # Create a group containing dataset level metadata and all the manifest arrays
        manifest_group = _construct_manifest_group(
            store=async_tiff_store, path=url, ifd=self._ifd, endian=endian
        )
        # Convert to a manifest store
        return ManifestStore(registry=registry, group=manifest_group)
