from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Literal, Tuple
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

from virtual_tiff.codecs import ChunkyCodec, HorizontalDeltaCodec, TruncateCodec
from virtual_tiff.constants import COMPRESSORS, ENDIAN, GEO_KEYS, SAMPLE_DTYPES
from virtual_tiff.imagecodecs import FloatPredCodec, ZstdCodec
from virtual_tiff.utils import (
    _is_nested_sequence,
    convert_obstore_to_async_tiff_store,
    gdal_metadata_to_dict,
)

if TYPE_CHECKING:
    from async_tiff import TIFF, GeoKeyDirectory, ImageFileDirectory
    from obstore.store import (
        ObjectStore,
    )


GDAL_METADATA_TAG = 42112
GDAL_NODATA_TAG = 42113
ZSTD_LEVEL_TAG = "65564"
DEFAULT_ZSTD_LEVEL = 9


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
        return ZstdCodec(level=ifd.other_tags.get("ZSTD_LEVEL_TAG", DEFAULT_ZSTD_LEVEL))
    else:
        return codec()


def _get_dtype(
    sample_format: tuple[int, ...], bits_per_sample: tuple[int, ...], endian: str
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
        result = np.dtype(dtype)
        if result.itemsize > 1:
            byteorder: Literal[">", "<"] = ">" if endian == "big" else "<"
            result = result.newbyteorder(byteorder)
        return result
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
) -> tuple[tuple[int, ...] | list[list[int]], list[int], list[int]]:
    rows_per_strip: int = ifd.rows_per_strip
    chunks: tuple[int, ...] | list[list[int]]
    if rows_per_strip > image_height:
        chunks = (image_height, ifd.image_width)
    elif (remainder := image_height % rows_per_strip) > 0:
        quotient = image_height // rows_per_strip
        chunks = [[rows_per_strip] * quotient + [remainder], [ifd.image_width]]
    else:
        chunks = (rows_per_strip, ifd.image_width)
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
    if _is_nested_sequence(chunks):
        codecs.append(TruncateCodec())
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
    if gdal_metadata := ifd.gdal_metadata:
        attrs = {**attrs, **gdal_metadata_to_dict(gdal_metadata)}
    if fill_value := ifd.gdal_nodata:
        attrs["gdal_no_data"] = fill_value
    return attrs


def _add_dim_for_samples_per_pixel(
    ifd: ImageFileDirectory,
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | list[list[int]],
) -> tuple[tuple[int, ...], tuple[int, ...] | list[list[int]]]:
    sample_dim_length = int(ifd.samples_per_pixel)
    shape = (sample_dim_length,) + shape
    if ifd.planar_configuration == 2:
        # For PlanarConfiguration = 2, the offsets for each component plane are stored
        # separately. Each plane has its own set of offsets, ordered by component.
        if _is_nested_sequence(chunks):
            assert isinstance(chunks, list)
            chunks = [[1] * sample_dim_length] + chunks
        else:
            assert isinstance(chunks, tuple)
            chunks = (1,) + chunks
    else:
        # Check if rectilinear or regular chunk grid
        if _is_nested_sequence(chunks):
            assert isinstance(chunks, list)
            chunks = [[sample_dim_length]] + chunks
        else:
            assert isinstance(chunks, tuple)
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
    chunk_manifest_shape = tuple(
        math.ceil(a / b) if isinstance(b, int) else len(b)
        for a, b in zip(shape, chunks)
    )
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
        sample_format=ifd.sample_format,
        bits_per_sample=ifd.bits_per_sample,
        endian=endian,
    )
    dimension_names: Tuple[str, ...] = ("y", "x")  # Following rioxarray's behavior
    chunks: tuple[int, ...] | list[list[int]]
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
    if isinstance(chunks[0], int):
        chunk_grid = {
            "name": "regular",
            "configuration": {"chunk_shape": chunks},
        }
    else:
        chunk_grid = {
            "name": "rectilinear",
            "configuration": {"chunk_shapes": chunks, "kind": "inline"},
        }

    metadata = ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid=chunk_grid,
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
    ifd_layout: Literal["flat", "nested"] = "flat",
) -> ManifestGroup:
    """Construct a ManifestGroup from TIFF IFDs.

    Args:
        store: Object store for reading the TIFF
        path: Full URL path to the TIFF file
        endian: Byte order ('little' or 'big')
        ifd: Specific IFD index to process, or None for all IFDs
        ifd_layout: How to organize IFDs - 'flat' for single group, 'nested' for group per IFD

    Returns:
        ManifestGroup containing the processed TIFF data
    """
    # TODO: Make an async approach
    urlpath = urlparse(path).path
    tiff = sync(_open_tiff(store=store, path=urlpath))

    # Build manifest arrays from selected IFDs
    manifest_arrays = _build_manifest_arrays(tiff, path, endian, ifd)

    # Organize into appropriate group structure
    attrs: dict[str, Any] = {}
    if ifd_layout == "flat":
        return _create_flat_group(manifest_arrays, attrs)
    elif ifd_layout == "nested":
        return _create_nested_group(manifest_arrays, attrs)
    else:
        raise ValueError(
            f"Expected 'flat' or 'nested' for ifd_layout; got {ifd_layout}"
        )


def _build_manifest_arrays(
    tiff: TIFF,
    path: str,
    endian: str,
    ifd_index: int | None,
) -> dict[str, ManifestArray]:
    """Build manifest arrays from TIFF IFDs.

    Args:
        tiff: Opened TIFF file
        path: Full URL path to the TIFF file
        endian: Byte order
        ifd_index: Specific IFD to process, or None for all

    Returns:
        Dictionary mapping IFD indices (as strings) to ManifestArrays
    """
    manifest_arrays = {}

    if ifd_index is not None:
        # Process single specified IFD
        manifest_arrays[str(ifd_index)] = _construct_manifest_array(
            ifd=tiff.ifds[ifd_index], path=path, endian=endian
        )
    else:
        # Process all IFDs
        for idx, ifd in enumerate(tiff.ifds):
            manifest_arrays[str(idx)] = _construct_manifest_array(
                ifd=ifd, path=path, endian=endian
            )

    return manifest_arrays


def _create_flat_group(
    manifest_arrays: dict[str, ManifestArray],
    attrs: dict[str, Any],
) -> ManifestGroup:
    """Create a flat group with all arrays at the same level."""
    return ManifestGroup(arrays=manifest_arrays, attributes=attrs)


def _create_nested_group(
    manifest_arrays: dict[str, ManifestArray],
    attrs: dict[str, Any],
) -> ManifestGroup:
    """Create a nested group with each array in its own subgroup."""
    from packaging.version import Version
    from virtualizarr import __version__ as _vz_version

    if Version(_vz_version) < Version("2.2.0"):
        raise ImportError(
            "The 'nested' ifd_layout requires VirtualiZarr >= 2.2.0, "
            f"but you have version {_vz_version}."
        )
    groups = {
        ifd_key: ManifestGroup(
            arrays={ifd_key: array}, attributes=array._metadata.attributes
        )
        for ifd_key, array in manifest_arrays.items()
    }
    return ManifestGroup(groups=groups, attributes=attrs)


class VirtualTIFF:
    def __init__(
        self, ifd: int | None = None, ifd_layout: Literal["flat", "nested"] = "flat"
    ) -> None:
        """Configure VirtualTIFF parser.

        Args:
            ifd : IFD within the TIFF file to virtualize. Defaults to None, meaning that all IFDs will be virtualized as Zarr groups.
            ifd_layout : How to map TIFF IFDs to Zarr groups/arrays. In all cases, an IFD is mapped to a Zarr array. Choose
                "flat" for all arrays to be contained in a single group. Choose "nested" for each array to be contained in a
                different group. "nested" is compatible with Xarray's DataTree model, because
                each node in the DataTree needs to be a Dataset (i.e., group) rather than Dataarray (i.e., array). Default is "flat".
        """
        self._ifd = ifd
        self.ifd_layout = ifd_layout

    def __call__(self, url: str, registry: ObjectStoreRegistry) -> ManifestStore:
        """Produce a ManifestStore from a file path and object store instance.

        Args:
            url : URL to the TIFF.
            registry : ObjectStoreRegistry to use for reading the TIFF.

        Returns:
            ms : ManifestStore containing ChunkManifests and Array metadata for the specified IFDs, along with an ObjectStore instance for loading any data.
        """
        parsed = urlparse(url)
        urlpath = parsed.path
        store, path_in_store = registry.resolve(url)
        endian = ENDIAN[store.get_range(urlpath, start=0, end=2).to_bytes()]
        async_tiff_store = convert_obstore_to_async_tiff_store(store)
        # Create a group containing dataset level metadata and all the manifest arrays
        manifest_group = _construct_manifest_group(
            store=async_tiff_store,
            path=url,
            ifd=self._ifd,
            endian=endian,
            ifd_layout=self.ifd_layout,
        )
        # Convert to a manifest store
        return ManifestStore(registry=registry, group=manifest_group)
