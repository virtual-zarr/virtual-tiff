from __future__ import annotations

import functools
import importlib.util
import math
import warnings
from typing import TYPE_CHECKING, Any, Iterable, Literal, Tuple

import numpy as np
from async_tiff import TIFF
from async_tiff.enums import Endianness
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from zarr.abc.codec import BaseCodec
from zarr.codecs import BytesCodec, TransposeCodec
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.dtype import parse_data_type

from virtual_tiff.codecs import ChunkyCodec, HorizontalDeltaCodec
from virtual_tiff.constants import COMPRESSORS, GEO_KEYS, SAMPLE_DTYPES
from virtual_tiff.imagecodecs import FloatPredCodec, ZstdCodec
from virtual_tiff.utils import (
    check_no_partial_strips,
    gdal_metadata_to_dict,
)
from virtual_tiff.vendor.xarray.zarr import FillValueCoder
from virtual_tiff.vendor.zarr.chunk_grids import _is_rectilinear_chunks


@functools.cache
def _has_unified_chunk_grid() -> bool:
    """Check if zarr has the unified ChunkGrid with is_regular support.

    Defers the actual import so zarr stays lazy at module load time.
    """
    if importlib.util.find_spec("zarr.core.chunk_grids") is None:
        return False
    from zarr.core.chunk_grids import ChunkGrid

    return hasattr(ChunkGrid, "is_regular")


if TYPE_CHECKING:
    from async_tiff import TIFF, GeoKeyDirectory, ImageFileDirectory
    from obspec import GetRange
    from obstore.store import (
        ObjectStore,
    )


GDAL_METADATA_TAG = 42112
GDAL_NODATA_TAG = 42113
ZSTD_LEVEL_TAG = "65564"
DEFAULT_ZSTD_LEVEL = 9
_ENDIANNESS_TO_STR = {
    Endianness.LittleEndian: "little",
    Endianness.BigEndian: "big",
}


_MSVC_INF_MAP = {
    "1.#INF": "inf",
    "-1.#INF": "-inf",
    "1.#QNAN": "nan",
    "-1.#QNAN": "nan",
    "1.#IND": "nan",
    "-1.#IND": "nan",
}


def _parse_fill_value(value: str, dtype: np.dtype) -> Any:
    """Parse a string fill value into the correct numpy scalar for the given dtype.

    Parameters
    ----------
    value
        String representation of the fill value (e.g. "-9999" or "nan").
    dtype
        Target numpy dtype.

    Returns
    -------
    Numpy scalar matching dtype.

    Raises
    ------
    ValueError
        If the string cannot be parsed as the target dtype.
    """
    # Normalize MSVC-style infinity/NaN representations
    normalized = _MSVC_INF_MAP.get(value, value)
    try:
        return np.dtype(dtype).type(normalized)
    except (ValueError, OverflowError) as e:
        raise ValueError(f"Cannot parse fill value {value!r} as {dtype}: {e}") from e


def _consolidate_fill_value(
    attrs: dict[str, Any], dtype: np.dtype
) -> tuple[dict[str, Any], Any]:
    """Extract, validate, and consolidate fill values from TIFF attributes.

    TIFF files (especially those converted from NetCDF) may carry fill value
    information in multiple places, all as strings:

    - ``gdal_no_data``: from the GDAL_NODATA tag (tag 42113)
    - ``_FillValue``: from GDAL metadata XML (CF convention)
    - ``missing_value``: from GDAL metadata XML (older CF convention)

    This function:

    1. Extracts fill values from all three sources
    2. Parses them from strings into the array's dtype
    3. Validates that all present values are consistent
    4. Returns a properly typed Zarr fill_value and cleaned-up attributes
       with ``_FillValue`` encoded for xarray's ``FillValueCoder``
    5. Removes per-variable prefixed duplicates (e.g. ``swe#_FillValue``)
       when they match the top-level value

    Parameters
    ----------
    attrs
        Mutable dictionary of array attributes. Will be modified in place
        to remove raw string fill values and add encoded versions.
    dtype
        The numpy dtype of the array.

    Returns
    -------
    attrs
        The cleaned-up attributes dictionary with properly encoded
        ``_FillValue`` (if applicable).
    fill_value
        A numpy scalar to use as the Zarr ``fill_value``, or None if no
        fill value was found (caller should fall back to dtype default).
    """
    # Collect raw string values from all sources
    _FILL_VALUE_KEYS = ("_FillValue", "missing_value", "gdal_no_data")
    raw_values: dict[str, str] = {}
    for key in _FILL_VALUE_KEYS:
        if key in attrs and isinstance(attrs[key], str):
            raw_values[key] = attrs[key]

    if not raw_values:
        return attrs, None

    # Parse all values into the target dtype
    parsed: dict[str, Any] = {}
    for key, raw in raw_values.items():
        parsed[key] = _parse_fill_value(raw, dtype)

    # Validate consistency — all parsed values must be equal
    unique_values = set()
    for v in parsed.values():
        if dtype.kind == "f" and np.isnan(v):
            unique_values.add("nan")
        else:
            unique_values.add(v)

    if len(unique_values) > 1:
        detail = ", ".join(f"{k}={v!r}" for k, v in parsed.items())
        warnings.warn(
            f"Conflicting fill values found: {detail}. "
            f"Using gdal_no_data as the authoritative source.",
            stacklevel=3,
        )

    # Determine the authoritative fill value (gdal_no_data takes priority as
    # it is the TIFF-native source; _FillValue and missing_value are
    # reconstructed from GDAL metadata XML and may have lost precision)
    if "gdal_no_data" in parsed:
        fill_value = parsed["gdal_no_data"]
    elif "_FillValue" in parsed:
        fill_value = parsed["_FillValue"]
    else:
        fill_value = parsed["missing_value"]

    # Encode _FillValue for xarray's FillValueCoder.
    # Only _FillValue gets this encoding — the Zarr backend decodes it via
    # FillValueCoder but passes missing_value through as-is, so
    # missing_value must be a plain numeric value.
    encoded = FillValueCoder.encode(fill_value, dtype)
    attrs["_FillValue"] = encoded

    if "missing_value" in parsed:
        attrs["missing_value"] = (
            fill_value.item() if hasattr(fill_value, "item") else fill_value
        )

    # Remove per-variable prefixed duplicates (e.g. swe#_FillValue, swe#missing_value)
    # when they carry the same raw string as the top-level value
    prefixed_to_remove = []
    for attr_key in list(attrs.keys()):
        if "#" not in attr_key:
            continue
        suffix = attr_key.split("#", 1)[1]
        if suffix in _FILL_VALUE_KEYS and isinstance(attrs[attr_key], str):
            prefixed_parsed = _parse_fill_value(attrs[attr_key], dtype)
            if prefixed_parsed == fill_value or (
                dtype.kind == "f" and np.isnan(prefixed_parsed) and np.isnan(fill_value)
            ):
                prefixed_to_remove.append(attr_key)
    for key in prefixed_to_remove:
        del attrs[key]

    return attrs, fill_value


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
        return ZstdCodec(level=ifd.other_tags.get(ZSTD_LEVEL_TAG, DEFAULT_ZSTD_LEVEL))
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


def _decompress_strip(compression: int, data: bytes) -> int:
    """Decompress raw TIFF strip data and return the decompressed byte count."""
    import lzma as _lzma
    import zlib as _zlib

    if compression in (8, 32946):  # Zlib/Deflate
        return len(_zlib.decompress(data))
    elif compression == 34925:  # LZMA
        return len(_lzma.decompress(data))

    codec_cls = COMPRESSORS.get(compression)
    if codec_cls is None:
        raise ValueError(f"Unknown compression tag: {compression}")

    from zarr.registry import get_numcodec

    numcodec = get_numcodec({"id": codec_cls.codec_name})
    result = numcodec.decode(data)
    if isinstance(result, np.ndarray):
        return result.nbytes
    return len(result)


def _fetch_last_strip(
    ifd: ImageFileDirectory,
    object_store: GetRange,
    store_path: str,
) -> bytes | None:
    """Fetch the last strip's compressed bytes if needed for padding detection.

    Returns None when padding detection doesn't require a fetch: tiles,
    single-strip images, evenly divisible strips, or uncompressed data.
    """
    if ifd.tile_height or not ifd.rows_per_strip:
        return None
    if ifd.rows_per_strip >= ifd.image_height:
        return None
    if ifd.image_height % ifd.rows_per_strip == 0:
        return None
    if ifd.compression <= 1:
        return None

    last_offset = ifd.strip_offsets[-1]
    last_byte_count = ifd.strip_byte_counts[-1]
    return bytes(
        object_store.get_range(
            store_path, start=last_offset, end=last_offset + last_byte_count
        )
    )


def _is_last_strip_padded(
    ifd: ImageFileDirectory,
    rows_per_strip: int,
    last_strip_data: bytes | None,
) -> bool:
    """Detect whether the last strip is padded to full rows_per_strip size.

    For uncompressed TIFFs, compares the last strip's byte count against the
    expected size for a full strip. For compressed TIFFs, decompresses the
    provided last strip data to determine its actual size.

    Returns True if the last strip is padded (use regular grid),
    False if unpadded (use rectilinear grid).
    """
    byte_counts = ifd.strip_byte_counts
    if not byte_counts or len(byte_counts) < 2:
        return False

    bytes_per_row = ifd.image_width * sum(ifd.bits_per_sample) // 8
    full_strip_bytes = rows_per_strip * bytes_per_row

    if ifd.compression <= 1:
        # Uncompressed: compare byte counts directly
        return byte_counts[-1] >= full_strip_bytes

    # Compressed: decompress the last strip to check its actual size.
    # Some codecs (e.g. JPEG with external quantization tables) cannot
    # decompress a strip independently.
    if last_strip_data is not None:
        try:
            decompressed_size = _decompress_strip(ifd.compression, last_strip_data)
        except Exception as e:
            raise NotImplementedError(
                f"Cannot determine last strip padding for compression tag "
                f"{ifd.compression}: {e}"
            ) from e
        return decompressed_size >= full_strip_bytes

    return False


def _get_chunks_from_strips(
    ifd: ImageFileDirectory,
    image_height: int,
    last_strip_data: bytes | None,
) -> tuple[tuple[int, ...] | list[list[int]], list[int], list[int]]:
    rows_per_strip: int = ifd.rows_per_strip
    chunks: tuple[int, ...] | list[list[int]]
    if rows_per_strip > image_height:
        chunks = (image_height, ifd.image_width)
    elif (remainder := image_height % rows_per_strip) > 0:
        if _is_last_strip_padded(ifd, rows_per_strip, last_strip_data):
            # Last strip is padded — use regular grid. The codec will receive
            # full-strip-sized buffers and zarr's boundary clipping handles
            # the data trimming.
            chunks = (rows_per_strip, ifd.image_width)
        elif _has_unified_chunk_grid():
            # Last strip is unpadded — use rectilinear grid with actual sizes
            quotient = image_height // rows_per_strip
            chunks = [[rows_per_strip] * quotient + [remainder], [ifd.image_width]]
        else:
            # No rectilinear support and unpadded partial strip
            check_no_partial_strips(
                image_height=image_height, rows_per_strip=rows_per_strip
            )
    else:
        chunks = (rows_per_strip, ifd.image_width)
    offsets = ifd.strip_offsets
    byte_counts = ifd.strip_byte_counts
    return chunks, offsets, byte_counts


def _get_codecs(
    ifd: ImageFileDirectory,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | list[list[int]],
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
        if (value := getattr(geo_key_directory, key)) is not None:
            attrs[key] = value
    return attrs


def _get_attributes(ifd: ImageFileDirectory) -> dict[str, Any]:
    attrs = {}
    if ifd.geo_key_directory:
        attrs = _parse_geo_key_directory(ifd.geo_key_directory)
    else:
        attrs = {}
    extra_keys = [
        "model_pixel_scale",
        "model_tiepoint",
        "photometric_interpretation",
        "model_transformation",
    ]
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
        if _is_rectilinear_chunks(chunks):
            assert isinstance(chunks, list)
            chunks = [[1] * sample_dim_length] + chunks
        else:
            assert isinstance(chunks, tuple)
            chunks = (1,) + chunks
    else:
        # Check if rectilinear or regular chunk grid
        if _is_rectilinear_chunks(chunks):
            assert isinstance(chunks, list)
            chunks = [[sample_dim_length]] + chunks
        else:
            assert isinstance(chunks, tuple)
            chunks = (sample_dim_length,) + chunks

    return shape, chunks


def _construct_chunk_manifest(
    *,
    url: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | list[list[int]],
    offsets: Iterable[int],
    byte_counts: Iterable[int],
) -> ChunkManifest:
    if isinstance(chunks, tuple):
        chunk_manifest_shape = tuple(math.ceil(a / b) for a, b in zip(shape, chunks))
    else:
        chunk_manifest_shape = tuple(len(b) for b in chunks)
    offsets = np.array(offsets, dtype=np.uint64)
    byte_counts = np.array(byte_counts, dtype=np.uint64)

    if np.all(offsets == 0) or np.all(byte_counts == 0):
        raise NotImplementedError(
            "TIFFs without byte counts and offsets aren't supported"
        )
    offsets = offsets.reshape(chunk_manifest_shape)
    byte_counts = byte_counts.reshape(chunk_manifest_shape)
    urls = np.full_like(offsets, url, dtype=np.dtypes.StringDType)
    return ChunkManifest.from_arrays(
        paths=urls,
        offsets=offsets,
        lengths=byte_counts,
    )


async def _open_tiff(*, path: str, store: ObjectStore) -> TIFF:
    return await TIFF.open(path, store=store)


def _construct_manifest_array(
    *,
    ifd: ImageFileDirectory,
    url: str,
    endian: str,
    last_strip_data: bytes | None,
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
    if ifd.compression == 50001 and ifd.extra_samples:
        # WEBP compression may omit alpha channels from the encoded data.
        # Reconstructing the omitted channels is not yet supported.
        raise NotImplementedError(
            "WEBP compression with extra samples (alpha) is not yet supported."
        )
    chunks: tuple[int, ...] | list[list[int]]
    if ifd.tile_height:
        chunks, offsets, byte_counts = _get_chunks_from_tiles(ifd)
    elif ifd.rows_per_strip:
        chunks, offsets, byte_counts = _get_chunks_from_strips(
            ifd, image_height=shape[0], last_strip_data=last_strip_data
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
        url=url, shape=shape, chunks=chunks, offsets=offsets, byte_counts=byte_counts
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
            "configuration": {"kind": "inline", "chunk_shapes": chunks},
        }

    attributes, fill_value = _consolidate_fill_value(attributes, dtype)
    if fill_value is None:
        fill_value = zdtype.default_scalar()
    metadata = ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid=chunk_grid,
        chunk_key_encoding={"name": "default"},
        fill_value=fill_value,
        codecs=codecs,
        attributes=attributes,
        dimension_names=dimension_names,
        storage_transformers=None,
    )
    return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)


def _construct_manifest_group(
    url: str,
    store: ObjectStore,
    path: str,
    *,
    ifd: int | None = None,
    ifd_layout: Literal["flat", "nested"] = "flat",
) -> ManifestGroup:
    """Construct a ManifestGroup from TIFF IFDs.

    Args:
        store: obstore ObjectStore for reading the TIFF
        path: Path within the store to the TIFF file
        ifd: Specific IFD index to process, or None for all IFDs
        ifd_layout: How to organize IFDs - 'flat' for single group, 'nested' for group per IFD

    Returns:
        ManifestGroup containing the processed TIFF data
    """
    # TODO: Make an async approach.
    # `async_tiff.TIFF.open` accepts any object implementing the obspec async
    # read protocol (see async_tiff/python/tests/test_obspec.py), so we hand
    # the obstore-or-wrapper through directly. Wrapping a Python store here
    # is fine: TIFF reads issue range requests via the obspec protocol, which
    # gives wrappers (caching, tracing) a chance to intercept.
    tiff = sync(_open_tiff(store=store, path=path))
    endian = _ENDIANNESS_TO_STR[tiff.endianness]

    # Build manifest arrays from selected IFDs
    manifest_arrays = _build_manifest_arrays(tiff, url, endian, ifd, store, path)

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
    url: str,
    endian: str,
    ifd_index: int | None,
    object_store: ObjectStore,
    store_path: str,
) -> dict[str, ManifestArray]:
    """Build manifest arrays from TIFF IFDs.

    Fetches the last strip's compressed bytes for each IFD that needs
    padding detection, then delegates to store-agnostic functions.

    Args:
        tiff: Opened TIFF file
        url: Full URL path to the TIFF file
        endian: Byte order
        ifd_index: Specific IFD to process, or None for all
        object_store: Object store for fetching raw bytes
        store_path: Path within the store

    Returns:
        Dictionary mapping IFD indices (as strings) to ManifestArrays
    """
    manifest_arrays = {}

    if ifd_index is not None:
        # Process single specified IFD
        ifd = tiff.ifds[ifd_index]
        last_strip_data = _fetch_last_strip(ifd, object_store, store_path)
        manifest_arrays[str(ifd_index)] = _construct_manifest_array(
            ifd=ifd, url=url, endian=endian, last_strip_data=last_strip_data
        )
    else:
        # Process all IFDs
        for idx, ifd in enumerate(tiff.ifds):
            last_strip_data = _fetch_last_strip(ifd, object_store, store_path)
            manifest_arrays[str(idx)] = _construct_manifest_array(
                ifd=ifd, url=url, endian=endian, last_strip_data=last_strip_data
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
        store, path_in_store = registry.resolve(url)
        # Create a group containing dataset level metadata and all the manifest arrays
        manifest_group = _construct_manifest_group(
            url,
            store=store,
            path=path_in_store,
            ifd=self._ifd,
            ifd_layout=self.ifd_layout,
        )
        # Convert to a manifest store
        return ManifestStore(registry=registry, group=manifest_group)
