# Adapted from https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/codecs/bytes.py and https://github.com/zarr-developers/zarr-python/pull/3332
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Self, cast, overload

import numpy as np
from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
)
from zarr.codecs.bytes import Endian
from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer
from zarr.core.common import JSON
from zarr.registry import register_codec

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec


def check_codecjson_v2(data: object) -> bool:
    return isinstance(data, Mapping) and "id" in data and isinstance(data["id"], str)


ZarrFormat = Literal[2, 3]


def _parse_endian(data: object) -> Endian | None:
    if data is None:
        return None
    if isinstance(data, Endian):
        return data
    if isinstance(data, str) and data in ("little", "big"):
        return Endian(data)
    raise ValueError(
        f"Invalid endian value: {data!r}. Expected 'little', 'big', or None."
    )


@dataclass(frozen=True)
class ChunkyCodec(ArrayBytesCodec):
    is_fixed_size = True

    endian: Endian | None

    def __init__(self, *, endian: Endian | str | None = "little") -> None:
        object.__setattr__(self, "endian", _parse_endian(endian))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return cast(dict[str, JSON], self.to_json(zarr_format=3))

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if isinstance(data, Mapping):
            return cls(endian=data.get("endian"))
        raise ValueError(f"Invalid JSON: {data}")

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if isinstance(data, str):
            return cls()
        if isinstance(data, Mapping):
            config = data.get("configuration", {})
            return cls(**config)
        raise ValueError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> CodecJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> CodecJSON_V2 | CodecJSON_V3:
        if zarr_format == 2:
            if self.endian is not None:
                return {"id": "ChunkyCodec", "endian": self.endian.value}  # type: ignore[return-value, typeddict-item]
            return {"id": "ChunkyCodec"}  # type: ignore[return-value]
        elif zarr_format == 3:
            if self.endian is not None:
                return {
                    "name": "ChunkyCodec",
                    "configuration": {"endian": self.endian.value},
                }
            return {"name": "ChunkyCodec"}
        raise ValueError(
            f"Unsupported Zarr format {zarr_format}. Expected 2 or 3."
        )  # pragma: no cover

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if array_spec.dtype.item_size == 0:
            if self.endian is not None:
                return replace(self, endian=None)
        elif self.endian is None:
            raise ValueError(
                "The `endian` configuration needs to be specified for multi-byte data types."
            )
        return self

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        assert isinstance(chunk_bytes, Buffer)
        if chunk_spec.dtype.item_size > 0:
            if self.endian == Endian.little:
                prefix = "<"
            else:
                prefix = ">"
            dtype = np.dtype(f"{prefix}{chunk_spec.dtype.to_native_dtype().str[1:]}")
        else:
            dtype = np.dtype(f"|{chunk_spec.dtype.to_native_dtype().str[1:]}")

        as_array_like = chunk_bytes.as_array_like()
        if isinstance(as_array_like, NDArrayLike):
            as_nd_array_like = as_array_like
        else:
            as_nd_array_like = np.asanyarray(as_array_like)
        chunk_array = chunk_spec.prototype.nd_buffer.from_ndarray_like(
            as_nd_array_like.view(dtype=dtype)
        )

        # ensure correct chunk shape
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.__class__(
                np.ascontiguousarray(
                    chunk_array._data.reshape(chunk_spec.shape, order="F")
                )
            )
        return chunk_array

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        assert isinstance(chunk_array, NDBuffer)
        if (
            chunk_array.dtype.itemsize > 1
            and self.endian is not None
            and self.endian != chunk_array.byteorder
        ):
            # type-ignore is a numpy bug
            # see https://github.com/numpy/numpy/issues/26473
            new_dtype = chunk_array.dtype.newbyteorder(self.endian.name)  # type: ignore[arg-type]
            chunk_array = chunk_array.astype(new_dtype)

        nd_array = chunk_array.as_ndarray_like()
        # Flatten the nd-array (only copy if needed) and reinterpret as bytes
        nd_array = nd_array.ravel(order="F").view(dtype="B")
        return chunk_spec.prototype.buffer.from_array_like(nd_array)

    def compute_encoded_size(
        self, input_byte_length: int, _chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class HorizontalDeltaCodec(ArrayArrayCodec):
    """TIFF Predictor=2 (horizontal differencing) codec.

    The TIFF spec defines differencing on raw sample values in file byte order,
    before any endian conversion. Since this is an ArrayArrayCodec (runs after
    BytesCodec swaps to native endian), we must swap to file byte order before
    the cumulative sum and back to native after, for non-native-endian data.
    """

    is_fixed_size = True
    endian: Endian

    def __init__(self, endian: Endian | str = Endian.little) -> None:
        object.__setattr__(self, "endian", _parse_endian(endian) or Endian.little)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return cast(dict[str, JSON], self.to_json(zarr_format=3))

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if isinstance(data, Mapping):
            endian = data.get("endian", "little")
            return cls(endian=endian)
        return cls()

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if isinstance(data, Mapping):
            config = data.get("configuration", {})
            if isinstance(config, Mapping):
                endian = config.get("endian", "little")
                return cls(endian=endian)
        return cls()

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> CodecJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> CodecJSON_V2 | CodecJSON_V3:
        if zarr_format == 2:
            return {"id": "HorizontalDeltaCodec", "endian": self.endian.value}  # type: ignore[return-value, typeddict-item]
        elif zarr_format == 3:
            return {
                "name": "HorizontalDeltaCodec",
                "configuration": {"endian": self.endian.value},
            }
        raise ValueError(
            f"Unsupported Zarr format {zarr_format}. Expected 2 or 3."
        )  # pragma: no cover

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        # TIFF Predictor=2 (horizontal differencing) encodes by subtracting
        # consecutive samples as raw unsigned integers, regardless of the
        # actual data type. Decoding reverses this via cumulative sum.
        #
        # Three subtleties require care:
        # 1. Float data: the differences are of the uint bit patterns, not
        #    float values (e.g. float32 diffs are uint32 subtractions).
        # 2. Integer overflow: numpy.cumsum upcasts small unsigned types
        #    (e.g. uint16 → uint64), losing modular wrapping arithmetic.
        #    Passing dtype= forces the accumulation in the original width.
        # 3. Endianness: the TIFF spec defines differencing on raw sample
        #    values in file byte order. By the time this ArrayArrayCodec
        #    runs, BytesCodec has already swapped to native endian. For
        #    non-native data, we must swap to file order before cumsum
        #    and back to native after, so the wrapping arithmetic matches.
        data = chunk_array._data
        dtype = data.dtype
        itemsize = dtype.itemsize
        native_uint = np.dtype(f"u{itemsize}")

        # Determine if file endian differs from the array's current byte order.
        # After BytesCodec, the array is native-endian. dtype.byteorder is
        # '<', '>', '=' or '|'. Normalize to compare against self.endian.
        bo = dtype.byteorder
        if bo == "=":
            bo = "<" if np.little_endian else ">"
        elif bo == "|":
            bo = "<"  # single-byte: endian doesn't matter
        native_is_file_order = (bo == "<" and self.endian == Endian.little) or (
            bo == ">" and self.endian == Endian.big
        )

        if itemsize > 1 and not native_is_file_order:
            # Non-native endian: swap to file order, cumsum, swap back
            file_prefix = "<" if self.endian == Endian.little else ">"
            file_uint = np.dtype(f"{file_prefix}u{itemsize}")
            result = data.view(native_uint).byteswap().view(file_uint)
            result = result.cumsum(axis=-1, dtype=file_uint)
            result = result.view(native_uint).byteswap().view(dtype)
        else:
            result = data.view(native_uint)
            result = result.cumsum(axis=-1, dtype=native_uint).view(dtype)

        return chunk_array.__class__(result)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        raise NotImplementedError()

    def compute_encoded_size(
        self, input_byte_length: int, _chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


register_codec("ChunkyCodec", ChunkyCodec)
register_codec("HorizontalDeltaCodec", HorizontalDeltaCodec)
