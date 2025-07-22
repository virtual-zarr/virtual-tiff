# Copied and modified from https://github.com/zarr-developers/zarr-python/blob/bb55f0c58320a6d27be3a0ba918feee398a53db4/src/zarr/codecs/bytes.py

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from zarr.abc.codec import ArrayBytesCodec, ArrayArrayCodec
from zarr.codecs.bytes import Endian
from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer
from zarr.core.common import JSON, parse_enum, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec


@dataclass(frozen=True)
class ChunkyCodec(ArrayBytesCodec):
    is_fixed_size = True

    endian: Endian | None

    def __init__(self, *, endian: Endian | str | None = "little") -> None:
        endian_parsed = None if endian is None else parse_enum(endian, Endian)
        object.__setattr__(self, "endian", endian_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "ChunkyCodec", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "ChunkyCodec"}

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
    is_fixed_size = True

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "HorizontalDeltaCodec", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "HorizontalDeltaCodec"}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return chunk_array.__class__(chunk_array._data.cumsum(axis=-1))

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
