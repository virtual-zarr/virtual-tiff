# Copied and slightly adapted from https://github.com/zarr-developers/numcodecs/blob/main/numcodecs/zarr3.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Self, TypeVar
from warnings import warn

import numpy as np

import numcodecs
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, parse_named_configuration

CODEC_PREFIX = "imagecodecs."
_imagecodecs_numcodecs = {
    "lzw": {"id": "imagecodecs_lzw"},
    "delta": {"id": "imagecodecs_delta"},
}


def _expect_name_prefix(codec_name: str) -> str:
    if not codec_name.startswith(CODEC_PREFIX):
        raise ValueError(
            f"Expected name to start with '{CODEC_PREFIX}'. Got {codec_name} instead."
        )  # pragma: no cover
    return codec_name.removeprefix(CODEC_PREFIX)


def _parse_codec_configuration(data: dict[str, JSON]) -> dict[str, JSON]:
    parsed_name, parsed_configuration = parse_named_configuration(data)
    if not parsed_name.startswith(CODEC_PREFIX):
        raise ValueError(
            f"Expected name to start with '{CODEC_PREFIX}'. Got {parsed_name} instead."
        )  # pragma: no cover
    id = _expect_name_prefix(parsed_name)
    return {"id": id, **parsed_configuration}


@dataclass(frozen=True)
class _NumcodecsCodec:
    codec_name: str
    codec_config: dict[str, JSON]

    def __init__(self, **codec_config: dict[str, JSON]) -> None:
        if not self.codec_name:
            raise ValueError(
                "The codec name needs to be supplied through the `codec_name` attribute."
            )  # pragma: no cover
        unprefixed_codec_name = _expect_name_prefix(self.codec_name)

        if "id" not in codec_config:
            codec_config = {"id": unprefixed_codec_name, **codec_config}
        elif codec_config["id"] != unprefixed_codec_name:
            raise ValueError(
                f"Codec id does not match {unprefixed_codec_name}. Got: {codec_config['id']}."
            )  # pragma: no cover

        object.__setattr__(self, "codec_config", codec_config)
        warn(
            "Numcodecs codecs are not in the Zarr version 3 specification and "
            "may not be supported by other zarr implementations.",
            category=UserWarning,
        )

    @cached_property
    def _codec(self) -> numcodecs.abc.Codec:
        return numcodecs.get_codec(_imagecodecs_numcodecs[self.codec_config["id"]])

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        codec_config = _parse_codec_configuration(data)
        return cls(**codec_config)

    def to_dict(self) -> JSON:
        codec_config = self.codec_config.copy()
        return {
            "name": self.codec_name,
            "configuration": codec_config,
        }

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError  # pragma: no cover


class _NumcodecsBytesBytesCodec(_NumcodecsCodec, BytesBytesCodec):
    def __init__(self, **codec_config: dict[str, JSON]) -> None:
        super().__init__(**codec_config)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._codec.decode,
            chunk_bytes,
            chunk_spec.prototype,
        )

    def _encode(self, chunk_bytes: Buffer, prototype: BufferPrototype) -> Buffer:
        encoded = self._codec.encode(chunk_bytes.as_array_like())
        if isinstance(encoded, np.ndarray):  # Required for checksum codecs
            return prototype.buffer.from_bytes(encoded.tobytes())
        return prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(self._encode, chunk_bytes, chunk_spec.prototype)


class _NumcodecsArrayArrayCodec(_NumcodecsCodec, ArrayArrayCodec):
    def __init__(self, **codec_config: dict[str, JSON]) -> None:
        super().__init__(**codec_config)

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        chunk_ndarray = chunk_array.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.decode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(
            out.reshape(chunk_spec.shape)
        )

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        chunk_ndarray = chunk_array.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)


class _NumcodecsArrayBytesCodec(_NumcodecsCodec, ArrayBytesCodec):
    def __init__(self, **codec_config: dict[str, JSON]) -> None:
        super().__init__(**codec_config)

    async def _decode_single(
        self, chunk_buffer: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        chunk_bytes = chunk_buffer.to_bytes()
        out = await asyncio.to_thread(self._codec.decode, chunk_bytes)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(
            out.reshape(chunk_spec.shape)
        )


# array-to-array codecs ("filters")
class Delta(_NumcodecsArrayArrayCodec):
    codec_name = f"{CODEC_PREFIX}delta"

    def __init__(self, **codec_config: dict[str, JSON]) -> None:
        super().__init__(**codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return replace(chunk_spec, dtype=np.dtype(astype))  # type: ignore[call-overload]
        return chunk_spec


T = TypeVar("T", bound=_NumcodecsCodec)


def _add_docstring(cls: type[T], ref_class_name: str) -> type[T]:
    cls.__doc__ = f"""
        See :class:`{ref_class_name}` for more details and parameters.
        """
    return cls


def _make_bytes_bytes_codec(
    codec_name: str, cls_name: str
) -> type[_NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(_NumcodecsBytesBytesCodec):
        codec_name = _codec_name

        def __init__(self, **codec_config: dict[str, JSON]) -> None:
            super().__init__(**codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def _make_array_array_codec(
    codec_name: str, cls_name: str
) -> type[_NumcodecsArrayArrayCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(_NumcodecsArrayArrayCodec):
        codec_name = _codec_name

        def __init__(self, **codec_config: dict[str, JSON]) -> None:
            super().__init__(**codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def _make_array_bytes_codec(
    codec_name: str, cls_name: str
) -> type[_NumcodecsArrayBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _Codec(_NumcodecsArrayBytesCodec):
        codec_name = _codec_name

        def __init__(self, **codec_config: dict[str, JSON]) -> None:
            super().__init__(**codec_config)

    _Codec.__name__ = cls_name
    return _Codec


def _make_checksum_codec(
    codec_name: str, cls_name: str
) -> type[_NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_name = CODEC_PREFIX + codec_name

    class _ChecksumCodec(_NumcodecsBytesBytesCodec):
        codec_name = _codec_name

        def __init__(self, **codec_config: dict[str, JSON]) -> None:
            super().__init__(**codec_config)

        def compute_encoded_size(
            self, input_byte_length: int, chunk_spec: ArraySpec
        ) -> int:
            return input_byte_length + 4  # pragma: no cover

    _ChecksumCodec.__name__ = cls_name
    return _ChecksumCodec


LZW = _add_docstring(_make_bytes_bytes_codec("lzw", "LZW"), "numcodecs.imagecodecs_lzw")

__all__ = [
    "Delta",
    "LZW",
]
