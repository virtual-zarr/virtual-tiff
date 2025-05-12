# Copied and slightly adapted from https://github.com/zarr-developers/numcodecs/blob/main/numcodecs/zarr3.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Self
from warnings import warn

import numpy as np

import numcodecs
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON

CODEC_PREFIX = "imagecodecs_"


def _expect_name_prefix(codec_name: str) -> str:
    if not codec_name.startswith(CODEC_PREFIX):
        raise ValueError(
            f"Expected name to start with '{CODEC_PREFIX}'. Got {codec_name} instead."
        )  # pragma: no cover
    return codec_name.removeprefix(CODEC_PREFIX)


@dataclass(frozen=True)
class _ImageCodecsCodec:
    codec_name: str
    codec_config: JSON

    def __init_subclass__(cls, *, codec_name: str | None = None, **kwargs):
        """To be used only when creating the actual public-facing codec class."""
        super().__init_subclass__(**kwargs)
        if codec_name is not None:
            namespace = codec_name

            cls_name = f"{CODEC_PREFIX}{namespace}.{cls.__name__}"
            cls.codec_name = f"{CODEC_PREFIX}{namespace}"
            cls.__doc__ = f"""
            See :class:`{cls_name}` for more details and parameters.
            """

    def __init__(self, **codec_config: JSON) -> None:
        if not self.codec_name:
            raise ValueError(
                "The codec name needs to be supplied through the `codec_name` attribute."
            )  # pragma: no cover
        unprefixed_codec_name = _expect_name_prefix(self.codec_name)

        if "id" not in codec_config:
            codec_config = {
                "id": unprefixed_codec_name,  # type: ignore
                **codec_config,
            }
        elif codec_config["id"] != unprefixed_codec_name:
            raise ValueError(
                f"Codec id does not match {unprefixed_codec_name}. Got: {codec_config['id']}."
            )  # pragma: no cover

        object.__setattr__(self, "codec_config", codec_config)
        warn(
            "Imagecodecs codecs are not in the Zarr version 3 specification and "
            "may not be supported by other zarr implementations.",
            category=UserWarning,
        )

    @cached_property
    def _codec(self) -> numcodecs.abc.Codec:
        codec_config = self.codec_config["configuration"]
        codec_config["id"] = self.codec_config["name"]
        return numcodecs.get_codec(codec_config)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**data)

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


class _ImageCodecsBytesBytesCodec(_ImageCodecsCodec, BytesBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
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


class _ImageCodecsArrayArrayCodec(_ImageCodecsCodec, ArrayArrayCodec):
    def __init__(self, **codec_config: JSON) -> None:
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


class _ImageCodecsArrayBytesCodec(_ImageCodecsCodec, ArrayBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
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
class DeltaCodec(_ImageCodecsArrayArrayCodec, codec_name="delta"):
    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return replace(chunk_spec, dtype=np.dtype(astype))  # type: ignore[call-overload]
        return chunk_spec


class FloatPredCodec(_ImageCodecsArrayArrayCodec, codec_name="floatpred"):
    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return replace(chunk_spec, dtype=np.dtype(astype))  # type: ignore[call-overload]
        return chunk_spec


# bytes-to-bytes codecs


class DeflateCodec(_ImageCodecsBytesBytesCodec, codec_name="deflate"):
    pass


class JetRawCodec(_ImageCodecsBytesBytesCodec, codec_name="jetraw"):
    pass


class JpegCodec(_ImageCodecsBytesBytesCodec, codec_name="jpeg"):
    pass


class Jpeg8Codec(_ImageCodecsBytesBytesCodec, codec_name="jpeg8"):
    pass


class Jpeg2KCodec(_ImageCodecsBytesBytesCodec, codec_name="jpeg2k"):
    pass


class JpegXRCodec(_ImageCodecsBytesBytesCodec, codec_name="jpegxr"):
    pass


class JpegXLCodec(_ImageCodecsBytesBytesCodec, codec_name="jpegxl"):
    pass


class LercCodec(_ImageCodecsBytesBytesCodec, codec_name="lerc"):
    pass


class LZWCodec(_ImageCodecsBytesBytesCodec, codec_name="lzw"):
    pass


class PackBitsCodec(_ImageCodecsBytesBytesCodec, codec_name="packbits"):
    pass


class PngCodec(_ImageCodecsBytesBytesCodec, codec_name="png"):
    pass


class WebpCodec(_ImageCodecsBytesBytesCodec, codec_name="webp"):
    pass


class ZstdCodec(_ImageCodecsBytesBytesCodec, codec_name="zstd"):
    pass


__all__ = [
    "DeflateCodec",
    "DeltaCodec",
    "FloatPredCodec",
    "JetRawCodec",
    "JpegCodec",
    "Jpeg8Codec",
    "Jpeg2KCodec",
    "JpegXLCodec",
    "JpegXRCodec",
    "LercCodec",
    "LZWCodec",
    "PackBitsCodec",
    "PngCodec",
    "WebpCodec",
    "ZstdCodec",
]
