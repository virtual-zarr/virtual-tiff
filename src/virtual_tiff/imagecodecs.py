# Adapted from https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/codecs/numcodecs/_codecs.py
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    overload,
)
from warnings import warn

import numpy as np
from zarr.abc.codec import (
    ArrayArrayCodec,
    BytesBytesCodec,
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
)
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON
from zarr.registry import get_numcodec

from virtual_tiff.codecs import check_codecjson_v2

if TYPE_CHECKING:
    from zarr.abc.numcodec import Numcodec


ZarrFormat = Literal[2, 3]


@dataclass(frozen=True)
class _ImageCodecsCodec:
    codec_name: str
    _codec: Numcodec
    codec_config: Mapping[str, Any]

    def __init__(self, **codec_config: Any) -> None:
        codec = get_numcodec(
            {
                "id": self.codec_name,
                **{k: v for k, v in codec_config.items() if k != "id"},
            }  # type: ignore[typeddict-item]
        )
        object.__setattr__(self, "_codec", codec)
        object.__setattr__(self, "codec_config", codec.get_config())
        warn(
            "Imagecodecs codecs are not in the Zarr version 3 specification and "
            "may not be supported by other zarr implementations.",
            category=UserWarning,
            stacklevel=2,
        )

    def to_dict(self) -> dict[str, JSON]:
        return cast(dict[str, JSON], self.to_json(zarr_format=3))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data)  # type: ignore[arg-type]

    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        if isinstance(data, str):
            return cls()
        return cls(**data.get("configuration", {}))

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)  # type: ignore[arg-type]
        return cls._from_json_v3(data)  # type: ignore[arg-type]

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> CodecJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> CodecJSON_V2 | CodecJSON_V3:
        codec_config = {k: v for k, v in self.codec_config.items() if k != "id"}
        if zarr_format == 2:
            return {"id": self.codec_name, **codec_config}  # type: ignore[return-value, typeddict-item]
        else:
            if codec_config:
                return {"name": self.codec_name, "configuration": codec_config}
            return {"name": self.codec_name}


class _ImageCodecsBytesBytesCodec(_ImageCodecsCodec, BytesBytesCodec):
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

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


class _ImageCodecsArrayArrayCodec(_ImageCodecsCodec, ArrayArrayCodec):
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

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


# array-to-array codecs ("filters")
class DeltaCodec(_ImageCodecsArrayArrayCodec):
    codec_name = "imagecodecs_delta"

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return replace(chunk_spec, dtype=np.dtype(astype))  # type: ignore[call-overload]
        return chunk_spec


class FloatPredCodec(_ImageCodecsArrayArrayCodec):
    codec_name = "imagecodecs_floatpred"

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return replace(chunk_spec, dtype=np.dtype(astype))  # type: ignore[call-overload]
        return chunk_spec


# bytes-to-bytes codecs


class DeflateCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_deflate"


class JetRawCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_jetraw"


class JpegCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_jpeg"


class Jpeg8Codec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_jpeg8"


class Jpeg2KCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_jpeg2k"


class JpegXRCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_jpegxr"


class JpegXLCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_jpegxl"


class LercCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_lerc"


class LZWCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_lzw"


class PackBitsCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_packbits"


class PngCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_png"


class WebpCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_webp"


class ZstdCodec(_ImageCodecsBytesBytesCodec):
    codec_name = "imagecodecs_zstd"


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
