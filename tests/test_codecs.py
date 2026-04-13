from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from zarr.codecs.bytes import Endian
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.buffer.cpu import NDBuffer
from zarr.core.dtype import Float32, Float64, UInt8, UInt16

from virtual_tiff.codecs import (
    ChunkyCodec,
    HorizontalDeltaCodec,
    _parse_endian,
    check_codecjson_v2,
)
from virtual_tiff.imagecodecs import (
    DeflateCodec,
    DeltaCodec,
    FloatPredCodec,
    LZWCodec,
    ZstdCodec,
)
from virtual_tiff.parser import (
    ZSTD_LEVEL_TAG,
    _get_compression,
    _open_tiff,
)

_DEFAULT_CONFIG = ArrayConfig(order="C", write_empty_chunks=True)


def _make_spec(shape, dtype, fill_value=0):
    return ArraySpec(
        shape=shape,
        dtype=dtype,
        fill_value=fill_value,
        config=_DEFAULT_CONFIG,
        prototype=default_buffer_prototype(),
    )


@dataclass
class FakeIFD:
    """Stub for async_tiff.ImageFileDirectory."""

    image_width: int = 256
    image_height: int = 256
    samples_per_pixel: int = 1
    bits_per_sample: tuple[int, ...] = (8,)
    sample_format: tuple[int, ...] = (1,)
    compression: int = 1
    predictor: int = 1
    planar_configuration: int = 1
    photometric_interpretation: Any = 1
    tile_height: int | None = 256
    tile_width: int | None = 256
    tile_offsets: list[int] = field(default_factory=lambda: [1000])
    tile_byte_counts: list[int] = field(default_factory=lambda: [65536])
    strip_offsets: list[int] | None = None
    strip_byte_counts: list[int] | None = None
    rows_per_strip: int | None = None
    geo_key_directory: Any = None
    model_pixel_scale: list[float] | None = None
    model_tiepoint: list[float] | None = None
    gdal_metadata: str | None = None
    gdal_nodata: str | None = None
    jpeg_tables: bytes | None = None
    other_tags: dict = field(default_factory=dict)


# --- check_codecjson_v2 tests ---


def test_check_codecjson_v2_with_id():
    assert check_codecjson_v2({"id": "foo"}) is True


def test_check_codecjson_v2_with_name():
    assert check_codecjson_v2({"name": "foo"}) is False


def test_check_codecjson_v2_with_string():
    assert check_codecjson_v2("foo") is False


def test_check_codecjson_v2_with_non_string_id():
    assert check_codecjson_v2({"id": 123}) is False


def test_check_codecjson_v2_with_empty_dict():
    assert check_codecjson_v2({}) is False


# --- _parse_endian tests ---


def test_parse_endian_none():
    assert _parse_endian(None) is None


def test_parse_endian_string_little():
    assert _parse_endian("little") == Endian.little


def test_parse_endian_string_big():
    assert _parse_endian("big") == Endian.big


def test_parse_endian_enum_passthrough():
    assert _parse_endian(Endian.big) is Endian.big


def test_parse_endian_invalid():
    with pytest.raises(ValueError, match="Invalid endian value"):
        _parse_endian("middle")


def test_parse_endian_invalid_type():
    with pytest.raises(ValueError, match="Invalid endian value"):
        _parse_endian(42)


# --- ChunkyCodec tests ---


def test_chunky_codec_default_endian():
    codec = ChunkyCodec()
    assert codec.endian == Endian.little


@pytest.mark.parametrize("endian", ["big", "little"])
def test_chunky_codec_roundtrip_preserves_endian(endian):
    """ChunkyCodec.to_dict must include the endian config so that
    big-endian data is not silently reinterpreted as little-endian
    after a serialization round-trip through zarr metadata."""
    codec = ChunkyCodec(endian=endian)
    restored = ChunkyCodec.from_dict(codec.to_dict())
    assert restored.endian == codec.endian


@pytest.mark.parametrize("endian", ["big", "little"])
def test_chunky_codec_to_json_v3(endian):
    codec = ChunkyCodec(endian=endian)
    v3 = codec.to_json(zarr_format=3)
    assert v3["name"] == "ChunkyCodec"
    assert v3["configuration"]["endian"] == endian
    restored = ChunkyCodec.from_json(v3)
    assert restored.endian == codec.endian


@pytest.mark.parametrize("endian", ["big", "little"])
def test_chunky_codec_to_json_v2(endian):
    codec = ChunkyCodec(endian=endian)
    v2 = codec.to_json(zarr_format=2)
    assert v2["id"] == "ChunkyCodec"
    assert v2["endian"] == endian
    restored = ChunkyCodec.from_json(v2)
    assert restored.endian == codec.endian


def test_chunky_codec_from_json_auto_detects_format():
    codec = ChunkyCodec(endian="big")
    v2 = codec.to_json(zarr_format=2)
    v3 = codec.to_json(zarr_format=3)
    assert ChunkyCodec.from_json(v2).endian == codec.endian
    assert ChunkyCodec.from_json(v3).endian == codec.endian


def test_chunky_codec_none_endian_to_json_v3():
    codec = ChunkyCodec(endian=None)
    v3 = codec.to_json(zarr_format=3)
    assert v3 == {"name": "ChunkyCodec"}
    assert "configuration" not in v3


def test_chunky_codec_none_endian_to_json_v2():
    codec = ChunkyCodec(endian=None)
    v2 = codec.to_json(zarr_format=2)
    assert v2 == {"id": "ChunkyCodec"}
    assert "endian" not in v2


def test_chunky_codec_from_json_v3_string_only():
    """A v3 codec can be just a name string with no configuration."""
    restored = ChunkyCodec._from_json_v3("ChunkyCodec")
    assert restored.endian == Endian.little  # default


def test_chunky_codec_from_json_v2_invalid():
    with pytest.raises(ValueError, match="Invalid JSON"):
        ChunkyCodec._from_json_v2(42)


def test_chunky_codec_from_json_v3_invalid():
    with pytest.raises(ValueError, match="Invalid JSON"):
        ChunkyCodec._from_json_v3(42)


def test_chunky_codec_compute_encoded_size():
    codec = ChunkyCodec()
    assert codec.compute_encoded_size(100, _make_spec((10,), UInt8())) == 100


@pytest.mark.asyncio
async def test_chunky_codec_decode_single_little_endian():
    codec = ChunkyCodec(endian="little")
    data = np.array([1, 2, 3], dtype="<u2")
    raw = default_buffer_prototype().buffer.from_bytes(data.tobytes())
    spec = _make_spec((3,), UInt16())
    result = await codec._decode_single(raw, spec)
    np.testing.assert_array_equal(result.as_ndarray_like(), data)


@pytest.mark.asyncio
async def test_chunky_codec_decode_single_big_endian():
    codec = ChunkyCodec(endian="big")
    data = np.array([1, 2, 3], dtype=">u2")
    raw = default_buffer_prototype().buffer.from_bytes(data.tobytes())
    spec = _make_spec((3,), UInt16())
    result = await codec._decode_single(raw, spec)
    np.testing.assert_array_equal(result.as_ndarray_like(), data)


@pytest.mark.asyncio
async def test_chunky_codec_encode_decode_roundtrip():
    codec = ChunkyCodec(endian="little")
    original = np.array([[1, 2], [3, 4]], dtype="<u2")
    nd_buf = NDBuffer.from_ndarray_like(original)
    spec = _make_spec((2, 2), UInt16())
    encoded = await codec._encode_single(nd_buf, spec)
    decoded = await codec._decode_single(encoded, spec)
    np.testing.assert_array_equal(decoded.as_ndarray_like(), original)


def test_chunky_codec_evolve_from_array_spec_single_byte():
    """endian should be preserved for single-byte dtypes (item_size > 0)."""
    codec = ChunkyCodec(endian="little")
    evolved = codec.evolve_from_array_spec(_make_spec((10,), UInt8()))
    assert evolved.endian == Endian.little


def test_chunky_codec_evolve_from_array_spec_multi_byte():
    """endian should be preserved for multi-byte dtypes."""
    codec = ChunkyCodec(endian="big")
    evolved = codec.evolve_from_array_spec(_make_spec((10,), UInt16()))
    assert evolved.endian == Endian.big


def test_chunky_codec_evolve_from_array_spec_none_endian_multi_byte():
    """endian=None with multi-byte dtype should raise."""
    codec = ChunkyCodec(endian=None)
    with pytest.raises(ValueError, match="endian"):
        codec.evolve_from_array_spec(_make_spec((10,), UInt16()))


# --- HorizontalDeltaCodec tests ---


def test_horizontal_delta_to_json_v3():
    codec = HorizontalDeltaCodec()
    v3 = codec.to_json(zarr_format=3)
    assert v3 == {"name": "HorizontalDeltaCodec"}
    restored = HorizontalDeltaCodec.from_json(v3)
    assert isinstance(restored, HorizontalDeltaCodec)


def test_horizontal_delta_to_json_v2():
    codec = HorizontalDeltaCodec()
    v2 = codec.to_json(zarr_format=2)
    assert v2["id"] == "HorizontalDeltaCodec"
    restored = HorizontalDeltaCodec.from_json(v2)
    assert isinstance(restored, HorizontalDeltaCodec)


def test_horizontal_delta_roundtrip():
    codec = HorizontalDeltaCodec()
    restored = HorizontalDeltaCodec.from_dict(codec.to_dict())
    assert isinstance(restored, HorizontalDeltaCodec)


def test_horizontal_delta_from_json_auto_detects_format():
    codec = HorizontalDeltaCodec()
    v2 = codec.to_json(zarr_format=2)
    v3 = codec.to_json(zarr_format=3)
    assert isinstance(HorizontalDeltaCodec.from_json(v2), HorizontalDeltaCodec)
    assert isinstance(HorizontalDeltaCodec.from_json(v3), HorizontalDeltaCodec)


@pytest.mark.asyncio
async def test_horizontal_delta_decode_cumsum():
    """HorizontalDeltaCodec decodes via cumulative sum along last axis."""
    codec = HorizontalDeltaCodec()
    # Input: differences [1, 2, 3] -> cumsum -> [1, 3, 6]
    diffs = np.array([[1, 2, 3]], dtype=np.int32)
    nd_buf = NDBuffer.from_ndarray_like(diffs)
    spec = _make_spec((1, 3), Float32())
    result = await codec._decode_single(nd_buf, spec)
    expected = np.array([[1, 3, 6]], dtype=np.int32)
    np.testing.assert_array_equal(result.as_ndarray_like(), expected)


@pytest.mark.asyncio
async def test_horizontal_delta_encode_raises():
    codec = HorizontalDeltaCodec()
    data = np.array([[1, 2, 3]], dtype=np.int32)
    nd_buf = NDBuffer.from_ndarray_like(data)
    spec = _make_spec((1, 3), Float32())
    with pytest.raises(NotImplementedError):
        await codec._encode_single(nd_buf, spec)


def test_horizontal_delta_compute_encoded_size():
    codec = HorizontalDeltaCodec()
    assert codec.compute_encoded_size(100, _make_spec((10,), UInt8())) == 100


def test_horizontal_delta_evolve_from_array_spec():
    codec = HorizontalDeltaCodec()
    spec = _make_spec((10,), UInt8())
    assert codec.evolve_from_array_spec(spec) is codec


# --- ImageCodecs tests ---


def test_imagecodecs_roundtrip():
    """Codec metadata must survive from_dict/to_dict cycles."""
    codec = LZWCodec()
    output = codec.to_dict()
    roundtripped = LZWCodec.from_dict(output)
    assert codec.codec_config == roundtripped.codec_config
    output = roundtripped.to_dict()
    roundtripped = LZWCodec.from_dict(output)
    assert codec.codec_config == roundtripped.codec_config


@pytest.mark.parametrize("codec_cls", [LZWCodec, DeflateCodec])
def test_imagecodecs_to_json_v3(codec_cls):
    codec = codec_cls()
    v3 = codec.to_json(zarr_format=3)
    assert v3["name"] == codec.codec_name
    restored = codec_cls.from_json(v3)
    assert restored.codec_config["id"] == codec.codec_config["id"]


@pytest.mark.parametrize("codec_cls", [LZWCodec, DeflateCodec])
def test_imagecodecs_to_json_v2(codec_cls):
    codec = codec_cls()
    v2 = codec.to_json(zarr_format=2)
    assert v2["id"] == codec.codec_name
    restored = codec_cls.from_json(v2)
    assert restored.codec_config["id"] == codec.codec_config["id"]


@pytest.mark.parametrize("codec_cls", [LZWCodec, DeflateCodec])
def test_imagecodecs_from_json_auto_detects_format(codec_cls):
    codec = codec_cls()
    v2 = codec.to_json(zarr_format=2)
    v3 = codec.to_json(zarr_format=3)
    assert codec_cls.from_json(v2).codec_config["id"] == codec.codec_config["id"]
    assert codec_cls.from_json(v3).codec_config["id"] == codec.codec_config["id"]


@pytest.mark.parametrize("codec_cls", [LZWCodec, DeflateCodec])
def test_imagecodecs_from_dict_ignores_wrapper_keys(codec_cls):
    """from_dict should extract only the configuration, not store
    the zarr wrapper keys (name, configuration) in codec_config."""
    codec = codec_cls()
    data = codec.to_dict()
    restored = codec_cls.from_dict(data)
    assert "name" not in restored.codec_config
    assert "configuration" not in restored.codec_config
    assert "id" in restored.codec_config


def test_imagecodecs_from_json_v3_string_only():
    """A v3 codec specified as just a name string should work."""
    restored = LZWCodec._from_json_v3("imagecodecs_lzw")
    assert restored.codec_config["id"] == "imagecodecs_lzw"


def test_imagecodecs_with_configuration():
    """Codecs with extra config should preserve it through roundtrips."""
    codec = DeflateCodec(level=6)
    assert codec.codec_config["level"] == 6
    v3 = codec.to_json(zarr_format=3)
    assert "configuration" in v3
    assert v3["configuration"]["level"] == 6
    restored = DeflateCodec.from_json(v3)
    assert restored.codec_config["level"] == 6


def test_imagecodecs_v2_with_configuration():
    """v2 format should inline config alongside the id."""
    codec = DeflateCodec(level=6)
    v2 = codec.to_json(zarr_format=2)
    assert v2["id"] == "imagecodecs_deflate"
    assert v2["level"] == 6
    restored = DeflateCodec.from_json(v2)
    assert restored.codec_config["level"] == 6


def test_imagecodecs_v3_no_configuration():
    """Codecs with only the 'id' in config should omit 'configuration' key in v3."""
    codec = LZWCodec()
    v3 = codec.to_json(zarr_format=3)
    assert "configuration" not in v3
    assert v3 == {"name": "imagecodecs_lzw"}


def test_imagecodecs_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LZWCodec()
        assert len(w) == 1
        assert "not in the Zarr version 3 specification" in str(w[0].message)


def test_imagecodecs_compute_encoded_size_raises():
    codec = LZWCodec()
    with pytest.raises(NotImplementedError):
        codec.compute_encoded_size(100, _make_spec((10,), UInt8()))


@pytest.mark.asyncio
async def test_imagecodecs_bytes_bytes_encode_decode_roundtrip():
    """LZW encode -> decode should recover the original bytes."""
    codec = LZWCodec()
    original = np.arange(256, dtype=np.uint8)
    buf = default_buffer_prototype().buffer.from_bytes(original.tobytes())
    spec = _make_spec((256,), UInt8())
    encoded = await codec._encode_single(buf, spec)
    assert encoded.to_bytes() != original.tobytes()  # compressed
    decoded = await codec._decode_single(encoded, spec)
    np.testing.assert_array_equal(
        np.frombuffer(decoded.to_bytes(), dtype=np.uint8), original
    )


@pytest.mark.asyncio
async def test_imagecodecs_zstd_encode_decode_roundtrip():
    codec = ZstdCodec()
    data = np.arange(100, dtype=np.float32)
    buf = default_buffer_prototype().buffer.from_bytes(data.tobytes())
    spec = _make_spec((100,), Float32(), fill_value=0.0)
    encoded = await codec._encode_single(buf, spec)
    decoded = await codec._decode_single(encoded, spec)
    np.testing.assert_array_equal(
        np.frombuffer(decoded.to_bytes(), dtype=np.float32), data
    )


def test_imagecodecs_delta_resolve_metadata_no_astype():
    codec = DeltaCodec()
    spec = _make_spec((10,), UInt8())
    result = codec.resolve_metadata(spec)
    assert result is spec


def test_imagecodecs_floatpred_resolve_metadata_no_astype():
    codec = FloatPredCodec(shape=(10,), dtype="f4")
    spec = _make_spec((10,), Float32(), fill_value=0.0)
    result = codec.resolve_metadata(spec)
    assert result is spec


class TestZstdLevelTag:
    def test_zstd_level_tag_constant_value(self):
        """The constant should be the numeric tag ID (int, matching async-tiff >= 0.7.0)."""
        assert ZSTD_LEVEL_TAG == 65564
        assert isinstance(ZSTD_LEVEL_TAG, int)

    def test_zstd_level_read_from_ifd(self):
        """_get_compression should use the ZSTD level from ifd.other_tags."""
        ifd = FakeIFD(
            compression=50000,
            other_tags={ZSTD_LEVEL_TAG: 3},  # int key 65564 -> level 3
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            codec = _get_compression(ifd, compression=50000)
        assert codec.codec_config["level"] == 3


class TestHorizontalDeltaFloat:
    """TIFF Predictor=2 operates on raw bit patterns as unsigned integers,
    regardless of the sample data type. The codec correctly handles this
    by performing cumsum on a uint view of the data."""

    @pytest.mark.asyncio
    async def test_float32_bit_level_diff(self):
        """HorizontalDeltaCodec correctly decodes float32 data with
        bit-level (Predictor=2) differencing."""
        codec = HorizontalDeltaCodec()

        # Original float32 values
        original = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        # TIFF Predictor=2 encodes by subtracting uint32 bit patterns:
        #   encoded[i] = uint32(raw[i]) - uint32(raw[i-1])
        raw_bits = original.view(np.uint32)
        encoded_bits = np.empty_like(raw_bits)
        encoded_bits[0, 0] = raw_bits[0, 0]
        encoded_bits[0, 1] = raw_bits[0, 1] - raw_bits[0, 0]
        encoded_bits[0, 2] = raw_bits[0, 2] - raw_bits[0, 1]

        # After BytesCodec, the data arrives as float32 (the array dtype)
        encoded_as_float = encoded_bits.view(np.float32)

        nd_buf = NDBuffer.from_ndarray_like(encoded_as_float)
        spec = _make_spec((1, 3), Float32(), fill_value=0.0)
        result = await codec._decode_single(nd_buf, spec)

        # Decoding performs cumsum on the uint32 bit patterns,
        # then views back as float32
        np.testing.assert_array_equal(result.as_ndarray_like(), original)

    @pytest.mark.asyncio
    async def test_float64_bit_level_diff(self):
        """Same bit-level differencing works for float64 data."""
        codec = HorizontalDeltaCodec()

        original = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        raw_bits = original.view(np.uint64)
        encoded_bits = np.empty_like(raw_bits)
        encoded_bits[0, 0] = raw_bits[0, 0]
        encoded_bits[0, 1] = raw_bits[0, 1] - raw_bits[0, 0]
        encoded_bits[0, 2] = raw_bits[0, 2] - raw_bits[0, 1]
        encoded_as_float = encoded_bits.view(np.float64)

        nd_buf = NDBuffer.from_ndarray_like(encoded_as_float)
        spec = _make_spec((1, 3), Float64(), fill_value=0.0)
        result = await codec._decode_single(nd_buf, spec)
        np.testing.assert_array_equal(result.as_ndarray_like(), original)

    @pytest.mark.asyncio
    async def test_uint16_wrapping(self):
        """Cumsum correctly wraps within uint16 range, matching libtiff's
        native-width arithmetic for Predictor=2."""
        codec = HorizontalDeltaCodec()

        # Original: [200, 10] — the diff wraps in uint16 arithmetic
        original = np.array([[200, 10]], dtype=np.uint16)
        encoded = np.empty_like(original)
        encoded[0, 0] = original[0, 0]
        # Compute wrapping diff the same way TIFF/libtiff does: uint16 subtraction
        encoded[0, 1] = original[0, 1] - original[0, 0]
        # 10 - 200 wraps to 65346 in uint16 numpy arithmetic

        nd_buf = NDBuffer.from_ndarray_like(encoded)
        spec = _make_spec((1, 2), UInt16())
        result = await codec._decode_single(nd_buf, spec)

        # Integer wrapping should give back the original values
        np.testing.assert_array_equal(result.as_ndarray_like(), original)

    @pytest.mark.asyncio
    async def test_int_cumsum_is_correct(self):
        """Integer predictor=2 decoding via cumsum works for normal values."""
        codec = HorizontalDeltaCodec()

        original = np.array([[10, 20, 35]], dtype=np.uint16)
        encoded = np.array(
            [[10, 10, 15]], dtype=np.uint16
        )  # diffs: 10, 20-10=10, 35-20=15

        nd_buf = NDBuffer.from_ndarray_like(encoded)
        spec = _make_spec((1, 3), UInt16())
        result = await codec._decode_single(nd_buf, spec)
        np.testing.assert_array_equal(result.as_ndarray_like(), original)


class TestOpenTiffWarnings:
    """Tests for _open_tiff warning paths when GeoTIFF metadata is present."""

    @pytest.mark.asyncio
    async def test_warns_when_async_geotiff_not_installed(self):
        """_open_tiff warns and returns plain TIFF when file has GeoKeyDirectory
        but async-geotiff is not installed."""
        fake_ifd = FakeIFD(geo_key_directory=object())
        fake_tiff = MagicMock()
        fake_tiff.ifds = [fake_ifd]
        with (
            patch("virtual_tiff.parser.HAS_ASYNC_GEOTIFF", False),
            patch(
                "virtual_tiff.parser.TIFF.open", new=AsyncMock(return_value=fake_tiff)
            ),
            patch(
                "virtual_tiff.parser.convert_obstore_to_async_tiff_store",
                return_value=MagicMock(),
            ),
        ):
            with pytest.warns(UserWarning, match="async-geotiff.*is not installed"):
                result = await _open_tiff(path="fake.tif", store=MagicMock())
        assert result is fake_tiff
