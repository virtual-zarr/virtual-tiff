import pytest

from virtual_tiff.codecs import ChunkyCodec, HorizontalDeltaCodec
from virtual_tiff.imagecodecs import DeflateCodec, LZWCodec

# --- ChunkyCodec tests ---


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
