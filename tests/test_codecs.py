import pytest

from virtual_tiff.codecs import ChunkyCodec
from virtual_tiff.imagecodecs import DeflateCodec, LZWCodec


@pytest.mark.parametrize("endian", ["big", "little"])
def test_chunky_codec_roundtrip_preserves_endian(endian):
    """ChunkyCodec.to_dict must include the endian config so that
    big-endian data is not silently reinterpreted as little-endian
    after a serialization round-trip through zarr metadata."""
    codec = ChunkyCodec(endian=endian)
    restored = ChunkyCodec.from_dict(codec.to_dict())
    assert restored.endian == codec.endian


def test_imagecodecs_roundtrip():
    """Codec metadata must from_dict/to_dict cycles."""
    codec = LZWCodec()
    output = codec.to_dict()
    roundtripped = LZWCodec.from_dict(output)
    assert codec == roundtripped
    assert "name" not in output.get("configuration", {})
    output = roundtripped.to_dict()
    roundtripped = LZWCodec.from_dict(output)
    assert codec == roundtripped
    assert "name" not in output.get("configuration", {})


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
