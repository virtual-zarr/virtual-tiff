import pytest

from virtual_tiff.codecs import ChunkyCodec


@pytest.mark.parametrize("endian", ["big", "little"])
def test_chunky_codec_roundtrip_preserves_endian(endian):
    """ChunkyCodec.to_dict must include the endian config so that
    big-endian data is not silently reinterpreted as little-endian
    after a serialization round-trip through zarr metadata."""
    codec = ChunkyCodec(endian=endian)
    restored = ChunkyCodec.from_dict(codec.to_dict())
    assert restored.endian == codec.endian
