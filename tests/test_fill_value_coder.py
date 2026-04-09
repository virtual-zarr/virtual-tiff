"""Tests for the vendored FillValueCoder.

Covers encode/decode roundtrips for all supported dtype kinds,
numpy scalar types, edge cases (inf, nan, zero), and error paths.
"""

import base64
import struct

import numpy as np
import pytest

from virtual_tiff.parser import _parse_fill_value
from virtual_tiff.vendor.xarray.zarr import FillValueCoder

# --- Roundtrip tests: encode then decode should recover the original value ---


class TestRoundtripFloat:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        "value",
        [0.0, 1.5, -9999.0, 1e-38, 1e38],
    )
    def test_finite(self, value, dtype):
        v = dtype(value)
        encoded = FillValueCoder.encode(v, np.dtype(dtype))
        decoded = FillValueCoder.decode(encoded, np.dtype(dtype))
        assert decoded == pytest.approx(float(v))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_inf(self, dtype):
        for v in [dtype(np.inf), dtype(-np.inf)]:
            encoded = FillValueCoder.encode(v, np.dtype(dtype))
            decoded = FillValueCoder.decode(encoded, np.dtype(dtype))
            np.testing.assert_equal(np.array(decoded, dtype=dtype), np.array(v))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nan(self, dtype):
        v = dtype(np.nan)
        encoded = FillValueCoder.encode(v, np.dtype(dtype))
        decoded = FillValueCoder.decode(encoded, np.dtype(dtype))
        assert np.isnan(decoded)

    def test_python_float(self):
        encoded = FillValueCoder.encode(3.14, np.dtype("float64"))
        decoded = FillValueCoder.decode(encoded, np.dtype("float64"))
        assert decoded == pytest.approx(3.14)

    def test_int_as_float(self):
        """int value should be accepted for float dtype."""
        encoded = FillValueCoder.encode(42, np.dtype("float64"))
        decoded = FillValueCoder.decode(encoded, np.dtype("float64"))
        assert decoded == pytest.approx(42.0)


class TestRoundtripInteger:
    @pytest.mark.parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ],
    )
    @pytest.mark.parametrize("value", [0, 1, -1, 127])
    def test_values(self, value, dtype):
        if np.dtype(dtype).kind == "u" and value < 0:
            pytest.skip("unsigned dtype cannot represent negative value")
        v = dtype(value)
        encoded = FillValueCoder.encode(v, np.dtype(dtype))
        decoded = FillValueCoder.decode(encoded, np.dtype(dtype))
        assert decoded == int(v)
        assert isinstance(decoded, int)

    def test_python_int(self):
        encoded = FillValueCoder.encode(99, np.dtype("int32"))
        decoded = FillValueCoder.decode(encoded, np.dtype("int32"))
        assert decoded == 99


class TestRoundtripBoolean:
    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_values(self, value):
        encoded = FillValueCoder.encode(value, np.dtype("bool"))
        decoded = FillValueCoder.decode(encoded, np.dtype("bool"))
        assert decoded == bool(value)
        assert isinstance(decoded, bool)


class TestRoundtripString:
    @pytest.mark.parametrize("value", ["", "hello", "hello world", "nan", "-9999"])
    def test_unicode(self, value):
        encoded = FillValueCoder.encode(value, np.dtype("U10"))
        decoded = FillValueCoder.decode(encoded, "string")
        assert decoded == value
        assert isinstance(decoded, str)


class TestRoundtripBytes:
    @pytest.mark.parametrize("value", [b"", b"\x00", b"hello", b"\xff\xfe"])
    def test_byte_strings(self, value):
        encoded = FillValueCoder.encode(value, np.dtype("S10"))
        decoded = FillValueCoder.decode(encoded, "bytes")
        assert decoded == value
        assert isinstance(decoded, bytes)


class TestRoundtripComplex:
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize(
        "value",
        [0 + 0j, 1 + 2j, -3.5 + 4.5j],
    )
    def test_finite(self, value, dtype):
        v = dtype(value)
        encoded = FillValueCoder.encode(v, np.dtype(dtype))
        decoded = FillValueCoder.decode(encoded, np.dtype(dtype))
        np.testing.assert_equal(np.array(decoded, dtype=dtype), np.array(v))

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_nan(self, dtype):
        v = dtype(complex("nan+nanj"))
        encoded = FillValueCoder.encode(v, np.dtype(dtype))
        decoded = FillValueCoder.decode(encoded, np.dtype(dtype))
        assert np.isnan(decoded.real)
        assert np.isnan(decoded.imag)

    def test_python_complex(self):
        encoded = FillValueCoder.encode(1 + 2j, np.dtype("complex128"))
        decoded = FillValueCoder.decode(encoded, np.dtype("complex128"))
        assert decoded == 1 + 2j

    def test_real_as_complex(self):
        """float value should be accepted for complex dtype."""
        encoded = FillValueCoder.encode(5.0, np.dtype("complex128"))
        decoded = FillValueCoder.decode(encoded, np.dtype("complex128"))
        assert decoded == 5.0 + 0j


# --- Encoding format tests ---


class TestEncodeFormat:
    def test_float_is_base64(self):
        encoded = FillValueCoder.encode(1.0, np.dtype("float64"))
        assert isinstance(encoded, str)
        raw = base64.standard_b64decode(encoded)
        assert len(raw) == 8
        assert struct.unpack("<d", raw)[0] == 1.0

    def test_int_is_python_int(self):
        encoded = FillValueCoder.encode(np.int32(42), np.dtype("int32"))
        assert encoded == 42
        assert isinstance(encoded, int)

    def test_bool_is_python_bool(self):
        encoded = FillValueCoder.encode(np.bool_(True), np.dtype("bool"))
        assert encoded is True

    def test_bytes_is_base64_string(self):
        encoded = FillValueCoder.encode(b"\x01\x02", np.dtype("S2"))
        assert isinstance(encoded, str)
        assert base64.standard_b64decode(encoded) == b"\x01\x02"

    def test_complex_is_list_of_two_base64(self):
        encoded = FillValueCoder.encode(1 + 2j, np.dtype("complex128"))
        assert isinstance(encoded, list)
        assert len(encoded) == 2
        real = struct.unpack("<d", base64.standard_b64decode(encoded[0]))[0]
        imag = struct.unpack("<d", base64.standard_b64decode(encoded[1]))[0]
        assert real == 1.0
        assert imag == 2.0


# --- Error tests ---


class TestEncodeErrors:
    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            FillValueCoder.encode(0, np.dtype("datetime64[ns]"))

    def test_bytes_value_for_float_dtype_raises(self):
        with pytest.raises(AssertionError):
            FillValueCoder.encode(b"\x00", np.dtype("float64"))

    def test_string_value_for_int_dtype_raises(self):
        with pytest.raises(AssertionError):
            FillValueCoder.encode("hello", np.dtype("int32"))


class TestDecodeErrors:
    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            FillValueCoder.decode(0, np.dtype("datetime64[ns]"))

    def test_int_value_for_float_decode_raises(self):
        with pytest.raises(AssertionError):
            FillValueCoder.decode(42, np.dtype("float64"))


# --- _parse_fill_value tests ---


class TestParseFillValue:
    @pytest.mark.parametrize(
        "value,dtype,expected",
        [
            ("-9999", np.dtype("float32"), np.float32(-9999.0)),
            ("0", np.dtype("uint8"), np.uint8(0)),
            ("nan", np.dtype("float64"), np.float64("nan")),
            ("inf", np.dtype("float32"), np.float32(np.inf)),
            ("-inf", np.dtype("float64"), np.float64(-np.inf)),
        ],
    )
    def test_valid(self, value, dtype, expected):
        result = _parse_fill_value(value, dtype)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == expected
        assert type(result) is type(expected)

    @pytest.mark.parametrize(
        "value,dtype",
        [
            ("-32768", np.dtype("uint8")),
            ("256", np.dtype("uint8")),
            ("not_a_number", np.dtype("float32")),
            ("hello", np.dtype("int16")),
        ],
    )
    def test_out_of_range_raises(self, value, dtype):
        with pytest.raises(ValueError, match="Cannot parse fill value"):
            _parse_fill_value(value, dtype)

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1.#INF", np.float32(np.inf)),
            ("-1.#INF", np.float32(-np.inf)),
            ("1.#QNAN", np.float32(np.nan)),
            ("-1.#QNAN", np.float32(np.nan)),
            ("1.#IND", np.float32(np.nan)),
            ("-1.#IND", np.float32(np.nan)),
        ],
    )
    def test_msvc_normalization(self, value, expected):
        result = _parse_fill_value(value, np.dtype("float32"))
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            np.testing.assert_equal(result, expected)
