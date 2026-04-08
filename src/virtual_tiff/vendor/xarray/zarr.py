import base64
import struct
from typing import Any

import numpy as np


class FillValueCoder:
    """Handle custom logic to safely encode and decode fill values in Zarr.
    Possibly redundant with logic in xarray/coding/variables.py but needs to be
    isolated from NetCDF-specific logic.

    Expanded from https://github.com/pydata/xarray/blob/main/xarray/backends/zarr.py
    """

    @classmethod
    def encode(
        cls, value: int | float | complex | str | bytes, dtype: np.dtype[Any]
    ) -> Any:
        if dtype.kind == "S":
            # byte string, this implies that 'value' must also be `bytes` dtype.
            assert isinstance(value, bytes)
            return base64.standard_b64encode(value).decode()
        elif dtype.kind == "b":
            # boolean
            return bool(value)
        elif dtype.kind in "iu":
            assert isinstance(value, int | float | np.integer | np.floating)
            return int(value)
        elif dtype.kind == "f":
            assert isinstance(value, int | float | np.integer | np.floating)
            return base64.standard_b64encode(struct.pack("<d", float(value))).decode()
        elif dtype.kind == "c":
            # complex - encode each component as base64, matching float encoding
            assert isinstance(
                value,
                complex | int | float | np.integer | np.floating | np.complexfloating,
            )
            c = complex(value)
            return [
                base64.standard_b64encode(struct.pack("<d", c.real)).decode(),
                base64.standard_b64encode(struct.pack("<d", c.imag)).decode(),
            ]
        elif dtype.kind == "U":
            return str(value)
        else:
            raise ValueError(f"Failed to encode fill_value. Unsupported dtype {dtype}")

    @classmethod
    def decode(
        cls, value: int | float | str | bytes | list, dtype: str | np.dtype[Any]
    ):
        if dtype == "string":
            # zarr V3 string type
            return str(value)
        elif dtype == "bytes":
            # zarr V3 bytes type
            assert isinstance(value, str | bytes)
            return base64.standard_b64decode(value)
        np_dtype = np.dtype(dtype)
        if np_dtype.kind == "f":
            assert isinstance(value, str | bytes)
            return struct.unpack("<d", base64.standard_b64decode(value))[0]
        elif np_dtype.kind == "c":
            # complex - decode each component from base64, matching float decoding
            assert isinstance(value, list | tuple) and len(value) == 2
            real = struct.unpack("<d", base64.standard_b64decode(value[0]))[0]
            imag = struct.unpack("<d", base64.standard_b64decode(value[1]))[0]
            return complex(real, imag)
        elif np_dtype.kind == "b":
            return bool(value)
        elif np_dtype.kind in "iu":
            assert isinstance(value, int | float | np.integer | np.floating)
            return int(value)
        else:
            raise ValueError(f"Failed to decode fill_value. Unsupported dtype {dtype}")
