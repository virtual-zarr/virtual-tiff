import base64
import numpy as np
from typing import Any
import struct


class FillValueCoder:
    """Handle custom logic to safely encode and decode fill values in Zarr.
    Possibly redundant with logic in xarray/coding/variables.py but needs to be
    isolated from NetCDF-specific logic.

    Copied from https://github.com/pydata/xarray/blob/main/xarray/backends/zarr.py
    """

    @classmethod
    def encode(cls, value: int | float | str | bytes, dtype: np.dtype[Any]) -> Any:
        if dtype.kind in "S":
            # byte string, this implies that 'value' must also be `bytes` dtype.
            assert isinstance(value, bytes)
            return base64.standard_b64encode(value).decode()
        elif dtype.kind in "b":
            # boolean
            return bool(value)
        elif dtype.kind in "iu":
            # todo: do we want to check for decimals?
            return int(value)
        elif dtype.kind in "f":
            return base64.standard_b64encode(struct.pack("<d", float(value))).decode()
        elif dtype.kind in "U":
            return str(value)
        else:
            raise ValueError(f"Failed to encode fill_value. Unsupported dtype {dtype}")

    @classmethod
    def decode(cls, value: int | float | str | bytes, dtype: str | np.dtype[Any]):
        if dtype == "string":
            # zarr V3 string type
            return str(value)
        elif dtype == "bytes":
            # zarr V3 bytes type
            assert isinstance(value, str | bytes)
            return base64.standard_b64decode(value)
        np_dtype = np.dtype(dtype)
        if np_dtype.kind in "f":
            assert isinstance(value, str | bytes)
            return struct.unpack("<d", base64.standard_b64decode(value))[0]
        elif np_dtype.kind in "b":
            return bool(value)
        elif np_dtype.kind in "iu":
            return int(value)
        else:
            raise ValueError(f"Failed to decode fill_value. Unsupported dtype {dtype}")
