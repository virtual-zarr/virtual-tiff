from pathlib import Path
from urllib.parse import urlparse

import imagecodecs
import numpy as np
import pytest
import rioxarray
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtual_tiff import VirtualTIFF

requires_network = pytest.mark.network

LZW_CLEAR_CODE = 256
LZW_EOI_CODE = 257


def lzw_encode_literals(
    data: bytes, *, with_eoi: bool = True, trailing: bytes = b""
) -> bytes:
    """Encode ``data`` as a TIFF-LZW stream of 9-bit literal codes.

    A clear code every 200 bytes keeps the dictionary below 512 entries, so every
    code stays 9 bits wide and the encoder needs no code-width logic.
    """
    out = bytearray()
    acc = nbits = 0

    def write(code: int) -> None:
        nonlocal acc, nbits
        acc = (acc << 9) | code
        nbits += 9
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1

    write(LZW_CLEAR_CODE)
    since_clear = 0
    for byte in data:
        if since_clear >= 200:
            write(LZW_CLEAR_CODE)
            since_clear = 0
        write(byte)
        since_clear += 1
    if with_eoi:
        write(LZW_EOI_CODE)
    if nbits:  # pad the final byte with zero bits
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out) + trailing


def lzw_unsized_decode_fails(stream: bytes, nbytes: int) -> bool:
    """Whether imagecodecs mis-decodes ``stream`` when not told the output size."""
    try:
        return len(imagecodecs.lzw_decode(stream)) != nbytes
    except imagecodecs.LzwError:
        return True


# Pytest configuration
def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="runs tests requiring a network connection",
    )


def pytest_runtest_setup(item):
    """Skip network tests unless explicitly enabled."""
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip(
            "set --run-network-tests to run tests requiring an internet connection"
        )


@pytest.fixture
def geotiff_file(tmp_path: Path) -> str:
    """Create a NetCDF4 file with air temperature data."""
    filepath = tmp_path / "air.tif"
    with xr.tutorial.open_dataset("air_temperature") as ds:
        ds.isel(time=0).rio.to_raster(filepath, driver="COG", COMPRESS="DEFLATE")
    return str(filepath)


def resolve_folder(folder: str):
    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent
    return repo_root / folder


def list_tiffs(folder):
    tif_files = list(folder.glob("*.tif"))
    return [file.name for file in tif_files]


def github_examples():
    data_dir = resolve_folder("tests/data/github")
    return list_tiffs(data_dir)


def gdal_examples():
    """Recursively find all .tif files under tests/data/gdal/, returning paths relative to gdal/."""
    data_dir = resolve_folder("tests/data/gdal")
    tif_files = sorted(data_dir.rglob("*.tif"))
    return [str(f.relative_to(data_dir)) for f in tif_files]


def geotiff_test_data_examples():
    """Recursively find all .tif files under tests/data/geotiff-test-data/, returning paths relative to geotiff-test-data/."""
    data_dir = resolve_folder("tests/data/geotiff-test-data")
    tif_files = sorted(data_dir.rglob("*.tif"))
    return [str(f.relative_to(data_dir)) for f in tif_files]


def loadable_dataset(filepath, registry, mask_and_scale=True):
    parser = VirtualTIFF(ifd=0)
    ms = parser(filepath, registry=registry)
    return xr.open_dataset(
        ms,
        engine="zarr",
        consolidated=False,
        zarr_format=3,
        mask_and_scale=mask_and_scale,
    ).load()


def rioxarray_comparison(
    filepath, registry: ObjectStoreRegistry = None, mask_and_scale=True
):
    if not registry:
        registry = ObjectStoreRegistry({filepath: LocalStore()})
    ds = loadable_dataset(filepath, registry, mask_and_scale=mask_and_scale)
    assert isinstance(ds, xr.Dataset)
    expected = rioxarray.open_rasterio(filepath, masked=mask_and_scale)
    filepath = urlparse(filepath).path
    if isinstance(expected, xr.DataArray):
        np.testing.assert_allclose(ds["0"].data.squeeze(), expected.data.squeeze())
    elif isinstance(expected, xr.Dataset):
        expected = expected[filepath.replace("/", "_").lstrip("_")]
        np.testing.assert_allclose(ds["0"].data.squeeze(), expected.data.squeeze())
    elif isinstance(expected, list):
        expected = expected[0][filepath.replace("/", "_").lstrip("_")]
        np.testing.assert_allclose(ds["0"].data.squeeze(), expected.data.squeeze())
    else:
        raise ValueError(
            f"Unexpected type returned from rioxarray.open_rasterio{filepath}"
        )
