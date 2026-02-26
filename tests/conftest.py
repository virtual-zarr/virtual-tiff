from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
import rioxarray
import xarray as xr
from obstore.store import LocalStore
from virtualizarr.registry import ObjectStoreRegistry

from virtual_tiff import VirtualTIFF

requires_network = pytest.mark.network


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


def loadable_dataset(filepath, registry):
    parser = VirtualTIFF(ifd=0)
    ms = parser(filepath, registry=registry)
    return xr.open_dataset(ms, engine="zarr", consolidated=False, zarr_format=3).load()


def rioxarray_comparison(filepath, registry: ObjectStoreRegistry = None):
    if not registry:
        registry = ObjectStoreRegistry({filepath: LocalStore()})
    ds = loadable_dataset(filepath, registry)
    assert isinstance(ds, xr.Dataset)
    expected = rioxarray.open_rasterio(filepath)
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
