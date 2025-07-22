from pathlib import Path

import pytest
import xarray as xr
import numpy as np
import rioxarray
from virtual_tiff import VirtualTIFF
from obstore.store import LocalStore
from virtualizarr.registry import ObjectStoreRegistry
from urllib.parse import urlparse


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
    data_dir = resolve_folder("tests/dvc/github")
    return list_tiffs(data_dir)


def gdal_autotest_examples():
    data_dir = resolve_folder("tests/dvc/gdal_autotest")
    return list_tiffs(data_dir)


def gdal_gcore_examples():
    data_dir = resolve_folder("tests/dvc/gdal_gcore")
    return list_tiffs(data_dir)


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
