import pytest
import numpy as np
import xarray as xr
from virtual_tiff import VirtualTIFF
from virtualizarr.registry import ObjectStoreRegistry
import rioxarray
from conftest import loadable_dataset, github_examples, resolve_folder
from obstore.store import LocalStore

failures = {
    "IBCSO_v2_ice-surface_cog.tif": "ValueError: Invalid range requested, start: 0 end: 0",
    "40613.tif": "Rust panic",
}

large_files = [
    "20250331090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif",
    "50N_120W.tif",
    "IBCSO_v2_ice-surface_cog.tif",
    "TCI.tif",
]


def test_simple_load_dataset_against_rioxarray(geotiff_file):
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    ds = loadable_dataset(f"file://{geotiff_file}", registry=registry)
    assert isinstance(ds, xr.Dataset)
    expected = rioxarray.open_rasterio(geotiff_file)
    observed = ds["0"]
    np.testing.assert_allclose(observed.data.squeeze(), expected.data.squeeze())


@pytest.mark.parametrize("filename", github_examples())
def test_load_dataset_against_rioxarray(filename):
    if filename in failures.keys():
        pytest.xfail(failures[filename])
    if filename in large_files:
        pytest.skip("Too slow")
    filepath = f"{resolve_folder('tests/dvc/github/')}/{filename}"
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    ds = loadable_dataset(f"file://{filepath}", registry=registry)
    assert isinstance(ds, xr.Dataset)
    da = ds["0"]
    da_expected = rioxarray.open_rasterio(filepath)
    np.testing.assert_allclose(da.data, da_expected.data.squeeze())


@pytest.mark.parametrize("filename", github_examples())
def test_virtual_dataset_from_tiff(filename):
    if filename in failures.keys():
        pytest.xfail(failures[filename])
    filepath = f"{resolve_folder('tests/dvc/github')}/{filename}"
    parser = VirtualTIFF(ifd=0)
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    ms = parser(f"file://{filepath}", registry=registry)
    ds = ms.to_virtual_dataset()
    assert isinstance(ds, xr.Dataset)
    # TODO: Add more property tests
