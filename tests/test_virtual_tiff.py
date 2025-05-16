import pytest
import numpy as np
import xarray as xr
from virtual_tiff import VirtualTIFF
import rioxarray
from conftest import loadable_dataset, github_examples, resolve_folder
from obstore.store import LocalStore
from virtualizarr.v2.api import open_virtual_dataset

failures = {
    "IBCSO_v2_ice-surface_cog.tif": "ValueError: Invalid range requested, start: 0 end: 0",
}

large_files = [
    "20250331090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif",
    "50N_120W.tif",
    "IBCSO_v2_ice-surface_cog.tif",
    "TCI.tif",
]


def test_simple_load_dataset_against_rioxarray(geotiff_file):
    ds = loadable_dataset(geotiff_file, store=LocalStore())
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
    ds = loadable_dataset(filepath, store=LocalStore())
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
    store = LocalStore()
    ms = parser(filepath, store)
    ds = ms.to_virtual_dataset()
    assert isinstance(ds, xr.Dataset)
    # TODO: Add more property tests


def test_using_VirtualTIFF_class_as_parser():
    filepath = f"{resolve_folder('tests/dvc/github')}/test_reference.tif"
    store = LocalStore()
    ds = open_virtual_dataset(
        filepath=filepath, object_reader=store, parser=VirtualTIFF
    )
    assert isinstance(ds, xr.Dataset)
    # TODO: Add more property tests
