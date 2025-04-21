import pytest
import numpy as np
import xarray as xr
from virtual_tiff.reader import create_manifest_store
import rioxarray
from conftest import dataset_from_local_file, github_examples, resolve_folder

failures = {
    "IBCSO_v2_ice-surface_cog.tif": "ValueError: Invalid range requested, start: 0 end: 0",
}


def test_simple_load_dataset_against_rioxarray(geotiff_file):
    ds = dataset_from_local_file(geotiff_file)
    assert isinstance(ds, xr.Dataset)
    expected = rioxarray.open_rasterio(geotiff_file).data.squeeze()
    observed = ds["0"].data.squeeze()
    np.testing.assert_allclose(observed, expected)


@pytest.mark.parametrize("filename", github_examples())
def test_load_dataset_against_rioxarray(filename):
    if filename in failures.keys():
        pytest.xfail(failures[filename])
    filepath = f"{resolve_folder('tests/dvc/github/')}/{filename}"
    ds = dataset_from_local_file(filepath)
    assert isinstance(ds, xr.Dataset)
    da = ds["0"]
    da_expected = rioxarray.open_rasterio(filepath)
    np.testing.assert_allclose(da.data, da_expected.data.squeeze())


@pytest.mark.parametrize("filename", github_examples())
def test_virtual_dataset_from_tiff(filename):
    if filename in failures.keys():
        pytest.xfail(failures[filename])
    filepath = f"{resolve_folder('tests/dvc/github')}/{filename}"
    ms = create_manifest_store(filepath, group="0")
    ds = ms.to_virtual_dataset()
    assert isinstance(ds, xr.Dataset)
    # TODO: Add more property tests
