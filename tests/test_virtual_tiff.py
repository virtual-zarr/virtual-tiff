import numpy as np
import pytest
import rioxarray
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtual_tiff import VirtualTIFF

from .conftest import (
    geotiff_test_data_examples,
    github_examples,
    loadable_dataset,
    resolve_folder,
)

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


@pytest.mark.parametrize("mask_and_scale", [True, False])
def test_simple_load_dataset_against_rioxarray(geotiff_file, mask_and_scale):
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    ds = loadable_dataset(
        f"file://{geotiff_file}", registry=registry, mask_and_scale=mask_and_scale
    )
    assert isinstance(ds, xr.Dataset)
    expected = rioxarray.open_rasterio(geotiff_file, masked=mask_and_scale)
    observed = ds["0"]
    np.testing.assert_allclose(observed.data.squeeze(), expected.data.squeeze())


@pytest.mark.parametrize("mask_and_scale", [True, False])
@pytest.mark.parametrize("filename", github_examples())
def test_load_dataset_against_rioxarray(filename, mask_and_scale):
    if filename in failures.keys():
        pytest.xfail(failures[filename])
    if filename in large_files:
        pytest.skip("Too slow")
    filepath = f"{resolve_folder('tests/data/github/')}/{filename}"
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    ds = loadable_dataset(
        f"file://{filepath}", registry=registry, mask_and_scale=mask_and_scale
    )
    assert isinstance(ds, xr.Dataset)
    da = ds["0"]
    da_expected = rioxarray.open_rasterio(filepath, masked=mask_and_scale)
    np.testing.assert_allclose(da.data, da_expected.data.squeeze())


@pytest.mark.parametrize("filename", github_examples())
def test_virtual_dataset_from_tiff(filename):
    if filename in failures.keys():
        pytest.xfail(failures[filename])
    filepath = f"{resolve_folder('tests/data/github')}/{filename}"
    parser = VirtualTIFF(ifd=0)
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    ms = parser(f"file://{filepath}", registry=registry)
    ds = ms.to_virtual_dataset()
    assert isinstance(ds, xr.Dataset)
    # TODO: Add more property tests


geotiff_test_data_failures = {
    "rasterio_generated/fixtures/uint8_rgba_webp_block64_cog.tif": "ValueError: cannot reshape array of size 12288 into shape (4,64,64)",
    "real_data/hot-oam/68077a72c46a9912474701ef.tif": "NotImplementedError: YCbCr PhotometricInterpretation is not yet supported.",
    "real_data/vantor/maxar_opendata_yellowstone_visual.tif": "NotImplementedError: YCbCr PhotometricInterpretation is not yet supported.",
    "real_data/rio-tiler/cog_rgb_with_stats.tif": "NotImplementedError: YCbCr PhotometricInterpretation is not yet supported.",
    "real_data/rio-tiler/non-tiled.tif": "ValueError: Zarr's default chunk grid expects all chunks to be equal size, but this TIFF has an uneven last strip.",
}

# Failures only when mask_and_scale=True; raw data passes. Tracked in
# https://github.com/virtual-zarr/virtual-tiff/issues/90.
geotiff_test_data_mask_and_scale_failures = {
    "rasterio_generated/fixtures/cog_uint8_rgba.tif": "Alpha band masking not yet supported (issue #90, Category 1).",
    "rasterio_generated/fixtures/cog_uint8_rgb_mask.tif": "Internal mask IFDs not yet supported (issue #90, Category 2).",
    "rasterio_generated/fixtures/uint8_1band_deflate_block128_unaligned_mask.tif": "Internal mask IFDs not yet supported (issue #90, Category 2).",
}


@pytest.mark.parametrize("mask_and_scale", [True, False])
@pytest.mark.parametrize("rel_path", geotiff_test_data_examples())
def test_geotiff_test_data_load(rel_path, mask_and_scale):
    if rel_path in geotiff_test_data_failures:
        pytest.xfail(geotiff_test_data_failures[rel_path])
    if mask_and_scale and rel_path in geotiff_test_data_mask_and_scale_failures:
        pytest.xfail(geotiff_test_data_mask_and_scale_failures[rel_path])
    filepath = f"{resolve_folder('tests/data/geotiff-test-data')}/{rel_path}"
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    ds = loadable_dataset(
        f"file://{filepath}", registry=registry, mask_and_scale=mask_and_scale
    )
    assert isinstance(ds, xr.Dataset)
    da = ds["0"]
    da_expected = rioxarray.open_rasterio(filepath, masked=mask_and_scale)
    np.testing.assert_allclose(da.data, da_expected.data.squeeze())


def test_local_store_with_prefix():
    data_dir = resolve_folder("tests/data/github").absolute()
    filepath = data_dir / "test_reference.tif"
    parser = VirtualTIFF(ifd=0)
    registry = ObjectStoreRegistry({"file://": LocalStore(data_dir)})
    ms = parser(f"file://{filepath}", registry=registry)
    ds = ms.to_virtual_dataset()
    assert isinstance(ds, xr.Dataset)
