from pathlib import Path
import pytest
import numpy as np
import xarray as xr
from virtual_tiff.reader import create_manifest_store


def resolve_filepath(file, folder):
    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent
    return str(repo_root / folder / file)


def example_tiffs():
    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent
    data_dir = repo_root / "tests" / "data"

    # Find all .tif files in the data directory
    tif_files = list(data_dir.glob("*.tif"))
    return [file.name for file in tif_files]


def gdal_samples():
    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent
    data_dir = repo_root / "tests" / "data" / "gdal_autotest"

    # Find all .tif files in the data directory
    tif_files = list(data_dir.glob("*.tif"))
    return [file.name for file in tif_files]


def dataset_from_local_file(filepath):
    from obstore.store import LocalStore

    ms = create_manifest_store(
        filepath=filepath, group="0", file_id="file://", store=LocalStore()
    )
    return xr.open_dataset(ms, engine="zarr", consolidated=False, zarr_format=3).load()


class TestVirtualTIFF:
    def test_synthetic_example(self, geotiff_file):
        import rioxarray
        from obstore.store import LocalStore

        ms = create_manifest_store(
            filepath=geotiff_file,
            group="0",
            file_id="file://",
            object_store=LocalStore(),
        )
        ds = xr.open_dataset(
            ms, engine="zarr", consolidated=False, zarr_format=3
        ).load()
        assert isinstance(ds, xr.Dataset)
        expected = rioxarray.open_rasterio(geotiff_file).data.squeeze()
        observed = ds["0"].data.squeeze()
        np.testing.assert_allclose(observed, expected)

    @pytest.mark.parametrize("filename", example_tiffs())
    def test_real_examples(self, filename):
        import rioxarray

        filepath = resolve_filepath(filename, folder="tests/data")
        ds = dataset_from_local_file(filepath)
        assert isinstance(ds, xr.Dataset)
        da_expected = rioxarray.open_rasterio(filepath)
        np.testing.assert_allclose(ds["0"].data, da_expected.data.squeeze())

    @pytest.mark.parametrize("filename", gdal_samples())
    def test_gdal_examples(self, filename):
        import rioxarray

        filepath = resolve_filepath(filename, folder="tests/data/gdal_autotest")
        ds = dataset_from_local_file(filepath)
        assert isinstance(ds, xr.Dataset)
        da_expected = rioxarray.open_rasterio(filepath)
        np.testing.assert_allclose(ds["0"].data.squeeze(), da_expected.data.squeeze())
