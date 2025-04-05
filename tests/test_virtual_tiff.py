import numpy as np
import xarray as xr
from virtual_tiff.reader import create_manifest_store

def test_read_geotiff(geotiff_file):
    import rioxarray
    from obstore.store import LocalStore

    ms = create_manifest_store(
        filepath=geotiff_file, group="0", file_id="file://", object_store=LocalStore()
    )
    ds = xr.open_dataset(ms, engine="zarr", consolidated=False, zarr_format=3).load()
    assert isinstance(ds, xr.Dataset)
    expected = rioxarray.open_rasterio(geotiff_file).data.squeeze()
    observed = ds["0"].data.squeeze()
    np.testing.assert_allclose(observed, expected)
