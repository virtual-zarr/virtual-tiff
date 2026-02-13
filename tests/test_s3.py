import os

import pytest
import xarray as xr
from conftest import rioxarray_comparison
from obstore.store import S3Store, from_url
from virtualizarr.registry import ObjectStoreRegistry

from virtual_tiff import VirtualTIFF


def test_load_s3_dataset_against_rioxarray():
    filepath = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/B04.tif"
    os.environ["AWS_NO_SIGN_REQUEST"] = "True"
    store = S3Store(
        bucket="sentinel-cogs",
        client_options={"allow_http": True},
        skip_signature=True,
        virtual_hosted_style_request=False,
        region="us-west-2",
    )
    registry = ObjectStoreRegistry({"s3://sentinel-cogs/sentinel-s2-l2a-cogs": store})
    rioxarray_comparison(filepath, registry=registry)


def test_open_datatree():
    from packaging.version import Version
    from virtualizarr import __version__ as _vz_version

    filepath = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/B04.tif"
    store = S3Store(
        bucket="sentinel-cogs",
        client_options={"allow_http": True},
        skip_signature=True,
        virtual_hosted_style_request=False,
        region="us-west-2",
    )
    registry = ObjectStoreRegistry({"s3://sentinel-cogs/sentinel-s2-l2a-cogs": store})
    parser = VirtualTIFF(ifd_layout="nested")

    if Version(_vz_version) < Version("2.2.0"):
        # Should raise ImportError for versions before 2.2.0
        with pytest.raises(ImportError, match="nested.*requires VirtualiZarr >= 2.2.0"):
            manifest_store = parser(filepath, registry=registry)
    else:
        # Should work properly for versions 2.2.0 and above
        manifest_store = parser(filepath, registry=registry)
        dt = xr.open_datatree(
            manifest_store, engine="zarr", zarr_format=3, consolidated=False
        )
        assert isinstance(dt, xr.DataTree)


def test_unknown_geokey():
    # Configuration
    bucket_url = "s3://prd-tnm/"
    file_url = f"{bucket_url}StagedProducts/Elevation/13/TIFF/current/s14w171/USGS_13_s14w171.tif"

    # Setup and open dataset
    s3_store = from_url(bucket_url, region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({bucket_url: s3_store})

    parser = VirtualTIFF(ifd=0)
    manifest_store = parser(url=file_url, registry=registry)
    ds = xr.open_zarr(manifest_store, zarr_format=3, consolidated=False)
    assert ds.load()
