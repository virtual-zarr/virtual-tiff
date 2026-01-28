import xarray as xr
from obstore.store import S3Store
from virtualizarr.registry import ObjectStoreRegistry

from virtual_tiff import VirtualTIFF


def test_multiband_planar_tiff_from_source_coop():
    """Test multi-band TIFF with PlanarConfiguration=2 and non-RGB PhotometricInterpretation.

    This tests the fix for the TIFF 6.0 spec compliance issue where PlanarConfiguration=2
    was only handled for RGB images (PhotometricInterpretation=2).

    The AEF embedding TIFFs have:
    - PhotometricInterpretation = 1 (BlackIsZero)
    - SamplesPerPixel = 64 (embedding dimensions)
    - PlanarConfiguration = 2 (separate planes)
    """
    filepath = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2023/10N/xjtqldak16clgy5os-0000000000-0000008192.tiff"
    store = S3Store(
        bucket="us-west-2.opendata.source.coop",
        skip_signature=True,
        region="us-west-2",
    )
    registry = ObjectStoreRegistry({"s3://us-west-2.opendata.source.coop/": store})
    parser = VirtualTIFF(ifd=0)
    ms = parser(filepath, registry=registry)
    ds = xr.open_zarr(ms, zarr_format=3, consolidated=False)

    assert isinstance(ds, xr.Dataset)
    assert "band" in ds["0"].dims
    assert ds["0"].sizes["band"] == 64
