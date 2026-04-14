import earthaccess
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obspec_utils.stores import AiohttpStore
from virtualizarr import open_virtual_dataset

from virtual_tiff import VirtualTIFF

from .conftest import requires_network


@requires_network
def test_load_podaac_http_dataset():
    """Test loading a PO.DAAC OPERA DSWX-S1 GeoTIFF via HTTP with earthaccess auth."""
    parser = VirtualTIFF(ifd=0)

    # Get earthaccess token
    auth = earthaccess.login(strategy="netrc")
    token = auth.token["access_token"]

    # PO.DAAC Store
    granule_server = "https://archive.podaac.earthdata.nasa.gov"
    http_store = AiohttpStore(
        granule_server,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
    )

    # PO.DAAC Registry
    http_registry = ObjectStoreRegistry[AiohttpStore]({granule_server: http_store})

    # Construct the file url
    granule_path = "podaac-ops-cumulus-protected/OPERA_L3_DSWX-S1_V1/OPERA_L3_DSWx-S1_T32UPU_20260326T170644Z_20260327T195326Z_S1C_30_v1.0_B01_WTR.tif"
    http_file_url = f"{granule_server}/{granule_path}"

    # Open the file as a virtual dataset
    virtual_http_ds = open_virtual_dataset(
        url=http_file_url,
        registry=http_registry,
        parser=parser,
    )
    assert isinstance(virtual_http_ds, xr.Dataset)
