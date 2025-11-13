from urllib.parse import urlparse

from conftest import rioxarray_comparison
from obstore.store import HTTPStore
from virtualizarr.registry import ObjectStoreRegistry


def test_load_https_dataset_against_rioxarray():
    url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/B04.tif"
    parsed = urlparse(url)
    url_base = f"{parsed.scheme}://{parsed.netloc}"
    store = HTTPStore.from_url(url_base)
    registry = ObjectStoreRegistry({url_base: store})
    rioxarray_comparison(url, registry=registry)
