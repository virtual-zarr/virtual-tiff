import os

import obstore
from virtualizarr.registry import ObjectStoreRegistry

from .conftest import requires_network, rioxarray_comparison


@requires_network
def test_load_gcs_dataset_against_rioxarray():
    bucket_url = "gs://gcp-public-data-landsat/"
    filepath = f"{bucket_url}LC08/01/044/034/LC08_L1TP_044034_20131228_20170307_01_T1/LC08_L1TP_044034_20131228_20170307_01_T1_B3.TIF"
    os.environ["GS_NO_SIGN_REQUEST"] = "YES"
    store = obstore.store.from_url(bucket_url, skip_signature=True)
    registry = ObjectStoreRegistry({bucket_url: store})
    rioxarray_comparison(filepath, registry=registry)
