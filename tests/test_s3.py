from conftest import rioxarray_comparison
import os
from obstore.store import S3Store


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
    rioxarray_comparison(filepath, store=store)
