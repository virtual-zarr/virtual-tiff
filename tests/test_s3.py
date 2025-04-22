from conftest import rioxarray_comparison


def test_load_https_dataset_against_rioxarray():
    filepath = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/B04.tif"
    rioxarray_comparison(filepath)
