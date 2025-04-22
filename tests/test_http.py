from conftest import rioxarray_comparison


def test_load_https_dataset_against_rioxarray():
    filepath = "https://dagshub.com/maxrjones/virtual-tiff/raw/7b49ee10de50f571d786606e7c5a549bb7a82e24/tests/dvc/github/test_reference.tif"
    rioxarray_comparison(filepath)
