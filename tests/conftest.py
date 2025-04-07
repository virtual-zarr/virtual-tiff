from pathlib import Path

import pytest
import xarray as xr


@pytest.fixture
def geotiff_file(tmp_path: Path) -> str:
    """Create a NetCDF4 file with air temperature data."""
    filepath = tmp_path / "air.tif"
    with xr.tutorial.open_dataset("air_temperature") as ds:
        ds.isel(time=0).rio.to_raster(filepath, driver="COG", COMPRESS="DEFLATE")
    return str(filepath)


def download_files():
    import pandas as pd
    import pooch

    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent

    # download files from gdal autotest (https://github.com/OSGeo/gdal/tree/master/autotest) using list in https://gist.githubusercontent.com/mdsumner/29b22ece80c829ae4aefbecbf4eef531/raw/a8ea6f13f8c96b8434b7fc3ae0629814914135c0/autotest_tif.txt
    pooch.retrieve(
        "https://gist.githubusercontent.com/mdsumner/29b22ece80c829ae4aefbecbf4eef531/raw/a8ea6f13f8c96b8434b7fc3ae0629814914135c0/autotest_tif.txt",
        None,
        path=f"{repo_root}/tests/data",
        fname="gdal_autotest_files.txt",
    )
    df = pd.read_csv(
        f"{repo_root}/tests/data/gdal_autotest_files.txt", header=None, names=["file"]
    )
    outpath = f"{repo_root}/tests/data/gdal_autotest"
    for row in df.iterrows():
        file = f"https://raw.githubusercontent.com/OSGeo/gdal/refs/heads/master/autotest/{row[1].file}"
        outname = file.split("/")[-1]
        pooch.retrieve(file, known_hash=None, path=outpath, fname=outname)

    # download files from https://github.com/zarr-developers/VirtualiZarr/issues/526#issuecomment-2773597088 and https://github.com/zarr-developers/VirtualiZarr/issues/526#issuecomment-2773732236 and https://github.com/zarr-developers/VirtualiZarr/issues/526#issuecomment-2777745891
    files = [
        (
            "https://gitlab.com/Richard.Scott1/raster-analysis-goals/-/raw/main/test_reference.tif?ref_type=heads&inline=false",
            "ae9918a73b06e9246b081b01883acd10904a408edf6962b0198e399d5a6f2e09",
        ),
        (
            "https://data.eodc.eu/collections/SENTINEL1_SIG0_20M/V1M1R2/EQUI7_SA020M/E048N063T3/SIG0_20250404T234439__VH_A091_E048N063T3_SA020M_V1M1R2_S1AIWGRDH_TUWIEN.tif",
            "1bf8722f0a9e0897091c0ffc55c60ba1ef57d82865924666963b73d2c388614b",
        ),
        (
            "https://github.com/mdsumner/rema-ovr/raw/refs/heads/main/rema_mosaic_1km_v2.0_filled_cop30_dem.tif",
            "84227a0ead3140a62cce9c7310c76ff39a68a62c576972dab759312b663214bd",
        ),
        (
            "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/55/G/EN/2025/3/S2C_T55GEN_20250324T000834_L2A/TCI.tif",
            "92a2641890b3fa91a9623aee1f188c5b6604f50974cdee36defbfd8fd40e189d",
        ),
        (
            "https://data.source.coop/ausantarctic/ghrsst-mur-v2/2025/03/31/20250331090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif",
            "3878419f2bebd4a5b3b91d090976f3d64cc6ce55fb54f80688d6473719c65d24",
        ),
        (
            "https://github.com/mdsumner/ibcso-cog/raw/main/IBCSO_v2_ice-surface_cog.tif",
            "47d035bbb246ef5188abafbba089f8a3dcff807de5b84ddfeb22d18e6d536826",
        ),
    ]
    outpath = f"{repo_root}/tests/data"
    for f in files:
        file = f[0]
        outname = file.split("/")[-1].split("?")[0]
        pooch.retrieve(file, known_hash=None, path=outpath, fname=outname)

    # download 4 GB file from https://github.com/zarr-developers/VirtualiZarr/issues/526#issuecomment-2773732236
    pooch.retrieve(
        "https://projects.pawsey.org.au/idea-gebco-tif/GEBCO_2024.tif",
        "9992b941b3f1e2ecd39b4d79b96abd1b06c65a070d38c065312e4f3e80026cc3",
        path=f"{outpath}/xlarge_files/",
        fname="GEBCO_2024.tif",
    )


if __name__ == "__main__":
    download_files()
