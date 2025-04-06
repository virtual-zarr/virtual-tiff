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

    df = pd.read_csv(
        f"{repo_root}/tests/data/gdal_autotest/files.csv", header=None, names=["file"]
    )
    outpath = f"{repo_root}/tests/data/gdal_autotest"
    for row in df.iterrows():
        file = f"https://raw.githubusercontent.com/OSGeo/gdal/refs/heads/master/autotest/{row[1].file}"
        outname = file.split("/")[-1]
        pooch.retrieve(file, known_hash=None, path=outpath, fname=outname)


if __name__ == "__main__":
    download_files()
