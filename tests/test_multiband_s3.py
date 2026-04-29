import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import S3Store

from virtual_tiff import VirtualTIFF

from .conftest import requires_network

AEF_FILEPATH = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2023/10N/xjtqldak16clgy5os-0000000000-0000008192.tiff"


def _aef_registry() -> tuple[str, ObjectStoreRegistry]:
    store = S3Store(
        bucket="us-west-2.opendata.source.coop",
        skip_signature=True,
        region="us-west-2",
    )
    registry = ObjectStoreRegistry({"s3://us-west-2.opendata.source.coop/": store})
    return AEF_FILEPATH, registry


@requires_network
def test_multiband_planar_tiff_from_source_coop():
    """Test multi-band TIFF with PlanarConfiguration=2 and non-RGB PhotometricInterpretation.

    This tests the fix for the TIFF 6.0 spec compliance issue where PlanarConfiguration=2
    was only handled for RGB images (PhotometricInterpretation=2).

    The AEF embedding TIFFs have:
    - PhotometricInterpretation = 1 (BlackIsZero)
    - SamplesPerPixel = 64 (embedding dimensions)
    - PlanarConfiguration = 2 (separate planes)
    """
    filepath, registry = _aef_registry()
    parser = VirtualTIFF(ifd=0)
    ms = parser(filepath, registry=registry)
    ds = xr.open_zarr(ms, zarr_format=3, consolidated=False)

    assert isinstance(ds, xr.Dataset)
    assert "band" in ds["0"].dims
    assert ds["0"].sizes["band"] == 64


@requires_network
def test_aef_tiff_has_model_transformation():
    """Test that model_transformation is exposed in attributes when available.

    The AEF TIFFs use ModelTransformationTag instead of
    ModelPixelScale + ModelTiepoint for georeferencing.
    """
    filepath, registry = _aef_registry()
    parser = VirtualTIFF(ifd=0)
    ms = parser(filepath, registry=registry)
    ds = xr.open_zarr(ms, zarr_format=3, consolidated=False)

    attrs = ds["0"].attrs
    assert "model_transformation" in attrs
    assert isinstance(attrs["model_transformation"], list)
    assert len(attrs["model_transformation"]) == 16
    assert "model_pixel_scale" not in attrs
    assert "model_tiepoint" not in attrs


@requires_network
def test_aef_band_indices_selects_subset():
    """Test that band_indices filters the band dimension at the manifest level.

    With band_indices=[0, 3, 63], the resulting dataset should only have 3 bands
    and the spatial dimensions should be unchanged.
    """
    filepath, registry = _aef_registry()
    selected = [0, 3, 63]
    parser = VirtualTIFF(ifd=0, band_indices=selected)
    ms = parser(filepath, registry=registry)
    ds = xr.open_zarr(ms, zarr_format=3, consolidated=False)

    assert ds["0"].sizes["band"] == len(selected)
    assert ds["0"].sizes["y"] == 8192
    assert ds["0"].sizes["x"] == 8192


@requires_network
def test_aef_band_indices_none_returns_all():
    """Test that band_indices=None (default) returns all 64 bands."""
    filepath, registry = _aef_registry()
    parser = VirtualTIFF(ifd=0, band_indices=None)
    ms = parser(filepath, registry=registry)
    ds = xr.open_zarr(ms, zarr_format=3, consolidated=False)

    assert ds["0"].sizes["band"] == 64


@requires_network
def test_aef_band_indices_out_of_range():
    """Test that out-of-range band index raises IndexError."""
    filepath, registry = _aef_registry()
    parser = VirtualTIFF(ifd=0, band_indices=[0, 64])
    with pytest.raises(IndexError, match="Band index 64 out of range"):
        parser(filepath, registry=registry)
