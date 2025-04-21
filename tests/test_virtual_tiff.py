from pathlib import Path
import pytest
import numpy as np
import xarray as xr
from virtual_tiff.reader import create_manifest_store
import rioxarray


def resolve_folder(folder: str):
    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent
    return repo_root / folder


def list_tiffs(folder):
    tif_files = list(folder.glob("*.tif"))
    return [file.name for file in tif_files]


def github_examples():
    data_dir = resolve_folder("tests/dvc/github")
    return list_tiffs(data_dir)


def gdal_autotest_examples():
    data_dir = resolve_folder("tests/dvc/gdal_autotest")
    return list_tiffs(data_dir)


def gdal_gcore_examples():
    data_dir = resolve_folder("tests/dvc/gdal_gcore")
    return list_tiffs(data_dir)


def manifest_store_from_local_file(filepath):
    from obstore.store import LocalStore

    return create_manifest_store(
        filepath=filepath, group="0", file_id="file://", store=LocalStore()
    )


def dataset_from_local_file(filepath):
    ms = manifest_store_from_local_file(filepath)
    return xr.open_dataset(ms, engine="zarr", consolidated=False, zarr_format=3).load()


def rioxarray_comparison(filepath):
    ds = dataset_from_local_file(filepath)
    assert isinstance(ds, xr.Dataset)
    da_expected = rioxarray.open_rasterio(filepath)
    np.testing.assert_allclose(ds["0"].data.squeeze(), da_expected.data.squeeze())


class TestVirtualTIFF:
    def test_simple(self, geotiff_file):
        ds = dataset_from_local_file(geotiff_file)
        assert isinstance(ds, xr.Dataset)
        expected = rioxarray.open_rasterio(geotiff_file).data.squeeze()
        observed = ds["0"].data.squeeze()
        np.testing.assert_allclose(observed, expected)

    @pytest.mark.parametrize("filename", github_examples())
    def test_against_rioxarray(self, filename):
        if filename in failures:
            pytest.xfail("Known failure")
        filepath = f"{resolve_folder('tests/dvc/github/')}/{filename}"
        ds = dataset_from_local_file(filepath)
        assert isinstance(ds, xr.Dataset)
        da = ds["0"]
        da_expected = rioxarray.open_rasterio(filepath)
        np.testing.assert_allclose(da.data, da_expected.data.squeeze())

    @pytest.mark.parametrize("filename", github_examples())
    def test_virtual_dataset_real_examples(self, filename):
        if filename in failures:
            pytest.xfail("Known failure")
        filepath = f"{resolve_folder('tests/dvc/github')}/{filename}"
        ms = manifest_store_from_local_file(filepath)
        ds = ms.to_virtual_dataset()
        assert isinstance(ds, xr.Dataset)
        # TODO: Add more property tests


class TestVirtualTIFFGDALData:
    @pytest.mark.parametrize("filename", gdal_autotest_examples())
    def test_against_rioxarray_data1(self, filename):
        if filename in failures:
            pytest.xfail("Known failure")
        filepath = f"{resolve_folder('tests/dvc/gdal_autotest')}/{filename}"
        rioxarray_comparison(filepath)

    @pytest.mark.parametrize("filename", gdal_gcore_examples())
    def test_against_rioxarray_data2(self, filename):
        if filename in failures:
            pytest.xfail("Known failure")
        filepath = f"{resolve_folder('tests/dvc/gdal_gcore')}/{filename}"
        rioxarray_comparison(filepath)


# Generated with the assistance of Claude
xfail_samples_per_pixel = [
    "TCI.tif",
    "20250331090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif",
    "rgbsmall_LZMA_tiled_separate.tif",
    "rgbsmall.tif",
    "rgbsmall_DEFLATE_separate.tif",
    "stefan_full_greyalpha_byte_LZW_predictor_2.tif",
    "rgbsmall_NONE.tif",
    "rgbsmall_int16_bigendian_lzw_predictor_2.tif",
    "whiteblackred.tif",
    "huge_line.tif",
    "rgbsmall_LZMA.tif",
    "rgbsmall_JXL_tiled.tif",
    "rgbsmall_ZSTD_tiled_separate.tif",
    "ca_nrc_NAD83v70VG.tif",
    "rgbsmall_LERC_ZSTD_tiled.tif",
    "expected_TILES.tif",
    "expected_MAP.tif",
    "rgbsmall_JPEG_separate.tif",
    "bug1488.tif",
    "test3658.tif",
    "rgbsmall_JPEG.tif",
    "rgbsmall_JXL.tif",
    "rgbsmall_JXL_separate.tif",
    "rgbsmall_JPEG_tiled_separate.tif",
    "rgbsmall_WEBP.tif",
    "rgbsmall_LERC_ZSTD_separate.tif",
    "IMG-md_alos.tif",
    "rgbsmall_ZSTD_separate.tif",
    "rgbsmall_LZW.tif",
    "rgbsmall_uint16_LZW_predictor_2.tif",
    "rgbsmall_NONE_separate.tif",
    "stefan_full_greyalpha_uint16_LZW_predictor_2.tif",
    "sparse_tiled_separate.tif",
    "rgbsmall_LZW_separate.tif",
    "rgbsmall_NONE_tiled.tif",
    "webp_lossless_rgba_alpha_fully_opaque.tif",
    "rgbsmall_LERC_DEFLATE.tif",
    "rgbsmall_LERC_DEFLATE_tiled_separate.tif",
    "rgbsmall_LERC_DEFLATE_separate.tif",
    "rgbsmall_LERC.tif",
    "missing_extrasamples.tif",
    "rgbsmall_DEFLATE.tif",
    "rgbsmall_WEBP_tiled.tif",
    "rgbsmall_ZSTD_tiled.tif",
    "rgbsmall_LERC_tiled.tif",
    "unstable_rpc_with_dem_blank_output.tif",
    "rgbsmall_ZSTD.tif",
    "0.tif",
    "rgbsmall_DEFLATE_tiled.tif",
    "rgbsmall_DEFLATE_tiled_separate.tif",
    "rgbsmall_LZMA_separate.tif",
    "rgbsmall_LERC_DEFLATE_tiled.tif",
    "1_0_0.tif",
    "rgbsmall_LZMA_tiled.tif",
    "rgbsmall_LERC_ZSTD.tif",
    "rgbsmall_LERC_separate.tif",
    "rgbsmall_LERC_ZSTD_tiled_separate.tif",
    "small_world.tif",
    "esri_geodataxform_no_resolutionunit.tif",
    "test_gdal2tiles_exclude_transparent.tif",
    "byte_5_bands_LZW_predictor_2.tif",
    "rgbsmall_byte_LZW_predictor_2.tif",
    "rgbsmall_LZW_tiled.tif",
    "stefan_full_greyalpha_byte_LZW_predictor_2.tif",
    "rgbsmall_int16_bigendian_lzw_predictor_2.tif",
    "rgbsmall_JPEG_ycbcr.tif",
    "stefan_full_greyalpha_uint64_LZW_predictor_2.tif",
    "rgbsmall_LERC_tiled_separate.tif",
    "rgbsmall_uint32_LZW_predictor_2.tif",
    "stefan_full_greyalpha_uint32_LZW_predictor_2.tif",
    "jxl-rgbi.tif",
    "md_spot.tif",
    "sparse_tiled_contig.tif",
    "rgbsmall_JPEG_tiled.tif",
    "rgbsmall_NONE_tiled_separate.tif",
    "rgbsmall_uint64_LZW_predictor_2.tif",
    "projection_from_esri_xml.tif",
    "rgbsmall_LZW_tiled_separate.tif",
    "2_0_0.tif",
    "2_0_1.tif",
    "stefan_full_rgba_LZW_predictor_2.tif",
    "rgbsmall_JXL_tiled_separate.tif",
    "test3_with_mask_8bit.tif",
    "stefan_full_rgba_photometric_rgb.tif",
    "test_11555.tif",
    "test_nodatavalues.tif",
    "bug4468.tif",
    "cielab.tif",
    "quad-lzw-old-style.tif",
    "ycbcr_12_lzw.tif",
    "rgba.tif",
    "exif_and_gps.tif",
    "ycbcr_24_lzw.tif",
    "md_dg.tif",
    "zackthecat_corrupted.tif",
    "mandrilmini_12bitjpeg.tif",
    "separate_tiled.tif",
    "6band_wrong_number_extrasamples.tif",
    "sstgeo.tif",
    "seperate_strip.tif",
    "ycbcr_44_lzw.tif",
    "tif_jpeg_ycbcr_too_big_last_stripe.tif",
    "complex_int32.tif",
    "rgba_with_alpha_0_and_255.tif",
    "md_ov.tif",
    "md_kompsat.tif",
    "md_rdk1.tif",
    "sasha.tif",
    "test_gf.tif",
    "ycbcr_11_lzw.tif",
    "ycbcr_14_lzw.tif",
    "contig_strip.tif",
    "ycbcr_22_lzw.tif",
    "contig_tiled.tif",
    "stefan_full_rgba_jpeg_contig.tif",
    "test3_with_mask_1bit_and_ovr.tif",
    "3376.tif",
    "rgbsmall_cmyk.tif",
    "stefan_full_rgba_jpeg_separate.tif",
    "toomanyblocks_separate.tif",
    "geoloc_triangles.tif",
    "ycbcr_41_lzw.tif",
    "int12_ycbcr_contig.tif",
    "md_ge_rgb_0010000.tif",
    "ycbcr_42_lzw_optimized.tif",
    "vrtmisc16_tile1.tif",
    "ycbcr_44_lzw_optimized.tif",
    "md_ls_b1.tif",
    "stefan_full_rgba.tif",
    "tif_webp.tif",
    "scanline_more_than_2GB.tif",
    "zackthecat.tif",
    "ycbcr_21_lzw.tif",
    "stefan_full_greyalpha.tif",
    "WGS_1984_Web_Mercator.tif",
    "md_dg_2.tif",
    "md_re.tif",
    "cog_strile_arrays_zeroified_when_possible.tif",
    "next_literalrow.tif",
    "cint16.tif",
    "test3_with_mask_1bit.tif",
    "reproduce_average_issue.tif",
    "test3_with_1mask_1bit.tif",
    "tiled_bad_offset.tif",
    "1bit_2bands.tif",
    "tif_webp_huge_single_strip.tif",
    "ycbcr_with_mask.tif",
    "ycbcr_42_lzw.tif",
    "oddsize_1bit2b.tif",
    "md_eros.tif",
    "test_hgrid_with_subgrid.tif",
]
xfail_byte_counts = [
    "VH.tif",
    "sparse_nodata_one.tif",
    "geog_arc_second.tif",
    "VV.tif",
    "tiff_dos_strip_chop.tif",
    "empty1bit.tif",
    "unknown_compression.tif",
    "block_width_above_32bit.tif",
    "arcgis93_geodataxform_gcp.tif",
    "image_width_above_32bit.tif",
    "tiff_dos_strip_chop.tif",
    "hugeblocksize.tif",
]
xfail_compression = [
    "byte_JXL_tiled.tif",
    "unsupported_codec_unknown.tif",
    "byte_LERC_DEFLATE_tiled.tif",
    "byte_LERC.tif",
    "byte_LERC_DEFLATE.tif",
    "byte_JPEG_tiled.tif",
    "byte_LERC_ZSTD.tif",
    "byte_JXL.tif",
    "byte_LERC_ZSTD_tiled.tif",
    "byte_JPEG.tif",
    "byte_ovr_jpeg_tablesmode1.tif",
    "byte_ovr_jpeg_tablesmode_not_correctly_set_on_ovr.tif",
    "byte_ovr_jpeg_tablesmode0.tif",
    "tif_jpeg_too_big_last_stripe.tif",
    "byte_ovr_jpeg_tablesmode2.tif",
    "byte_ovr_jpeg_tablesmode3.tif",
    "byte_jpg_tablesmodezero.tif",
    "byte_jpg_unusual_jpegtable.tif",
    "slim_g4.tif",
    "byte_jxl_dng_1_7_52546.tif",
    "byte_jxl_deprecated_50002.tif",
    "byte_LZMA_tiled.tif",
    "unsupported_codec_jp2000.tif",
    "byte_LZMA.tif",
    "byte_LERC_tiled.tif",
    "thunder.tif",
    "next_literalspan.tif",
    "next_default_case.tif",
    "byte_lerc.tif",
    "irregular_tile_size_jpeg_in_tiff.tif",
]

xfail_reshape = [
    "isis3_geotiff.tif",
    "bug_6526_input.tif",
    "pyramid_shaded_ref.tif",
    "melb-small.tif",
    "utmsmall.tif",
    "n43.tif",
    "geos_vrtwarp.tif",
    "unstable_rpc_with_dem_source.tif",
    "warp_52_dem.tif",
    "excessive-memory-TIFFFillTile.tif",
    "toomanyblocks.tif",
    "dstsize_larger_than_source.tif",
    "oddsize1bit.tif",
    "utilities_utmsmall.tif",
    "huge4GB.tif",
    "vrtmisc16_tile2.tif",
    "size_of_stripbytecount_at_1_and_lower_than_stripcount.tif",
    "transformer_13_dem.tif",
    "vrtmisc16_tile1.tif",
    "size_of_stripbytecount_lower_than_stripcount.tif",
    "int10.tif",
    "int12.tif",
]
xfail_assert = [
    "byte_LZW_predictor_2.tif",
    "int16_big_endian.tif",
    "unstable_rpc_with_dem_elevation.tif",
    "uint16_LZW_predictor_2.tif",
]
xfail_panic = [
    "missing_tilebytecounts_and_offsets.tif",
    "nodata_precision_issue_float32.tif",
    "projected_GTCitationGeoKey_with_underscore_and_GeogTOWGS84GeoKey.tif",
    "byte_coord_epoch.tif",
    "polygonize_check_area.tif",
    "nodata_precision_issue_float32.tif",
    "epsg_27563_allgeokeys.tif",
    "many_blocks_truncated.tif",
    "weird_mercator_2sp.tif",
    "byte_bigtiff_invalid_slong8_for_stripoffsets.tif",
    "corrupted_deflate_singlestrip.tif",
    "excessive-memory-TIFFFillStrip.tif",
    "packbits-not-enough-data.tif",
    "excessive-memory-TIFFFillStrip2.tif",
    "huge-implied-number-strips.tif",
    "spaf27_markedcorrect.tif",
    "bigtiff_header_extract.tif",
    "minimum_tiff_tags_with_warning.tif",
    "gtiff_towgs84_override.tif",
    "corrupted_gtiff_tags.tif",
    "uint16_sgilog.tif",
    "minimum_tiff_tags_no_warning.tif",
    "byte_user_defined_geokeys.tif",
    "huge-number-strips.tif",
]
xfail_dtype = [
    "cint32.tif",
    "cint32_big_endian.tif",
    "int24.tif",
    "cint_sar.tif",
    "float24.tif",
    "uint33.tif",
]
xfail_other = [
    "lzw_corrupted.tif",
    "uint32_LZW_predictor_2.tif",
    "uint64_LZW_predictor_2.tif",
    "float32_LZW_predictor_2.tif",
    "IBCSO_v2_ice-surface_cog.tif",
    "byte_little_endian_blocksize_16_predictor_standard_golden.tif",
    "cog_sparse_strile_arrays_zeroified_when_possible.tif",
    "float64_LZW_predictor_2.tif",
    "strip_larger_than_2GB_header.tif",
    "tiff_with_subifds.tif",
    "byte_buggy_packbits.tif",
    "twoimages.tif",
    "one_strip_nobytecount.tif",
    "leak-ZIPSetupDecode.tif",
    "byte_truncated.tif",
    "byte_zstd_corrupted.tif",
    "byte_zstd_corrupted2.tif",
]
failures = (
    xfail_byte_counts
    + xfail_compression
    + xfail_assert
    + xfail_dtype
    + xfail_other
    + xfail_panic
    + xfail_samples_per_pixel
    + xfail_reshape
)
