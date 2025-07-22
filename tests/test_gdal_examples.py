import pytest
from conftest import (
    gdal_autotest_examples,
    gdal_gcore_examples,
    resolve_folder,
    rioxarray_comparison,
)
from virtualizarr import open_virtual_dataset
from virtual_tiff import VirtualTIFF
from obstore.store import LocalStore
from virtualizarr.registry import ObjectStoreRegistry


def match_error(filepath, error, match):
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    with pytest.raises(
        error,
        match=match,
    ):
        open_virtual_dataset(
            url=f"file://{filepath}", registry=registry, parser=VirtualTIFF()
        )


def run_gdal_test(filename, filepath):
    if filename in skip:
        pytest.xfail("Known failure")
    filepath = f"{resolve_folder(filepath)}/{filename}"
    if filename in unknown_compressor:
        match_error(
            filepath,
            ValueError,
            r"TIFF has compressor tag .+, which is not recognized\. Please raise an issue for support\.",
        )
    elif filename in jpeg_tables:
        match_error(
            filepath,
            NotImplementedError,
            "JPEG compression with quantization tables is not yet supported.",
        )
    elif filename in YCbCr:
        match_error(
            filepath,
            NotImplementedError,
            "YCbCr PhotometricInterpretation is not yet supported.",
        )
    elif filename in partial_chunks:
        match_error(
            filepath,
            ValueError,
            r"Zarr's default chunk grid expects all chunks to be equal size, but this TIFF has an image height of (.*?)",
        )
    elif filename in byte_counts:
        match_error(
            filepath,
            NotImplementedError,
            "TIFFs without byte counts and offsets aren't supported",
        )
    elif filename in dtype:
        match_error(
            filepath,
            ValueError,
            r"Unrecognized datatype, got sample_format (.*?)",
        )
    elif filename in nested:
        match_error(
            filepath,
            NotImplementedError,
            r"Nested grids are not supported, but file has (.*?)",
        )
    else:
        rioxarray_comparison(f"file://{filepath}")


@pytest.mark.parametrize("filename", gdal_autotest_examples())
def test_against_rioxarray_gdal_autotest(filename):
    run_gdal_test(filename, "tests/dvc/gdal_autotest")


@pytest.mark.parametrize("filename", gdal_gcore_examples())
def test_against_rioxarray_gdal_gcore(filename):
    run_gdal_test(filename, "tests/dvc/gdal_gcore")


corrupted = [
    "lzw_corrupted.tif",
    "byte_buggy_packbits.tif",
    "byte_zstd_corrupted2.tif",
    "byte_zstd_corrupted.tif",
    "unsupported_codec_jp2000.tif",
]
jpeg_tables = [
    "rgbsmall_JPEG.tif",
    "byte_JPEG_tiled.tif",
    "byte_JPEG.tif",
    "byte_ovr_jpeg_tablesmode1.tif",
    "rgbsmall_JPEG_tiled.tif",
    "rgbsmall_JPEG_tiled_separate.tif",
    "byte_ovr_jpeg_tablesmode_not_correctly_set_on_ovr.tif",
    # "tif_jpeg_too_big_last_stripe.tif",
    "byte_ovr_jpeg_tablesmode2.tif",
    "byte_ovr_jpeg_tablesmode3.tif",
    "irregular_tile_size_jpeg_in_tiff.tif",
    "byte_jpg_unusual_jpegtable.tif",
    # "stefan_full_rgba_jpeg_contig.tif",
    # "stefan_full_rgba_jpeg_separate.tif",
    "rgbsmall_JPEG_separate.tif",
]
unknown_compressor = [
    "unsupported_codec_unknown.tif",
    "thunder.tif",
    "next_default_case.tif",
    "next_literalspan.tif",
    "slim_g4.tif",
    "scanline_more_than_2GB.tif",
    "next_literalrow.tif",
]
YCbCr = [
    "rgbsmall_JPEG_ycbcr.tif",
    "zackthecat_corrupted.tif",
    # "tif_jpeg_ycbcr_too_big_last_stripe.tif",
    "sasha.tif",
    "zackthecat.tif",
    "ycbcr_with_mask.tif",
    "mandrilmini_12bitjpeg.tif",
    "ycbcr_42_lzw.tif",
    "ycbcr_21_lzw.tif",
    "ycbcr_44_lzw_optimized.tif",
    "ycbcr_42_lzw_optimized.tif",
    "ycbcr_41_lzw.tif",
    "ycbcr_22_lzw.tif",
    "ycbcr_14_lzw.tif",
    "ycbcr_11_lzw.tif",
    "ycbcr_44_lzw.tif",
    "ycbcr_24_lzw.tif",
    "ycbcr_12_lzw.tif",
]
slow_tests = [
    "bug1488.tif",
]
nested = ["test_hgrid_with_subgrid.tif"]
partial_chunks = [
    "isis3_geotiff.tif",
    "bug_6526_input.tif",
    "rgbsmall_uint16_LZW_predictor_2.tif",
    "stefan_full_greyalpha_uint16_LZW_predictor_2.tif",
    "pyramid_shaded_ref.tif",
    "melb-small.tif",
    "utmsmall.tif",
    "n43.tif",
    "test_gdal2tiles_exclude_transparent.tif",
    "unstable_rpc_with_dem_source.tif",
    "warp_52_dem.tif",
    "rgbsmall_uint32_LZW_predictor_2.tif",
    "rgbsmall_uint64_LZW_predictor_2.tif",
    "utilities_utmsmall.tif",
    "stefan_full_rgba_LZW_predictor_2.tif",
    "stefan_full_rgba_photometric_rgb.tif",
    "dstsize_larger_than_source.tif",
    "sstgeo.tif",
    "tif_jpeg_ycbcr_too_big_last_stripe.tif",
    "test_gf.tif",
    "utmsmall.tif",
    "stefan_full_rgba_jpeg_contig.tif",
    "stefan_full_rgba_jpeg_separate.tif",
    "vrtmisc16_tile2.tif",
    "tif_jpeg_too_big_last_stripe.tif",
    "transformer_13_dem.tif",
    "vrtmisc16_tile1.tif",
    "stefan_full_rgba.tif",
    "VH.tif",
    "VV.tif",
    "geog_arc_second.tif",
    "rgbsmall_int16_bigendian_lzw_predictor_2.tif",
    "quad-lzw-old-style.tif",
]
byte_counts = [
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
    "huge_line.tif",
    "sparse_tiled_contig.tif",
    "projection_from_esri_xml.tif",
    "cog_strile_arrays_zeroified_when_possible.tif",
    "one_strip_nobytecount.tif",
    "rgbsmall_cmyk.tif",
    "sparse_tiled_separate.tif",
    "toomanyblocks.tif",
]
dtype = [
    "cint32.tif",
    "cint32_big_endian.tif",
    "int24.tif",
    "cint_sar.tif",
    "float24.tif",
    "uint33.tif",
    "cint16.tif",
    "complex_int32.tif",
]
xfail_int64 = [
    "int64_full_range.tif",
    "uint64_full_range.tif",
    "uint64.tif",
    "int64.tif",
    "rgbsmall_uint64_LZW_predictor_2.tif",
]
xfail_pred2 = ["float32_LZW_predictor_2.tif"]
# Generated with the assistance of Claude
xfail_byte_range = [
    "strip_larger_than_2GB_header.tif",
    "byte_truncated.tif",
    "cog_sparse_strile_arrays_zeroified_when_possible.tif",
    "tiled_bad_offset.tif",
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

xfail_gdal_cannot_read = [
    "uint64_LZW_predictor_2.tif",
    "stefan_full_greyalpha_uint64_LZW_predictor_2.tif",
    "float64_LZW_predictor_2.tif",
    "leak-ZIPSetupDecode.tif",
    # Cannot open TIFF file due to missing codec JXL
    "rgbsmall_JXL_tiled.tif",
    "byte_jxl_dng_1_7_52546.tif",
    "byte_jxl_deprecated_50002.tif",
    "byte_JXL_tiled.tif",
    "rgbsmall_JXL.tif",
    "byte_JXL.tif",
    "jxl-rgbi.tif",
    "rgbsmall_JXL_tiled_separate.tif",
    "rgbsmall_JXL_separate.tif",
]
xfail_subifd = [
    "tiff_with_subifds.tif",
]
xfail_photometric = [
    "cielab.tif",
    "int12_ycbcr_contig.tif",
]
xfail_reshape = [
    "webp_lossless_rgba_alpha_fully_opaque.tif",
    "geos_vrtwarp.tif",
    "excessive-memory-TIFFFillTile.tif",
    "oddsize1bit.tif",
    "huge4GB.tif",
    "toomanyblocks_separate.tif",
    "size_of_stripbytecount_at_1_and_lower_than_stripcount.tif",
    "size_of_stripbytecount_lower_than_stripcount.tif",
    "int10.tif",
    "int12.tif",
    "1bit_2bands.tif",
    "oddsize_1bit2b.tif",
    "contig_tiled.tif",
    "separate_tiled.tif",
]
skip = (
    slow_tests
    + corrupted
    + xfail_byte_range
    + xfail_pred2
    + xfail_panic
    + xfail_gdal_cannot_read
    + xfail_subifd
    + xfail_photometric
    + xfail_reshape
    + xfail_int64
)
