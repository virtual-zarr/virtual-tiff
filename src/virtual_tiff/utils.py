from __future__ import annotations

import xml.etree.ElementTree as ET


def gdal_metadata_to_dict(xml_string: str) -> dict[str, str]:
    """
    Convert GDAL metadata XML to a dictionary.
    """
    root = ET.fromstring(xml_string)
    metadata_dict = {}
    for item in root.findall("Item"):
        name = item.get("name")
        if name:
            value = item.text.strip() if item.text else ""
            metadata_dict[name] = value
    return metadata_dict


def check_no_partial_strips(image_height: int, rows_per_strip: int):
    """Check that there are no partial chunks based on the image height and rows per strip"""
    if image_height % rows_per_strip > 0:
        raise ValueError(
            "Zarr's default chunk grid expects all chunks to be equal size, but this TIFF has an image height of "
            f"{image_height} which isn't evenly divisible by its rows per strip {rows_per_strip}. "
            "See https://github.com/developmentseed/virtual-tiff/issues/24 for more details."
        )
