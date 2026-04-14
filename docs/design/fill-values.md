# Fill Value Handling

This document describes how virtual-tiff handles fill values when mapping TIFF metadata to the Zarr data model, and the design decisions behind the approach.

## The problem

TIFF files — especially those converted from NetCDF via GDAL — can carry fill value information in multiple redundant places, all stored as strings:

| Source | TIFF location | Example |
|--------|---------------|---------|
| `gdal_no_data` | GDAL_NODATA tag (42113) | `"-9999"` |
| `_FillValue` | GDAL metadata XML (tag 42112) | `"-9999"` |
| `missing_value` | GDAL metadata XML (tag 42112) | `"-9999"` |
| Per-variable prefixed | GDAL metadata XML | `"swe#_FillValue"`, `"swe#missing_value"` |

Meanwhile, the Zarr data model has a single, typed `fill_value` field on each array, and CF-aware readers like xarray expect `_FillValue` attributes to be encoded in a specific format.

Naively passing the TIFF's string attributes through to Zarr causes several failures:

1. **Type mismatch**: xarray's `FillValueCoder.decode()` expects base64-encoded bytes for float `_FillValue` attributes, not a plain string like `"-9999"`. Passing the raw string causes a `struct.error`.
2. **Zarr fill_value defaults to zero**: Without parsing the nodata value, the Zarr `fill_value` is set to the dtype default (e.g., `0.0`), which is semantically wrong — uninitialized chunks should return the nodata value, not zero.
3. **Conflicting attributes**: If `_FillValue` and `missing_value` are both present with inconsistent encoding, xarray raises a "multiple fill values" warning or error.

## Two distinct fill value concepts

The Zarr and CF data models use "fill value" for two different purposes:

**Zarr `fill_value`** (storage-level)
:   The default value returned for uninitialized or missing chunks. Set in `ArrayV3Metadata.fill_value`. Zarr handles its own serialization — no special encoding needed.

**CF `_FillValue` attribute** (data-level)
:   A sentinel value that CF-aware readers (xarray, rioxarray) use to mask individual data points as missing within chunks that _do_ contain data. Stored as an array attribute and must be encoded for xarray's `FillValueCoder`.

In the original NetCDF files, these are often the same value (e.g., `-9999`), serving both roles. The TIFF conversion collapses both into string metadata, losing the distinction. Virtual-tiff must reconstruct it.

## The conversion chain

Understanding the data flow explains why type information is lost:

```
NetCDF (typed attributes)
  _FillValue: float32(-9999.0)
  missing_value: float32(-9999.0)
      │
      ▼  GDAL conversion (gdal_translate, rio.to_raster, etc.)
GeoTIFF (string metadata)
  GDAL_NODATA tag: "-9999"          ← ASCII string
  GDAL metadata XML:
    _FillValue: "-9999"             ← text in XML
    missing_value: "-9999"          ← text in XML
    swe#_FillValue: "-9999"         ← per-variable duplicate
      │
      ▼  virtual-tiff parser
Virtual Zarr store
  fill_value: float32(-9999.0)      ← properly typed
  _FillValue: "AAAAAICHw8A="       ← base64-encoded for xarray
  missing_value: -9999.0            ← plain numeric for xarray
```

## Consolidation approach

The `_consolidate_fill_value` function in `parser.py` handles this mapping. It runs after `_get_attributes()` extracts the raw TIFF metadata and before `ArrayV3Metadata` is constructed.

### Step 1: Extract

Collect string values from the three known fill value attribute keys (`_FillValue`, `missing_value`, `gdal_no_data`). Only string values are collected — if an attribute is already numeric or encoded, it is left untouched.

### Step 2: Parse

Each string value is normalized and cast to the array's numpy dtype. MSVC-style infinity and NaN representations (e.g., `"-1.#INF"`, `"1.#QNAN"`, `"-1.#IND"`) are normalized to their standard forms (`"-inf"`, `"nan"`) before casting. The cast uses `np.dtype(dtype).type(value)` — for example, `"-9999"` for a `float32` array becomes `np.float32(-9999.0)`.

If a value cannot be represented in the target dtype (e.g., a GDAL_NODATA value of `-32768` on a uint8 array), a `ValueError` is raised. This is treated as an error rather than silently ignored because an out-of-range nodata value indicates a mismatch between the file's metadata and its storage dtype that should be investigated.

### Step 3: Validate

All parsed values are compared. If they disagree, a warning is emitted. This can happen when:

- The source file used different nodata values for storage vs. masking
- A conversion tool introduced a discrepancy
- Precision was lost in the string round-trip (e.g., `float32` → string → `float64`)

### Step 4: Select authoritative value

Priority order:

1. **`gdal_no_data`** — the TIFF-native nodata value from the GDAL_NODATA tag. This is the most direct representation of the file's intended nodata.
2. **`_FillValue`** — CF convention attribute, reconstructed from GDAL metadata XML.
3. **`missing_value`** — older CF convention attribute.

The selected value becomes both the Zarr `fill_value` and the basis for encoded attributes.

### Step 5: Encode and clean up

- **`_FillValue`** is encoded via `FillValueCoder.encode()` (e.g., base64 for floats) because xarray's Zarr backend decodes this attribute through `FillValueCoder.decode()`.
- **`missing_value`** is set as a plain Python numeric (via `.item()`) because xarray passes this attribute through without decoding — it must already be a native numeric type.
- **`gdal_no_data`** is left as the original string, since neither xarray nor rioxarray process it.
- **Per-variable prefixed duplicates** (e.g., `swe#_FillValue`, `swe#missing_value`) are removed when they carry the same value as the top-level attribute, to reduce clutter.

## Downstream reader compatibility

### xarray

xarray recognizes `_FillValue` and `missing_value` as CF masking attributes. Its Zarr backend decodes `_FillValue` via `FillValueCoder` but passes `missing_value` through as-is. Both must decode to the same numeric value, or xarray raises a "multiple fill values" warning.

The `use_zarr_fill_value_as_mask` option (default: `None`/`False` for `open_zarr`) controls whether the Zarr-level `fill_value` is also treated as a masking sentinel. With our approach, the Zarr `fill_value` and the decoded `_FillValue` attribute are the same value, so both paths produce correct results.

### rioxarray

rioxarray's `nodata` property checks attributes in this order: `_FillValue`, `missing_value`, `fill_value`, `nodata`, then falls back to rasterio's `DatasetReader.nodata`. It does **not** recognize `gdal_no_data` by name. When data is accessed through a virtual Zarr store (no rasterio file manager), rioxarray relies entirely on the attribute chain — so properly encoding `_FillValue` is essential.

### xarray's `FillValueCoder` dtype support

Not all dtypes are supported by xarray's `FillValueCoder`. The vendored `FillValueCoder` in virtual-tiff supports the following types:

| dtype kind | Encoding format |
|------------|----------------|
| `f` (float) | base64-encoded little-endian double |
| `c` (complex) | list of two base64-encoded little-endian doubles `[real, imag]` |
| `iu` (integer) | Python `int` |
| `b` (boolean) | Python `bool` |
| `U` (unicode string) | Python `str` |
| `S` (byte string) | base64-encoded bytes |

Complex dtype support has been added to the vendored `FillValueCoder` but has not yet been merged into xarray's released version. Structured, compound, and datetime dtypes are not supported and will raise a `ValueError` during encoding or decoding.

## When no fill value is present

If none of the three fill value keys are present as string attributes, `_consolidate_fill_value` returns `None` and the caller falls back to the dtype's default scalar (e.g., `0.0` for `float32`, `0` for `int16`). This is correct for TIFF files that have no nodata concept — the Zarr `fill_value` is only meaningful for uninitialized chunks, and no `_FillValue` attribute is emitted.
