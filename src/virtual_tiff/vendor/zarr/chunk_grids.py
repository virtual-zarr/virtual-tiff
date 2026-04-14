# Vendored from zarr-python (zarr.core.chunk_grids._is_rectilinear_chunks)
# for compatibility with zarr versions that don't include rectilinear chunk support.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeGuard


def _is_rectilinear_chunks(chunks: Any) -> TypeGuard[Sequence[Sequence[int]]]:
    """Check if chunks is a nested sequence (e.g. [[10, 20], [5, 5]]).

    Returns True for inputs like [[10, 20], [5, 5]] or [(10, 20), (5, 5)].
    Returns False for flat sequences like (10, 10) or [10, 10].
    """
    if isinstance(chunks, (str, int)):
        return False
    if not hasattr(chunks, "__iter__"):
        return False
    try:
        first_elem = next(iter(chunks), None)
        if first_elem is None:
            return False
        return hasattr(first_elem, "__iter__") and not isinstance(
            first_elem, (str, bytes, int)
        )
    except (TypeError, StopIteration):
        return False
