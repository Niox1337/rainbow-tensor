"""Index validation, slice expansion, and selection computation.

This module validates indexing expressions, expands slices into integer
positions, computes the selected coordinates, and computes the result shape
after indexing. Basic indexing supports negative integer indices, negative
slice bounds and steps, ``Ellipsis`` (``...``), and ``None`` (newaxis).
Advanced indexing supports a full-shape boolean mask, integer index arrays
of any shape that broadcast together, and per-axis boolean arrays that act
like their nonzero integer arrays; see :func:`advanced_index`.
"""

import itertools
from operator import index as integer_index

from .explanations import t
from .shape import coordinates, format_shape


def _resolve_int(entry, size, axis):
    """Resolve a possibly negative integer index against an axis size."""
    resolved = entry + size if entry < 0 else entry
    if not 0 <= resolved < size:
        raise IndexError(
            f"axis {axis}: index {entry} is out of range for size {size}"
        )
    return resolved


def _shape_of(obj):
    """Read an array shape or validate the rectangular shape of a nested list.

    Every sibling must have the same shape. Inferring only from the first
    child would silently drop values from longer siblings in a ragged input.
    """
    if hasattr(obj, "shape"):
        return tuple(integer_index(d) for d in obj.shape)
    if not isinstance(obj, list):
        return ()
    if not obj:
        return (0,)
    child_shape = _shape_of(obj[0])
    if any(_shape_of(child) != child_shape for child in itertools.islice(obj, 1, None)):
        raise ValueError("index arrays must be rectangular, got a ragged nested list")
    return (len(obj),) + child_shape


def _get(obj, coord):
    """Read ``obj[coord]`` for a tuple coord, for arrays and nested lists."""
    for i in coord:
        obj = obj[i]
    return obj


def _is_array_like(entry):
    """True for an index-array entry: a list or a shaped (non scalar) object.

    Ints, slices, ``None``, and ``Ellipsis`` are not array entries.
    """
    if isinstance(entry, list):
        return True
    return hasattr(entry, "shape") and _shape_of(entry) != ()


def _is_bool(value):
    """True for a Python bool or a NumPy boolean scalar, without importing it."""
    return isinstance(value, bool) or type(value).__name__.startswith("bool")


def _first_leaf(entry):
    """Drill into a nested array-like and return its first scalar element."""
    cur = entry
    while _is_array_like(cur):
        if len(cur) == 0:
            return None
        cur = cur[0]
    return cur


def _is_bool_array(entry):
    """Identify boolean arrays even when empty, without importing a backend.

    A declared dtype preserves the kind of an empty array. Lists have no dtype,
    so every leaf must be boolean. Mixed lists such as ``[True, 2]`` are integer
    indices rather than masks.
    """
    if not _is_array_like(entry):
        return False
    dtype_kind = getattr(getattr(entry, "dtype", None), "kind", None)
    if dtype_kind is not None:
        return dtype_kind == "b"
    return _is_bool(_first_leaf(entry)) and all(
        _is_bool(_get(entry, coord)) for coord in coordinates(_shape_of(entry))
    )


def validate_index(index, shape):
    """Validate and normalize an index tuple against a tensor shape.

    The index must be a tuple. Entries may be integers (negative allowed),
    slices, a single ``Ellipsis``, or ``None`` (newaxis). The returned tuple
    has the ellipsis expanded into full slices and integers resolved to their
    non-negative position, with ``None`` kept in place to mark an inserted
    size 1 axis. Integer scalars follow Python's integer index protocol. Omitted
    trailing axes are filled with full slices, matching NumPy indexing.
    """
    if not isinstance(index, tuple):
        raise TypeError(
            f"index must be a tuple, got {type(index).__name__}. "
            f"For a single axis use a one element tuple such as (0,)"
        )

    if sum(e is Ellipsis for e in index) > 1:
        raise IndexError("an index can have at most one ellipsis '...'")

    normalized = []
    for entry in index:
        if entry is not None and entry is not Ellipsis and not isinstance(entry, slice):
            if not _is_bool(entry):
                try:
                    entry = integer_index(entry)
                except TypeError:
                    pass  # The validation below reports the unsupported entry.
        normalized.append(entry)
    index = tuple(normalized)

    consuming = sum(
        isinstance(e, slice) or (isinstance(e, int) and not isinstance(e, bool))
        for e in index
    )
    if consuming > len(shape):
        raise IndexError(
            f"too many indices for tensor of rank {len(shape)}: "
            f"{consuming} axes were indexed"
        )
    fill = [slice(None)] * (len(shape) - consuming)

    tokens = []
    axis = 0
    for entry in index:
        if entry is Ellipsis:
            tokens.extend(fill)
            axis += len(fill)
        elif entry is None:
            tokens.append(None)
        elif _is_bool(entry):
            raise TypeError(f"axis {axis}: boolean indices are not supported")
        elif isinstance(entry, int):
            tokens.append(_resolve_int(entry, shape[axis], axis))
            axis += 1
        elif isinstance(entry, slice):
            tokens.append(entry)
            axis += 1
        else:
            raise TypeError(
                f"axis {axis}: unsupported index entry {entry!r}. "
                f"Only integers, slices, Ellipsis, and None are supported"
            )

    tokens.extend(slice(None) for _ in range(len(shape) - axis))
    return tuple(tokens)


def expand_slice(sl, size):
    """Expand a slice into the list of integer positions it selects.

    Negative bounds and negative steps are handled by ``slice.indices``.
    """
    return list(range(*sl.indices(size)))


def _range_length(positions):
    """Count a range with Python integers, without the platform limit of ``len``.

    Shape metadata can exceed ``sys.maxsize`` even when a later zero dimension
    makes the whole tensor empty. No positions are enumerated to obtain the count.
    """
    distance = positions.stop - positions.start
    if positions.step < 0:
        distance = -distance
    stride = abs(positions.step)
    return max(0, (distance + stride - 1) // stride)


def selected_coordinates(shape, index):
    """Compute every coordinate selected by ``index`` in row-major order.

    ``None`` entries insert a new axis in the result and select nothing in the
    source tensor, so they are skipped here. Empty selections return before
    expanding any axis, even when other dimensions are very large.
    """
    tokens = validate_index(index, shape)
    axes = []
    axis = 0
    for tok in tokens:
        if tok is None:
            continue
        size = shape[axis]
        axis += 1
        if isinstance(tok, int):
            axes.append(range(tok, tok + 1))
        else:
            axes.append(range(*tok.indices(size)))
    if any(not axis for axis in axes):
        return []
    return list(itertools.product(*axes))


def result_shape(shape, index):
    """Compute the result shape after indexing.

    Integer entries remove their axis. Slice entries keep their axis with a
    size equal to the number of selected positions. ``None`` inserts a size 1
    axis where it appears.
    """
    tokens = validate_index(index, shape)
    out = []
    axis = 0
    for tok in tokens:
        if tok is None:
            out.append(1)
            continue
        size = shape[axis]
        axis += 1
        if isinstance(tok, slice):
            out.append(_range_length(range(*tok.indices(size))))
    return tuple(out)


def format_slice(sl):
    """Format a slice the way it would appear in indexing notation."""
    if sl.start is None and sl.stop is None and sl.step is None:
        return ":"
    start = "" if sl.start is None else str(sl.start)
    stop = "" if sl.stop is None else str(sl.stop)
    if sl.step is None:
        return f"{start}:{stop}"
    return f"{start}:{stop}:{sl.step}"


def format_token(entry):
    """Format a single raw index entry as it would appear in notation."""
    if isinstance(entry, slice):
        return format_slice(entry)
    if entry is None:
        return "None"
    if entry is Ellipsis:
        return "..."
    if _is_array_like(entry):
        return _format_array(entry)
    return str(entry)


def _format_array(entry):
    """Format a possibly nested integer index array as ``[[0, 1], [2, 3]]``."""
    if _is_array_like(entry):
        return "[" + ", ".join(_format_array(entry[i]) for i in range(len(entry))) + "]"
    if _is_bool(entry):
        return "True" if entry else "False"
    return str(int(entry))


def format_index(index):
    """Format an index tuple as a readable string such as ``0, :, 1``."""
    if not index:
        return "()"
    return ", ".join(format_token(entry) for entry in index)


def explain_index(shape, index):
    """Build the lines explaining how an index transforms a shape."""
    tokens = validate_index(index, shape)
    lines = [
        t("common.original_shape", shape=format_shape(shape)),
        t("common.index", index=format_index(index)),
        t("common.result_shape", shape=format_shape(result_shape(shape, index))),
    ]
    axis = 0  # source axis
    out = 0  # result axis position
    for tok in tokens:
        if tok is None:
            lines.append(t("index.new_axis", pos=out))
            out += 1
            continue
        if isinstance(tok, int):
            lines.append(t("index.axis_removed", axis=axis, index=tok))
        else:
            lines.append(t("index.axis_kept", axis=axis, slice=format_slice(tok)))
            out += 1
        axis += 1
    return lines


def is_advanced(index, shape):
    """True when ``index`` uses a boolean mask or integer index arrays."""
    if not isinstance(index, tuple):
        return _is_array_like(index)
    return any(_is_array_like(e) for e in index)


def advanced_index(shape, index):
    """Materialize unique source highlights for a supported advanced index.

    Returns ``(selected_coords, result_shape, explanation_lines)`` and retains
    the historical list return type. Arrays broadcast with NumPy's axis placement,
    including integer scalars, slices, ellipsis, and ``None``. Boolean masks may
    span consecutive source axes. Boolean scalar indices remain unsupported.

    This compatibility helper explicitly enumerates the source selection. Views
    should use :class:`rainbow_tensor.index_mapping.IndexMapping` to resolve
    individual result coordinates and query compact highlights without expanding
    slice products. Repeated output picks are preserved by that mapping separately.
    """
    from .index_mapping import IndexMapping

    mapping = IndexMapping(shape, index)
    return list(mapping.selection), mapping.result_shape, mapping.explanation
