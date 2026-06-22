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
from math import prod

from .explanations import t
from .ops import broadcast_result_shape, broadcast_source_coord
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
    """Best-effort shape of an array-like or a nested list."""
    if hasattr(obj, "shape"):
        return tuple(int(d) for d in obj.shape)
    dims = []
    cur = obj
    while isinstance(cur, list):
        dims.append(len(cur))
        cur = cur[0] if cur else None
    return tuple(dims)


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
    """True for an index-array entry whose elements are booleans."""
    return _is_array_like(entry) and _is_bool(_first_leaf(entry))


def validate_index(index, shape):
    """Validate and normalize an index tuple against a tensor shape.

    The index must be a tuple. Entries may be integers (negative allowed),
    slices, a single ``Ellipsis``, or ``None`` (newaxis). The returned tuple
    has the ellipsis expanded into full slices and integers resolved to their
    non-negative position, with ``None`` kept in place to mark an inserted
    size 1 axis. The axis consuming entries (integers and slices) must cover
    the tensor rank exactly.
    """
    if not isinstance(index, tuple):
        raise TypeError(
            f"index must be a tuple, got {type(index).__name__}. "
            f"For a single axis use a one element tuple such as (0,)"
        )

    if sum(e is Ellipsis for e in index) > 1:
        raise IndexError("an index can have at most one ellipsis '...'")

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
        elif isinstance(entry, bool):
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

    if axis != len(shape):
        raise ValueError(
            f"index covers {axis} axes but tensor has rank {len(shape)}. "
            f"Use '...' to fill the remaining axes"
        )
    return tuple(tokens)


def expand_slice(sl, size):
    """Expand a slice into the list of integer positions it selects.

    Negative bounds and negative steps are handled by ``slice.indices``.
    """
    return list(range(*sl.indices(size)))


def selected_coordinates(shape, index):
    """Compute every coordinate selected by ``index`` in row-major order.

    ``None`` entries insert a new axis in the result and select nothing in the
    source tensor, so they are skipped here.
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
            axes.append([tok])
        else:
            axes.append(expand_slice(tok, size))
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
            out.append(len(expand_slice(tok, size)))
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
    """Compute selection, result shape, and explanation for advanced indexing.

    Returns ``(selected_coords, result_shape, explanation_lines)``. A standalone
    boolean mask of matching shape selects every True position. Integer index
    arrays gather coordinates and follow NumPy: the gathered axis stays in place
    when the advanced axes are contiguous and moves to the front when a slice
    separates them. Index arrays may have any shape and broadcast against each
    other, with the broadcast shape forming the gathered block. A boolean array
    on one or more consecutive axes acts like its nonzero integer arrays.
    """
    if not isinstance(index, tuple):
        return _mask_index(shape, index)
    return _array_index(shape, index)


def _mask_index(shape, mask):
    mshape = _shape_of(mask)
    if mshape != tuple(shape):
        raise IndexError(
            f"boolean mask shape {mshape} does not match tensor shape {tuple(shape)}"
        )
    selected = []
    for coord in coordinates(shape):
        value = _get(mask, coord)
        if not _is_bool(value):
            raise TypeError(
                "a standalone array index must be a boolean mask; got value of "
                f"type {type(value).__name__}"
            )
        if value:
            selected.append(coord)
    result = (len(selected),)
    plural = "s" if len(selected) != 1 else ""
    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        t("index.mask"),
        t("index.mask_keeps", count=len(selected), plural=plural),
        t("common.result_shape", shape=format_shape(result)),
    ]
    return selected, result, explanation


def _array_index(shape, index):
    ndim = len(shape)
    if sum(e is Ellipsis for e in index) > 1:
        raise IndexError("an index can have at most one ellipsis '...'")
    if any(e is None for e in index):
        raise IndexError("newaxis (None) is not supported with advanced indexing")

    def consumes(e):
        # A boolean array consumes one axis per dimension, like its nonzero arrays.
        if _is_bool_array(e):
            return len(_shape_of(e))
        if e is Ellipsis or e is None:
            return 0
        return 1

    consuming = sum(consumes(e) for e in index)
    if consuming > ndim:
        raise IndexError(
            f"too many indices for tensor of rank {ndim}: {consuming} axes indexed"
        )
    fill = ndim - consuming

    # One ("int"|"slice"|"arr", value) entry per source axis, in axis order.
    entries = []
    axis = 0
    for e in index:
        if e is Ellipsis:
            entries.extend(("slice", slice(None)) for _ in range(fill))
            axis += fill
        elif isinstance(e, bool):
            raise TypeError(f"axis {axis}: boolean scalar indices are not supported")
        elif isinstance(e, int):
            entries.append(("int", _resolve_int(e, shape[axis], axis)))
            axis += 1
        elif isinstance(e, slice):
            entries.append(("slice", e))
            axis += 1
        elif _is_bool_array(e):
            # A boolean array spanning d axes acts like its nonzero integer
            # arrays: d paired index arrays selecting each True position.
            bshape_ = _shape_of(e)
            d = len(bshape_)
            expected = tuple(shape[axis:axis + d])
            if bshape_ != expected:
                raise IndexError(
                    f"boolean index on axes {axis}..{axis + d - 1} has shape "
                    f"{format_shape(bshape_)} but the tensor axes are "
                    f"{format_shape(expected)}"
                )
            nz = [c for c in coordinates(bshape_) if _get(e, c)]
            for col in range(d):
                resolved = {(k,): nz[k][col] for k in range(len(nz))}
                entries.append(("arr", ((len(nz),), resolved)))
            axis += d
        elif _is_array_like(e):
            ishape = _shape_of(e)
            resolved = {}
            for c in coordinates(ishape):
                resolved[c] = _resolve_int(int(_get(e, c)), shape[axis], axis)
            entries.append(("arr", (ishape, resolved)))
            axis += 1
        else:
            raise TypeError(f"axis {axis}: unsupported index entry {e!r}")

    if axis != ndim:
        raise ValueError(
            f"index covers {axis} axes but tensor has rank {ndim}. "
            f"Use '...' to fill the remaining axes"
        )

    arr_axes = [a for a, (kind, _) in enumerate(entries) if kind == "arr"]
    # NumPy keeps the gathered axis where the advanced block is when the
    # advanced axes are contiguous, but moves it to the front when a slice
    # separates them. An int between arrays does not count as a separator.
    between = entries[arr_axes[0]: arr_axes[-1] + 1]
    contiguous = not any(kind == "slice" for kind, _ in between)

    # The index arrays broadcast together to one block of axes, just like NumPy.
    ishapes = [entries[a][1][0] for a in arr_axes]
    try:
        bshape = broadcast_result_shape(ishapes)
    except ValueError:
        raise IndexError(
            f"index arrays could not be broadcast together: shapes "
            f"{[format_shape(s) for s in ishapes]}"
        ) from None

    slice_axes = [a for a, (kind, _) in enumerate(entries) if kind == "slice"]
    slice_pos = {a: expand_slice(entries[a][1], shape[a]) for a in slice_axes}

    if contiguous:
        result = []
        placed = False
        for axis_, (kind, _) in enumerate(entries):
            if kind == "slice":
                result.append(len(slice_pos[axis_]))
            elif kind == "arr" and not placed:
                result.extend(bshape)
                placed = True
        result_shape_ = tuple(result)
    else:
        result_shape_ = tuple(bshape) + tuple(len(slice_pos[a]) for a in slice_axes)

    selected = []
    seen = set()
    for bidx in coordinates(bshape):
        fixed = {}
        for a, (kind, value) in enumerate(entries):
            if kind == "int":
                fixed[a] = value
            elif kind == "arr":
                ishape, resolved = value
                fixed[a] = resolved[broadcast_source_coord(bidx, ishape)]
        for combo in itertools.product(*[slice_pos[a] for a in slice_axes]):
            full = dict(fixed)
            full.update(zip(slice_axes, combo))
            key = tuple(full[a] for a in range(ndim))
            if key not in seen:
                seen.add(key)
                selected.append(key)

    count = prod(bshape)
    axes_text = ", ".join(str(a) for a in arr_axes)
    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        t("common.index", index=format_index(index)),
        t("index.arrays"),
        t(
            "index.gather",
            axes=axes_text,
            count=count,
            plural="s" if count != 1 else "",
            shape=format_shape(bshape),
        ),
    ]
    if not contiguous:
        explanation.append(t("index.slice_separates"))
    explanation.append(t("common.result_shape", shape=format_shape(result_shape_)))
    return selected, result_shape_, explanation
