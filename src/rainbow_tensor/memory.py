"""Explain public NumPy-style storage metadata without importing an array backend."""

from operator import index

from .views.shapes import shape

_MISSING = object()


def _attribute(obj, name):
    """Read optional metadata without invoking backend conversion methods."""
    try:
        value = getattr(obj, name)
    except (AttributeError, NotImplementedError):
        return _MISSING
    return _MISSING if callable(value) else value


def _integer(value):
    if isinstance(value, bool):
        return None
    try:
        return index(value)
    except TypeError:
        return None


def _flag(flags, name):
    value = _attribute(flags, name)
    if value is _MISSING:
        try:
            value = flags[name.upper()]
        except (KeyError, TypeError, IndexError, AttributeError):
            return None
    if isinstance(value, bool):
        return value
    value = _integer(value)
    return bool(value) if value in (0, 1) else None


def _metadata(array, normalized):
    metadata = {
        "shape": normalized,
        "dtype": None,
        "itemsize": None,
        "strides": None,
        "c_contiguous": None,
        "f_contiguous": None,
        "owns_data": None,
        "has_base": None,
    }
    if isinstance(array, tuple):
        return metadata

    dtype = _attribute(array, "dtype")
    if dtype is not _MISSING and dtype is not None:
        metadata["dtype"] = str(dtype)

    itemsize = _integer(_attribute(array, "itemsize"))
    if itemsize is not None and itemsize > 0:
        metadata["itemsize"] = itemsize

    strides = _attribute(array, "strides")
    if strides is not _MISSING and strides is not None:
        try:
            strides = tuple(_integer(value) for value in strides)
        except TypeError:
            strides = ()
        if len(strides) == len(normalized) and all(value is not None for value in strides):
            metadata["strides"] = strides

    flags = _attribute(array, "flags")
    if flags is not _MISSING:
        metadata["c_contiguous"] = _flag(flags, "c_contiguous")
        metadata["f_contiguous"] = _flag(flags, "f_contiguous")
        metadata["owns_data"] = _flag(flags, "owndata")

    base = _attribute(array, "base")
    if base is not _MISSING:
        metadata["has_base"] = base is not None
    return metadata


def _yes_no(value):
    return "unknown" if value is None else ("yes" if value else "no")


def memory(array, theme=None, precision=2, renderer=None):
    """Draw a tensor and explain the storage metadata it exposes.

    Accepts an array-like object with a ``shape`` or a literal shape tuple.
    Optional metadata follows NumPy conventions: ``strides`` is a sequence
    of byte offsets, ``itemsize`` is the number of bytes per element, and
    ``flags`` reports ``c_contiguous``, ``f_contiguous``, and ``owndata``.
    Backends with different conventions are not converted or imported.
    ``theme``, ``precision``, and ``renderer`` behave as in :func:`shape`.

    Returns a :class:`~rainbow_tensor.visual.TensorVisual` containing the
    logical shape drawing, English explanations, and a ``metadata`` dictionary.
    Its keys are ``shape``, ``dtype``, ``itemsize``, ``strides``,
    ``c_contiguous``, ``f_contiguous``, ``owns_data``, and ``has_base``.
    Unavailable fields are ``None``, including storage fields for shape tuples.

    The drawing shows logical indices, not physical memory addresses. Metadata
    is reported rather than inferred from shape or values. Ownership and a
    base object do not identify which prior operation created a view or copy,
    and do not prove whether two arbitrary arrays share memory. Object dtypes
    describe element slots, not the storage of their referenced Python objects.
    """
    visual = shape(array, theme=theme, precision=precision, renderer=renderer)
    metadata = _metadata(array, visual.shape)
    lines = ["Memory layout metadata. The figure shows logical indices, not physical addresses."]
    dtype = metadata["dtype"]
    lines.append(f"Dtype: {dtype if dtype is not None else 'unknown'}.")
    itemsize = metadata["itemsize"]
    lines.append(f"Item size: {f'{itemsize} bytes' if itemsize is not None else 'unknown'}.")
    strides = metadata["strides"]
    if strides is None:
        lines.append("Byte strides: unknown.")
    else:
        lines.append(f"Byte strides: {strides}.")
        for axis, stride in enumerate(strides):
            lines.append(f"Axis {axis}: a one-index step changes the byte offset by {stride}.")
        if any(stride < 0 for stride in strides):
            lines.append("A negative stride walks backwards through storage along that axis.")
        if any(stride == 0 for stride in strides):
            lines.append("A zero stride reuses the same storage along that axis.")
    lines.append(
        f"C-contiguous: {_yes_no(metadata['c_contiguous'])}. "
        f"F-contiguous: {_yes_no(metadata['f_contiguous'])}."
    )
    lines.append(
        f"Owns data: {_yes_no(metadata['owns_data'])}. "
        f"Base object present: {_yes_no(metadata['has_base'])}."
    )
    if metadata["owns_data"] is False and metadata["has_base"] is True:
        lines.append("The reported flags and base describe a view backed by another object.")
    lines.append("These attributes do not identify which earlier operation made a view or copy.")
    visual.metadata = metadata
    visual.explanation = visual.explanation + lines
    visual.text = "\n".join(visual.explanation)
    return visual
