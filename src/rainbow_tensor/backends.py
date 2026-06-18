"""Value access helpers for array-like backends.

The core renderer only needs a shape and a value at a coordinate. Backends
mostly agree on ``.shape``, but scalar extraction varies. These helpers keep
that duck typing in one small place so the notebook layer stays readable.
"""


def value_at_coordinate(array, coord):
    """Return the Python value at ``coord`` from an array-like object."""
    value = _index_value(array, coord)
    return scalar_value(value)


def _index_value(array, coord):
    """Index ``array`` by a coordinate tuple with a chained fallback."""
    try:
        return array[coord]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        value = array
        try:
            for axis_index in coord:
                value = value[axis_index]
        except (IndexError, KeyError, TypeError, ValueError):
            raise exc
        return value


def scalar_value(value):
    """Convert a backend scalar to a plain Python value where possible."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass

    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        try:
            return scalar_value(numpy())
        except (TypeError, ValueError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            listed = tolist()
        except (TypeError, ValueError):
            pass
        else:
            if not isinstance(listed, list):
                return listed

    return value
