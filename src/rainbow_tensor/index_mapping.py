"""Lazy indexing provenance and compact source highlights.

Result coordinates retain gather order and repeated picks. Source highlights
describe unique coordinates separately, so drawing a preview never expands the
Cartesian product of selected slices or broadcasted index arrays.
"""

from collections import Counter
from itertools import zip_longest
from math import prod
from operator import index as integer_index
from typing import Protocol, runtime_checkable

from .explanations import t
from .indexing import (
    _get,
    _is_array_like,
    _is_bool,
    _is_bool_array,
    _range_length,
    _resolve_int,
    _shape_of,
    explain_index,
    format_index,
    is_advanced,
    result_shape,
)
from .ops import broadcast_result_shape, broadcast_source_coord
from .selection import BasicSelection
from .shape import flat_index, format_shape
from .tracing import _normalize_focus


@runtime_checkable
class CompactSelection(Protocol):
    """Source-highlight queries that renderers can use without enumeration."""

    def __bool__(self): ...

    def __contains__(self, coordinate): ...

    def matches_prefix(self, prefix): ...

    def axis_pins(self, axis, limit): ...


def _coordinates(shape):
    """Iterate a Cartesian shape without pooling its axis ranges."""
    for position in range(prod(shape)):
        coordinate = []
        for size in reversed(shape):
            position, value = divmod(position, size)
            coordinate.append(value)
        yield tuple(reversed(coordinate))


def _pins(positions, limit):
    """Keep bounded head and tail positions with room for intervening gaps."""
    if not positions or limit <= 0:
        return ()
    if isinstance(positions, range) and positions.step < 0:
        positions = positions[::-1]
    count = _range_length(positions) if isinstance(positions, range) else len(positions)
    contiguous = isinstance(positions, range) and positions.step == 1
    slots = count if contiguous else 2 * count - 1
    if slots <= limit:
        return tuple(positions)
    keep = max(1, limit - 1 if contiguous else (limit + 1) // 2)
    return tuple(positions[:(keep + 1) // 2]) + tuple(positions[count - keep // 2:])


class _ArrayIndex:
    """Store one index array and inverted positions in space proportional to it."""

    def __init__(self, shape, values):
        self.shape = shape
        self.values = values
        self.positions = {}
        for coordinate, value in values.items():
            self.positions.setdefault(value, []).append(coordinate)
        self.distinct_values = tuple(sorted(self.positions))

    def at(self, broadcast_coordinate):
        return self.values[broadcast_source_coord(broadcast_coordinate, self.shape)]


def _positions_compatible(constraints, broadcast_rank):
    """Join candidate positions through their shared, non-singleton axes.

    Each candidate list is scanned at most once per query to index projections
    by the axes already bound by earlier arrays. Coordinates on private axes
    cannot constrain another array and are dropped, along with duplicate
    projections. Disconnected groups of broadcast axes are checked separately,
    so their matches never form a Cartesian product. The indexes use space
    proportional to the candidate inputs, rather than to their broadcast product.

    Multiway joins can still branch over distinct compatible projections. This
    is not a linear-time guarantee for arbitrary cyclic constraints, but later
    arrays never rescan an entire candidate list for every earlier position.
    """
    if len(constraints) < 2:
        return True
    aligned = [
        (
            {
                broadcast_rank - len(array.shape) + axis: axis
                for axis, size in enumerate(array.shape) if size != 1
            },
            matches,
        )
        for array, matches in constraints
    ]
    uses = Counter(axis for axes, _ in aligned for axis in axes)
    components = []
    for axes, matches in aligned:
        shared = {axis: local for axis, local in axes.items() if uses[axis] > 1}
        if not shared:
            continue
        members = [(shared, matches)]
        member_axes = set(shared)
        separate = []
        for group, group_axes in components:
            if member_axes.isdisjoint(group_axes):
                separate.append((group, group_axes))
            else:
                members.extend(group)
                member_axes.update(group_axes)
        components = separate + [(members, member_axes)]

    def join_component(members):
        bound = set()
        indexed = []
        pending = list(members)
        while pending:
            # Resolve already-bound filters first, then extend connected axes.
            position = min(
                range(len(pending)),
                key=lambda i: (
                    not pending[i][0].keys() <= bound,
                    bound.isdisjoint(pending[i][0]),
                    len(pending[i][1]),
                ),
            )
            axes, matches = pending.pop(position)
            common = tuple(axis for axis in axes if axis in bound)
            added = tuple(axis for axis in axes if axis not in bound)
            buckets = {}
            for coordinate in matches:
                key = tuple(coordinate[axes[axis]] for axis in common)
                values = tuple(coordinate[axes[axis]] for axis in added)
                buckets.setdefault(key, set()).add(values)
            indexed.append((common, added, buckets))
            bound.update(axes)

        def compatible(position, bindings):
            if position == len(indexed):
                return True
            common, added, buckets = indexed[position]
            key = tuple(bindings[axis] for axis in common)
            for values in buckets.get(key, ()):
                current = dict(zip(added, values))
                if compatible(position + 1, bindings | current):
                    return True
            return False

        return compatible(0, {})

    return all(join_component(members) for members, _ in components)


class _AdvancedSelection:
    """Query unique source highlights without expanding selected slice products.

    Iteration is an explicit request to enumerate the selection. It follows the
    legacy gather-then-slice order and deduplicates gather groups as they appear.
    There is deliberately no ``len`` operation that would trigger that work.
    Use the mapping's ``result_count`` for the exact output size, including repeats.
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def __bool__(self):
        return self.mapping.result_count != 0

    def __contains__(self, coordinate):
        try:
            return len(coordinate) == len(self.mapping.source_shape) and self.matches_prefix(
                coordinate
            )
        except TypeError:
            return False

    def matches_prefix(self, prefix):
        """Match slice bounds and compatible index-array positions for a frame."""
        mapping = self.mapping
        if not self or len(prefix) > len(mapping.source_shape):
            return False
        try:
            prefix = tuple(integer_index(value) for value in prefix)
        except TypeError:
            return False
        constraints = []
        for axis, value in enumerate(prefix):
            if axis in mapping.fixed:
                if value != mapping.fixed[axis]:
                    return False
            elif axis in mapping.slices:
                if value not in mapping.slices[axis]:
                    return False
            else:
                array = mapping.arrays[axis]
                matches = array.positions.get(value, ())
                if not matches:
                    return False
                constraints.append((array, matches))
        return _positions_compatible(constraints, len(mapping.broadcast_shape))

    def axis_pins(self, axis, limit):
        if not self:
            return ()
        mapping = self.mapping
        if axis in mapping.fixed:
            value = mapping.fixed[axis]
            positions = range(value, value + 1)
        elif axis in mapping.slices:
            positions = mapping.slices[axis]
        else:
            positions = mapping.arrays[axis].distinct_values
        return _pins(positions, limit)

    def __iter__(self):
        if not self:
            return
        mapping = self.mapping
        array_axes = tuple(mapping.arrays)
        slice_axes = tuple(mapping.slices)
        slice_shape = tuple(_range_length(mapping.slices[axis]) for axis in slice_axes)
        seen = set()
        for broadcast_coordinate in _coordinates(mapping.broadcast_shape):
            gathered = tuple(mapping.arrays[axis].at(broadcast_coordinate) for axis in array_axes)
            if gathered in seen:
                continue
            seen.add(gathered)
            fixed = dict(mapping.fixed)
            fixed.update(zip(array_axes, gathered))
            for coordinate in _coordinates(slice_shape):
                source = dict(fixed)
                source.update(
                    (axis, mapping.slices[axis][value])
                    for axis, value in zip(slice_axes, coordinate)
                )
                yield tuple(source[axis] for axis in range(len(mapping.source_shape)))

    def __eq__(self, other):
        if not isinstance(other, (list, tuple)):
            return NotImplemented
        missing = object()
        return all(a == b for a, b in zip_longest(self, other, fillvalue=missing))


class IndexMapping:
    """Map result coordinates to source coordinates without building the result.

    ``result_shape`` and ``result_count`` describe all output positions, including
    duplicate gathers and empty results. ``source_coord`` resolves one position.
    ``selection`` separately supports compact source highlighting, while iteration
    over the mapping yields result-ordered source coordinates with repeats intact.

    Index arrays are validated and copied once. Boolean masks require a full scan
    to determine the output shape. Neither selected slices nor broadcast products
    are expanded during construction or rendering.
    """

    def __init__(self, source_shape, index):
        self.source_shape = tuple(source_shape)
        self.is_advanced = is_advanced(index, self.source_shape)
        if not self.is_advanced:
            self.selection = BasicSelection(self.source_shape, index)
            self.result_shape = result_shape(self.source_shape, index)
            self.result_count = prod(self.result_shape)
            self.explanation = explain_index(self.source_shape, index)
            return
        self.fixed = {}
        self.slices = {}
        self.arrays = {}
        if isinstance(index, tuple):
            self._parse_arrays(index)
        else:
            self._parse_mask(index)
        self.result_count = prod(self.result_shape)
        self.selection = _AdvancedSelection(self)

    def source_coord(self, result_coordinate):
        """Resolve one result coordinate, accepting negative positions on its axes."""
        if not isinstance(result_coordinate, tuple):
            raise TypeError("result coordinate must be a tuple of integers")
        if self.result_count == 0:
            raise IndexError("cannot address an empty indexing result")
        coordinate = _normalize_focus(result_coordinate, self.result_shape)
        if not self.is_advanced:
            return self.selection[flat_index(coordinate, self.result_shape)]
        start = self.broadcast_start
        broadcast_coordinate = coordinate[start:start + len(self.broadcast_shape)]
        source = []
        for axis in range(len(self.source_shape)):
            if axis in self.fixed:
                source.append(self.fixed[axis])
            elif axis in self.slices:
                source.append(self.slices[axis][coordinate[self.slice_outputs[axis]]])
            else:
                source.append(self.arrays[axis].at(broadcast_coordinate))
        return tuple(source)

    def __iter__(self):
        for coordinate in _coordinates(self.result_shape):
            yield self.source_coord(coordinate)

    def _parse_mask(self, mask):
        shape = _shape_of(mask)
        if len(shape) != len(self.source_shape) or any(
            size not in (0, expected) for size, expected in zip(shape, self.source_shape)
        ):
            raise IndexError(
                f"boolean mask shape {shape} does not match tensor shape "
                f"{self.source_shape}"
            )
        if not _is_bool_array(mask):
            raise TypeError("a standalone array index must be a boolean mask")
        positions = []
        for coordinate in _coordinates(shape):
            value = _get(mask, coordinate)
            if not _is_bool(value):
                raise TypeError("a standalone array index must be a boolean mask")
            if value:
                positions.append(coordinate)
        count = len(positions)
        for axis in range(len(self.source_shape)):
            self.arrays[axis] = _ArrayIndex((count,), {
                (i,): coordinate[axis] for i, coordinate in enumerate(positions)
            })
        self.broadcast_shape = self.result_shape = (count,)
        self.broadcast_start = 0
        self.slice_outputs = {}
        self.explanation = [
            t("common.original_shape", shape=format_shape(self.source_shape)),
            t("index.mask"),
            t("index.mask_keeps", count=count, plural="s" if count != 1 else ""),
            t("common.result_shape", shape=format_shape(self.result_shape)),
        ]

    def _parse_arrays(self, index):
        if sum(entry is Ellipsis for entry in index) > 1:
            raise IndexError("an index can have at most one ellipsis '...'")

        boolean = [_is_bool_array(entry) for entry in index]
        consuming = sum(
            len(_shape_of(entry)) if is_boolean else 1
            for entry, is_boolean in zip(index, boolean)
            if entry is not None and entry is not Ellipsis
        )
        ndim = len(self.source_shape)
        if consuming > ndim:
            raise IndexError(
                f"too many indices for tensor of rank {ndim}: {consuming} axes indexed"
            )
        fill = ndim - consuming
        entries = []
        array_specs = {}
        axis = 0

        def add_slice(token):
            nonlocal axis
            self.slices[axis] = range(*token.indices(self.source_shape[axis]))
            entries.append(("slice", axis))
            axis += 1

        for entry, is_boolean in zip(index, boolean):
            if entry is None:
                entries.append(("newaxis", None))
            elif entry is Ellipsis:
                for _ in range(fill):
                    add_slice(slice(None))
            elif _is_bool(entry):
                raise TypeError(f"axis {axis}: boolean scalar indices are not supported")
            elif isinstance(entry, slice):
                add_slice(entry)
            elif not _is_array_like(entry) and hasattr(entry, "__index__"):
                self.fixed[axis] = _resolve_int(integer_index(entry), self.source_shape[axis], axis)
                entries.append(("int", axis))
                axis += 1
            elif _is_array_like(entry):
                dtype_kind = getattr(getattr(entry, "dtype", None), "kind", None)
                if dtype_kind is not None and dtype_kind not in "biu":
                    raise IndexError("index arrays must have an integer or boolean dtype")
                shape = _shape_of(entry)
                if is_boolean:
                    expected = self.source_shape[axis:axis + len(shape)]
                    if any(size not in (0, target) for size, target in zip(shape, expected)):
                        raise IndexError(
                            f"boolean index has shape {format_shape(shape)} but the tensor axes "
                            f"have shape {format_shape(expected)}"
                        )
                    positions = [coord for coord in _coordinates(shape) if _get(entry, coord)]
                    for column in range(len(shape)):
                        values = {(i,): coord[column] for i, coord in enumerate(positions)}
                        array_specs[axis] = ((len(positions),), values)
                        entries.append(("arr", axis))
                        axis += 1
                else:
                    values = {}
                    for coordinate in _coordinates(shape):
                        try:
                            value = integer_index(_get(entry, coordinate))
                        except TypeError:
                            raise IndexError("index arrays must contain integers") from None
                        values[coordinate] = value
                    array_specs[axis] = (shape, values)
                    entries.append(("arr", axis))
                    axis += 1
            else:
                raise TypeError(f"axis {axis}: unsupported index entry {entry!r}")
        if not any(entry is Ellipsis for entry in index):
            for _ in range(fill):
                add_slice(slice(None))

        block = [
            position for position, entry in enumerate(index)
            if entry is not None and entry is not Ellipsis and not isinstance(entry, slice)
        ]
        between = index[block[0]:block[-1] + 1]
        contiguous = not any(
            entry is None or entry is Ellipsis or isinstance(entry, slice) for entry in between
        )
        shapes = [shape for shape, _ in array_specs.values()]
        try:
            self.broadcast_shape = broadcast_result_shape(shapes)
        except ValueError:
            raise IndexError(
                f"index arrays could not be broadcast together: shapes "
                f"{[format_shape(shape) for shape in shapes]}"
            ) from None
        # An empty broadcast performs no gathers, so array bounds are irrelevant.
        # Types and scalar indices were still validated above, as NumPy requires.
        for source_axis, (shape, values) in array_specs.items():
            if 0 not in self.broadcast_shape:
                for coordinate, value in values.items():
                    values[coordinate] = _resolve_int(
                        value, self.source_shape[source_axis], source_axis
                    )
            self.arrays[source_axis] = _ArrayIndex(shape, values)
        basic_axes = []
        insertion = None
        for kind, source_axis in entries:
            if kind in ("slice", "newaxis"):
                basic_axes.append(source_axis if kind == "slice" else None)
            elif kind == "arr" and insertion is None:
                insertion = len(basic_axes)
        self.broadcast_start = insertion if contiguous else 0
        basic_shape = tuple(
            1 if axis is None else _range_length(self.slices[axis]) for axis in basic_axes
        )
        start = self.broadcast_start
        self.result_shape = basic_shape[:start] + self.broadcast_shape + basic_shape[start:]
        self.slice_outputs = {
            axis: position if position < start else position + len(self.broadcast_shape)
            for position, axis in enumerate(basic_axes) if axis is not None
        }
        count = prod(self.broadcast_shape)
        self.explanation = [
            t("common.original_shape", shape=format_shape(self.source_shape)),
            t("common.index", index=format_index(index)),
            t("index.arrays"),
            t(
                "index.gather", axes=", ".join(str(axis) for axis in self.arrays),
                count=count, plural="s" if count != 1 else "",
                shape=format_shape(self.broadcast_shape),
            ),
        ]
        if not contiguous:
            self.explanation.append(t("index.slice_separates"))
        self.explanation.append(t("common.result_shape", shape=format_shape(self.result_shape)))
