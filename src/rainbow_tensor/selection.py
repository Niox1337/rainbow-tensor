"""Compact coordinate selections for basic tensor indexing."""

from math import prod
from operator import index as integer_index

from .indexing import _range_length, validate_index


class BasicSelection:
    """Represent a basic index as source-axis ranges instead of coordinate lists.

    Iteration follows the index's slice order, including negative steps. Integer
    lookup and membership do not enumerate preceding coordinates. ``count`` is
    an arbitrary-size integer, while Python's ``len`` can overflow for selections
    larger than ``sys.maxsize``. Use ``list(selection)`` or a slice explicitly
    when materialized coordinates are needed. Renderers can inspect membership,
    prefixes, and bounded axis pins without iterating over the selection.
    """

    def __init__(self, shape, index):
        tokens = validate_index(index, shape)
        axes = []
        for token in tokens:
            if token is None:
                continue
            size = shape[len(axes)]
            if isinstance(token, int):
                axes.append(range(token, token + 1))
            else:
                axes.append(range(*token.indices(size)))
        self.axes = tuple(axes)
        self.count = 0 if any(not axis for axis in self.axes) else prod(
            _range_length(axis) for axis in self.axes
        )

    def __bool__(self):
        return self.count != 0

    def __len__(self):
        return self.count

    def __iter__(self):
        for position in range(self.count):
            yield self[position]

    def __getitem__(self, position):
        if isinstance(position, slice):
            return [self[i] for i in range(*position.indices(self.count))]
        position = integer_index(position)
        if position < 0:
            position += self.count
        if not 0 <= position < self.count:
            raise IndexError("selection index out of range")
        coordinate = []
        for axis in reversed(self.axes):
            position, offset = divmod(position, _range_length(axis))
            coordinate.append(axis[offset])
        return tuple(reversed(coordinate))

    def __eq__(self, other):
        if isinstance(other, BasicSelection):
            return self.axes == other.axes or not self and not other
        if isinstance(other, (list, tuple)):
            return self.count == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    def __contains__(self, coordinate):
        try:
            return len(coordinate) == len(self.axes) and self.matches_prefix(coordinate)
        except TypeError:
            return False

    def matches_prefix(self, prefix):
        """Whether any selected coordinate starts with the given frame prefix."""
        if not self or len(prefix) > len(self.axes):
            return False
        try:
            prefix = tuple(integer_index(value) for value in prefix)
        except TypeError:
            return False
        return all(value in axis for value, axis in zip(prefix, self.axes))

    def axis_pins(self, axis, limit):
        """Sample selected positions at both ends within an axis's display budget."""
        if not self or limit <= 0:
            return ()
        positions = self.axes[axis]
        if positions.step < 0:
            positions = positions[::-1]
        count = _range_length(positions)
        slots = count if positions.step == 1 else count * 2 - 1
        if slots <= limit:
            return tuple(positions)
        keep = max(1, limit - 1 if positions.step == 1 else (limit + 1) // 2)
        head = (keep + 1) // 2
        tail = keep // 2
        return tuple(positions[:head]) + tuple(positions[count - tail:])
