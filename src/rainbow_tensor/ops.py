"""Shape changing and combining operations.

This module computes result shapes and the coordinate mappings for the
operations a learner meets when rearranging, summarising, combining, or
stretching tensors. Everything here is pure Python over shapes and integer
coordinates, so no tensor library is imported. The notebook layer turns these
mappings into values to draw, and the tests cross check every result against
NumPy.
"""

from itertools import product
from math import prod
from string import ascii_letters

from .shape import flat_index


def unflatten(flat, shape):
    """Return the coordinate at row major flat position ``flat`` in ``shape``."""
    coord = []
    for dim in reversed(shape):
        coord.append(flat % dim)
        flat //= dim
    return tuple(reversed(coord))


# Reshape ------------------------------------------------------------------


def reshape_result_shape(old_shape, new_shape):
    """Resolve ``new_shape`` against ``old_shape`` and validate the element count.

    One axis may be ``-1`` and is inferred from the remaining axes, matching
    the NumPy convention. The total number of elements must stay the same.
    """
    new = tuple(new_shape)
    neg = [i for i, d in enumerate(new) if d == -1]
    if len(neg) > 1:
        raise ValueError("can only specify one unknown dimension with -1")
    for d in new:
        if d < -1 or d == 0:
            raise ValueError(f"invalid reshape dimension {d}")
    total = prod(old_shape)
    if neg:
        rest = prod(d for d in new if d != -1)
        if rest == 0 or total % rest != 0:
            raise ValueError(
                f"cannot reshape size {total} into {new}"
            )
        new = tuple(total // rest if d == -1 else d for d in new)
    if prod(new) != total:
        raise ValueError(
            f"cannot reshape tensor of size {total} into shape {tuple(new_shape)}"
        )
    return new


def reshape_source_coord(result_coord, old_shape, new_shape):
    """Map a destination coordinate back to its source coordinate.

    Reshape keeps row major order, so the element at flat position ``k`` in the
    new layout is the same element at flat position ``k`` in the old layout.
    """
    return unflatten(flat_index(result_coord, new_shape), old_shape)


# Transpose and permute ----------------------------------------------------


def transpose_axes(ndim, axes):
    """Validate ``axes`` and return the permutation as a tuple.

    ``None`` reverses the axis order, matching ``numpy.transpose``.
    """
    if axes is None:
        return tuple(reversed(range(ndim)))
    axes = tuple(axes)
    if sorted(axes) != list(range(ndim)):
        raise ValueError(
            f"axes {axes} is not a permutation of the {ndim} tensor axes"
        )
    return axes


def transpose_result_shape(shape, axes):
    """Return the shape after moving each axis to its permuted position."""
    return tuple(shape[a] for a in axes)


def transpose_source_coord(result_coord, axes):
    """Map a destination coordinate back to its source coordinate.

    Result axis ``r`` is drawn from source axis ``axes[r]``, so the source
    coordinate places ``result_coord[r]`` at source axis ``axes[r]``.
    """
    source = [0] * len(axes)
    for r, a in enumerate(axes):
        source[a] = result_coord[r]
    return tuple(source)


# Swap axes ----------------------------------------------------------------


def swapaxes_axes(ndim, axis1, axis2):
    """Return the transpose permutation for swapping two axes."""
    first = _check_axis(axis1, ndim)
    second = _check_axis(axis2, ndim)
    axes = list(range(ndim))
    axes[first], axes[second] = axes[second], axes[first]
    return tuple(axes)


# Einsum -------------------------------------------------------------------


def parse_einsum_subscripts(subscripts, operand_count):
    """Parse einsum text into input labels and output labels.

    The supported syntax covers named axes with ASCII letters and an optional
    explicit output after ``->``. Ellipsis is rejected so the visual stays
    honest about every axis it draws.
    """
    compact = "".join(str(subscripts).split())
    if "..." in compact or "." in compact:
        raise ValueError("einsum ellipsis is not supported yet")
    if compact.count("->") > 1:
        raise ValueError("einsum subscripts may contain only one output arrow")

    if "->" in compact:
        input_text, output_text = compact.split("->")
        explicit = True
    else:
        input_text = compact
        output_text = ""
        explicit = False

    inputs = tuple(tuple(part) for part in input_text.split(","))
    if len(inputs) != operand_count:
        raise ValueError(
            f"einsum expected {len(inputs)} operands from subscripts, got {operand_count}"
        )

    for labels in inputs:
        _validate_einsum_labels(labels, "input")

    if explicit:
        output = tuple(output_text)
        _validate_einsum_labels(output, "output")
        if len(set(output)) != len(output):
            raise ValueError("einsum output labels may not repeat")
        input_labels = {label for labels in inputs for label in labels}
        for label in output:
            if label not in input_labels:
                raise ValueError(f"einsum output label {label!r} does not appear in input")
    else:
        counts = {}
        for labels in inputs:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        output = tuple(sorted(label for label, count in counts.items() if count == 1))

    return inputs, output


def _validate_einsum_labels(labels, role):
    """Validate one side of an einsum subscript."""
    for label in labels:
        if label not in ascii_letters:
            raise ValueError(f"einsum {role} labels must be ASCII letters, got {label!r}")


def einsum_index_sizes(input_axes, shapes):
    """Return the size for each subscript label after validating operands."""
    if len(input_axes) != len(shapes):
        raise ValueError("einsum inputs and shapes must have the same length")

    sizes = {}
    for operand, (labels, shape) in enumerate(zip(input_axes, shapes)):
        if len(labels) != len(shape):
            raise ValueError(
                f"einsum operand {operand} has {len(labels)} labels for rank {len(shape)}"
            )
        for label, size in zip(labels, shape):
            if label in sizes and sizes[label] != size:
                raise ValueError(
                    f"einsum label {label!r} has inconsistent sizes "
                    f"{sizes[label]} and {size}"
                )
            sizes[label] = size
    return sizes


def einsum_contracted_labels(input_axes, output_axes):
    """Return labels that are summed away by an einsum."""
    output = set(output_axes)
    contracted = []
    seen = set()
    for labels in input_axes:
        for label in labels:
            if label not in output and label not in seen:
                contracted.append(label)
                seen.add(label)
    return tuple(contracted)


def einsum_result_shape(subscripts, shapes):
    """Derive the einsum output shape from labels and operand shapes."""
    input_axes, output_axes = parse_einsum_subscripts(subscripts, len(shapes))
    sizes = einsum_index_sizes(input_axes, shapes)
    return tuple(sizes[label] for label in output_axes)


def einsum_selected_coords(input_axes, output_axes, shapes):
    """Return source coordinates that feed the first output element.

    These coordinates give the renderer a small concrete contraction to
    highlight. For matrix multiplication this marks the row in the first
    operand and the column in the second operand.
    """
    sizes = einsum_index_sizes(input_axes, shapes)
    contracted = einsum_contracted_labels(input_axes, output_axes)
    fixed = {label: 0 for label in output_axes}
    selected = [[] for _ in input_axes]
    ranges = [range(sizes[label]) for label in contracted]

    for values in product(*ranges):
        assignment = dict(fixed)
        assignment.update(zip(contracted, values))
        for operand, labels in enumerate(input_axes):
            selected[operand].append(tuple(assignment[label] for label in labels))

    if not contracted:
        for operand, labels in enumerate(input_axes):
            selected[operand].append(tuple(fixed[label] for label in labels))

    return [sorted(set(coords)) for coords in selected]


# Axis reductions ----------------------------------------------------------


def reduce_result_shape(shape, axis):
    """Return the shape after collapsing ``axis``."""
    axis = _check_axis(axis, len(shape))
    return tuple(s for i, s in enumerate(shape) if i != axis)


def reduce_source_coords(result_coord, shape, axis):
    """Yield every source coordinate that collapses into ``result_coord``."""
    axis = _check_axis(axis, len(shape))
    for k in range(shape[axis]):
        yield result_coord[:axis] + (k,) + result_coord[axis:]


def _check_axis(axis, ndim):
    """Resolve a possibly negative axis and bounds check it."""
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise ValueError(f"axis {axis} is out of range for a rank {ndim} tensor")
    return resolved


# Concatenate --------------------------------------------------------------


def concatenate_result_shape(shapes, axis):
    """Return the shape after joining ``shapes`` along ``axis``.

    Every operand must match on every axis except the joined one.
    """
    if not shapes:
        raise ValueError("need at least one tensor to concatenate")
    ndim = len(shapes[0])
    axis = _check_axis(axis, ndim)
    for s in shapes:
        if len(s) != ndim:
            raise ValueError("all tensors must have the same rank to concatenate")
        for i in range(ndim):
            if i != axis and s[i] != shapes[0][i]:
                raise ValueError(
                    f"all tensors must match on axis {i} to concatenate along "
                    f"axis {axis}, got {s[i]} and {shapes[0][i]}"
                )
    out = list(shapes[0])
    out[axis] = sum(s[axis] for s in shapes)
    return tuple(out)


def concatenate_source(result_coord, shapes, axis):
    """Map a result coordinate to its ``(operand_index, source_coord)``."""
    axis = _check_axis(axis, len(shapes[0]))
    pos = result_coord[axis]
    for i, s in enumerate(shapes):
        if pos < s[axis]:
            coord = result_coord[:axis] + (pos,) + result_coord[axis + 1:]
            return i, coord
        pos -= s[axis]
    raise IndexError("result coordinate is past the end of the joined axis")


# Stack --------------------------------------------------------------------


def stack_result_shape(shapes, axis):
    """Return the shape after stacking ``shapes`` on a new ``axis``.

    Every operand must have the same shape. The new axis may sit anywhere from
    ``0`` to the operand rank.
    """
    if not shapes:
        raise ValueError("need at least one tensor to stack")
    base = shapes[0]
    for s in shapes:
        if tuple(s) != tuple(base):
            raise ValueError(
                f"all tensors must have the same shape to stack, got {tuple(s)} "
                f"and {tuple(base)}"
            )
    axis = _check_axis(axis, len(base) + 1)
    out = list(base)
    out.insert(axis, len(shapes))
    return tuple(out)


def stack_source(result_coord, axis, ndim):
    """Map a result coordinate to its ``(operand_index, source_coord)``."""
    axis = _check_axis(axis, ndim + 1)
    i = result_coord[axis]
    coord = result_coord[:axis] + result_coord[axis + 1:]
    return i, coord


# Broadcasting -------------------------------------------------------------


def broadcast_result_shape(shapes):
    """Return the broadcast shape of ``shapes`` using the NumPy rules.

    Axes are aligned from the right. On each axis the sizes must be equal or
    one of them must be ``1``, which stretches to match the other.
    """
    ndim = max(len(s) for s in shapes)
    result = []
    for offset in range(1, ndim + 1):
        sizes = [s[-offset] for s in shapes if offset <= len(s)]
        big = [n for n in sizes if n != 1]
        if len(set(big)) > 1:
            raise ValueError(
                f"operands could not be broadcast together: axis sizes {sizes}"
            )
        result.append(big[0] if big else 1)
    return tuple(reversed(result))


def broadcast_source_coord(result_coord, operand_shape):
    """Map a result coordinate back to ``operand_shape``.

    Leading axes the operand does not have are dropped, and a stretched axis of
    size one always reads position zero.
    """
    extra = len(result_coord) - len(operand_shape)
    coord = []
    for i, size in enumerate(operand_shape):
        pos = result_coord[extra + i]
        coord.append(0 if size == 1 else pos)
    return tuple(coord)


def broadcast_stretched_axes(operand_shape, result_shape):
    """Return the result axes where ``operand_shape`` is stretched.

    This is every leading axis the operand gains, plus every size one axis that
    grows to match the result.
    """
    extra = len(result_shape) - len(operand_shape)
    stretched = list(range(extra))
    for i, size in enumerate(operand_shape):
        if size == 1 and result_shape[extra + i] != 1:
            stretched.append(extra + i)
    return stretched
