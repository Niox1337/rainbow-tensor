"""Shape math for einsum.

Pure Python parsing of einsum subscripts plus the result shape and the source
coordinates a contraction selects. No tensor library is imported, so the views
layer turns these mappings into values to draw and the tests cross check every
result against NumPy.
"""

from itertools import product
from math import prod
from string import ascii_letters


def _expand_ellipsis(compact, shapes):
    """Replace each ``...`` with generated labels for the broadcast axes.

    The ellipsis stands for the axes a subscript does not name. Those axes
    broadcast across the operands aligned from the right, exactly like NumPy, so
    they are expanded into a shared run of fresh labels. The widest run sets the
    number of broadcast axes, and a shorter operand uses the rightmost labels of
    that run. The expansion always emits an explicit ``->`` output.
    """
    if "->" in compact:
        input_text, output_text = compact.split("->")
        has_output = True
    else:
        input_text, output_text, has_output = compact, "", False

    input_parts = input_text.split(",")
    if len(input_parts) != len(shapes):
        raise ValueError(
            f"einsum expected {len(input_parts)} operands from subscripts, got {len(shapes)}"
        )

    ell_lens = []
    for part, shape in zip(input_parts, shapes):
        if part.count("...") > 1:
            raise ValueError("einsum subscript may contain at most one ellipsis per operand")
        if "..." in part:
            named = part.replace("...", "")
            if "." in named:
                raise ValueError("einsum '.' is only valid as part of an ellipsis '...'")
            covered = len(shape) - len(named)
            if covered < 0:
                raise ValueError(
                    f"einsum operand has {len(named)} labels for rank {len(shape)}"
                )
            ell_lens.append(covered)
        else:
            if "." in part:
                raise ValueError("einsum '.' is only valid as part of an ellipsis '...'")
            ell_lens.append(None)

    max_e = max([e for e in ell_lens if e is not None], default=0)
    used = {c for c in compact if c.isalpha()}
    pool = [c for c in ascii_letters if c not in used]
    if max_e > len(pool):
        raise ValueError("einsum ellipsis needs more labels than are available")
    ell_labels = pool[:max_e]

    new_parts = []
    for part, covered in zip(input_parts, ell_lens):
        if covered is None:
            new_parts.append(part)
        else:
            new_parts.append(part.replace("...", "".join(ell_labels[max_e - covered:])))
    new_input = ",".join(new_parts)

    if has_output:
        if output_text.count("...") > 1:
            raise ValueError("einsum output may contain at most one ellipsis")
        named_out = output_text.replace("...", "")
        if "." in named_out:
            raise ValueError("einsum '.' is only valid as part of an ellipsis '...'")
        new_output = output_text.replace("...", "".join(ell_labels))
    else:
        counts = {}
        for part in new_parts:
            for label in part:
                counts[label] = counts.get(label, 0) + 1
        singles = sorted(label for label, c in counts.items() if c == 1 and label not in ell_labels)
        new_output = "".join(ell_labels) + "".join(singles)

    return new_input + "->" + new_output


def parse_einsum_subscripts(subscripts, operand_count, shapes=None):
    """Parse einsum text into input labels and output labels.

    The supported syntax covers named axes with ASCII letters, an optional
    explicit output after ``->``, and ellipsis (``...``) notation. An ellipsis
    stands for the broadcast axes a subscript leaves unnamed and is expanded
    against ``shapes``, so the operand shapes are required whenever ``...``
    appears. Implicit and explicit output are both supported.
    """
    compact = "".join(str(subscripts).split())
    if "..." in compact or "." in compact:
        if shapes is None:
            raise ValueError("einsum ellipsis requires the operand shapes to expand")
        compact = _expand_ellipsis(compact, shapes)
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
    """Return broadcast label sizes while preserving each operand's diagonals.

    A label shared across operands may broadcast a singleton dimension. When
    a label repeats within one operand, its axes must have exactly equal sizes
    because those axes select a diagonal rather than broadcast independently.
    """
    if len(input_axes) != len(shapes):
        raise ValueError("einsum inputs and shapes must have the same length")

    sizes = {}
    for operand, (labels, shape) in enumerate(zip(input_axes, shapes)):
        if len(labels) != len(shape):
            raise ValueError(
                f"einsum operand {operand} has {len(labels)} labels for rank {len(shape)}"
            )
        operand_sizes = {}
        for label, size in zip(labels, shape):
            if label in operand_sizes and operand_sizes[label] != size:
                raise ValueError(
                    f"einsum label {label!r} has inconsistent sizes "
                    f"{operand_sizes[label]} and {size} within operand {operand}"
                )
            operand_sizes[label] = size
            previous = sizes.get(label, 1)
            if previous != size and previous != 1 and size != 1:
                raise ValueError(
                    f"einsum label {label!r} has inconsistent sizes "
                    f"{previous} and {size}"
                )
            sizes[label] = size if previous == 1 else previous
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
    input_axes, output_axes = parse_einsum_subscripts(subscripts, len(shapes), shapes)
    sizes = einsum_index_sizes(input_axes, shapes)
    return tuple(sizes[label] for label in output_axes)


def einsum_term_count(input_axes, output_axes, shapes):
    """Count products contributing to one output without enumerating them."""
    sizes = einsum_index_sizes(input_axes, shapes)
    contracted = einsum_contracted_labels(input_axes, output_axes)
    return prod(sizes[label] for label in contracted)


def einsum_source_terms(input_axes, output_axes, shapes, output_coord):
    """Yield ordered operand coordinates for each product in one output.

    Contracted labels follow their first appearance in the inputs, with the
    final label changing fastest. A singleton operand axis always reads zero,
    retaining repeated factors when that axis broadcasts over contracted terms.
    """
    sizes = einsum_index_sizes(input_axes, shapes)
    contracted = einsum_contracted_labels(input_axes, output_axes)
    fixed = dict(zip(output_axes, output_coord))
    # itertools.product pools every input range before its first yield. Decode
    # a flat term position instead so a bounded trace stays small for long axes.
    for position in range(prod(sizes[label] for label in contracted)):
        assignment = dict(fixed)
        for label in reversed(contracted):
            position, assignment[label] = divmod(position, sizes[label])
        yield tuple(
            tuple(0 if size == 1 else assignment[label] for label, size in zip(labels, shape))
            for labels, shape in zip(input_axes, shapes)
        )


def einsum_selected_coords(input_axes, output_axes, shapes, output_coord=None):
    """Return unique source coordinates that feed one chosen output element.

    The first output remains the default for existing callers. Each operand's
    selection is enumerated independently, avoiding the full Cartesian product
    of contracted labels that never occur in that operand.
    """
    sizes = einsum_index_sizes(input_axes, shapes)
    if 0 in sizes.values():
        return [[] for _ in shapes]
    contracted = einsum_contracted_labels(input_axes, output_axes)
    if output_coord is None:
        output_coord = (0,) * len(output_axes)
    fixed = dict(zip(output_axes, output_coord))
    selected = []
    for labels, shape in zip(input_axes, shapes):
        operand_sizes = dict(zip(labels, shape))
        active = [label for label in contracted if operand_sizes.get(label, 1) != 1]
        ranges = [range(operand_sizes[label]) for label in active]
        coords = []
        for values in product(*ranges):
            assignment = dict(fixed)
            assignment.update(zip(active, values))
            coords.append(
                tuple(0 if size == 1 else assignment[label] for label, size in zip(labels, shape))
            )
        selected.append(sorted(coords))
    return selected
