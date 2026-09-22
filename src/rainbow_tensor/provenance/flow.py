"""Record tensor operation recipes without constructing intermediate arrays."""

from ..index_mapping import IndexMapping
from ..ops import (
    broadcast_result_shape,
    broadcast_source_coord,
    concatenate_result_shape,
    concatenate_source,
    expand_dims_axes,
    expand_dims_source_coord,
    matmul_result_shape,
    moveaxis_axes,
    normalize_reduction_axes,
    reduce_result_shape,
    reduce_source_coords,
    reduce_term_count,
    repeat_source_coord,
    repeat_source_lookup,
    reshape_result_shape,
    reshape_source_coord,
    squeeze_axes,
    squeeze_source_coord,
    stack_result_shape,
    stack_source,
    swapaxes_axes,
    take_axis_and_indices,
    take_source_coord,
    transpose_axes,
    transpose_result_shape,
    transpose_source_coord,
    validate_keepdims,
)
from ..ops.einsum import (
    einsum_index_sizes,
    einsum_source_terms,
    einsum_term_count,
    parse_einsum_subscripts,
)
from ..ops.reductions import iter_matmul_source_terms
from ..shape import _check_axis, extract_shape
from ..visual import TensorVisual
from .model import TrackedTensor


class Flow:
    """Connect explicitly recorded operations to their original tensor inputs.

    Start with ``flow.input(array, name="X")`` and pass its tracked result to
    operation methods such as ``flow.index`` and ``flow.sum``. Construction
    records shapes and coordinate mappings only. It neither evaluates values
    nor allocates an intermediate tensor. Input arrays remain live references,
    while shapes and operation parameters are captured when each step is added.
    """

    def __init__(self):
        """Create an empty graph with names local to this flow."""
        self._nodes = {}
        self._names = set()
        self._sequence = 0

    def _operands(self, arrays):
        """Capture operands and reject implicit or cross-flow dependencies."""
        operands = tuple(arrays)
        if not operands:
            raise ValueError("an operation needs at least one tracked input")
        for array in operands:
            if not isinstance(array, TrackedTensor):
                raise TypeError("operation inputs must be tracked tensors from this flow")
            if array.flow is not self:
                raise ValueError("operation inputs must belong to the same flow")
            if self._nodes.get((array.node_id, array.output_port)) is not array:
                raise ValueError("operation input is not registered in this flow")
        return operands

    def _identity(self, operation, name, ports=1):
        """Reserve a stable graph identity and unique display names."""
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("name must be a nonempty string")
        sequence = self._sequence
        while True:
            display = name if name is not None else f"{operation}_{sequence}"
            names = (display,) if ports == 1 else tuple(
                f"{display}[{port}]" for port in range(ports)
            )
            reserved = {display, *names}
            if not self._names.intersection(reserved):
                break
            if name is not None:
                raise ValueError(f"name {name!r} is already used in this flow")
            sequence += 1
        self._sequence = sequence + 1
        self._names.update(reserved)
        return f"n{sequence}", names

    def _node(self, operation, inputs, shape, terms, term_count=1, *, divisor=1,
              name=None, source=None, origin=None):
        """Register one immutable recipe after all operation validation succeeds."""
        node_id, names = self._identity(operation, name)
        node = TrackedTensor(
            flow=self, node_id=node_id, output_port=0, name=names[0],
            operation=operation, shape=tuple(shape), inputs=inputs,
            term_count=term_count, divisor=divisor, term_factory=terms, source=source,
            origin=origin,
        )
        self._nodes[(node_id, 0)] = node
        return node

    def _mapped(self, operation, inputs, shape, origin, *, name=None):
        """Record an identity-valued operation with a coordinate origin resolver."""
        def terms(coordinate):
            operand, source_coordinate = origin(coordinate)
            yield (inputs[operand].ref(source_coordinate),)

        return self._node(operation, inputs, shape, terms, name=name, origin=origin)

    def input(self, array_or_shape, *, name=None):
        """Register an array or a shape tuple as an explicit provenance boundary.

        Array values are read only during a later value or visual query. A shape
        tuple represents generated row-major values, matching the static views.
        Recording the same array twice creates two distinct logical inputs.
        """
        if isinstance(array_or_shape, (TrackedTensor, TensorVisual)):
            raise TypeError("input expects an array or a shape, not a tracked tensor or visual")
        shape = extract_shape(array_or_shape)
        source = array_or_shape if hasattr(array_or_shape, "shape") else shape
        return self._node("input", (), shape, lambda coordinate: iter(()), 0,
                          name=name, source=source)

    def index(self, array, selection, *, name=None):
        """Record slicing or advanced indexing, preserving repeated result picks.

        Index arrays and masks are copied during mapping construction. Later
        edits to a selection cannot change the recorded operation's origins.
        """
        inputs = self._operands((array,))
        mapping = IndexMapping(array.shape, selection)
        return self._mapped("index", inputs, mapping.result_shape,
                            lambda coord: (0, mapping.source_coord(coord)), name=name)

    def reshape(self, array, new_shape, *, name=None):
        """Record a row-major reshape, resolving an inferred dimension once."""
        inputs = self._operands((array,))
        shape = reshape_result_shape(array.shape, new_shape)
        return self._mapped("reshape", inputs, shape,
                            lambda coord: (0, reshape_source_coord(coord, array.shape, shape)),
                            name=name)

    def _permute(self, array, axes, operation, name):
        """Share coordinate inversion across the three axis-permutation methods."""
        inputs = self._operands((array,))
        shape = transpose_result_shape(array.shape, axes)
        return self._mapped(operation, inputs, shape,
                            lambda coord: (0, transpose_source_coord(coord, axes)), name=name)

    def transpose(self, array, axes=None, *, name=None):
        """Record an axis permutation, reversing axes when none are specified."""
        self._operands((array,))
        return self._permute(array, transpose_axes(len(array.shape), axes), "transpose", name)

    def swapaxes(self, array, axis1, axis2, *, name=None):
        """Record exchanging two axes while preserving each element's origin."""
        self._operands((array,))
        axes = swapaxes_axes(len(array.shape), axis1, axis2)
        return self._permute(array, axes, "swapaxes", name)

    def moveaxis(self, array, source, destination, *, name=None):
        """Record moving axes while keeping all remaining axes in relative order."""
        self._operands((array,))
        axes = moveaxis_axes(len(array.shape), source, destination)
        return self._permute(array, axes, "moveaxis", name)

    def squeeze(self, array, axis=None, *, name=None):
        """Record removal of singleton axes without changing the source values."""
        inputs = self._operands((array,))
        removed = squeeze_axes(array.shape, axis)
        shape = tuple(size for i, size in enumerate(array.shape) if i not in removed)
        return self._mapped("squeeze", inputs, shape,
                            lambda coord: (0, squeeze_source_coord(coord, removed)), name=name)

    def expand_dims(self, array, axis, *, name=None):
        """Record insertion of singleton axes at normalized result positions."""
        inputs = self._operands((array,))
        inserted = expand_dims_axes(len(array.shape), axis)
        source = iter(array.shape)
        shape = tuple(1 if i in inserted else next(source)
                      for i in range(len(array.shape) + len(inserted)))
        return self._mapped("expand_dims", inputs, shape,
                            lambda coord: (0, expand_dims_source_coord(coord, inserted)),
                            name=name)

    def repeat(self, array, repeats, axis=0, *, name=None):
        """Record adjacent copies with storage bounded by the input repeat counts."""
        inputs = self._operands((array,))
        original = array.shape
        promoted = original or (1,)
        axis = _check_axis(axis, len(promoted))
        lookup = repeat_source_lookup(promoted[axis], repeats)
        shape = promoted[:axis] + (lookup.count,) + promoted[axis + 1:]
        return self._mapped("repeat", inputs, shape,
                            lambda coord: (0, repeat_source_coord(coord, lookup, axis, original)),
                            name=name)

    def take(self, array, indices, axis=0, *, name=None):
        """Record scalar or sequence gathers with normalized, captured indices."""
        inputs = self._operands((array,))
        original = array.shape
        axis, indices = take_axis_and_indices(original, indices, axis)
        promoted = original or (1,)
        replacement = () if isinstance(indices, int) else (len(indices),)
        shape = promoted[:axis] + replacement + promoted[axis + 1:]
        return self._mapped("take", inputs, shape,
                            lambda coord: (0, take_source_coord(coord, indices, axis, original)),
                            name=name)

    def concatenate(self, arrays, axis=0, *, name=None):
        """Record a joined axis and retain which input supplies each result cell."""
        inputs = self._operands(arrays)
        shapes = tuple(array.shape for array in inputs)
        shape = concatenate_result_shape(shapes, axis)
        axis = _check_axis(axis, len(shape))
        return self._mapped("concatenate", inputs, shape,
                            lambda coord: concatenate_source(coord, shapes, axis), name=name)

    def stack(self, arrays, axis=0, *, name=None):
        """Record a new operand axis without losing the identity of each input."""
        inputs = self._operands(arrays)
        shape = stack_result_shape(tuple(array.shape for array in inputs), axis)
        axis = _check_axis(axis, len(shape))
        return self._mapped("stack", inputs, shape,
                            lambda coord: stack_source(coord, axis, len(inputs[0].shape)),
                            name=name)

    def broadcast(self, *arrays, name=None):
        """Return one tracked output per operand stretched to a common shape.

        Outputs share an operation identity but have distinct output ports.
        Each output retains its own values and traces to its matching input.
        """
        inputs = self._operands(arrays)
        shape = broadcast_result_shape(tuple(array.shape for array in inputs))
        node_id, names = self._identity("broadcast", name, len(inputs))
        outputs = []
        for port, array in enumerate(inputs):
            def terms(coordinate, source=array):
                yield (source.ref(broadcast_source_coord(coordinate, source.shape)),)

            node = TrackedTensor(
                flow=self, node_id=node_id, output_port=port, name=names[port],
                operation="broadcast", shape=shape, inputs=inputs, term_count=1,
                divisor=1, term_factory=terms, source=None,
                origin=lambda coord, source=array, port=port: (
                    port, broadcast_source_coord(coord, source.shape)
                ),
            )
            self._nodes[(node_id, port)] = node
            outputs.append(node)
        return tuple(outputs)

    def _reduce(self, array, axis, keepdims, operation, name):
        """Preserve reduction grouping and a mean's divisor as a separate recipe."""
        inputs = self._operands((array,))
        axes = normalize_reduction_axes(axis, len(array.shape),
                                        allow_scalar_axis=operation == "sum")
        keepdims = validate_keepdims(keepdims)
        shape = reduce_result_shape(array.shape, axes, keepdims=keepdims)
        count = reduce_term_count(array.shape, axes)

        def terms(coordinate):
            for source in reduce_source_coords(coordinate, array.shape, axes, keepdims=keepdims):
                yield (array.ref(source),)

        return self._node(operation, inputs, shape, terms, count,
                          divisor=count if operation == "mean" else 1, name=name)

    def sum(self, array, axis=None, *, keepdims=False, name=None):
        """Record an ordered sum over all axes or a normalized subset of axes."""
        return self._reduce(array, axis, keepdims, "sum", name)

    def mean(self, array, axis=None, *, keepdims=False, name=None):
        """Record a sum followed by division, including an undefined empty mean."""
        return self._reduce(array, axis, keepdims, "mean", name)

    def matmul(self, a, b, *, name=None):
        """Record ordered products and their sum for batched matrix multiplication."""
        inputs = self._operands((a, b))
        shape = matmul_result_shape(a.shape, b.shape)

        def terms(coordinate):
            for ac, bc in iter_matmul_source_terms(coordinate, a.shape, b.shape):
                yield (a.ref(ac), b.ref(bc))

        return self._node("matmul", inputs, shape, terms, a.shape[-1], name=name)

    def einsum(self, subscripts, *arrays, name=None):
        """Record labelled products, contractions, diagonals and broadcast factors."""
        inputs = self._operands(arrays)
        shapes = tuple(array.shape for array in inputs)
        input_axes, output_axes = parse_einsum_subscripts(subscripts, len(inputs), shapes)
        sizes = einsum_index_sizes(input_axes, shapes)
        shape = tuple(sizes[label] for label in output_axes)
        count = einsum_term_count(input_axes, output_axes, shapes)

        def terms(coordinate):
            for coordinates in einsum_source_terms(input_axes, output_axes, shapes, coordinate):
                yield tuple(array.ref(source) for array, source in zip(inputs, coordinates))

        return self._node("einsum", inputs, shape, terms, count, name=name)
