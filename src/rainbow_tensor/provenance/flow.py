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
from ..ops.elementwise import normalize_operand
from ..ops.reductions import iter_matmul_source_terms
from ..ops.selection import extrema_spec
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
              name=None, source=None, origin=None, binary_operator=None, selection_operator=None):
        """Register one immutable recipe after all operation validation succeeds."""
        node_id, names = self._identity(operation, name)
        node = TrackedTensor(
            flow=self, node_id=node_id, output_port=0, name=names[0],
            operation=operation, shape=tuple(shape), inputs=inputs,
            term_count=term_count, divisor=divisor, term_factory=terms, source=source,
            origin=origin, binary_operator=binary_operator, selection_operator=selection_operator,
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
        """Register an array, numeric literal, or shape tuple as an input.

        Array values are read only during a later value or visual query. A shape
        tuple represents generated row-major values, matching the static views.
        Recording the same array twice creates two distinct logical inputs.
        A numeric literal is a scalar. Register constants here before passing
        them to a binary operation, just as for an array input.
        """
        if isinstance(array_or_shape, (TrackedTensor, TensorVisual)):
            raise TypeError("input expects an array or a shape, not a tracked tensor or visual")
        array_or_shape = normalize_operand(array_or_shape)
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

    def _binary(self, operation, a, b, name):
        """Record one ordered scalar operation at each broadcast output coordinate."""
        inputs = self._operands((a, b))
        shape = broadcast_result_shape(tuple(array.shape for array in inputs))

        def operands(coordinate):
            yield tuple(array.ref(broadcast_source_coord(coordinate, array.shape))
                        for array in inputs)

        return self._node(operation, inputs, shape, operands,
                          name=name, binary_operator=operation)

    def add(self, a, b, *, name=None):
        """Record elementwise addition with independent left and right origins.

        Both inputs must be tracked in this Flow. Their shapes broadcast as
        in NumPy. Use ``flow.input(value)`` to register a scalar constant.
        Reusing the same input keeps two occurrence paths but reads each
        needed source element once within a numerical query.
        """
        return self._binary("add", a, b, name)

    def subtract(self, a, b, *, name=None):
        """Record broadcast subtraction, preserving the minuend and subtrahend.

        Inputs follow :meth:`add`. The traced expression keeps ``a - b`` in
        that order and never interprets origin counts as signed coefficients.
        """
        return self._binary("subtract", a, b, name)

    def multiply(self, a, b, *, name=None):
        """Record elementwise multiplication rather than a matrix contraction.

        Inputs follow :meth:`add`. Each output has exactly two ordered
        operand roles, including when both roles reach the same source.
        """
        return self._binary("multiply", a, b, name)

    def divide(self, a, b, *, name=None):
        """Record true division with separate numerator and denominator paths.

        Inputs follow :meth:`add`. Division uses Python scalar arithmetic,
        and a zero denominator raises ZeroDivisionError during evaluation.
        Recording and structural tracing do not read values or perform division.
        A row-normalization chain can reuse its input in the numerator and
        a keepdims sum in the denominator without flattening either expression.
        """
        return self._binary("divide", a, b, name)

    def greater(self, a, b, *, name=None):
        """Record broadcast ``a > b`` with ordered, value-free source references.

        Inputs must belong to this Flow. Comparison follows Python scalar
        rules, so ordered complex comparisons raise TypeError when evaluated.
        NaN compares false, matching ordinary real NumPy comparisons.
        """
        return self._binary("greater", a, b, name)

    def greater_equal(self, a, b, *, name=None):
        """Record broadcast ``a >= b`` with the same rules as :meth:`greater`."""
        return self._binary("greater_equal", a, b, name)

    def less(self, a, b, *, name=None):
        """Record broadcast ``a < b`` with the same rules as :meth:`greater`."""
        return self._binary("less", a, b, name)

    def less_equal(self, a, b, *, name=None):
        """Record broadcast ``a <= b`` with the same rules as :meth:`greater`."""
        return self._binary("less_equal", a, b, name)

    def equal(self, a, b, *, name=None):
        """Record broadcast equality, including complex values and false NaN equality."""
        return self._binary("equal", a, b, name)

    def not_equal(self, a, b, *, name=None):
        """Record broadcast inequality, including a true result for NaN against itself."""
        return self._binary("not_equal", a, b, name)

    def where(self, condition, x, y, *, name=None):
        """Record a broadcast conditional with all three structural candidates.

        The condition and both alternatives must be tracked in this Flow.
        Structural tracing never reads them or claims which branch wins.
        Numerical queries conservatively plan and evaluate both branches,
        including an unused branch that may fail, then record the selected
        source separately. Values retain their Python scalar type rather than
        undergoing NumPy's common dtype promotion.
        """
        inputs = self._operands((condition, x, y))
        shape = broadcast_result_shape(tuple(array.shape for array in inputs))

        def candidates(coordinate):
            yield tuple(array.ref(broadcast_source_coord(coordinate, array.shape))
                        for array in inputs)

        return self._node("where", inputs, shape, candidates, name=name,
                          selection_operator="where")

    def _extremum(self, array, axis, keepdims, operation, name):
        """Record real-valued candidate groups without choosing a winning source."""
        inputs = self._operands((array,))
        axes, keepdims, shape, count = extrema_spec(array.shape, axis, keepdims, operation)

        def candidates(coordinate):
            for source in reduce_source_coords(coordinate, array.shape, axes, keepdims=keepdims):
                yield (array.ref(source),)

        return self._node(operation, inputs, shape, candidates, count, name=name,
                          selection_operator=operation)

    def min(self, array, axis=None, *, keepdims=False, name=None):
        """Record a minimum over real scalar candidates with an explicit first winner.

        Axis may be None, one integer, or a tuple of distinct axes. Empty
        reduced groups are rejected. Structural traces list candidates only.
        Evaluation keeps the first equal minimum or the first NaN if present.
        It preserves scalar values rather than executing a backend kernel.
        """
        return self._extremum(array, axis, keepdims, "min", name)

    def max(self, array, axis=None, *, keepdims=False, name=None):
        """Record a maximum using the shape and first-occurrence rules of :meth:`min`."""
        return self._extremum(array, axis, keepdims, "max", name)

    def argmin(self, array, axis=None, *, keepdims=False, name=None):
        """Record the first minimum's integer position along one axis.

        Axis is None or one integer. None returns a flattened row-major
        position. The first NaN wins if present. Structural candidate traces
        stay value-free, while an evaluated choice identifies the source cell.
        """
        return self._extremum(array, axis, keepdims, "argmin", name)

    def argmax(self, array, axis=None, *, keepdims=False, name=None):
        """Record the first maximum's position using the rules of :meth:`argmin`."""
        return self._extremum(array, axis, keepdims, "argmax", name)

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
