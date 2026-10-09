# Agent instructions: teach NumPy with Rainbow Tensor

This page is a working reference for an LLM agent using the **rainbow-tensor
1.8.2 source checkout**. Read it directly when a user supplies this URL.
Generate runnable code that answers their NumPy question with a useful
visualization and a checked explanation. The user does not need to copy a
prompt or fill in a template.

Use the user's expression and data when supplied. Otherwise choose a small
deterministic NumPy array with distinct values. If the user supplies only this
URL, ask which operation or expression they want to understand.

## Build the lesson around one question

1. State what the learner should predict, such as the shape of a reduction or
   the source cells of `result[1]`.
2. Show the input values and shapes, then compute the real NumPy result.
3. Choose the control that answers the question. Use `explore` to select
   outputs, `walkthrough` to advance through contributions, or
   `reduction_playground` to change axes and predict the new shape.
4. Name the output coordinate and its source coordinates. Explain the local
   arithmetic as well as the colours. Keep repeated contributions and the
   order of subtraction or division.
5. Assert the shape and at least one value or source mapping against NumPy.
   Say which assertions you actually executed. Mark unexecuted checks as
   unverified.
6. Give one concrete next interaction and its expected answer. Do not leave
   the learner with an unexplained widget.

Return complete notebook cells with imports and short explanations between
them. Use `display(...)` for each figure or controller. Keep controls open
while the learner explores, then explain how to call `.close()`.
For a script, exported lesson, or host without widget support, use the current
static `.visual`, or the equivalent static operation.

Write the explanation in the user's language. Keep Python names and comments
in English. Leave the theme automatic unless requested otherwise. Call
`rt.available_languages()` before selecting figure text with
`rt.set_language(...)`. Do not invent locale codes or translation keys.

## Runtime and object contracts

Install the matching checkout in the environment used by the notebook kernel.
Run this command from its repository root. The version described here may
include features that have not yet been published to PyPI.

```bash
python -m pip install -e .
```

Import `rainbow_tensor as rt`. NumPy, IPython, `ipywidgets`, and `anywidget`
are required dependencies. Missing widget packages indicate an incomplete
installation. Widget imports alone do not prove that a notebook host can
display live controls.

Read `rt.capabilities()` before selecting an operation. Compare its package
version with this reference and use its actual signatures and controller lists.
The <a href="../_static/capabilities.json">capability reference</a> describes
the docs build. The [teaching contract](teaching-contract.md) includes runnable
positive and deliberately incorrect specimens. It verifies API usage and
selected-source claims without claiming to grade arbitrary prose.

| Object | Role | Public interface |
| --- | --- | --- |
| NumPy array | Execute native computation | Indexing, NumPy methods, `.shape` |
| `TensorVisual` | One static operation or selected lesson view | `.svg`, `.text`, `.save(path)`, `.result_shape`, `.trace` |
| `TrackedTensor` | An explicitly recorded Flow recipe | `.shape`, `.visualize(focus=...)`, `.trace(coord)`, `.value(coord)` |
| `FocusExplorer` | Select another result and inspect its sources | `.set_focus(coord)`, `.visual`, `.close()` |
| `Walkthrough` | Select an occurrence and contribution | `.snapshot`, `.next_term()`, `.select_term(index)`, `.select_occurrence(index)`, `.set_focus(coord)`, `.visual`, `.close()` |
| `ReductionPlayground` | Change axes and predict the output shape | `.set_parameters(...)`, `.prediction.value`, `.reveal()`, `.hide()`, `.explorer`, `.visual`, `.close()` |

A static call such as `rt.sum(x, axis=1)` returns a display object. Do not
index it or pass it to another tensor operation. For a single-input operation,
`visual.shape` describes the source and `visual.result_shape` the output.
A tracked tensor's `.shape` describes its logical result.

A shape tuple such as `(2, 3)` generates row-major example values from zero.
It does not provide literal data. Elementwise operands and `flow.input`
also accept numeric literals as scalar values. For a scalar shape view, use
`rt.shape(np.array(7))`.

Focus is a tuple of **output** coordinates, not an input index expression.
Use `()` for a scalar. Empty outputs have no focus coordinate, so omit it.
Keep the introductory example small enough to show every relevant cell.

## Translate a NumPy question into an API call

The names `x`, `a`, `b`, `selection`, and `focus` below are supplied by
the example. Check the input and output shapes before choosing a coordinate.

| NumPy concept | Rainbow Tensor call |
| --- | --- |
| Tensor axes and shape | `rt.shape(x)` |
| `x[selection]` | `rt.explore(rt.index, x, selection, focus=focus)` |
| `x.reshape((3, 2))` | `rt.explore(rt.reshape, x, (3, 2), focus=focus)` |
| `x.transpose((1, 0))` | `rt.explore(rt.transpose, x, (1, 0), focus=focus)` |
| `np.take(x, [1, 0, 1], axis=0)` | `rt.explore(rt.take, x, [1, 0, 1], axis=0, focus=focus)` |
| `np.repeat(x, 2, axis=0)` | `rt.explore(rt.repeat, x, 2, axis=0, focus=focus)` |
| `np.concatenate([a, b], axis=0)` | `rt.explore(rt.concatenate, [a, b], axis=0, focus=focus)` |
| `np.stack([a, b], axis=0)` | `rt.explore(rt.stack, [a, b], axis=0, focus=focus)` |
| Operand expansion | `rt.explore(rt.broadcast, a, b, focus=focus, focus_operand=0)` |
| `a + b`, `a - b` | `rt.explore(rt.add, a, b, focus=focus)` or `rt.explore(rt.subtract, a, b, focus=focus)` |
| `a * b`, `a / b` | `rt.explore(rt.multiply, a, b, focus=focus)` or `rt.explore(rt.divide, a, b, focus=focus)` |
| `a > b` and related comparisons | `rt.explore(rt.greater, a, b, focus=focus)` or the matching comparison function |
| `np.where(condition, x, y)` | `rt.explore(rt.where, condition, x, y, focus=focus)` |
| `x.sum(axis=1, keepdims=True)` | `rt.explore(rt.sum, x, axis=1, keepdims=True, focus=focus)` |
| `x.mean(axis=1)` | `rt.explore(rt.mean, x, axis=1, focus=focus)` |
| `x.max(axis=1)`, `x.argmax(axis=1)` | `rt.explore(rt.max, x, axis=1, focus=focus)` or `rt.explore(rt.argmax, x, axis=1, focus=focus)` |
| `a @ b` | `rt.explore(rt.matmul, a, b, focus=focus)` |
| `np.einsum("ij,jk->ik", a, b)` | `rt.explore(rt.einsum, "ij,jk->ik", a, b, focus=focus)` |
| Physical array storage | `rt.memory(x)` |

Pass the operation itself to `explore`. Do not pass an already rendered
visual. `shape` and `memory` are standalone inspection views.
Other shape operations are `swapaxes`, `moveaxis`, `squeeze`, and
`expand_dims`. See the [API reference](../api) for exact signatures.

Preserve these differences from NumPy:

- `repeat` and `take` default to `axis=0` and reject `axis=None`.
  Flatten first with `x.reshape(-1)`, or record
  `flow.reshape(node, (-1,))`.
- `reshape` has no `order` argument. Do not claim that it explains
  Fortran-order reshaping or establishes whether an array was copied.
- `sum` and `mean` accept axes and `keepdims`, but not NumPy's
  `dtype`, `out`, or `where` keywords.
- Indexing supports integers, slices, ellipsis, new axes, boolean masks, and
  advanced integer arrays. Boolean scalar indices are unsupported.
- `broadcast` explains expansion, not arithmetic. A static broadcast view
  takes two operands. Choose `focus_operand=0` or `1` for its result.
  `flow.broadcast` returns one tracked output per input. Unpack the outputs.
- No NumPy operation is intercepted automatically. For an unsupported
  operation, compute its result natively and show it with `rt.shape`.
  State that its internal operation and earlier history are not traced.

## Pattern 1: select a reduction output

For `[[1, 2, 3], [4, 5, 6]]`, reducing axis 1 produces `[6, 15]`.
Explain that the selected second output combines source coordinates
`(1, 0)`, `(1, 1)`, and `(1, 2)`. The next selection changes the source
row and gives `1 + 2 + 3 = 6`.

```{literalinclude} ../../examples/lessons/reduction.py
:language: python
```

Ask what changes with `axis=0`. The result becomes `[5, 7, 9]`.
With `keepdims=True` in the original example, the output shape is `(2, 1)`
and the second row's focus is `(1, 0)`.

## Pattern 2: preserve repeated selections

A repeated gather and a reverse slice make two output positions read the
same input. Keep the selection identical in NumPy and the visualization.

```{literalinclude} ../../examples/lessons/repeated_index.py
:language: python
```

Outputs `(0, 0)` and `(2, 0)` both read `X[1, 2] = 6`.
Output `(1, 0)` reads `X[0, 2] = 3`. Do not collapse repeated result
positions into one highlighted source.

## Pattern 3: explain expansion before addition

At output `(1, 2)`, shapes `(2, 1)` and `(1, 3)` supply values 20 and 3.
Changing to `(0, 2)` changes the left source but keeps the right source.

```{literalinclude} ../../examples/lessons/broadcast.py
:language: python
```

The two controls are independent. In this pattern NumPy performs addition.
Use Pattern 5 when the arithmetic itself should appear in the traced view.

## Pattern 4: follow a chain back to its original input

Record every relevant step through a single `Flow`. Use unique explicit
names and tracked operands from that Flow. Register constants with
`flow.input(value)` before using them in Flow arithmetic.

```{literalinclude} ../../examples/lessons/operation_origins.py
:language: python
```

The selected result is:

```text
Y[0] = T[0, 0] + T[0, 1] + T[0, 2]
     = X[1, 2] + X[0, 2] + X[1, 2]
     = 6 + 3 + 6
     = 15
```

Then follow `Y[1] = 5 + 2 + 5 = 12`. Three contributions reach two original
cells. With several original inputs, key counts by the full `root.reference`
rather than coordinates alone. Counts describe participation, not derivatives
or general arithmetic coefficients.

Flow does not run native backend kernels or materialise complete intermediate
arrays. Existing arrays carry no recoverable prior history. The recipe captures
parameters such as index arrays, while source values are read afresh on each
query. Keep input shapes fixed.

## Pattern 5: preserve ordered binary arithmetic

Binary views retain both source roles. Inspect
`visual.trace.expression.operator` and `.operands`.
Their `trace.terms` is empty. Do not treat subtraction or division as an
implicit sum of products.

```{literalinclude} ../../examples/lessons/elementwise.py
:language: python
```

At output `(1, 2)`, inputs 20 and 3 give 23 by addition, 17 by subtraction,
and 60 by elementwise multiplication. A literal denominator 2 is a scalar at
`()`. Explain `multiply` separately from matrix multiplication.

## Pattern 6: advance through contributions

`rt.walkthrough` accepts `rt.sum`, `rt.mean`, `rt.matmul`, or a tracked
Flow result. It does not accept every function supported by `explore`.

```{literalinclude} ../../examples/lessons/guided_terms.py
:language: python
```

Separate the mean's subtotals 4, 9, and 15 from its final value `15 / 3 = 5`.
In the matrix product, the products `1 * 3` and `2 * 4` sum to 11.
`snapshot.factor_values`, `term_value`, `subtotal`, and `output_value`
expose different quantities.

Tell the learner which term to select next. Term indices are zero-based.
Occurrence indices belong to the current trace and must be read again after
the output changes. Check `snapshot.trace.complete` and
`snapshot.numeric_complete` separately. An available subtotal must not be
described as including omitted terms.

## Pattern 7: inspect a denominator's own operation

For row normalization, record the row total with `keepdims=True`, then
record division. Inspect the denominator's sum as a separate occurrence.

```{literalinclude} ../../examples/lessons/row_normalization.py
:language: python
```

The output is `9 / (3 + 6 + 9) = 0.5`. The value 9 occurs on two paths.
That count is not a coefficient of two. Discover the sum occurrence from
the current trace rather than hardcoding an ID.

## Pattern 8: predict a shape before revealing it

`rt.reduction_playground` accepts `rt.sum` or `rt.mean`. It keeps the
source and corresponding NumPy expression visible while hiding the answer.

```{literalinclude} ../../examples/lessons/reduction_axes.py
:language: python
```

For shape `(2, 3, 4)`, reducing axes `(0, 2)` gives `(3,)`, or
`(1, 3, 1)` with `keepdims=True`. `axis=()` reduces no axes and
`axis=None` reduces all axes. Each parameter change resets the prediction.
A still-valid output focus is retained, otherwise it resets to the first
output or no focus for an empty output.

Use `set_parameters`, `reveal()`, and `hide()` for programmatic changes.
The hidden result remains accessible through `playground.visual`.
Predict mode hides the answer in the interface, not its numerical evaluation.
Calculation limits remain in force.

## Pattern 9: keep a comparison and its choice connected

When the comparison itself is part of the lesson, record it in a `Flow` and
pass that tracked result to `flow.where`. Do not turn the comparison into a
plain NumPy mask if the learner needs to inspect how the condition was formed.

```{literalinclude} ../../examples/lessons/conditional_choices.py
:language: python
```

The trace lists the condition, true branch, and false branch as candidates.
The evaluated choice is separate. The thick dashed border marks the branch
that supplied the focused output. Explain that Flow evaluates both recorded
branches within its shared budget.

## Pattern 10: distinguish an extreme value from its position

Use `rt.explore` for a direct operation. Use `Flow` and `walkthrough` when the
learner should inspect candidate values and the source of the winner.

```{literalinclude} ../../examples/lessons/extrema_positions.py
:language: python
```

For `[2, 8, 8]`, `max` returns 8 and `argmax` returns position 1. The first
maximum wins the tie, and its source cell gets the thick dashed border. Extrema
accept real scalar values and follow the row-major first-tie rule described in
the [conditions and extrema guide](conditions-and-extrema.md).

## Pattern 11: save only prepared lesson states

Capture prepared views or controller states for playback outside a live
notebook. The recording cannot calculate an output that was not captured.

```{literalinclude} ../../examples/lessons/portable_recording.py
:language: python
```

Use the offline file when live controls are unavailable. Tell the learner that
its navigation changes captured states only, not tensor inputs or parameters.

## Delivery and numerical boundaries

All patterns above use the canonical files run by
`python scripts/check_lessons.py`. See [verified lessons](verified-lessons.md)
and [guided lessons](guided-lessons.md) for exercises and checked answers.

- **Live controls.** Use a notebook with a running kernel and widget support.
  Output cells support click, Enter, and Space. Coordinate fields reach
  positions hidden by a large preview. Re-read `controller.visual` after
  changes, and keep controllers open until the learner finishes.
- **Static output.** `visual.save("lesson.svg")` saves the figure only.
  Save `visual.text` separately when its explanation is needed outside the
  notebook. SVG export does not produce PNG or retain live Python controls.
- **Portable output.** `rt.capture_lesson` saves prepared states as HTML or JSON.
  Its bounded offline player does not run Python or calculate new results.
- **Numerical semantics.** Preview arithmetic uses Python scalars. Native
  accumulation dtype, rounding, and overflow can differ. In particular,
  `divide` raises `ZeroDivisionError` for any zero denominator, including
  `0 / 0`. Use real NumPy results when teaching backend numerical behaviour.
- **Scalars and empties.** A scalar has one coordinate, `()`. An empty output
  has none. An empty reduction group can still produce a real output cell
  with sum 0 or undefined mean NaN.
- **Bounded traces.** Sum-of-products `OutputTrace.terms` stores at most eight
  terms. Check `.complete`. Binary expressions use `.expression` instead.
  Flow defaults to `max_depth=6`, `max_nodes=80`, and `max_edges=120`.
  Check its completeness and truncation reasons before claiming that a source
  list is exhaustive.
- **Bounded values.** Arithmetic and Flow queries default to
  `max_terms=10_000` and `max_total_terms=100_000`, with different scopes.
  A question mark means unevaluated, not zero. `TrackedTensor.value` raises
  `rt.ValueBudgetExceeded` if its plan exceeds a limit. Prefer a smaller
  teaching example to silently disabling limits.
- **Storage.** Logical origins do not prove whether an operation made a copy.
  Use `rt.memory` for the backend's available storage information.

Before returning a generated lesson, verify its imports, output shape,
focused coordinate, source coordinates, and local arithmetic. Check a second
selection or parameter choice. Preserve duplicates and local divisors.
Label generated values, partial traces, and skipped evaluation accurately.
Give the learner a concrete prediction and a way to check the answer.
