# LLM prompt for beginner tensor lessons

Use this page as context when asking an LLM to build a lesson with
rainbow-tensor 1.3.0. It includes a reusable prompt, the API choices that matter,
and examples with checked results. Give the model the whole page when possible,
then state the operation and the learner's experience.

## Copy this prompt

Replace the bracketed fields before sending it to an LLM. The instructions
work for a short notebook lesson or a single worked example.

```text
Write a beginner tensor lesson using rainbow-tensor 1.3.0.

Topic: [for example, repeated indexing followed by transpose and sum]
Learner: [for example, knows Python lists but is new to tensor axes]
Environment: [Jupyter with widgets, Jupyter without widgets, or a Python script]
Explanation language: [language]
Lesson length: [one worked example, or a short sequence of notebook cells]

Use these package facts:
- Install name: rainbow-tensor. Import: import rainbow_tensor as rt.
- Start with small, deterministic NumPy arrays, usually np.arange or np.array.
  A shape tuple such as (2, 3) creates generated row-major example values
  starting at zero. It does not supply the learner's actual data.
- rt.shape(x) shows structure. rt.index(x, selection, show_result=True)
  compares input and result. Add focus=(...) to follow one output coordinate.
- Static operation calls return TensorVisual, a display object, not an array.
  Do not index it, chain it into another tensor operation, or claim it holds
  an executable backend tensor. Use .result_shape for an operation's output
  shape. .shape describes its source. Read .svg and .text or call .save(path).
- Supported static operation names are index, reshape, transpose, swapaxes,
  moveaxis, squeeze, expand_dims, repeat, take, concatenate, stack, broadcast,
  sum, mean, matmul, and einsum. Use their documented arguments. rt.shape and
  rt.memory are structure and storage views, not arithmetic operations.
- repeat and take default to axis=0 and require an integer axis. They do not
  accept axis=None. Flatten a NumPy input with x.reshape(-1), or a tracked
  input with flow.reshape(node, (-1,)), before repeating or taking positions.
- For a chain, create flow = rt.Flow(), register each array with
  flow.input(array, name="X"), and call the corresponding flow methods.
  They return TrackedTensor objects with the logical output .shape.
  Keep operands in the same Flow. Each explicit name must be unique there.
  Use node.visualize(focus=(...)), node.trace((...)), or node.value((...)).
  These recipes do not intercept NumPy operations, recover earlier history,
  execute native backend kernels, or materialise intermediate arrays.
- focus is a tuple of coordinates in the output, not an input selection.
  Use () for a scalar. Empty outputs have no valid focus coordinate.
- rt.explore(rt.sum, x, axis=1) explores one operation.
  rt.explore(node) explores a recorded chain. Both need the interactive extra,
  a live notebook kernel, and widget support. Always supply a static fallback.
  Do not pass an already rendered TensorVisual to rt.explore.
- A static broadcast view takes two inputs and focus_operand=0 or 1 chooses
  its focused output. flow.broadcast(a, b) returns two tracked outputs,
  not their sum or product. Unpack them before recording another operation.
- Coordinate traces preserve repeated contributions. A source cell appearing
  twice can contribute twice even though the picture highlights it once.
  Root occurrence counts describe participation, not derivatives or general
  arithmetic coefficients. Keep products and each mean's divisor local to
  the operation that produced them.
- Values in arithmetic previews use Python scalar arithmetic. Native backend
  dtype accumulation, overflow, and rounding can differ. Compute native
  results separately when the lesson needs that numerical behaviour.
- Keep default budgets for beginner examples. A question mark means a value
  was not evaluated, not zero. Inspect trace completeness before calling a
  source list exhaustive. Large previews can omit cells.
- Python code, identifiers, and comments should be English. Write the teaching
  prose in the requested language. Use rt.available_languages() to inspect
  installed catalogs, then rt.set_language(...) for supported figure text.
  Never invent a locale code or translation key. Leave theme selection
  automatic unless the learner requests a specific appearance.

Teach in this order:
1. State one question the learner should be able to answer.
2. Show the input values and shapes. Explain which dimension each axis names.
3. Ask the learner to predict one output's shape or source coordinate.
4. Render a focused example. Name the output coordinate and its input
   coordinates in words as well as colour. Explain each axis change.
5. Write the local arithmetic or coordinate mapping for that output. Preserve
   duplicate terms and operation order. Check it with executable assertions.
6. Change one coordinate or parameter and ask a short follow-up question.

Return runnable cells with imports, expected results, and brief explanations
between cells. Use explicit display(...) calls if a cell must show several
figures. Otherwise put the figure last in its own notebook cell. For a Python
script, save an SVG and print its text explanation. Keep the first example
small enough that all relevant cells are visible. State which assertions you
actually ran. If execution is unavailable, label them as unverified.

Stay within the documented API. For an unsupported operation, compute its
native result separately and show it with rt.shape, stating that its internal
operation and earlier history are not traced. Do not invent a Flow method.
```

## Choose the right entry point

The [API reference](../api) lists full signatures. These are the choices
that most often change the explanation:

| Question | Entry point | What to inspect |
| --- | --- | --- |
| What does each axis contain? | `rt.shape(x)` | Shape, axis legend, nested frames |
| Which input position becomes this output? | `rt.index(x, selection, focus=(0, 0))` | `visual.index_mapping` and `visual.trace` |
| How does one transformation move elements? | `rt.reshape(x, (3, 2), focus=(2, 1))` | Source coordinate and `visual.result_shape` |
| Which values form this sum or product? | `rt.sum(x, axis=1, focus=(0,))` or `rt.matmul(a, b, focus=(0, 0))` | Ordered `visual.trace.terms` |
| How did several steps produce this value? | `node.visualize(focus=(0,))` | `visual.provenance`, plus `node.trace((0,))` |
| Can the learner choose another result? | `rt.explore(operation, *args, **kwargs)` or `rt.explore(node)` | Clickable result cells and coordinate fields |
| Is the physical array contiguous or a view? | `rt.memory(x)` | Backend-reported storage information |

The reshape example assumes a six-element input. Focus coordinates in the
other examples must match their own output shapes. Physical storage and logical
origins answer different questions. A traced transpose does not establish
whether a backend made a copy.

Static views and Flow share these operation families:

| Family | Operation names |
| --- | --- |
| Selection | `index`, `take`, `repeat` |
| Shape changes | `reshape`, `transpose`, `swapaxes`, `moveaxis`, `squeeze`, `expand_dims` |
| Combining inputs | `concatenate`, `stack`, `broadcast` |
| Arithmetic | `sum`, `mean`, `matmul`, `einsum` |

For `concatenate` and `stack`, pass a sequence of operands. For `einsum`, put
the subscript string first, for example `rt.einsum("ij,jk->ik", a, b)`.
`repeat` and `take` default to `axis=0` and require an integer axis. They do
not accept `axis=None`. To flatten first, use `x.reshape(-1)` on a NumPy array,
or record `flow.reshape(node, (-1,))` before calling a Flow method. Static
`broadcast` takes two operands.
`Flow.broadcast` also supports more than two and returns one tracked output
per input.

## Template: explain one index

Install the package and NumPy in the Python environment used by the notebook
kernel or script:

```bash
python -m pip install rainbow-tensor numpy
```

The question is: *which input value becomes output `(0, 0)`?* Row selection
`[1, 0, 1]` repeats the second input row. The slice reverses each selected row.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(1, 7).reshape(2, 3)
selection = ([1, 0, 1], slice(None, None, -1))
expected = x[selection]
assert expected.tolist() == [[6, 5, 4], [3, 2, 1], [6, 5, 4]]

visual = rt.index(x, selection, focus=(0, 0))
assert visual.shape == (2, 3)
assert visual.result_shape == (3, 3)
assert visual.index_mapping.source_coord((0, 0)) == (1, 2)
assert visual.index_mapping.source_coord((2, 0)) == (1, 2)
assert visual.trace.output_coord == (0, 0)
visual
```

Output `(0, 0)` reads `X[1, 2]`, whose value is 6. Output `(2, 0)` reads the
same input position. Ask the learner what changes when the focus moves to
`(1, 0)`. The answer is input `(0, 2)`, whose value is 3.

## Template: explain a chain

This example is self-contained. It records selection, transpose, and sum,
then asks why `Y[0]` is 15.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(1, 7).reshape(2, 3)
flow = rt.Flow()
source = flow.input(x, name="X")
selected = flow.index(source, ([1, 0, 1], slice(None, None, -1)), name="S")
transposed = flow.transpose(selected, name="T")
y = flow.sum(transposed, axis=1, name="Y")

assert selected.shape == (3, 3)
assert transposed.shape == (3, 3)
assert y.shape == (3,)
assert [y.value((i,)) for i in range(3)] == [15, 12, 9]

trace = y.trace((0,))
assert trace.complete
counts = {root.reference.coordinate: root.count for root in trace.roots}
assert counts == {(1, 2): 2, (0, 2): 1}

visual = y.visualize(focus=(0,))
assert visual.result_shape == (3,)
assert visual.provenance.complete
visual
```

Transpose turns the first column of `S` into the first row of `T`. Summing
axis 1 combines the entries along each row:

```text
Y[0] = T[0, 0] + T[0, 1] + T[0, 2]
     = S[0, 0] + S[1, 0] + S[2, 0]
     = X[1, 2] + X[0, 2] + X[1, 2]
     = 6 + 3 + 6
     = 15
```

Three terms reach two distinct original cells. The count of two for `(1, 2)`
must survive the explanation. A dictionary keyed only by coordinates is safe
here because the chain has one original input. With several inputs, retain
the full `root.reference` so equal coordinates in different inputs remain
distinct.

For the follow-up, focus on `(1,)` and predict `5 + 2 + 5 = 12`. Then change
the reduction to `flow.mean(transposed, axis=1, name="M")` and explain the
division by three at that step.

## Add interaction or export a static lesson

For live controls, install `rainbow-tensor[interactive]` in the kernel's
environment. The following cell reuses `y` from the chain:

```python
try:
    explorer = rt.explore(y, focus=(0,))
except ImportError:
    lesson_view = y.visualize(focus=(0,))
else:
    lesson_view = explorer
lesson_view
```

This fallback handles missing optional dependencies. A notebook host without
widget support may still need the explicit static call
`y.visualize(focus=(0,))`. Click a visible result cell or use its coordinate
fields to select another output. When finished with a live explorer, call
`explorer.close()` to release its widget connections.

For a script or an exported lesson, save the figure and its explanation
separately:

```python
from pathlib import Path

visual = y.visualize(focus=(0,))
visual.save("tensor-origins.svg")
Path("tensor-origins.txt").write_text(visual.text, encoding="utf-8")
print(visual.text)
```

`save` writes the figure, while `.text` contains the notebook's accompanying
explanation. The default renderer produces SVG. It does not save a PNG
screenshot, and a saved SVG has no live Python controls. A screenshot can be
captured separately by the notebook host or browser when required.

## Handle boundaries honestly

- **Scalar and empty results.** A scalar has shape `()` and one element at
  `focus=()`. An empty output has no element to select. Leave `focus=None`.
  Distinguish an empty output from an empty reduction group. The latter can
  produce a real output cell with sum 0 or mean NaN.
- **Partial traces.** A single-operation `OutputTrace` stores at most eight
  terms. Check `.complete`. Flow tracing defaults to `max_depth=6`,
  `max_nodes=80`, and `max_edges=120`. Check its `.complete` and
  `.truncated_reasons` before claiming to list every contribution. Root counts
  from an incomplete trace cover only paths actually reached.
- **Bounded evaluation.** Arithmetic previews and Flow value queries default
  to `max_terms=10_000` and `max_total_terms=100_000`. Their scopes differ, as
  described in the guides below. A budget-limited figure shows `?`.
  `TrackedTensor.value` raises `rt.ValueBudgetExceeded`. Do not silently
  remove limits to make a large example render.
- **Live values.** Flow records operation parameters, including copied index
  arrays. Input array values remain live and are read again on refresh.
  Keep input shapes fixed. Editing an input value can change the next answer.
- **Language and colour.** Use installed translation catalogs for figure
  text. Do not translate Python API names. Keep repository examples and code
  comments in English. Name coordinates and axes explicitly so the lesson
  remains understandable without relying on colour alone.

## Check the generated lesson

Before sharing it, verify that:

- Every cell runs in order with declared imports and dependencies.
- The expected output shape, focused coordinate, and source coordinates agree.
- At least one asserted value or mapping checks the lesson's main claim.
- Repeated contributions and local products or divisors remain visible.
- Truncation, generated values, and skipped arithmetic are labelled accurately.
- The static version is useful without widgets, and saved text accompanies SVG
  when the explanation is needed outside the notebook.
- The learner gets one concrete prediction question and a checked answer.

Use the [indexing guide](indexing.md) for coordinate rules,
[reductions and math](reductions-and-math.md) for numerical semantics,
[cross-operation origins](provenance.md) for Flow and its budgets,
[interactive focus](interactive.md) for widget behaviour, and
[translations](translations.md) for language configuration.
