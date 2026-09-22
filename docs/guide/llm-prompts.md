# Agent instructions: teach NumPy with Rainbow Tensor

This page is written for an LLM agent. When a user gives you this URL, use it
as the API and teaching reference for generating Rainbow Tensor code that helps
them understand NumPy visually. Read the instructions directly. The user does
not need to copy a prompt, fill in a template, or learn this reference first.

The examples target **rainbow-tensor 1.3.0**. Your output should answer the
user's NumPy question with runnable code, a useful visualization, and a short
explanation of how particular output elements were produced.

## Your task

Use the user's expression, data, and learning goal when supplied. Otherwise,
choose a small deterministic NumPy array with distinct values. Default to a
static notebook example. Add interaction only when it helps and the environment
supports it. If the user only supplies this URL without a topic, ask which
NumPy operation or expression they want to understand.

Follow this sequence when composing your answer:

1. Identify the NumPy concept and one concrete question to explain. For a
   reduction, this might be which input elements produce `result[1]`.
2. Write the real NumPy computation. Keep it separate from the visualization.
   Show the input values, input shape, and expected result shape.
3. Choose the matching Rainbow Tensor call from the reference below. Focus
   on one valid output coordinate so its source elements can be followed.
4. Explain the coordinate mapping or local arithmetic in words. Name the
   coordinates as well as their colours. Preserve repeated contributions.
5. Check the shape and at least one value or source mapping with assertions.
   Compare with the real NumPy result. Say which checks you actually ran.
6. Offer one small variation or prediction question, such as changing the
   focus, axis, or slice direction. Include enough information to check it.

Return complete, runnable cells with imports and brief explanations between
cells. Use `display(...)` when a cell contains multiple figures or other output.
For scripts, save SVG files and print their `.text` explanation. If you cannot
execute code, state that its checks are unverified.

Write the explanation in the user's language. Keep Python identifiers and
comments in English. Leave the theme automatic unless requested otherwise.
Use `rt.available_languages()` before selecting a figure language with
`rt.set_language(...)`. Do not invent locale codes or translation keys.

## Runtime and return types

Install into the environment used by the notebook kernel or Python script:

```bash
python -m pip install "rainbow-tensor==1.3.0"
```

The distribution name is `rainbow-tensor`. Import it as
`import rainbow_tensor as rt`. NumPy and IPython are dependencies. The optional
`rainbow-tensor[interactive]` extra provides notebook controls.

Keep these three kinds of objects distinct:

| Object | Purpose | How to use it |
| --- | --- | --- |
| NumPy array | Execute the computation and check NumPy semantics | Index it, call NumPy operations, inspect `.shape` |
| `TensorVisual` | Display one operation and its metadata | Use `display(visual)`, `.svg`, `.text`, `.save(path)`, `.result_shape` |
| `TrackedTensor` | Record an explicit operation chain within a `Flow` | Use `.shape`, `.visualize(focus=...)`, `.trace(coordinate)`, `.value(coordinate)` |

Static calls such as `rt.sum(x, axis=1)` return `TensorVisual`, not arrays.
Do not index that object or pass it as the input to another tensor operation.
For a single-input operation, `visual.shape` describes the source and
`visual.result_shape` describes the result. A shape tuple such as `(2, 3)`
creates generated row-major values starting at zero. Pass an actual array when
explaining the user's values.

`focus` is a tuple of **output** coordinates. It is not the input selection.
Use `focus=()` for a scalar output. Empty outputs have no valid coordinate,
so omit focus. Keep the first example small enough to show all relevant cells.

## Translate NumPy expressions into visualizations

Use this table to select an entry point. The rows are API patterns, with
`x`, `a`, `b`, `selection`, and `focus` supplied by your example.

| NumPy expression or concept | Rainbow Tensor call | What to explain |
| --- | --- | --- |
| `x.shape` | `rt.shape(x)` | Axis numbers, lengths, and nested structure |
| `x[selection]` | `rt.index(x, selection, show_result=True, focus=focus)` | Which source coordinate supplies the chosen output |
| `x.reshape((3, 2))` | `rt.reshape(x, (3, 2), focus=focus)` | Row-major positions before and after a shape change |
| `x.transpose((1, 0))` | `rt.transpose(x, (1, 0), focus=focus)` | How swapping the two axes moves coordinates |
| `np.take(x, [1, 0, 1], axis=0)` | `rt.take(x, [1, 0, 1], axis=0, focus=focus)` | Reordered and repeated selections |
| `np.repeat(x, 2, axis=0)` | `rt.repeat(x, 2, axis=0, focus=focus)` | Separate copies that share a source |
| `np.concatenate([a, b], axis=0)` | `rt.concatenate([a, b], axis=0, focus=focus)` | Joining an existing axis |
| `np.stack([a, b], axis=0)` | `rt.stack([a, b], axis=0, focus=focus)` | Introducing a new axis |
| `np.broadcast_arrays(a, b)` | `rt.broadcast(a, b, focus=focus, focus_operand=0)` | Expansion of one operand to the common shape |
| `x.sum(axis=1, keepdims=True)` | `rt.sum(x, axis=1, keepdims=True, focus=focus)` | Which axis is reduced and why a length-one axis remains |
| `x.mean(axis=1)` | `rt.mean(x, axis=1, focus=focus)` | The selected group's sum and its divisor |
| `a @ b` | `rt.matmul(a, b, focus=focus)` | The ordered products forming a chosen output |
| `np.einsum("ij,jk->ik", a, b)` | `rt.einsum("ij,jk->ik", a, b, focus=focus)` | Retained and contracted indices |
| `x.strides` and storage layout | `rt.memory(x)` | Backend-reported storage, separately from logical origins |

The reshape row requires six elements, the transpose and reduction rows assume
suitable two-dimensional arrays, and every focus must match its output shape.
Other supported shape operations are `swapaxes`, `moveaxis`, `squeeze`, and
`expand_dims`. Consult the [API reference](../api) for their exact signatures.

Do not assume that every NumPy keyword is supported:

- `repeat` and `take` default to `axis=0` and require an integer axis. NumPy's
  default flattening behaviour is different. To reproduce it, flatten the
  array first with `x.reshape(-1)` and visualize that flattened input.
- `reshape` has no `order` argument. Do not claim that a view explains a
  Fortran-order reshape or proves whether NumPy copied the array.
- `sum` and `mean` accept `axis=None`, an integer, or a tuple, plus `keepdims`.
  They do not accept NumPy's `dtype`, `out`, or `where` keywords.
- Indexing supports integer positions, slices, ellipsis, new axes, boolean
  masks, and advanced integer arrays. Boolean scalar indices are unsupported.
- `broadcast` explains operand expansion. It does not add or multiply the
  operands. Choose `focus_operand=0` or `1` to inspect the corresponding input.
- There is no general `rt.add`, `rt.multiply`, or automatic NumPy interception.
  For an unsupported operation, compute its result with NumPy and display
  that array with `rt.shape`. State that its internal operation is not traced.

## Pattern 1: generate a focused reduction explanation

Use this self-contained pattern when the user asks about reduction axes. Keep
the NumPy result visible in the generated answer and connect it to the figure.

```python
import numpy as np
import rainbow_tensor as rt
from IPython.display import display

x = np.arange(1, 7).reshape(2, 3)
result = x.sum(axis=1)
assert result.tolist() == [6, 15]

visual = rt.sum(x, axis=1, focus=(1,))
assert visual.result_shape == result.shape == (2,)
assert result[1] == x[1, 0] + x[1, 1] + x[1, 2] == 15

display(rt.shape(x))
display(visual)
print(result)
```

Explain that axis 0 selects rows and axis 1 moves through entries within a
row. Reducing axis 1 produces one sum per row. Output `(1,)` comes from
`X[1, 0]`, `X[1, 1]`, and `X[1, 2]`, giving `4 + 5 + 6 = 15`.

For a follow-up, switch to `axis=0`. The NumPy result is `[5, 7, 9]` with
shape `(3,)`. If you instead add `keepdims=True` to the original reduction,
the result shape is `(2, 1)` and the matching focus becomes `(1, 0)`.

## Pattern 2: generate an indexing explanation

Use a repeated gather with a reverse slice to show why distinct output
positions can read the same source. Keep the original selection unchanged
between the NumPy expression and the visual call.

```python
import numpy as np
import rainbow_tensor as rt
from IPython.display import display

x = np.arange(1, 7).reshape(2, 3)
selection = ([1, 0, 1], slice(None, None, -1))
result = x[selection]
assert result.tolist() == [[6, 5, 4], [3, 2, 1], [6, 5, 4]]

visual = rt.index(x, selection, show_result=True, focus=(0, 0))
assert visual.shape == x.shape == (2, 3)
assert visual.result_shape == result.shape == (3, 3)
assert visual.index_mapping.source_coord((0, 0)) == (1, 2)
assert visual.index_mapping.source_coord((2, 0)) == (1, 2)
assert visual.trace.output_coord == (0, 0)
display(visual)
```

Explain that outputs `(0, 0)` and `(2, 0)` both read `X[1, 2]`, whose value
is 6. Changing the focus to `(1, 0)` selects `X[0, 2]`, whose value is 3.
Do not collapse the two occurrences into one output position.

## Pattern 3: explain broadcasting before arithmetic

For a question about `a + b`, visualize how both operands expand, then compute
the addition in NumPy. The broadcast figure only explains the expansion.

```python
import numpy as np
import rainbow_tensor as rt
from IPython.display import display

a = np.array([[10], [20]])
b = np.array([[1, 2, 3]])
expanded_a, expanded_b = np.broadcast_arrays(a, b)
result = a + b
assert result.tolist() == [[11, 12, 13], [21, 22, 23]]
assert expanded_a[1, 2] == a[1, 0] == 20
assert expanded_b[1, 2] == b[0, 2] == 3
assert result[1, 2] == expanded_a[1, 2] + expanded_b[1, 2] == 23

display(rt.broadcast(a, b, focus=(1, 2), focus_operand=0))
display(rt.broadcast(a, b, focus=(1, 2), focus_operand=1))
display(rt.shape(result))
```

Explain right-aligned dimension matching: `(2, 1)` and `(1, 3)` both expand
to `(2, 3)`. At output `(1, 2)`, the values come from `a[1, 0]` and `b[0, 2]`.
The final shape view shows the addition's values without tracing the addition.

## Pattern 4: trace a NumPy expression across operations

Use `Flow` when the user wants to know how several steps produced a value.
Record each operation explicitly. Keep all tracked operands in the same Flow,
and use unique names there. NumPy operations on the original arrays do not
record history. Flow does not run native backend kernels or materialise
intermediate arrays.

This pattern explains `x[selection].T.sum(axis=1)` and checks it against NumPy:

```python
import numpy as np
import rainbow_tensor as rt
from IPython.display import display

x = np.arange(1, 7).reshape(2, 3)
selection = ([1, 0, 1], slice(None, None, -1))
result = x[selection].T.sum(axis=1)

flow = rt.Flow()
source = flow.input(x, name="X")
selected = flow.index(source, selection, name="S")
transposed = flow.transpose(selected, name="T")
y = flow.sum(transposed, axis=1, name="Y")

assert selected.shape == (3, 3)
assert transposed.shape == (3, 3)
assert y.shape == result.shape == (3,)
assert [y.value((i,)) for i in range(3)] == result.tolist() == [15, 12, 9]

trace = y.trace((0,))
assert trace.complete
counts = {root.reference.coordinate: root.count for root in trace.roots}
assert counts == {(1, 2): 2, (0, 2): 1}

visual = y.visualize(focus=(0,))
assert visual.result_shape == result.shape
assert visual.provenance.complete
display(visual)
```

Explain the local operations in order, retaining the repeated term:

```text
Y[0] = T[0, 0] + T[0, 1] + T[0, 2]
     = S[0, 0] + S[1, 0] + S[2, 0]
     = X[1, 2] + X[0, 2] + X[1, 2]
     = 6 + 3 + 6
     = 15
```

Three contributions reach two distinct original cells. Root occurrence counts
are participation counts, not derivatives or general arithmetic coefficients.
Keep products and each mean's divisor local to the operation that produced
them. With several original inputs, retain the full `root.reference` when
collecting counts so equal coordinates in different arrays stay distinct.

Flow provides the same operation families listed above, excluding the
standalone `shape` and `memory` views. `flow.broadcast(a, b)` returns a tuple
of tracked outputs, one per operand. Unpack it before recording another
operation. To flatten before `repeat` or `take`, record
`flow.reshape(node, (-1,))` first.

## Deliver notebook controls or portable output

Use static figures by default. When live interaction is appropriate,
`rt.explore(rt.sum, x, axis=1)` explores one operation and `rt.explore(node)`
explores a recorded chain. Do not pass an already rendered `TensorVisual`.

The following cell reuses `y` from Pattern 4 and falls back when optional
widget dependencies are missing:

```python
try:
    explorer = rt.explore(y, focus=(0,))
except ImportError:
    lesson_view = y.visualize(focus=(0,))
else:
    lesson_view = explorer
display(lesson_view)
```

A live kernel and a notebook host with widget support are also required.
An unsupported host may need the explicit static call even if the import
succeeds. Tell the learner that clicking a result cell or changing coordinate
fields selects another output. Call `explorer.close()` when the live explorer
is no longer needed, not immediately after displaying it.

For a script or portable output, reuse `y` from Pattern 4 and save the figure
and explanation separately:

```python
from pathlib import Path

visual = y.visualize(focus=(0,))
visual.save("tensor-origins.svg")
Path("tensor-origins.txt").write_text(visual.text, encoding="utf-8")
print(visual.text)
```

`save` writes the figure only. It does not save a PNG or embed the accompanying
text explanation. A saved SVG has no live Python controls.

## Guardrails for generated explanations

- **Numerical semantics.** Arithmetic previews use Python scalar arithmetic.
  NumPy dtype accumulation, overflow, and rounding can differ. Use the real
  NumPy result as the reference when those details are the lesson's subject.
- **Scalars and empties.** A scalar has one coordinate, `()`. An empty output
  has none. Distinguish it from an empty reduction group, which can produce a
  real output cell with sum 0 or mean NaN.
- **Partial traces.** A single-operation `OutputTrace` stores at most eight
  terms. Check `.complete`. Flow tracing defaults to `max_depth=6`,
  `max_nodes=80`, and `max_edges=120`. Check `.complete` and
  `.truncated_reasons` before calling a list of sources exhaustive.
- **Bounded evaluation.** Arithmetic previews and Flow value queries default
  to `max_terms=10_000` and `max_total_terms=100_000`. Their scopes differ.
  A budget-limited figure shows `?`, meaning unevaluated, not zero.
  `TrackedTensor.value` raises `rt.ValueBudgetExceeded`. Simplify a teaching
  example rather than silently removing its limits.
- **Live inputs.** Flow copies operation parameters such as index arrays.
  Input values remain live and are read again on refresh. Keep their shapes
  fixed. An edited input value can change the next answer.
- **Unsupported behaviour.** Do not invent methods, keywords, or tracing
  support. Displaying a NumPy result with `rt.shape` does not recover its
  earlier operations. Logical origins do not establish physical memory use.

Before returning generated code, check that it runs in order, compares the
NumPy result with the visualized operation, uses valid focus coordinates, and
explains at least one actual source mapping or calculation. Keep duplicate
contributions visible. Label generated values, incomplete traces, and
unverified checks accurately. The static version should remain useful when
widgets are unavailable.

For less common cases, consult the [API reference](../api),
[indexing rules](indexing.md), [numerical semantics](reductions-and-math.md),
[Flow and its budgets](provenance.md), [widget behaviour](interactive.md), and
[language configuration](translations.md).
