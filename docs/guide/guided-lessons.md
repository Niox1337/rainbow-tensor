# Guided arithmetic and reductions

Use these lessons after [the verified foundations](verified-lessons.md).
They introduce ordered elementwise expressions, one contribution at a time,
and reduction axes that the learner can change. Every Python listing is the
same source file executed by the lesson checker.

## Keep the two operands in order

At output `(1, 2)`, broadcasting pairs `a[1, 0] = 20` with
`b[0, 2] = 3`. Addition gives 23, subtraction gives 17, and multiplication
gives 60. Swapping the operands changes subtraction and division.

```{literalinclude} ../../examples/lessons/elementwise.py
:language: python
```

For elementwise arithmetic, inspect `visual.trace.expression.operator` and
`.operands`. The two references keep their left and right roles.
`trace.terms` is empty for a binary expression, so do not interpret it as a
sum of products.

Static `add`, `subtract`, `multiply`, and `divide` accept numeric literals
as scalar operands. A Flow still requires every operand to be registered,
including constants such as `flow.input(2)`. Shape tuples generate example
values. They are not literal tensor data.

Division follows Python scalar arithmetic. A zero denominator raises
`ZeroDivisionError`, including `0 / 0`. Use actual nonzero denominator
values in an introductory lesson.

## Build a result one contribution at a time

Before advancing a mean over `[4, 5, 6]`, predict its running subtotal.
The subtotals are 4, 9, and 15. The completed mean is `15 / 3 = 5`.
The subtotal and final output answer different questions.

```{literalinclude} ../../examples/lessons/guided_terms.py
:language: python
```

`rt.walkthrough` accepts `rt.sum`, `rt.mean`, `rt.matmul`, or a tracked
Flow result. The controls select an output, an occurrence in its recorded
history, and a term within that occurrence. Use Previous and Next to move
between available contributions.

The Python methods are `next_term()`, `previous_term()`,
`select_term(index)`, `select_occurrence(index)`, and `set_focus(coord)`.
Term indices are zero-based. Occurrence indices belong to the current trace.
Read them again after choosing another output.

`snapshot.factor_values` describes the selected contribution.
`snapshot.term_value` is its product or ordered binary result.
`snapshot.subtotal` accumulates available reduction products through the
selected term. It is measured before a mean's divisor is applied.
`snapshot.output_value` is the selected occurrence's complete value when
the numerical plan succeeds. A truncated trace does not imply that an
available subtotal includes omitted terms. Check `snapshot.trace.complete`
and `snapshot.numeric_complete` separately.

Walkthrough values use Flow's bounded Python scalar evaluator. Accumulation
and division can differ from a tensor backend's dtype and overflow behavior.
For numerical work, compute with the original array library and compare a
small result before using its visualization to teach the calculation.

Keep `mean_lesson` and `product_lesson` open while exploring, then call
their `.close()` methods. Their current `.visual` remains an exportable SVG
and text explanation.

## Follow the denominator of a normalized row

Why does the final score in the second row become 0.5? Its numerator is 9.
Its denominator is a separate sum, `3 + 6 + 9 = 18`.

```{literalinclude} ../../examples/lessons/row_normalization.py
:language: python
```

`keepdims=True` makes the row totals shape `(2, 1)`, so each total can
broadcast across its row. The source value 9 appears in two paths, once as
the numerator and once inside the denominator. That occurrence count is not
a coefficient of two. The operation remains `9 / (3 + 6 + 9)`.

Select the sum occurrence to inspect its three terms, then select a new
final output to return to the root expression. The example discovers the
sum occurrence from the current trace instead of assuming a fixed ID.

## Predict the shape before revealing it

For source shape `(2, 3, 4)`, reducing axes `(0, 2)` leaves shape `(3,)`.
With `keepdims=True`, it instead leaves `(1, 3, 1)`. The middle output
combines eight values and has the same sum, 92, in both cases.

```{literalinclude} ../../examples/lessons/reduction_axes.py
:language: python
```

The playground accepts `rt.sum` or `rt.mean`. Choose all axes, no axes, or
a selected tuple, then apply the parameters. In predict mode, enter a shape
tuple and reveal the result. The source figure and current NumPy expression
stay visible. `axis=None` reduces everything, while `axis=()` reduces
nothing.

From Python, call `set_parameters(axis=..., keepdims=...)`, `reveal()`,
and `hide()`. A parameter change starts a new prediction. A valid focus is
preserved. If its coordinate no longer fits the result, the explorer starts
at the first output or has no focus for an empty output.

Prediction mode hides the result controls. It does not postpone constructing
the result view, and `playground.visual` remains available to Python.
Set `predict=False` when you want every result shown immediately.
Calculation budgets remain in force as axes and focus change.

Call `playground.close()` when finished. For a static lesson, use the current
`.visual`, or call the matching operation with the selected axis and
`keepdims` values.

## Check the lessons

```bash
python scripts/check_lessons.py elementwise guided-terms row-normalization reduction-axes
```

Each lesson checks NumPy values, tensor shapes, selected sources, and at least
one state change. The checker closes controls and removes temporary exports.
Browser event tests separately cover the live controls.

For a notebook containing these four lessons, open
[16_guided_tensor_lessons.ipynb](../../examples/16_guided_tensor_lessons.ipynb).
