# Conditions, choices, and extrema

A conditional operation has several possible sources. Its structural trace
shows those candidates. Numerical evaluation separately identifies the source
that supplies the current result. Keep both ideas visible when teaching a
comparison followed by a choice.

## Compare, choose, then reduce

`greater`, `greater_equal`, `less`, `less_equal`, `equal`, and `not_equal`
compare broadcast operands in their original left-to-right order. They work
as ordinary views, with `rt.explore`, and as `Flow` methods.
`where(condition, x, y)` broadcasts all three inputs to a common result shape.
It chooses `x` where the condition is true and `y` elsewhere.

```{literalinclude} ../../examples/lessons/conditional_choices.py
:language: python
```

For the selected value 8, the condition is true and the original array supplies
the result. Selecting the value 3 changes the result to 0 because `3 > 3` is
false. The condition, true branch, and false branch remain distinct roles in
the path list even when two roles refer to the same original element.

`where` keeps the broadcast result shape. Boolean indexing instead selects a
variable number of positions. Indexing still accepts concrete mask arrays.
A tracked comparison is a recipe, so it cannot be passed directly as an index
mask. Use the explicit comparison and `where` chain when the lesson needs to
retain how a condition was computed.

## Read candidates and choices separately

`TrackedTensor.trace()` reads no array values. A selection step has a
`selection_operator` and ordered `candidates`, whose integers index other
occurrences in that trace. Its `terms` is empty. A candidate list must never be
interpreted as multiplication or addition.

The immediate visual trace uses `SelectionExpression`. A numerical visual
also exposes `metadata["evaluated_selections"]`. Each record identifies an
output, a selected source, the candidate position and count, and a reason.
A walkthrough's `snapshot.selected_source` provides the corresponding immutable
record for its current occurrence. It is `None` when numerical work is unavailable.
The selected source is outlined with a thick dashed border, including when a
bounded structural trace omits that candidate's path.

Structural completeness and numerical completeness are independent. A small
trace may omit candidate paths even when bounded evaluation finds the winner.
An incomplete trace's input counts describe only reached candidate paths.
They do not count exclusively selected values or express numerical sensitivity.

Conditional evaluation plans and evaluates the condition and both branches.
An error in an unused recorded branch still raises, and all branches consume
the shared evaluation budget. `where` uses Python scalar truth and preserves
the selected scalar value without NumPy's common dtype promotion.

## Distinguish values from positions

```{literalinclude} ../../examples/lessons/extrema_positions.py
:language: python
```

For `[2, 8, 8]`, `max` returns 8 and `argmax` returns position 1.
The two values tie, so the first candidate supplies the selected source.
The second row `[7, 7, 3]` has maximum 7 at position 0.

| Operation | Supported axis | Output |
| --- | --- | --- |
| `min`, `max` | `None`, one integer, or a tuple of distinct axes | Selected scalar value |
| `argmin`, `argmax` | `None` or one integer | Selected position |

All four support `keepdims`. With `axis=None`, an index result is a flattened
row-major position. Value extrema support `axis=()` to reduce no axes.
Scalar arrays accept integer axis 0 or -1. An empty reduced group raises an
error because extrema have no identity. An empty surviving output is valid.
These APIs do not accept `initial`, `where`, `out`, or `dtype` arguments.

Extrema accept real scalar values. The first equal value wins, including the
sign of a tied zero. The first NaN wins when any candidate is NaN. This is an
explicit teaching policy, not a reproduction of every backend kernel's
floating-point behavior. Complex extrema are unsupported.

For standalone exploration, use `rt.explore(rt.argmax, values, axis=1)`.
For contribution and path navigation, record the operation with `Flow` and
pass its tracked result to `rt.walkthrough`. A standalone
`rt.walkthrough(rt.argmax, ...)` is not part of the factory contract.
