# Shapes

`shape` draws the structure of a tensor. Every non-leaf axis becomes a coloured
frame, leaf elements sit in rounded cells, and a legend names each axis with its
size in the matching colour.

```python
import numpy as np
import rainbow_tensor as rt

rt.shape(np.arange(3))                    # a vector
rt.shape(np.arange(6).reshape(2, 3))      # a matrix
rt.shape(np.arange(8).reshape(2, 2, 2))   # a cube
rt.shape(np.arange(16).reshape(2, 2, 2, 2))
```

The same call works on a shape tuple or on any array-like object that exposes a
`.shape` attribute, so no array library is imported by the core.

## A scalar is one value, an empty tensor has none

Shape `()` has no axes and contains one value. Shape `(1,)` has one axis and
also contains one value. Shape `(0,)` has one axis but no values:

```python
from IPython.display import display

display(rt.shape(np.array(7)))    # scalar, shape ()
display(rt.shape(np.array([7])))  # one-element vector, shape (1,)
display(rt.shape(np.empty(0)))    # empty vector, shape (0,)
```

The scalar's cell coordinate is `()`, with flat index 0. Indexing it with `()`
keeps that single value:

```python
scalar = np.array(7)
visual = rt.index(scalar, (), show_result=True)
assert visual.result_shape == ()
assert visual.index_mapping.source_coord(()) == ()
visual
```

Any zero dimension makes the whole tensor empty. For example, `(2, 0, 3)`
contains no elements even though two of its axes have positive lengths. The
figure keeps the logical shape and shows **No elements**, without a numeric
cell or a read from the array. Large remaining dimensions are not expanded.

```python
display(rt.shape((2, 0, 3)))
display(rt.shape((1_000_000, 0, 1_000_000)))
display(rt.shape(()))  # a shape tuple supplies the scalar placeholder value 0
```

Dimensions must be nonnegative integers. Negative dimensions, booleans, and
floating-point dimensions are rejected. A tuple supplies a shape, while an
array supplies its own values. To display the scalar value 7, use `np.array(7)`
rather than the shape tuple `(7,)`.

### Reshape without changing the element count

A scalar can become a one-element vector and back. An empty array can change
shape as long as the result still has zero elements:

```python
display(rt.reshape(np.array(7), (1,)))
display(rt.reshape(np.array([7]), ()))
display(rt.reshape(np.empty((0, 3)), (-1, 3)))  # infers shape (0, 3)
display(rt.reshape(np.empty((0, 3)), (2, 0)))
```

An explicit zero dimension alongside `-1`, such as `(0, -1)`, is ambiguous.
The total would be zero for any inferred size, so `reshape` rejects it.
Provide all dimensions or keep the known dimensions nonzero when using `-1`.

## Colour scheme

Each axis has its own colour drawn from a rainbow ramp keyed by depth, so the
structure stays readable at any rank.

- Axis 0 is the outer frame, drawn red
- Axis 1 is the inner row frame, drawn orange
- Deeper axes move through lime, teal, blue, violet, and pink, so adjacent axes
  stay easy to tell apart
- Leaf elements sit in plain cells, and a selected element fills green
- The shape label, the index label, and the legend swatches all match

## Float precision

Floats format to a chosen number of decimals, right aligned for easy reading.

```python
x = np.linspace(0, 1, 6).reshape(2, 3)
rt.shape(x, precision=3)
```

## Saving to a file

Every result carries its SVG and can write it straight to disk.

```python
visual = rt.shape(np.arange(8).reshape(2, 2, 2))
visual.save("tensor.svg")
```

## Big tensor previews

Large tensors use two limits. `max_cells` caps one axis, while
`max_visible_cells` caps the whole preview, so a high rank tensor does not draw
a huge SVG. The preview keeps the head and tail of each hidden axis.

```python
rt.shape((100, 100, 100, 100))
```

## Explanation text

Explanations print as plain notebook output below the figure, and the same text
stays available in Python through the `text` attribute.

```python
visual = rt.index((2, 2, 2), (0, slice(None), 1))
visual.text
```
