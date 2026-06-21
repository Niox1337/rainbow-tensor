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
