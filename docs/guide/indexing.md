# Indexing

`index` shows how an indexing expression selects elements. The selected cells
fill green and the selected frames keep their colour, while the rest of the
tensor dims so the selected path stands out.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(8).reshape(2, 2, 2)
rt.index(x, (0, slice(None), 1))
```

For this array the index `(0, slice(None), 1)` selects the values `1` and `3`,
the selected coordinates are `(0, 0, 1)` and `(0, 1, 1)`, and the result shape
is `(2,)`.

## Integers and slices

An integer drops an axis. A slice keeps it. Slice bounds and steps may be
negative, so `slice(4, 1, -1)` walks backwards.

```python
rt.index(np.arange(5), (slice(4, 1, -1),))
```

## Ellipsis and a new axis

`Ellipsis` fills the axes you leave out, so `(0, ..., 1)` keeps the middle axis.
`None` inserts a size one axis that shows up in the result shape and label.

```python
rt.index(np.arange(12).reshape(2, 3, 2), (0, ..., 1))
rt.index(np.arange(6).reshape(2, 3), (None, 1, slice(None)))
```

## Boolean masks

A boolean mask of matching shape highlights every True position. A boolean
array on a single axis acts like its nonzero integer array and mixes with
slices and integer indices.

```python
x = np.arange(20).reshape(4, 5)
rt.index(x, x % 3 == 0)
```

## Fancy indexing

Integer arrays gather coordinates. Paired arrays pick one element each, and
multi-dimensional index arrays broadcast together and may mix with slices, with
the gathered block placed where NumPy puts it.

```python
x = np.arange(20).reshape(4, 5)
rt.index(x, ([0, 2, 3], [1, 4, 0]))
```

Integer scalars participate in the gathered block. If a slice or ellipsis
separates them from integer arrays, the gathered axes move to the front.
Even an ellipsis that fills zero axes can affect this ordering.
Floating-point index arrays are rejected rather than rounded or truncated.

## Large basic selections

Basic indexing stores source-axis ranges, so a large selection does not need
a list of every coordinate before drawing its preview.

```python
visual = rt.index((1_000_000, 1_000_000), (Ellipsis,))
visual.selected.count            # 1_000_000_000_000
visual.selected[0]               # (0, 0)
(10, 20) in visual.selected      # True
```

`visual.selected` supports lazy iteration, integer lookup, and explicit slices.
Use `.count` for selections larger than Python's `len` limit. Converting the
selection to a list still materializes it, so reserve that for small examples.
The lower-level `selected_coordinates` helper continues to return a list.

Advanced index arrays and masks currently materialize their selected source
coordinates. Their highlights represent unique source positions, so repeated
indices do not create duplicate source cells in the picture. Mixing `None`
with advanced indexing and boolean scalar indices remains unsupported.

## Clear errors

An out of range index points at the offending axis instead of failing deep
inside NumPy.

```python
x = np.arange(8).reshape(2, 2, 2)
try:
    rt.index(x, (2, slice(None), 1))
except IndexError as error:
    print(error)
```
