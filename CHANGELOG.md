# Changelog

## 1.0.1

### Changed

- Internal restructure, with no change to the public API. The shape maths and the display layer are each split by operation family into the `rainbow_tensor.ops` and `rainbow_tensor.views` packages (reshaping, reductions, combining, einsum, and shapes for the views), so `einsum` is no longer the only separated operation. Shared helpers moved into `shape.py` and `visual.py`, and the tests follow the same family layout. Every public import keeps working unchanged

## 0.15.0

### Added

- `moveaxis` draws moving one or more axes to new positions, the named axis movement operation between `transpose` and `swapaxes`. `source` and `destination` are an axis or a sequence of axes of equal length, the moved axes go to their destinations while the rest keep their order, and the result shape matches `numpy.moveaxis`. Each result axis keeps the colour of the source axis it came from, so the move stays traceable, and `transpose` and `swapaxes` are unchanged

## 0.14.0

### Added

- `repeat` draws repeating elements along one axis. A single count repeats every element, or one count per element repeats each by its own amount, matching `numpy.repeat`. Each source element and the adjacent run of copies it produces share one tint, so the result reads as materialised copies. This is the contrast with `broadcast`, which stretches a size one axis virtually without copying any values

## 0.13.0

### Added

- `take` draws a gather along one axis. A 1-D list of indices selects positions along the chosen axis, the result replaces that axis with the gathered positions, and the result shape matches `numpy.take`. Negative axes and negative indices both work. Each gathered source slice and the result slice it feeds share one tint, so repeated indices repeat a tint and reordered indices reorder them, making the gather visible. The existing advanced indexing behaviour is unchanged

## 0.12.0

### Added

- `matmul` draws a matrix multiplication `a @ b` as two operands and the output side by side. Vector, matrix, and batched matrix multiplication are all supported, with the result shape matching `numpy.matmul`. The shared inner axis is marked in the accent colour, and the row of the first operand and the column of the second operand that combine into the first output element are highlighted, so the contraction reads directly off the figure. The batch axes reuse the broadcast shape logic

## 0.11.0

### Added

- `squeeze` draws a tensor with its size one axes removed. With no axis it removes every size one axis, and an explicit axis or tuple removes only those, each of which must have size one, matching `numpy.squeeze`. The source panel marks the removed size one axes in the accent colour, every surviving axis keeps its source colour in the result, and the result shape matches NumPy
- `expand_dims` draws a tensor with a size one axis inserted. The axis is a position or tuple of positions in the result, with negative positions counting from the end, matching `numpy.expand_dims`. Each existing axis keeps its colour, the inserted size one axes are marked in the accent colour, and the result shape matches NumPy

## 0.10.8

### Added

- Einsum now parses ellipsis (`...`) notation. An ellipsis stands for the broadcast axes a subscript leaves unnamed, is expanded against the operand shapes and aligned from the right like NumPy, and supports both implicit and explicit output. The derived output shape matches NumPy, and an invalid ellipsis raises a clear error

## 0.10.7

### Fixed

- Einsum label colours now match between the figure and the caption. A leaf axis label used to be coloured only in the caption, and the contraction highlight dimmed unrelated frames, so the colours disagreed. A leaf axis now shows its label colour as the cell border and operand frames keep their colour

### Changed

- Einsum now colours every label by its role. Free, shared, and contracted labels are drawn from three distinct colour families, and each label keeps one colour across every operand caption, operand figure, and the output panel

## 0.10.6

### Changed

- A sum or mean result element now shares the same background as the source values that fold into it, so a result cell and its source group read as one unit. The first group stays highlighted on both panels, and the result shape and numeric result are unchanged

## 0.10.5

### Changed

- Sum and mean now colour the source values that fold into the same result element with one shared background and highlight the first group, so a reduction reads with the same clarity as the concatenate and stack views. The reduced axis, the result shape, and the numeric result are unchanged

## 0.10.4

### Added

- Advanced indexing now accepts per-axis boolean arrays. A boolean array on one or more consecutive axes acts like its nonzero integer arrays, mixes with slices and integer indices, and matches the NumPy selected coordinates and result shape. The full-shape boolean mask keeps working as before

## 0.10.3

### Added

- Advanced indexing now accepts multi-dimensional integer index arrays. Index arrays of any shape broadcast together, the gathered block takes the broadcast shape, and the selected coordinates and result shape match NumPy, including when a slice moves the block to the front

## 0.10.2

### Added

- A global axis colour scheme. `set_default_axis_colors` sets an axis colour ramp once for every later render, `get_default_axis_colors` reads it back, and `None` clears it. A per call `theme` still overrides the global scheme

### Changed

- The default axis colour ramp now spreads its hues so adjacent axes are easy to tell apart. After red and orange it jumps to lime and teal, then walks through blue, violet, and pink, in both the light and the dark theme

## 0.10.1

### Fixed

- Skipped tensor positions in a large preview now always draw the intended ellipsis glyphs. The horizontal, vertical, and midline ellipsis are defined with ASCII unicode escapes so a non-UTF-8 re-save of the source can no longer corrupt them into broken text

## 0.10.0

### Added

- Big tensor previews now respect a total visible cell budget, and selected positions stay visible when the preview budget allows it
- Explanation text now prints as standard notebook output and is available as `TensorVisual.text`

## 0.9.0

### Added

- Backend value adapters for NumPy style arrays plus Torch, JAX, and TensorFlow style scalar values
- A renderer registry with SVG as the default renderer and a public hook for later output backends
- A backend arrays notebook that runs even when optional backend libraries are not installed

## 0.8.0

### Added

- Einsum views parse subscripts, colour shared labels across operands, highlight labels that contract away, and show the derived output shape
- Swapaxes views draw the source and result side by side with the moved axes keeping their colours

## 0.7.0

### Added

- Shape changing views. Reshape draws the same values flowing from the old layout into the new one, transpose reorders the axes with each axis keeping its colour, and sum and mean collapse a chosen axis and mark the elements that fold into each result. The source and the result sit side by side in one figure through a new multi panel renderer
- Combining views. Concatenate joins operands along an existing axis and tints each operand so the seam is clear, and stack places operands onto a brand new axis. The result colours every cell by the operand it came from, and a mismatch in the operand shapes raises a clear error
- Broadcasting views. Broadcast stretches a smaller operand to match a larger one, drawing each operand in its own shape and again stretched to the common shape so the repeated values show, and marking every stretched axis in the accent colour. Incompatible shapes raise a clear error
- Render tensors of any rank. Frames nest to arbitrary depth, alternating across and down so a 4D or 5D tensor stays compact, with per axis truncation at every level
- Advanced indexing. A full-shape boolean mask highlights every True position, and integer index arrays highlight the gathered coordinates. The result shape matches NumPy for mixed basic and advanced indexing, including moving the gathered axis to the front when a slice separates the advanced axes

### Changed

- Renamed `show_shape` to `shape` and `show_index` to `index`. The first argument is an array-like object whose own values are rendered, or a shape tuple as before

## 0.2.1

### Fixed

- show_shape and show_index no longer render twice in a notebook cell. The result renders once through _repr_svg_ and is still returned for inspection
- The SVG width now grows to fit the label and explanation lines, so longer text is no longer clipped at the edge of the frame

## 0.2.0

### Changed

- Redesigned the visualisation so each axis has its own colour, with axis 0 frames red, axis 1 frames orange, and selected leaf values green
- The shape label numbers and the index label tokens are now coloured to match the frames and the selected values
- In an index view the unselected frames and values are drawn in a neutral dark tone instead of being faded

## 0.1.0

First version.

### Added

- Static SVG visualisation of tensor shape for 1D, 2D, and 3D tensors through `show_shape`
- Static SVG visualisation of basic indexing through `show_index`, with highlighted selections and de-emphasised context
- A plain text explanation of the result shape and of which axes are kept or removed
- Support for shape tuples and array-like objects with a `.shape` attribute, including NumPy arrays
- Integer indexing and basic slicing with `slice(None)`, `slice(start, stop)`, and `slice(start, stop, step)`

### Known limitations

- Interactive widgets are not included in this version
- Advanced indexing, boolean masks, `None`, `newaxis`, and ellipsis are not supported
- Tensors of 4D or higher are not supported

### Planned next steps

- Interactive controls in a notebook widget
- Higher dimensional layouts
- Optional support for more tensor libraries through shape duck typing
