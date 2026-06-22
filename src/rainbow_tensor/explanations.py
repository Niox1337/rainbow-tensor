"""Centralised explanation text for the notebook display layer.

Every caption line shown under a visual is built from a keyed template here,
instead of being inlined as an f-string next to the render logic. This keeps
the wording in one place to copy edit, and gives a single point to translate.

The active language lives in :mod:`rainbow_tensor.config` and is set with
:func:`set_language`. English (``"en"``) is the fallback at two levels: an
unknown language falls back to the whole ``en`` table, and a key missing from a
translation falls back to its ``en`` string, so a partial translation never
raises.
"""

from . import config

_DEFAULT = "en"

MESSAGES = {
    "en": {
        # shared across many ops
        "common.original_shape": "Original shape: {shape}",
        "common.new_shape": "New shape: {shape}",
        "common.result_shape": "Result shape: {shape}",
        "common.axes_order": "Axes order: {order}",
        "common.index": "Index: {index}",
        # large tensor preview
        "preview.large_single": (
            "Large preview for tensor {shape} draws at most {max} real cells "
            "from {total} total."
        ),
        "preview.large_multi": (
            "Large preview for tensor {i} {shape} draws at most {max} real cells "
            "from {total} total."
        ),
        "preview.hidden": "Hidden axes {hidden} keep the head, selected positions, and tail.",
        # reshape
        "reshape.row_major": (
            "Values keep their row major order, so element k stays element k."
        ),
        # transpose
        "transpose.keep_colour": "Each axis keeps its colour as it moves to its new position.",
        # swapaxes
        "swapaxes.swap": "Swapping axes {a} and {b}.",
        "swapaxes.keep_colour": "The two moved axes keep their source colours in the result.",
        # moveaxis
        "moveaxis.move": "Moving axes {source} to {destination}.",
        "moveaxis.keep_colour": "Each moved axis keeps its source colour in the result.",
        # squeeze
        "squeeze.removing": "Removing size one axes {axes}.",
        "squeeze.none": "No size one axes to remove.",
        "squeeze.keep_colour": (
            "Each surviving axis keeps its colour, and the removed size one axes are marked."
        ),
        # expand_dims
        "expand_dims.inserting": "Inserting size one axes at {axes}.",
        "expand_dims.keep_colour": (
            "Each existing axis keeps its colour, and the inserted size one axes are marked."
        ),
        # matmul
        "matmul.operands": "Operand shapes: {a} @ {b}",
        "matmul.dims": "Left rows: {rows}, right columns: {cols}, shared inner axis: {inner}.",
        "matmul.combine": (
            "The shared inner axis is marked, and the highlighted row and column "
            "combine into the first output element."
        ),
        # repeat
        "repeat.repeating": "Repeating {repeats} along axis {axis}.",
        "repeat.copies": (
            "Each source element is copied into its own run of result cells, so "
            "repeat materialises real copies, unlike broadcast which stretches a "
            "size one axis virtually."
        ),
        # take
        "take.taking": "Taking indices {indices} along axis {axis}.",
        "take.copies": (
            "Each result slice copies a source slice, so repeated indices repeat a "
            "slice and reordered indices reorder them."
        ),
        # reduce (sum, mean, ...)
        "reduce.reducing": "Reducing axis {axis} with {op}.",
        "reduce.combines": "Each result element combines {count} values from axis {axis}.",
        "reduce.share_background": (
            "Values that fold into the same result share one background with that "
            "result element, and the first group is highlighted."
        ),
        # einsum
        "einsum.operand_subscripts": "Operand subscripts: {subscripts}",
        "einsum.output_subscript": "Output subscript: {subscript}",
        "einsum.contracted": "Contracted labels: {labels}",
        "einsum.none_contracted": "No labels are contracted.",
        # concatenate
        "concatenate.operands": "Operand shapes: {shapes}",
        "concatenate.joining": "Joining along axis {axis}.",
        "concatenate.seam": (
            "Each operand keeps its tint, so the seam along the joined axis is clear."
        ),
        # stack
        "stack.operand": "Operand shape: {shape}",
        "stack.stacking": "Stacking {count} tensors on a new axis {axis}.",
        "stack.new_axis": "The new axis indexes the operands, each kept in its own tint.",
        # broadcast
        "broadcast.operands": "Operand shapes: {a} and {b}",
        "broadcast.stretches": "operand {i} {shape} stretches axes {axes}",
        "broadcast.matches": "operand {i} {shape} already matches",
        "broadcast.fill": (
            "A stretched axis repeats one value, so both operands fill the result shape."
        ),
        # index
        "index.new_axis": "A new size 1 axis is inserted by None at result position {pos}.",
        "index.axis_removed": "Axis {axis} is removed because integer index {index} is used.",
        "index.axis_kept": "Axis {axis} is kept because slice {slice} is used.",
        "index.mask": "Advanced indexing with a boolean mask.",
        "index.mask_keeps": "The mask keeps {count} element{plural} where it is True.",
        "index.arrays": "Advanced indexing with integer arrays.",
        "index.gather": "Axes {axes} gather {count} position{plural} in shape {shape}.",
        "index.slice_separates": (
            "A slice separates the gathered axes, so the gathered axis moves to the front."
        ),
    },
}


def set_language(lang):
    """Set the active language for all subsequent explanations."""
    config.language = lang


def get_language():
    """Return the active explanation language."""
    return config.language


def t(key, **kwargs):
    """Return the explanation for ``key`` in the active language.

    Falls back to English for an unknown language or a missing key.
    """
    table = MESSAGES.get(config.language, MESSAGES[_DEFAULT])
    template = table.get(key) or MESSAGES[_DEFAULT][key]
    return template.format(**kwargs)
