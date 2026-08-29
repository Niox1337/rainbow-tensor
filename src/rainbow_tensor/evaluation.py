"""Plan bounded numerical work before reading any values in a math preview."""

from functools import wraps
from operator import index as integer_index

from .layout import build_layout

DEFAULT_MAX_TERMS = 10_000
DEFAULT_MAX_TOTAL_TERMS = 100_000


def validate_max_terms(max_terms, name="max_terms"):
    """Accept positive integer limits or an explicit unbounded ``None``."""
    if max_terms is None:
        return None
    if isinstance(max_terms, bool) or type(max_terms).__name__ in ("bool", "bool_"):
        raise ValueError(f"{name} must be a positive integer or None")
    try:
        limit = integer_index(max_terms)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer or None") from None
    if limit <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return limit


def evaluation_plan(term_count, max_terms, max_total_terms, shape, selected, theme):
    """Plan visible output coordinates and decide whether to compute all of them.

    A term is one source contribution to a reduction or one product of operand
    values in a contraction. ``max_terms`` applies to one output, while
    ``max_total_terms`` applies to all planned outputs together. Source-panel
    values are separate from this calculation budget and remain bounded by the
    layout. A product may read more than one operand value per term.

    Planning uses the regular layout without a backend value callback. All
    planned outputs are computed or all are skipped, so focus and render order
    cannot change which numerical results fit. Custom renderers may choose
    different coordinates within the same planned number of output slots.
    """
    limit = validate_max_terms(max_terms)
    total_limit = validate_max_terms(max_total_terms, "max_total_terms")
    layout = build_layout(shape, selected=selected, theme=theme)
    coordinates = frozenset(cell.coord for cell in layout.cells if not cell.ellipsis)
    total_terms = term_count * len(coordinates)
    reason = None
    if coordinates and limit is not None and term_count > limit:
        reason = "max_terms"
    elif total_limit is not None and total_terms > total_limit:
        reason = "max_total_terms"
    evaluation = {
        "status": "skipped" if reason else "evaluated",
        "reason": reason,
        "term_count": term_count,
        "max_terms": limit,
        "max_total_terms": total_limit,
        "output_count": len(coordinates),
        "total_terms": total_terms,
        "scope": "per_output_cell_and_preview",
    }
    return evaluation


def budgeted_values(evaluation):
    """Cache complete outputs and cap custom renderers at the planned cell count.

    A renderer may request different output coordinates from the default layout.
    Each distinct coordinate uses one slot, while repeated requests reuse its
    value. Extra requests return ``?`` without reading source values, keeping
    the total term budget valid even when a renderer ignores the layout.
    Unknown values are not cached, so excess requests cannot grow the cache.
    """
    def decorate(value_fn):
        values = {}

        @wraps(value_fn)
        def value(coordinate):
            if coordinate in values:
                return values[coordinate]
            if evaluation["status"] == "skipped" or len(values) >= evaluation["output_count"]:
                return "?"
            result = value_fn(coordinate)
            values[coordinate] = result
            return result

        return value

    return decorate


def evaluation_explanation(evaluation, highlighted_terms=None):
    """Explain omitted output values and any bounded contribution highlights."""
    if evaluation["status"] != "skipped":
        return []
    count = evaluation["term_count"]
    if evaluation["reason"] == "max_terms":
        limit = evaluation["max_terms"]
        lines = [
            f"Each output cell requires {count:,} terms, exceeding max_terms={limit:,}. "
            "Output values are shown as '?' and were not computed. "
            "Set max_terms=None to remove the per-output limit."
        ]
    else:
        outputs = evaluation["output_count"]
        total = evaluation["total_terms"]
        limit = evaluation["max_total_terms"]
        lines = [
            f"The {outputs:,} visible output cells require {total:,} terms in total, "
            f"exceeding max_total_terms={limit:,}. "
            "Output values are shown as '?' and were not computed. "
            "Set max_total_terms=None to remove the total limit."
        ]
    if highlighted_terms is not None and highlighted_terms < count:
        lines.append(
            f"Source highlights sample {highlighted_terms} of {count:,} contribution terms."
        )
    return lines
