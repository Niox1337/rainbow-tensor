"""Per-output-cell limits for numerical evaluation in math previews."""

from operator import index as integer_index

DEFAULT_MAX_TERMS = 10_000


def validate_max_terms(max_terms):
    """Accept positive integer limits or an explicit unbounded ``None``."""
    if max_terms is None:
        return None
    if isinstance(max_terms, bool) or type(max_terms).__name__ in ("bool", "bool_"):
        raise ValueError("max_terms must be a positive integer or None")
    try:
        limit = integer_index(max_terms)
    except TypeError:
        raise ValueError("max_terms must be a positive integer or None") from None
    if limit <= 0:
        raise ValueError("max_terms must be a positive integer or None")
    return limit


def value_evaluation(term_count, max_terms):
    """Describe whether every visible output cell can be evaluated completely.

    A term is one source contribution to a reduction or one product of operand
    values in a contraction. The limit applies separately to each output cell,
    not to the total number of source reads across the rendered figure.
    """
    limit = validate_max_terms(max_terms)
    return {
        "status": "evaluated" if limit is None or term_count <= limit else "skipped",
        "term_count": term_count,
        "max_terms": limit,
        "scope": "per_output_cell",
    }


def evaluation_explanation(evaluation, highlighted_terms=None):
    """Explain omitted output values and any bounded contribution highlights."""
    if evaluation["status"] != "skipped":
        return []
    count = evaluation["term_count"]
    limit = evaluation["max_terms"]
    lines = [
        f"Each output cell requires {count:,} terms, exceeding max_terms={limit:,}. "
        "Output values are shown as '?' and were not computed. "
        "Set max_terms=None to compute all terms."
    ]
    if highlighted_terms is not None and highlighted_terms < count:
        lines.append(
            f"Source highlights sample {highlighted_terms} of {count:,} contribution terms."
        )
    return lines
