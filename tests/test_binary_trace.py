"""Binary traces preserve ordered roles without reinterpreting product terms."""

from dataclasses import FrozenInstanceError

import pytest

from rainbow_tensor.tracing import (
    BinaryExpression,
    OperandRef,
    OutputTrace,
    _trace_explanation,
)


@pytest.mark.parametrize("operation,symbol", [
    ("add", "+"), ("subtract", "-"), ("multiply", "*"), ("divide", "/"),
])
def test_binary_expression_explains_distinct_ordered_source_roles(operation, symbol):
    expression = BinaryExpression(operation, (OperandRef(0, (1, 0)), OperandRef(1, (0, 2))))
    trace = OutputTrace(operation, (1, 2), 1, (), True, expression=expression)
    assert trace.terms == ()
    assert trace.expression.operands[0].operand == 0
    assert trace.expression.operands[1].operand == 1
    assert f"A[1, 0] {symbol} B[0, 2]" in _trace_explanation(trace)[0]


def test_binary_scalar_references_remain_distinct_and_immutable():
    expression = BinaryExpression("divide", (OperandRef(0, ()), OperandRef(1, ())))
    trace = OutputTrace("divide", (), 1, (), True, expression=expression)
    assert "A[()] / B[()]" in _trace_explanation(trace)[0]
    with pytest.raises(FrozenInstanceError):
        expression.operator = "multiply"


def test_existing_product_terms_keep_their_original_interpretation():
    trace = OutputTrace("matmul", (0, 0), 1, ((OperandRef(0, (0, 1)), OperandRef(1, (1, 0))),),
                        True)
    assert trace.expression is None
    assert "A[0, 1] * B[1, 0]" in _trace_explanation(trace)[1]
