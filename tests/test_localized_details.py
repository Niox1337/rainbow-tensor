"""Dynamic explanations and notebook controls honor the active language."""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.evaluation import evaluation_explanation
from rainbow_tensor.numerics import numeric_explanation, numeric_semantics
from rainbow_tensor.tracing import _build_trace, _trace_explanation


@pytest.fixture(autouse=True)
def restore_language():
    """Keep locale changes local to each test, including assertion failures."""
    original = rt.get_language()
    yield
    rt.set_language(original)


@pytest.fixture
def explorers():
    pytest.importorskip("ipywidgets")
    created = []

    def make(operation, *args, **kwargs):
        explorer = rt.explore(operation, *args, **kwargs)
        created.append(explorer)
        return explorer

    yield make
    for explorer in created:
        explorer.close()


def test_english_trace_preserves_heading_and_source_equation():
    rt.set_language("en")
    trace = _build_trace("sum", (1,), 1, [((1, 0),)])
    assert _trace_explanation(trace) == [
        "Focus: output (1,) uses 1 term.",
        "output[1] = source[1, 0]",
    ]


def test_chinese_trace_localizes_bounded_equation_and_omitted_count():
    rt.set_language("zh-CN")
    trace = _build_trace("sum", (1,), 10, (((1, k),) for k in range(10)))
    heading, equation = _trace_explanation(trace)

    assert "聚焦：输出 (1,) 使用 10 项。" in heading
    assert "显示前 8 项，省略 2 项。" in heading
    assert equation.startswith("输出[1] = 来源[1, 0]")
    assert "来源[1, 7] + ..." in equation
    assert "来源[1, 8]" not in equation
    assert "Focus:" not in heading


def test_chinese_einsum_and_empty_traces_use_localized_names():
    rt.set_language("zh-CN")
    trace = _build_trace("einsum", (), 1, [((), ())])
    assert _trace_explanation(trace)[1] == "输出[()] = 操作数 0[()] * 操作数 1[()]"
    mean = _build_trace("mean", (), 0, (), divisor=0)
    assert _trace_explanation(mean)[1] == "输出[()] = NaN（空组的均值未定义）"
    assert _trace_explanation(None) == ["结果为空，没有可聚焦的输出坐标。"]


@pytest.mark.parametrize("reason", ["max_terms", "max_total_terms"])
def test_chinese_budget_explains_limits_and_sampled_highlights(reason):
    rt.set_language("zh-CN")
    evaluation = {
        "status": "skipped", "reason": reason, "term_count": 20_000,
        "max_terms": 10_000, "max_total_terms": 100_000,
        "output_count": 6, "total_terms": 120_000,
    }
    lines = evaluation_explanation(evaluation, highlighted_terms=8)

    assert "未进行计算" in lines[0]
    assert ("max_terms=10,000" if reason == "max_terms" else "max_total_terms=100,000") in lines[0]
    assert lines[1] == "来源高亮仅显示 20,000 个贡献项中的 8 项。"


def test_chinese_numeric_model_distinguishes_real_and_generated_operands():
    rt.set_language("zh-CN")
    semantics = numeric_semantics([np.ones(2), (2,)])
    lines = numeric_explanation(semantics)

    assert lines[0] == "数值模型：Python 标量运算。"
    assert "溢出行为可能与输入数组的后端不同" in lines[1]
    assert lines[2] == "操作数 1 使用根据形状生成的行优先占位值。"
    assert semantics["operand_sources"] == ["array_values", "generated_values"]


def test_chinese_memory_explains_strides_and_keeps_metadata_language_neutral():
    array = np.arange(6, dtype=np.int64).reshape(2, 3)[:, ::-1]
    rt.set_language("en")
    original = rt.memory(array)
    rt.set_language("zh-CN")
    translated = rt.memory(array)

    assert translated.metadata == original.metadata
    assert "数据类型：int64。" in translated.text
    assert "元素大小：8 字节。" in translated.text
    assert "字节步长：(24, -8)。" in translated.text
    assert "轴 1：索引每增加 1，字节偏移量变化 -8。" in translated.text
    assert "负步长表示沿该轴反向访问存储。" in translated.text
    assert "拥有数据：否。存在底层对象：是。" in translated.text
    assert "由另一对象提供存储的视图" in translated.text


def test_chinese_memory_reports_unknown_and_empty_storage_without_guessing():
    rt.set_language("zh-CN")
    unknown = rt.memory((2, 3))
    empty = rt.memory(np.empty((0, 3)))

    assert "数据类型：未知。" in unknown.text
    assert "字节步长：未知。" in unknown.text
    assert "拥有数据：未知。存在底层对象：未知。" in unknown.text
    assert "空张量没有可遍历的元素地址。" in empty.text


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("operation", [rt.sum, rt.einsum])
def test_translated_explorer_trace_appears_once_before_and_after_updates(
    explorers, explicit, operation,
):
    rt.set_language("zh-CN")
    options = {"focus": (1,)} if explicit else {}
    args = ((2, 3),) if operation is rt.sum else ("ij->i", (2, 3))
    if operation is rt.sum:
        options["axis"] = 1
    explorer = explorers(operation, *args, **options)

    assert explorer.explanation.value.count("聚焦：") == 1
    assert explorer.update_button.description == "更新聚焦"
    assert explorer.coordinates[0].description == "轴 0"
    explorer.coordinates[0].value = 1
    explorer.update_button.click()
    assert explorer.explanation.value.count("聚焦：") == 1
    assert explorer.status.value == "当前显示输出 (1,)。"
    assert explorer.visual.metadata["trace_in_explanation"] is True


def test_explorer_labels_refresh_when_language_changes(explorers):
    rt.set_language("en")
    explorer = explorers(rt.sum, (2, 3), axis=1)
    assert explorer.update_button.description == "Update focus"
    rt.set_language("zh-CN")
    explorer.set_focus((1,))

    assert explorer.update_button.description == "更新聚焦"
    assert explorer.coordinates[0].description == "轴 0"
    assert explorer.status.value == "当前显示输出 (1,)。"
    assert explorer.explanation.value.count("聚焦：") == 1


def test_empty_explorer_and_update_errors_are_localized(explorers):
    rt.set_language("zh-CN")
    empty = explorers(rt.sum, (0, 3), axis=1)
    assert empty.status.value == "结果为空：没有可聚焦的输出坐标。"
    assert empty.update_button.disabled

    array = np.arange(6).reshape(2, 3)
    explorer = explorers(rt.sum, array, axis=1)
    previous = explorer.visual
    array.shape = (3, 2)
    explorer.update_button.click()

    assert explorer.visual is previous
    assert explorer.status.value.startswith("无法更新聚焦：")
    assert "the output shape changed" in explorer.status.value
