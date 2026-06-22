"""Tests for the centralised explanation text and language switching."""

import pytest

from rainbow_tensor import get_language, set_language, swapaxes
from rainbow_tensor.explanations import MESSAGES, t


@pytest.fixture(autouse=True)
def _restore_language():
    yield
    set_language("en")
    MESSAGES.pop("xx", None)


def test_every_translation_only_uses_known_keys():
    # A typo in a translated key would silently fall back to en forever, so guard it.
    en_keys = set(MESSAGES["en"])
    for lang, table in MESSAGES.items():
        extra = set(table) - en_keys
        assert not extra, f"{lang} has keys absent from en: {extra}"


def test_unknown_language_falls_back_to_en():
    set_language("nope")
    assert t("common.result_shape", shape="(2,)") == "Result shape: (2,)"


def test_partial_translation_falls_back_per_key():
    MESSAGES["xx"] = {"swapaxes.swap": "swap {a} {b}"}
    set_language("xx")
    assert t("swapaxes.swap", a=0, b=2) == "swap 0 2"
    # key missing from the xx table still resolves through en
    assert t("common.result_shape", shape="(2,)") == "Result shape: (2,)"


def test_set_language_changes_a_real_visual():
    np = pytest.importorskip("numpy")
    MESSAGES["xx"] = {"swapaxes.swap": "Tausche {a} und {b}."}
    set_language("xx")
    assert get_language() == "xx"
    visual = swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2)
    assert "Tausche 0 und 2." in visual.text
