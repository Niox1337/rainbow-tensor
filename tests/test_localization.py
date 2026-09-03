"""Regression tests for locale discovery, fallback, and validated external catalogs."""

import ctypes
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rainbow_tensor import config
from rainbow_tensor import explanations as messages

_PLATFORM_LANGUAGE = messages._platform_language
_WINDOWS_LANGUAGE = messages._windows_ui_language


@pytest.fixture(autouse=True)
def isolated_languages(monkeypatch):
    """Keep catalog mutations and machine locale preferences inside each test."""
    saved_catalogs = {tag: dict(table) for tag, table in messages.MESSAGES.items()}
    saved_language = config.language
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(messages, "_platform_language", lambda: None)
    messages.set_language("en")
    yield
    messages.MESSAGES.clear()
    messages.MESSAGES.update(saved_catalogs)
    config.language = saved_language
    _WINDOWS_LANGUAGE.cache_clear()


def write_catalog(directory, filename, table):
    """Write an explicit external catalog without changing packaged resources."""
    path = directory / filename
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def test_shipped_chinese_covers_every_english_message():
    assert {"en", "zh"} <= set(messages.available_languages())
    assert set(messages.MESSAGES["zh"]) == set(messages.MESSAGES["en"])
    for key, template in messages.MESSAGES["zh"].items():
        assert messages._fields(template) == messages._fields(messages.MESSAGES["en"][key])
    messages.set_language("zh-CN")
    expected = messages.MESSAGES["zh"]["common.result_shape"].format(shape="(2,)")
    assert messages.t("common.result_shape", shape="(2,)") == expected
    assert expected != messages.MESSAGES["en"]["common.result_shape"].format(shape="(2,)")


@pytest.mark.parametrize(
    ("requested", "resolved"),
    [
        ("zh-CN", "zh"),
        ("zh_CN.UTF-8", "zh"),
        ("ZH_hans_CN@variant", "zh"),
        ("en_US.UTF-8", "en"),
        ("C", "en"),
        ("C.UTF-8", "en"),
        ("POSIX", "en"),
        ("zz", "en"),
    ],
)
def test_requested_spelling_and_resolved_catalog_are_distinct(requested, resolved):
    messages.set_language(requested)
    assert messages.get_language() == requested
    assert messages.get_resolved_language() == resolved


@pytest.mark.parametrize("value", [None, 1, True, [], {}])
def test_non_string_language_is_rejected_without_changing_state(value):
    with pytest.raises(TypeError, match="language must be a string"):
        messages.set_language(value)
    assert messages.get_language() == "en"


@pytest.mark.parametrize("value", ["", " ", "../zh", "English language", "-zh", "zh--CN"])
def test_invalid_language_is_rejected_without_changing_state(value):
    with pytest.raises(ValueError, match="language must be a locale tag"):
        messages.set_language(value)
    assert messages.get_language() == "en"


@pytest.mark.parametrize(
    ("environment", "platform", "expected"),
    [
        ({"LANGUAGE": "zz:zh_CN:en", "LC_ALL": "en"}, "en", "zh"),
        ({"LANGUAGE": "zz", "LC_ALL": "zh_CN"}, "en", "zh"),
        ({"LC_ALL": "en_US", "LC_MESSAGES": "zh_CN", "LANG": "zh_CN"}, "zh", "en"),
        ({"LC_MESSAGES": "zh_CN", "LANG": "en_US"}, "en", "zh"),
        ({"LANG": "zh_CN.UTF-8"}, "en", "zh"),
        ({"LANGUAGE": " : : "}, "zh_CN", "zh"),
        ({"LANGUAGE": "C:zh", "LANG": "zh_CN"}, "zh", "en"),
        ({"LC_ALL": "POSIX", "LANG": "zh_CN"}, "zh", "en"),
        ({"LANGUAGE": "invalid preference:zz"}, "zh_CN", "zh"),
        ({}, "zh_CN", "zh"),
        ({}, None, "en"),
    ],
)
def test_auto_language_preference_order(monkeypatch, environment, platform, expected):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(messages, "_platform_language", lambda: platform)
    messages.set_language("auto")
    assert messages.get_language() == "auto"
    assert messages.get_resolved_language() == expected


def test_explicit_language_overrides_environment_and_normalizes_auto(monkeypatch):
    monkeypatch.setenv("LANGUAGE", "zh_CN")
    messages.set_language("en")
    assert messages.get_resolved_language() == "en"
    messages.set_language(" AUTO ")
    assert messages.get_language() == "auto"
    assert messages.get_resolved_language() == "zh"


def test_supported_environment_does_not_query_platform(monkeypatch):
    def unexpected_platform_lookup():
        raise AssertionError("A resolved environment preference needs no platform lookup")

    monkeypatch.setenv("LANG", "zh_CN")
    monkeypatch.setattr(messages, "_platform_language", unexpected_platform_lookup)
    messages.set_language("auto")
    assert messages.get_resolved_language() == "zh"


def test_external_regional_catalog_falls_back_per_key(tmp_path):
    write_catalog(tmp_path, "fr.json", {"common.result_shape": "Base shape: {shape}"})
    write_catalog(tmp_path, "fr_CA.json", {"swapaxes.swap": "Regional swap {a} with {b}."})
    assert messages.load_translations(tmp_path) == ("fr", "fr-ca")
    messages.set_language("fr_CA.UTF-8")
    assert messages.get_resolved_language() == "fr-ca"
    assert messages.t("swapaxes.swap", a=0, b=1) == "Regional swap 0 with 1."
    assert messages.t("common.result_shape", shape="(2,)") == "Base shape: (2,)"
    key = next(
        key for key in messages.MESSAGES["en"] if not messages._fields(messages.MESSAGES["en"][key])
    )
    assert messages.t(key) == messages.MESSAGES["en"][key]
    assert {"fr", "fr-ca"} <= set(messages.available_languages())


def test_new_language_requires_only_a_catalog_file(tmp_path):
    catalog = write_catalog(tmp_path, "it.json", {"common.result_shape": "Loaded shape: {shape}"})
    assert messages.load_translations(catalog) == ("it",)
    messages.set_language("it-IT")
    assert messages.get_resolved_language() == "it"
    assert messages.t("common.result_shape", shape="()") == "Loaded shape: ()"


def test_loading_a_partial_override_preserves_other_messages(tmp_path):
    original = messages.MESSAGES["zh"]["swapaxes.swap"]
    catalog = write_catalog(tmp_path, "zh.json", {"common.result_shape": "Override {shape}"})
    messages.load_translations(catalog)
    messages.set_language("zh")
    assert messages.t("common.result_shape", shape="()") == "Override ()"
    assert messages.MESSAGES["zh"]["swapaxes.swap"] == original


def test_new_packaged_language_is_discovered_from_filename(tmp_path, monkeypatch):
    directory = tmp_path / "locales"
    directory.mkdir()
    write_catalog(directory, "en.json", messages.MESSAGES["en"])
    write_catalog(directory, "it_IT.json", {"common.result_shape": "Discovered {shape}"})
    monkeypatch.setattr(messages.resources, "files", lambda package: tmp_path)
    catalogs = messages._packaged_catalogs()
    assert set(catalogs) == {"en", "it-it"}
    messages.MESSAGES.clear()
    messages.MESSAGES.update(catalogs)
    messages.set_language("it-IT")
    assert messages.t("common.result_shape", shape="()") == "Discovered ()"


@pytest.mark.parametrize(
    ("table", "match"),
    [
        ({"unknown.message": "Unknown"}, "unknown keys"),
        ({"common.result_shape": "Missing parameter"}, "placeholders differ"),
        ({"common.result_shape": "{shape} {extra}"}, "placeholders differ"),
        ({"common.result_shape": "{shape"}, "Invalid translation"),
        ({"common.result_shape": "{}"}, "named fields"),
        ({"common.result_shape": "{shape.attr}"}, "named fields"),
        ({"common.result_shape": "{shape!x}"}, "valid conversions"),
        ({"common.result_shape": "{shape:{extra}}"}, "placeholders differ"),
        ({"common.result_shape": ""}, "nonempty string"),
        ({"common.result_shape": 42}, "nonempty string"),
        (["not", "an", "object"], "must be a JSON object"),
    ],
)
def test_invalid_catalog_is_rejected_before_installation(tmp_path, table, match):
    catalog = write_catalog(tmp_path, "fr.json", table)
    with pytest.raises(ValueError, match=match):
        messages.load_translations(catalog)
    assert "fr" not in messages.available_languages()


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        (
            '{"common.result_shape": "{shape}", "common.result_shape": "{shape}"}',
            "duplicate message key",
        ),
        ('{"common.result_shape":', "Cannot read translation catalog"),
    ],
)
def test_invalid_json_is_reported_with_catalog_path(tmp_path, contents, match):
    catalog = tmp_path / "fr.json"
    catalog.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=match) as error:
        messages.load_translations(catalog)
    assert str(catalog) in str(error.value)


def test_invalid_catalog_filename_is_rejected(tmp_path):
    catalog = write_catalog(tmp_path, "not a locale.json", {})
    with pytest.raises(ValueError, match="Invalid translation catalog filename"):
        messages.load_translations(catalog)


def test_duplicate_normalized_catalog_names_are_rejected(tmp_path):
    write_catalog(tmp_path, "fr_CA.json", {})
    write_catalog(tmp_path, "fr-CA.json", {})
    with pytest.raises(ValueError, match="Duplicate translation language: fr-ca"):
        messages.load_translations(tmp_path)
    assert "fr-ca" not in messages.available_languages()


def test_failed_batch_leaves_existing_catalogs_unchanged(tmp_path):
    before = {tag: dict(table) for tag, table in messages.MESSAGES.items()}
    write_catalog(tmp_path, "fr.json", {"common.result_shape": "Valid {shape}"})
    write_catalog(tmp_path, "zz.json", {"not.a.key": "Invalid"})
    with pytest.raises(ValueError, match="unknown keys"):
        messages.load_translations(tmp_path)
    assert messages.MESSAGES == before


def test_missing_path_and_empty_directory_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Expected a translation JSON"):
        messages.load_translations(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="No translation JSON files"):
        messages.load_translations(tmp_path)


def test_unknown_message_key_remains_an_error():
    with pytest.raises(KeyError, match="not.a.message"):
        messages.t("not.a.message")


def test_windows_ui_language_is_cached_and_auto_refreshes_it(monkeypatch):
    calls = []

    def native_language():
        return 0x0804

    def load_library(name):
        calls.append(name)
        return SimpleNamespace(GetUserDefaultUILanguage=native_language)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", load_library, raising=False)
    _WINDOWS_LANGUAGE.cache_clear()
    assert _WINDOWS_LANGUAGE() == "zh_CN"
    assert _WINDOWS_LANGUAGE() == "zh_CN"
    assert calls == ["kernel32"]
    messages.set_language("auto")
    assert _WINDOWS_LANGUAGE() == "zh_CN"
    assert calls == ["kernel32", "kernel32"]


def test_platform_fallback_reads_locale_without_mutating_it(monkeypatch):
    def forbidden_setlocale(*args, **kwargs):
        raise AssertionError("Language detection must not change the process locale")

    monkeypatch.setattr(messages, "_windows_ui_language", lambda: None)
    monkeypatch.setattr(messages.locale, "getlocale", lambda: ("fr_CA", "UTF-8"))
    monkeypatch.setattr(messages.locale, "setlocale", forbidden_setlocale)
    assert _PLATFORM_LANGUAGE() == "fr_CA"


def test_platform_lookup_handles_an_unavailable_locale(monkeypatch):
    def unavailable_locale():
        raise ValueError("Unsupported locale setting")

    monkeypatch.setattr(messages, "_windows_ui_language", lambda: None)
    monkeypatch.setattr(messages.locale, "getlocale", unavailable_locale)
    assert _PLATFORM_LANGUAGE() is None


def test_fresh_process_defaults_to_auto():
    environment = dict(os.environ)
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        environment.pop(variable, None)
    environment["LANGUAGE"] = "zh_CN"
    code = """
import json
from rainbow_tensor.explanations import get_language, get_resolved_language
print(json.dumps([get_language(), get_resolved_language()]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == ["auto", "zh"]
