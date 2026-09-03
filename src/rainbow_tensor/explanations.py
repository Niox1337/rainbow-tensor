"""File-based translation catalogs and automatic display-language selection.

Catalog filenames identify languages, for example ``en.json`` or ``pt-BR.json``.
Each UTF-8 JSON object maps message keys to format strings. Catalogs are discovered
inside the package and may be extended explicitly with ``load_translations``.
The mutable ``MESSAGES`` mapping remains available for existing integrations.
"""

import json
import locale
import os
import re
import sys
from functools import lru_cache
from importlib import resources
from pathlib import Path
from string import Formatter

from . import config

_DEFAULT = "en"
_LOCALE_VARIABLES = ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG")
_TAG = re.compile(r"[a-z]{2,8}(?:-[a-z0-9]{1,8})*\Z")
_FORMATTER = Formatter()


def _normalize_tag(value):
    """Normalize POSIX locale spellings and case without consulting global locale state."""
    if not isinstance(value, str):
        return None
    tag = value.strip().split(".", 1)[0].split("@", 1)[0].replace("_", "-").lower()
    if tag in {"c", "posix"}:
        return _DEFAULT
    return tag if _TAG.fullmatch(tag) and tag != "auto" else None


def _parents(tag):
    """Try a regional or script tag before progressively less specific parents."""
    while tag:
        yield tag
        tag = tag.rpartition("-")[0]


def _unique_object(pairs):
    """Reject duplicate JSON keys instead of silently keeping the last translation."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate message key {key!r}")
        result[key] = value
    return result


def _read_catalog(resource):
    """Read either a package resource or filesystem path with an actionable error."""
    try:
        return json.loads(resource.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Cannot read translation catalog {resource}: {error}") from error


def _fields(template):
    """Collect named fields, including nested format specifications, without evaluation."""
    fields = set()
    for _, name, specification, conversion in _FORMATTER.parse(template):
        if name is None:
            continue
        if not name.isidentifier() or conversion not in (None, "s", "r", "a"):
            raise ValueError("placeholders must use named fields and valid conversions")
        fields.add(name)
        fields.update(_fields(specification))
    return fields


def _validate_catalog(messages, name, english=None):
    """Check message types, known keys, and placeholder parity before installing a catalog."""
    if not isinstance(messages, dict):
        raise ValueError(f"Translation catalog {name} must be a JSON object")
    if english is not None:
        unknown = set(messages) - set(english)
        if unknown:
            raise ValueError(f"Translation catalog {name} has unknown keys: {sorted(unknown)}")
    for key, template in messages.items():
        if not isinstance(key, str) or not key or not isinstance(template, str) or not template:
            raise ValueError(f"Translation catalog {name} must contain nonempty string messages")
        try:
            fields = _fields(template)
            if english is not None and fields != _fields(english[key]):
                raise ValueError(f"placeholders differ from English for {key!r}")
        except ValueError as error:
            raise ValueError(f"Invalid translation in {name}, key {key!r}: {error}") from error
    return messages


def _catalog_tag(resource):
    """Derive a locale from the filename so new languages need no Python registration."""
    tag = _normalize_tag(resource.name[:-5])
    if tag is None:
        raise ValueError(f"Invalid translation catalog filename: {resource.name}")
    return tag


def _packaged_catalogs():
    """Discover package data with the resource API, including non-filesystem installs."""
    directory = resources.files("rainbow_tensor").joinpath("locales")
    files = sorted(
        (resource for resource in directory.iterdir() if resource.name.endswith(".json")),
        key=lambda resource: resource.name,
    )
    catalogs = {}
    for resource in files:
        tag = _catalog_tag(resource)
        if tag in catalogs:
            raise ValueError(f"Duplicate translation language: {tag}")
        catalogs[tag] = _read_catalog(resource)
    if _DEFAULT not in catalogs:
        raise ValueError("The English translation catalog is missing")
    english = _validate_catalog(catalogs[_DEFAULT], "en.json")
    for tag, messages in catalogs.items():
        _validate_catalog(messages, tag, english)
    return catalogs


MESSAGES = _packaged_catalogs()


def available_languages():
    """Return sorted catalog language tags, including explicitly loaded files."""
    return tuple(sorted({_normalize_tag(tag) for tag in MESSAGES} - {None}))


def set_language(lang):
    """Choose a locale tag or ``auto`` for subsequent visuals.

    Unknown but valid tags fall back to English. The requested spelling is kept
    by ``get_language``. ``get_resolved_language`` reports the catalog in use.
    """
    if not isinstance(lang, str):
        raise TypeError("language must be a string locale tag or 'auto'")
    requested = lang.strip()
    if requested.lower() == "auto":
        requested = "auto"
        _windows_ui_language.cache_clear()
    elif _normalize_tag(requested) is None:
        raise ValueError("language must be a locale tag such as 'en', 'zh-CN', or 'auto'")
    config.language = requested


def get_language():
    """Return the requested language setting, including ``auto`` when enabled."""
    return config.language


@lru_cache(maxsize=1)
def _windows_ui_language():
    """Read Windows' UI language once per session, refreshed by selecting auto again."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        function = ctypes.WinDLL("kernel32").GetUserDefaultUILanguage
        function.argtypes = []
        function.restype = ctypes.c_ushort
        return locale.windows_locale.get(function())
    except (AttributeError, OSError):
        return None


def _platform_language():
    """Prefer the Windows UI language, then query the existing Python locale unchanged."""
    language = _windows_ui_language()
    if language:
        return language
    try:
        return locale.getlocale()[0]
    except (TypeError, ValueError, OSError):
        return None


def _requested_languages():
    """Yield explicit or environment preferences before consulting the platform lazily."""
    requested = get_language()
    if requested != "auto":
        yield requested
        return
    for variable in _LOCALE_VARIABLES:
        for language in os.environ.get(variable, "").split(":"):
            if language.strip():
                yield language
    yield _platform_language()


def _catalog_map():
    """Include legacy direct MESSAGES edits while comparing canonical locale spellings."""
    return {_normalize_tag(tag): table for tag, table in MESSAGES.items()}


def _resolve_language(catalogs):
    """Find the first installed preference without using other languages for missing keys."""
    for language in _requested_languages():
        tag = _normalize_tag(language)
        if tag:
            for candidate in _parents(tag):
                if candidate in catalogs:
                    return candidate
    return _DEFAULT


def get_resolved_language():
    """Return the available catalog selected by the current language setting.

    Automatic mode tries LANGUAGE preferences, LC_ALL, LC_MESSAGES, LANG, and
    the platform UI locale in order. Regional tags fall back to parent tags.
    C and POSIX select English. This never changes the process locale.
    """
    return _resolve_language(_catalog_map())


def load_translations(path):
    """Load one JSON catalog or all JSON files in an explicit directory.

    Filenames determine language tags. Messages must use existing English keys
    and exactly the same named placeholders. Partial translations fall back per
    key. The complete batch is validated before it is merged, so a bad file
    leaves existing catalogs unchanged. Return the loaded language tags.
    """
    location = Path(path)
    if location.is_dir():
        files = sorted(location.glob("*.json"))
    elif location.is_file() and location.suffix == ".json":
        files = [location]
    else:
        raise ValueError(f"Expected a translation JSON file or directory: {location}")
    if not files:
        raise ValueError(f"No translation JSON files found in {location}")
    pending = {}
    english = MESSAGES[_DEFAULT]
    for resource in files:
        tag = _catalog_tag(resource)
        if tag in pending:
            raise ValueError(f"Duplicate translation language: {tag}")
        pending[tag] = _validate_catalog(_read_catalog(resource), resource.name, english)
    for tag, messages in pending.items():
        MESSAGES[tag] = {**MESSAGES.get(tag, {}), **messages}
    return tuple(sorted(pending))


def t(key, **kwargs):
    """Format a message, falling back through regional parents and English."""
    catalogs = _catalog_map()
    selected = _resolve_language(catalogs)
    for language in (*_parents(selected), _DEFAULT):
        template = catalogs.get(language, {}).get(key)
        if template:
            return template.format(**kwargs)
    raise KeyError(key)
