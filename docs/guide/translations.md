# Translations

Explanations, figure labels, memory details, and interactive controls use message
catalogs. English and Simplified Chinese are bundled. Python API names such as
`sum` and `reshape`, code notation, and validation exceptions stay in English.

## Choose a language

```python
import rainbow_tensor as rt

rt.available_languages()   # -> ("en", "zh") in a fresh installation
rt.set_language("zh-CN")
rt.get_language()           # -> "zh-CN"
rt.get_resolved_language()  # -> "zh"
rt.sum((2, 3), axis=1)
```

`get_language()` keeps the requested spelling. `get_resolved_language()` reports
the normalized tag of the catalog actually selected. For example, `zh-CN` uses
`zh` when there is no separate regional catalog. A valid language tag with no
available catalog or parent falls back to English.

The initial setting is `"auto"`. Use `rt.set_language("auto")` to restore it.
Automatic detection follows the Python kernel's environment and system locale,
not the browser language. The [configuration guide](themes-and-configuration)
describes the preference order and explicit overrides.

## Start with a small catalog

Each catalog is one UTF-8 JSON file containing a flat object. Its filename is the
language tag, for example `fr.json` or `pt-BR.json`. Keys identify messages and
values hold the translated text. Do not wrap the messages in a language object.

You can translate a few messages first. For example, save this as `fr.json`:

```json
{
  "common.result_shape": "Forme du résultat : {shape}",
  "interactive.update": "Actualiser le focus"
}
```

The English catalog at `src/rainbow_tensor/locales/en.json` is the reference for
available keys and placeholders. Copy only the entries you want to translate,
then replace their values. Omitted entries continue to use a parent catalog or
English, so a partial translation is usable immediately.

Keep every named placeholder from the English message. In the example above,
`{shape}` must remain `{shape}`. You may move placeholders to fit the sentence,
but you cannot add or remove their names. Preserve format specifications such
as `{count:,}` when they are present. Message values must be nonempty strings,
keys must exist in the English catalog, and duplicate JSON keys are rejected.

## Load a catalog in a notebook

Load a file explicitly, then choose its language:

```python
rt.load_translations("fr.json")  # -> ("fr",)
rt.set_language("fr")
rt.sum((2, 3), axis=1)
```

You can also pass a directory to load its `*.json` files:

```python
rt.load_translations("my-translations")
rt.available_languages()  # includes the languages just loaded
```

Directory loading reads files directly inside that directory. It does not
search nested folders. The whole batch is validated before any catalog changes,
so one invalid file leaves the current catalogs untouched. Two filenames that
normalize to the same language tag are rejected within a batch.

Loading merges the provided messages into any existing catalog for that
language. It replaces matching keys and keeps other messages already loaded.
It does not select the language for you. External files are read only when you
call `load_translations`, and loading an edited file again updates subsequent
visuals without restarting the kernel.

## Add a bundled language

Place the new file in `src/rainbow_tensor/locales/`, alongside `en.json` and
`zh.json`. A new language needs only its JSON file, with no Python registry
change. Catalogs are discovered by filename when the package is imported, and
the `locales/*.json` package-data rule includes them in distributions.

Restart the notebook kernel after adding or editing a bundled catalog. If you
are testing a catalog outside an installed package, use `load_translations`
instead. Create a new visual after changing the language or catalog to inspect
the updated text.

## Regional and missing-message fallback

Tags accept regional and script components. A request for `pt-BR` tries `pt-br`
first, then `pt`. POSIX spellings such as `zh_CN.UTF-8` are accepted too. Tags are
normalized for matching, while `get_language()` preserves the requested setting.

For each message, the selected catalog is checked first, followed by its parent
catalogs and then English. A partial `pt-BR.json` can therefore override regional
wording while sharing other messages from `pt.json`. Missing messages do not
switch to an unrelated language elsewhere in the system preference list.
