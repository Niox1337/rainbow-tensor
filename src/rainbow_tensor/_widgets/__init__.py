"""Browser widget loaded only when a notebook explorer is created."""

from importlib.resources import files

from anywidget import AnyWidget
from traitlets import Bool, Int, List, Tuple, Unicode


class ExplorerFigure(AnyWidget):
    """Display an SVG and send discrete result-cell activations to its kernel."""

    _esm = files(__package__).joinpath("explorer.js").read_text(encoding="utf-8")
    value = Unicode("").tag(sync=True)
    revision = Int(0).tag(sync=True)
    description = Unicode("").tag(sync=True)
    label = Unicode("").tag(sync=True)


class PathSelector(AnyWidget):
    """Expose a native listbox whose navigation keys do not trigger host shortcuts.

    Option values are bounded trace occurrence numbers. The browser preserves
    its select node across synchronized updates, while Python remains responsible
    for accepting a selection and restoring the prior state after a failed draw.
    """

    _esm = files(__package__).joinpath("paths.js").read_text(encoding="utf-8")
    options = List(Tuple(Unicode(), Int())).tag(sync=True)
    value = Int(default_value=None, allow_none=True).tag(sync=True)
    description = Unicode("").tag(sync=True)
    disabled = Bool(False).tag(sync=True)
