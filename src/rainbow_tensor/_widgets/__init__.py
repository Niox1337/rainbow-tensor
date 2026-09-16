"""Optional browser widget loaded only when a notebook explorer is created."""

from importlib.resources import files

from anywidget import AnyWidget
from traitlets import Int, Unicode


class ExplorerFigure(AnyWidget):
    """Display an SVG and send discrete result-cell activations to its kernel."""

    _esm = files(__package__).joinpath("explorer.js").read_text(encoding="utf-8")
    value = Unicode("").tag(sync=True)
    revision = Int(0).tag(sync=True)
