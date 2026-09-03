"""Keep display assertions independent of the machine running the tests."""

import pytest

from rainbow_tensor import config


@pytest.fixture(autouse=True)
def english_display(monkeypatch):
    """Pin existing text expectations while auto-language tests opt in explicitly."""
    monkeypatch.setattr(config, "language", "en")
