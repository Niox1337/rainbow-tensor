"""Legacy and package imports share the same notebook controller objects."""

from importlib import import_module

import pytest

import rainbow_tensor as rt
import rainbow_tensor.interactive as interactive


@pytest.mark.parametrize("module, class_name, factory", [
    ("focus", "FocusExplorer", "explore"),
    ("walkthrough", "Walkthrough", "walkthrough"),
    ("playground", "ReductionPlayground", "reduction_playground"),
])
def test_public_import_paths_preserve_controller_and_factory_identity(module, class_name, factory):
    implementation = import_module(f"rainbow_tensor.interactive.{module}")
    legacy_name = "interactive" if module == "focus" else module
    legacy = import_module(f"rainbow_tensor.{legacy_name}")
    for name in (class_name, factory):
        assert getattr(rt, name) is getattr(interactive, name)
        assert getattr(legacy, name) is getattr(implementation, name)
        assert getattr(rt, name) is getattr(implementation, name)
