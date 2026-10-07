"""Compatibility imports for reduction prediction controls.

New code can import these objects from :mod:`rainbow_tensor.interactive`.
Both paths expose the same objects and preserve existing controller imports.
"""

from .interactive.playground import ReductionPlayground, reduction_playground

__all__ = ["ReductionPlayground", "reduction_playground"]
