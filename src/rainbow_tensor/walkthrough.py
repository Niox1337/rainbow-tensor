"""Compatibility imports for guided contribution walkthroughs.

New code can import these objects from :mod:`rainbow_tensor.interactive`.
Both paths expose the same objects, so controller subclasses and patches retain
one implementation.
"""

from .interactive.walkthrough import Walkthrough, walkthrough

__all__ = ["Walkthrough", "walkthrough"]
