"""
rainbow-tensor: A Python package for IPython/Jupyter notebooks.

This package can only be used in IPython environments (Jupyter notebooks, IPython shell).
"""

def _check_ipython():
    """Check if running in IPython environment."""
    try:
        __IPYTHON__  # This variable is automatically defined in IPython
        return True
    except NameError:
        return False

# Check if running in IPython on import
if not _check_ipython():
    raise RuntimeError(
        "rainbow-tensor can only be used in IPython environments "
        "(Jupyter notebooks, IPython shell). "
        "Please run this code in a Jupyter notebook or IPython shell."
    )

# Import numpy for users
import numpy as np

__version__ = "0.1.0"
__all__ = ["np"]
