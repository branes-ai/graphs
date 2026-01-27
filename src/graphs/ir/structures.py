"""
Deprecated: Use graphs.core.structures instead.
This module will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.ir.structures is deprecated. "
    "Use graphs.core.structures instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

from graphs.core.structures import *
