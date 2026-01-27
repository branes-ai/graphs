"""
Deprecated: Use graphs.estimation.roofline instead.
This module will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.analysis.roofline_analyzer is deprecated. "
    "Use graphs.estimation.roofline instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

from graphs.estimation.roofline import *
