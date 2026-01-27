"""
Deprecated: Use graphs.estimation.architecture_comparator instead.
This module will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.analysis.architecture_comparator is deprecated. "
    "Use graphs.estimation.architecture_comparator instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

from graphs.estimation.architecture_comparator import *
