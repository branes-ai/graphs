"""
Deprecated: Use graphs.estimation.concurrency instead.
This module will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.analysis.concurrency is deprecated. "
    "Use graphs.estimation.concurrency instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

from graphs.estimation.concurrency import *
