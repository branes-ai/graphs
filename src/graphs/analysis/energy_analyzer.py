"""
Deprecated: Use graphs.estimation.energy instead.
This module will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.analysis.energy_analyzer is deprecated. "
    "Use graphs.estimation.energy instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

from graphs.estimation.energy import *
