"""
Deprecated: Use graphs.estimation.dynamo_characterizer instead.
This module will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.analysis.dynamo_characterizer is deprecated. "
    "Use graphs.estimation.dynamo_characterizer instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

from graphs.estimation.dynamo_characterizer import *
