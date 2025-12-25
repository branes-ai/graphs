"""Adapters for converting internal data structures to external formats.

This package provides adapters for converting UnifiedAnalysisResult to:
- Pydantic models (for embodied-schemas integration)
- JSON/dict formats
- Other output formats as needed
"""

from graphs.adapters.pydantic_output import (
    convert_to_pydantic,
    convert_roofline_to_pydantic,
    convert_energy_to_pydantic,
    convert_memory_to_pydantic,
    make_verdict,
)

__all__ = [
    "convert_to_pydantic",
    "convert_roofline_to_pydantic",
    "convert_energy_to_pydantic",
    "convert_memory_to_pydantic",
    "make_verdict",
]
