"""Individual validator modules.

Each module here registers one or more SKUValidator implementations
against ``graphs.hardware.sku_validators.default_registry`` at import
time. Importing this package triggers all of them; that's what
``load_validators()`` calls.

Phase 2b validator modules:

- consistency: tile_mix_consistency, cross_ref_consistency (INTERNAL)
- electrical:  power_profile_monotonicity (ELECTRICAL)
- area:        block_library_validity, area_self_consistency,
               composite_density_envelope (AREA)
- energy:      tops_per_watt_envelope (ENERGY)

Stage-2c / 2d / Stage-8 modules will be appended here as new validators
arrive.
"""

# Importing each module triggers the @register_class decorators inside.
from . import area  # noqa: F401
from . import consistency  # noqa: F401
from . import electrical  # noqa: F401
from . import energy  # noqa: F401
from . import reliability  # noqa: F401
from . import thermal  # noqa: F401
