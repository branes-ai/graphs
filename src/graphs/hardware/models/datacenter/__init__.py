"""
Datacenter hardware resource models.
"""

from .h100_pcie import h100_pcie_resource_model
from .v100_sxm2 import v100_sxm2_resource_model
from .a100_sxm4_80gb import a100_sxm4_80gb_resource_model
from .t4 import t4_resource_model
from .tpu_v4 import tpu_v4_resource_model
from .cpu_x86 import cpu_x86_resource_model
from .ampere_ampereone_192 import ampere_ampereone_192_resource_model
from .ampere_ampereone_128 import ampere_ampereone_128_resource_model
from .intel_xeon_platinum_8490h import intel_xeon_platinum_8490h_resource_model
from .intel_xeon_platinum_8592plus import intel_xeon_platinum_8592plus_resource_model
from .intel_granite_rapids import intel_granite_rapids_resource_model
from .amd_epyc_9654 import amd_epyc_9654_resource_model
from .amd_epyc_9754 import amd_epyc_9754_resource_model
from .amd_epyc_turin import amd_epyc_turin_resource_model

__all__ = [
    'h100_pcie_resource_model',
    'v100_sxm2_resource_model',
    'a100_sxm4_80gb_resource_model',
    't4_resource_model',
    'tpu_v4_resource_model',
    'cpu_x86_resource_model',
    'ampere_ampereone_192_resource_model',
    'ampere_ampereone_128_resource_model',
    'intel_xeon_platinum_8490h_resource_model',
    'intel_xeon_platinum_8592plus_resource_model',
    'intel_granite_rapids_resource_model',
    'amd_epyc_9654_resource_model',
    'amd_epyc_9754_resource_model',
    'amd_epyc_turin_resource_model',
]
