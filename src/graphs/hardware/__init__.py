"""
Hardware Modeling and Mapping

Models hardware resources and provides algorithms for mapping computational
graphs to specific hardware architectures.
"""

from .resource_model import (
    HardwareType,
    Precision,
    ClockDomain,
    ComputeResource,
    PrecisionProfile,
    HardwareResourceModel,
    HardwareAllocation,
    GraphHardwareAllocation,
    HardwareMapper,
)

from .operand_fetch import (
    OperandFetchBreakdown,
    OperandFetchEnergyModel,
    CPUOperandFetchModel,
    GPUOperandFetchModel,
    TPUOperandFetchModel,
    KPUOperandFetchModel,
    create_operand_fetch_model,
    compare_operand_fetch_energy,
    format_comparison_table,
)

__all__ = [
    'HardwareType',
    'Precision',
    'ClockDomain',
    'ComputeResource',
    'PrecisionProfile',
    'HardwareResourceModel',
    'HardwareAllocation',
    'GraphHardwareAllocation',
    'HardwareMapper',
    # Operand Fetch Energy Models
    'OperandFetchBreakdown',
    'OperandFetchEnergyModel',
    'CPUOperandFetchModel',
    'GPUOperandFetchModel',
    'TPUOperandFetchModel',
    'KPUOperandFetchModel',
    'create_operand_fetch_model',
    'compare_operand_fetch_energy',
    'format_comparison_table',
]
