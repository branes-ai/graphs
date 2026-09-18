"""SoC L0 analyzer: efficiency, mapping, schedule, power (graphs#269 Phase 3).

``graphs.hardware.soc`` composes a design into silicon at a node; this package
runs a pipeline workload on the result. See
``docs/plans/soc-phase3-execution-plan.md``.
"""

from .efficiency import (
    CLASS_FLOOR,
    PRECISION_ORDER,
    EfficiencyEntry,
    EfficiencyTable,
    KernelClass,
    KernelClassMap,
    PooledThroughput,
    StageKernel,
    execution_format,
    load_efficiency_tables,
    load_kernel_classes,
)
from .mapping import Engine, MappingFile, StageService, engines_of, find_mapping, load_mapping
from .power import PowerReport, roll_up
from .schedule import Schedule, schedule

__all__ = [
    "CLASS_FLOOR",
    "PRECISION_ORDER",
    "EfficiencyEntry",
    "EfficiencyTable",
    "KernelClass",
    "KernelClassMap",
    "PooledThroughput",
    "StageKernel",
    "execution_format",
    "load_efficiency_tables",
    "load_kernel_classes",
    "Engine",
    "MappingFile",
    "Schedule",
    "StageService",
    "engines_of",
    "find_mapping",
    "load_mapping",
    "schedule",
    "PowerReport",
    "roll_up",
]
