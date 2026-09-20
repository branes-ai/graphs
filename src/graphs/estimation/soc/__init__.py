"""SoC L0 analyzer: efficiency, mapping, schedule, power (graphs#269 Phase 3).

``graphs.hardware.soc`` composes a design into silicon at a node; this package
runs a pipeline workload on the result. See
``docs/plans/soc-phase3-execution-plan.md``.
"""

from .analyzer import SoCAnalysisResult, SoCAnalyzer
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
from .breakeven import (
    EngineRequirement,
    RequiredEfficiency,
    StageRequirement,
    capability_mapping,
    required_efficiency,
)
from .domainflow import (
    Ceiling,
    Fabric,
    FabricCeilings,
    fabric_ceilings,
    fabrics_of,
)
from .mapping import Engine, MappingFile, StageService, engines_of, find_mapping, load_mapping
from .power import PowerReport, roll_up
from .schedule import Schedule, schedule
from .study import Override, Study, SweepRow, load_study, run_study

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
    "Ceiling",
    "Engine",
    "EngineRequirement",
    "Fabric",
    "FabricCeilings",
    "fabric_ceilings",
    "fabrics_of",
    "MappingFile",
    "RequiredEfficiency",
    "StageRequirement",
    "capability_mapping",
    "required_efficiency",
    "Schedule",
    "StageService",
    "engines_of",
    "find_mapping",
    "load_mapping",
    "schedule",
    "PowerReport",
    "roll_up",
    "SoCAnalysisResult",
    "SoCAnalyzer",
    "Override",
    "Study",
    "SweepRow",
    "load_study",
    "run_study",
]
