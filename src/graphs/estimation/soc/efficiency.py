"""Engine efficiency tables for the SoC L0 analyzer (graphs#269 PR 3.1).

A stage's service time on an engine is its ops in each precision class divided
by the engine's peak in the format that class runs in, times an efficiency:
what fraction of that peak this *kind of kernel* attains there. The tables
here carry those efficiencies, keyed ``(kernel_class, engine_kind, precision)``.

Two kinds of table, kept apart (decision P3-D1):

* **pooled** -- the Data Annex's own model: the whole SoC as one machine with
  a stated effective throughput per precision class (A / B / C = 2,000 / 300 /
  15 GOP/s). ``annex_v1`` is this, and it is a regression target: with it the
  analyzer must reproduce the annex, adding nothing and losing nothing.
* **per_engine** -- one efficiency per kernel class, engine kind and format.
  ``default_v1`` carries a number only where one is measured; every other
  entry is explicitly unknown. An unknown efficiency is a gap the analyzer
  reports, never a default it fills in -- the same discipline as unanchored
  silicon in the composition (P2-D3).

Whether an engine *can* run a class is a different question from how well,
and is answered structurally by ``execution_format``: a class runs in the
lowest format the engine has at or above the class's floor. An engine with
no such format cannot run the class at all. Nothing is promoted to a
capability the engine does not have (parent plan, risk "silent precision
promotion").
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, PositiveFloat, model_validator

from graphs.core.pipeline_workload import CLASS_NAMES, EffectiveThroughput
from graphs.hardware.soc.ip_block import Confidence, EngineKind


class KernelClass(str, Enum):
    """What a stage's arithmetic looks like to an engine (parent plan 5.4)."""

    DENSE_CONV_GEMM = "dense_conv_gemm"
    ATTENTION_PREFILL = "attention_prefill"
    WEIGHT_STREAM_DECODE = "weight_stream_decode"
    ELEMENTWISE_NORM = "elementwise_norm"
    COST_VOLUME_DP = "cost_volume_dp"
    FFT = "fft"
    FEATURE_TRACK = "feature_track"
    SPARSE_HASH_SCATTER = "sparse_hash_scatter"
    KNN_TREE = "knn_tree"
    RAYCAST = "raycast"
    WAVEFRONT = "wavefront"
    SMALL_DENSE_LINALG = "small_dense_linalg"
    SMALL_QP = "small_qp"
    GRAPH_SEARCH = "graph_search"
    PIXEL_FIXED_FUNCTION = "pixel_fixed_function"


#: Formats in increasing precision. A class may run in any format at or above
#: its floor; never below.
PRECISION_ORDER: Tuple[str, ...] = ("int8", "fp16", "fp32", "fp64")

#: The floor of each precision class (``pipeline_workload``): A is
#: INT8-eligible, B needs FP16, C needs FP32. The annex's Class C is
#: "FP32/FP64" without saying which stage needs which, so FP32 is the floor
#: taken -- optimistic for a stage that truly needs FP64 (decision P3-D6).
CLASS_FLOOR: Dict[str, str] = {"A": "int8", "B": "fp16", "C": "fp32"}


def execution_format(precision_class: str, formats: Mapping[str, float]) -> Optional[str]:
    """The format a class runs in on an engine with these peak formats.

    The lowest format at or above the class's floor that the engine has a
    positive peak in, or ``None`` when the engine cannot run the class.
    """
    floor = PRECISION_ORDER.index(CLASS_FLOOR[precision_class])
    for fmt in PRECISION_ORDER[floor:]:
        if formats.get(fmt, 0.0) > 0:
            return fmt
    return None


class EfficiencyEntry(BaseModel):
    """Fraction of an engine's dense peak a kernel class attains in a format."""

    kernel_class: KernelClass
    engine_kind: EngineKind
    precision: str
    #: ``None`` means unknown: the pair is supported but nothing measures it.
    compute_eff: Optional[float] = Field(None, gt=0, le=1)
    #: The spread of the measurements ``compute_eff`` summarizes.
    eff_range: Optional[Tuple[float, float]] = None
    confidence: Confidence
    source: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _consistent(self) -> "EfficiencyEntry":
        tag = f"{self.kernel_class.value}/{self.engine_kind.value}/{self.precision}"
        if self.precision not in PRECISION_ORDER:
            raise ValueError(f"{tag}: unknown precision {self.precision!r}")
        if self.compute_eff is None:
            if self.confidence != Confidence.UNKNOWN or self.eff_range is not None:
                raise ValueError(f"{tag}: an unknown efficiency has no range and confidence unknown")
            return self
        if self.confidence == Confidence.UNKNOWN:
            raise ValueError(f"{tag}: a stated efficiency needs a confidence above unknown")
        if self.eff_range is not None:
            lo, hi = self.eff_range
            if not 0 < lo <= self.compute_eff <= hi <= 1:
                raise ValueError(f"{tag}: range {self.eff_range} must bracket {self.compute_eff}")
        return self

    @property
    def key(self) -> Tuple[KernelClass, EngineKind, str]:
        return (self.kernel_class, self.engine_kind, self.precision)

    @property
    def known(self) -> bool:
        return self.compute_eff is not None


class PooledThroughput(BaseModel):
    """Effective GOP/s per precision class for the SoC as one machine."""

    a: PositiveFloat
    b: PositiveFloat
    c: PositiveFloat
    confidence: Confidence
    source: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}

    def as_effective_throughput(self) -> EffectiveThroughput:
        return EffectiveThroughput(a=self.a, b=self.b, c=self.c)


class EfficiencyTable(BaseModel):
    """One named efficiency model: pooled or per engine."""

    id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    name: str
    kind: Literal["pooled", "per_engine"]
    #: A per-engine table can layer over another: it inherits the base's
    #: entries and overrides those it restates (a measured table over
    #: default_v1). Resolved by ``load_efficiency_tables``.
    base: Optional[str] = None
    pooled: Optional[PooledThroughput] = None
    entries: List[EfficiencyEntry] = Field(default_factory=list)
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _shape(self) -> "EfficiencyTable":
        if self.kind == "pooled":
            if self.pooled is None or self.entries:
                raise ValueError(f"table {self.id!r}: a pooled table has pooled rates and no entries")
            if self.base is not None:
                raise ValueError(f"table {self.id!r}: a pooled table cannot layer over another")
            return self
        if self.pooled is not None or not self.entries:
            raise ValueError(f"table {self.id!r}: a per-engine table has entries and no pooled rates")
        if self.base == self.id:
            raise ValueError(f"table {self.id!r} cannot be its own base")
        keys = [e.key for e in self.entries]
        dup = sorted({f"{k[0].value}/{k[1].value}/{k[2]}" for k in keys if keys.count(k) > 1})
        if dup:
            raise ValueError(f"table {self.id!r}: duplicate entries {dup}")
        return self

    def lookup(
        self, kernel_class: KernelClass, engine_kind: EngineKind, precision: str
    ) -> Optional[EfficiencyEntry]:
        """The entry for this key, or ``None`` when the table has no row.

        A missing row and a row with ``compute_eff: null`` mean the same thing
        to the analyzer -- no service time -- but the row says *why*.
        """
        if self.kind != "per_engine":
            raise TypeError(f"table {self.id!r} is pooled; it has no per-engine entries")
        for entry in self.entries:
            if entry.key == (kernel_class, engine_kind, precision):
                return entry
        return None

    @property
    def known_entries(self) -> Tuple[EfficiencyEntry, ...]:
        return tuple(e for e in self.entries if e.known)


class StageKernel(BaseModel):
    """A workload stage's kernel class, and why."""

    kernel_class: KernelClass
    reason: str = Field(..., min_length=20)

    model_config = {"extra": "forbid"}


class KernelClassMap(BaseModel):
    """Kernel classes for the stages of one workload (decision P3-D2)."""

    workload: str
    stages: Dict[str, StageKernel]

    model_config = {"extra": "forbid"}

    def check_against(self, stage_keys) -> None:
        """Every stage classified, and nothing classified that is not a stage."""
        missing = sorted(set(stage_keys) - set(self.stages))
        extra = sorted(set(self.stages) - set(stage_keys))
        if missing or extra:
            raise KeyError(f"kernel classes for {self.workload!r}: missing {missing}, unknown {extra}")

    def of(self, stage_key: str) -> KernelClass:
        return self.stages[stage_key].kernel_class


SOC_DATA_DIR = Path(__file__).resolve().parents[4] / "soc_designs"
DEFAULT_EFFICIENCY_DIR = SOC_DATA_DIR / "efficiency"
DEFAULT_KERNEL_CLASS_PATH = SOC_DATA_DIR / "kernel_classes" / "branes_7tier_v1.yaml"


def resolve_layers(tables: Mapping[str, EfficiencyTable]) -> Dict[str, EfficiencyTable]:
    """Flatten every ``base:`` chain: a layered table's entries are its
    base's, overridden key by key by its own."""
    resolved: Dict[str, EfficiencyTable] = {}

    def flatten(table_id: str, seen: Tuple[str, ...]) -> EfficiencyTable:
        if table_id in resolved:
            return resolved[table_id]
        if table_id in seen:
            raise ValueError(f"efficiency tables layer in a cycle: {' -> '.join(seen + (table_id,))}")
        table = tables[table_id]
        if table.base is None:
            resolved[table_id] = table
            return table
        if table.base not in tables:
            raise KeyError(f"table {table_id!r}: base {table.base!r} is not a table")
        base = flatten(table.base, seen + (table_id,))
        if base.kind != "per_engine":
            raise ValueError(f"table {table_id!r}: base {base.id!r} is not per-engine")
        own = {e.key for e in table.entries}
        merged = [e for e in base.entries if e.key not in own] + list(table.entries)
        resolved[table_id] = table.model_copy(update={"entries": merged})
        return resolved[table_id]

    for table_id in tables:
        flatten(table_id, ())
    return resolved


def load_efficiency_tables(path: Optional[Path] = None) -> Dict[str, EfficiencyTable]:
    """Every ``*.yaml`` table under ``path``, keyed by id, layers resolved."""
    root = Path(path) if path else DEFAULT_EFFICIENCY_DIR
    tables: Dict[str, EfficiencyTable] = {}
    for file in sorted(root.glob("*.yaml")):
        table = EfficiencyTable.model_validate(yaml.safe_load(file.read_text()))
        if table.id != file.stem:
            raise ValueError(f"{file.name}: id {table.id!r} must match the file name")
        tables[table.id] = table
    return resolve_layers(tables)


def load_kernel_classes(path: Optional[Path] = None) -> KernelClassMap:
    file = Path(path) if path else DEFAULT_KERNEL_CLASS_PATH
    return KernelClassMap.model_validate(yaml.safe_load(file.read_text()))


__all__ = [
    "CLASS_FLOOR",
    "CLASS_NAMES",
    "EfficiencyEntry",
    "EfficiencyTable",
    "KernelClass",
    "KernelClassMap",
    "PRECISION_ORDER",
    "PooledThroughput",
    "StageKernel",
    "execution_format",
    "load_efficiency_tables",
    "load_kernel_classes",
    "resolve_layers",
]
