"""
On-Chip Fabric Model (Layer 6)

Describes the on-chip interconnect that moves data between cores, caches,
memory controllers, and compute clusters within a single chip: crossbar
(GPU SM-to-L2), ring/mesh (x86 CPUs), 2D-mesh (tile accelerators such as
KPU and Cerebras), CLOS, and full-mesh topologies.

Layer 6 captures the TRANSPORT cost (per-hop latency, per-flit energy,
bandwidth saturation) of packets traversing the fabric. It does NOT capture
cache capacity, hit-rate, or coherence protocol behavior - those belong to
Layer 5 (L3/LLC).

See ``docs/plans/microarch-model-delivery-plan.md`` (M0 scaffolding, M6 SoC
data movement milestone).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
import json
from pathlib import Path


class Topology(Enum):
    """On-chip fabric topology."""
    CROSSBAR = "crossbar"
    RING = "ring"
    MESH_2D = "mesh_2d"
    CLOS = "clos"
    FULL_MESH = "full_mesh"
    UNKNOWN = "unknown"


@dataclass
class SoCFabricModel:
    """
    On-chip fabric model for Layer 6 (SoC data movement).

    Fields default to zero / empty / UNKNOWN so an instance round-trips
    without any populated content. M0 ships the scaffolding; per-SKU
    values are filled in by M6.

    Attributes:
        topology: Interconnect topology enum.
        hop_latency_ns: Per-hop latency in nanoseconds (quiet fabric).
        pj_per_flit_per_hop: Energy per flit per hop, picojoules.
        bisection_bandwidth_gbps: Chip-wide bisection bandwidth, gigabits
            per second (Gbps). Per the field name, values are in bits/s;
            consumers converting to GB/s must divide by 8.
        controller_count: Number of memory controllers on the chip.
        flit_size_bytes: Flit width in bytes (typical 32-64 on modern chips).
        max_injection_gbps: Saturation injection rate per source,
            gigabits per second (Gbps). Bits/s, not bytes/s.
        provenance: Source tag for these values (datasheet URL, commit, etc.).
    """
    topology: Topology = Topology.UNKNOWN
    hop_latency_ns: float = 0.0
    pj_per_flit_per_hop: float = 0.0
    bisection_bandwidth_gbps: float = 0.0
    controller_count: int = 0
    flit_size_bytes: int = 0
    max_injection_gbps: Optional[float] = None
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology.value,
            "hop_latency_ns": self.hop_latency_ns,
            "pj_per_flit_per_hop": self.pj_per_flit_per_hop,
            "bisection_bandwidth_gbps": self.bisection_bandwidth_gbps,
            "controller_count": self.controller_count,
            "flit_size_bytes": self.flit_size_bytes,
            "max_injection_gbps": self.max_injection_gbps,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoCFabricModel":
        data = dict(data)
        if "topology" in data and isinstance(data["topology"], str):
            data["topology"] = Topology(data["topology"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "SoCFabricModel":
        return cls.from_dict(json.loads(Path(path).read_text()))
