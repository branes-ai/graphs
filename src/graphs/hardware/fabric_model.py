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

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple
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
        mesh_dimensions: For ``MESH_2D`` topology, the (width, height) tile
            grid. Used by ``hop_count_avg`` and the KPU-NoC formula.
        routing_distance_factor: Multiplier on shortest-path hop count to
            account for non-Manhattan routes (~1.0 for ideal XY routing,
            ~1.2 for typical real NoC routing). Replaces the
            ``KPUTileEnergyModel.l3_routing_distance_factor`` constant.
        low_confidence: Set when datasheet sources for the topology are
            insufficient. Per issue M6 constraint, the panel surfaces the
            assumption rather than leaving fields blank.
    """
    topology: Topology = Topology.UNKNOWN
    hop_latency_ns: float = 0.0
    pj_per_flit_per_hop: float = 0.0
    bisection_bandwidth_gbps: float = 0.0
    controller_count: int = 0
    flit_size_bytes: int = 0
    max_injection_gbps: Optional[float] = None
    provenance: str = ""

    # M6 additions
    mesh_dimensions: Optional[Tuple[int, int]] = None
    routing_distance_factor: float = 1.0
    low_confidence: bool = False

    # ------------------------------------------------------------------
    # Analytical hop-count curve
    # ------------------------------------------------------------------
    def hop_count_avg(self, node_count: Optional[int] = None) -> float:
        """
        Average hop count between two random nodes for this topology.

        ``node_count`` defaults to the implied count from
        ``mesh_dimensions`` for ``MESH_2D``, ``controller_count`` for
        bus-style topologies. The result is the analytical average,
        not a worst case, and includes ``routing_distance_factor``.

        Returns 0.0 when topology is UNKNOWN or node_count cannot be
        inferred.
        """
        # Resolve node_count from topology-specific defaults
        if node_count is None:
            if self.topology is Topology.MESH_2D and self.mesh_dimensions:
                w, h = self.mesh_dimensions
                node_count = w * h
            elif self.controller_count > 0:
                node_count = self.controller_count
            else:
                return 0.0

        if node_count <= 1:
            return 0.0

        topo = self.topology
        rdf = self.routing_distance_factor

        if topo is Topology.CROSSBAR or topo is Topology.FULL_MESH:
            # Single-hop across the fabric.
            return 1.0 * rdf
        if topo is Topology.RING:
            # Bidirectional ring: average path = N/4.
            return (node_count / 4.0) * rdf
        if topo is Topology.MESH_2D:
            # Average Manhattan distance on a w x h mesh:
            #   E[|x1-x2|] + E[|y1-y2|] = (w + h) / 3.
            if self.mesh_dimensions:
                w, h = self.mesh_dimensions
                return ((w + h) / 3.0) * rdf
            # Fall back to sqrt-N approximation
            side = math.sqrt(node_count)
            return ((2 * side) / 3.0) * rdf
        if topo is Topology.CLOS:
            # 3-stage Clos: O(log N) avg path; use ceil(log2(N)) as
            # a rough proxy.
            return max(1.0, math.log2(node_count)) * rdf
        # UNKNOWN
        return 0.0

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
            "mesh_dimensions": (
                list(self.mesh_dimensions) if self.mesh_dimensions else None
            ),
            "routing_distance_factor": self.routing_distance_factor,
            "low_confidence": self.low_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoCFabricModel":
        data = dict(data)
        if "topology" in data and isinstance(data["topology"], str):
            data["topology"] = Topology(data["topology"])
        # Validate mesh_dimensions shape early so a malformed JSON
        # blob fails at deserialization rather than deep inside
        # hop_count_avg() where the (w, h) unpack would crash.
        dims = data.get("mesh_dimensions")
        if dims is not None:
            if isinstance(dims, list):
                dims = tuple(dims)
            if (
                not isinstance(dims, tuple)
                or len(dims) != 2
                or not all(isinstance(v, int) and v > 0 for v in dims)
            ):
                raise ValueError(
                    f"mesh_dimensions must be a pair of positive "
                    f"integers; got {data.get('mesh_dimensions')!r}"
                )
            data["mesh_dimensions"] = dims
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "SoCFabricModel":
        return cls.from_dict(json.loads(Path(path).read_text()))
