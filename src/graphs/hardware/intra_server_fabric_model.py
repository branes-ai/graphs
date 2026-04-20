"""
Intra-Server Fabric Model (Layer 8)

Describes the load/store-semantic fabric inside a single server or pod:
NVLink + NVSwitch on a DGX 8-GPU board, PCIe host-to-device, UPI between
Xeon sockets, Infinity Fabric between EPYC CCDs, TPU ICI between chips
in one pod. Addresses are shared, coherent, or at minimum DMA-reachable
without a network stack.

Layer 8 is distinct from Layer 9 (cluster interconnect): Layer 8 is
load/store-semantic within one chassis; Layer 9 is message-passing between
chassis.

See ``docs/plans/microarch-model-delivery-plan.md`` and
``docs/plans/bottom-up-microbenchmark-plan.md`` (Phase 8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import json
from pathlib import Path


class IntraServerTopology(Enum):
    """Intra-server (or intra-pod) fabric topology."""
    BUS = "bus"
    CROSSBAR = "crossbar"
    NVSWITCH_FULL_MESH = "nvswitch_full_mesh"
    PCIE_TREE = "pcie_tree"
    MULTI_SOCKET_MESH = "multi_socket_mesh"
    TPU_ICI = "tpu_ici"
    UNKNOWN = "unknown"


@dataclass
class IntraServerFabricModel:
    """
    Layer 8 fabric model: tightly-coupled, load/store-semantic.

    All fields default to zero / empty / UNKNOWN so M0 scaffolding
    round-trips cleanly. Per-SKU values populate later.

    Attributes:
        topology: Intra-server topology enum.
        device_count: Number of coherent devices in the fabric
            (e.g., 8 for DGX H100 NVSwitch mesh).
        per_link_bandwidth_gbps: Bandwidth per directional link, GB/s.
        link_energy_pj_per_bit: Energy per bit per link traversal.
        link_count: Total directional link count
            (e.g., 64 for NVSwitch all-to-all).
        round_trip_latency_us: Device-to-device round-trip latency, us.
        provenance: Source tag.
    """
    topology: IntraServerTopology = IntraServerTopology.UNKNOWN
    device_count: int = 0
    per_link_bandwidth_gbps: float = 0.0
    link_energy_pj_per_bit: float = 0.0
    link_count: int = 0
    round_trip_latency_us: Optional[float] = None
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology.value,
            "device_count": self.device_count,
            "per_link_bandwidth_gbps": self.per_link_bandwidth_gbps,
            "link_energy_pj_per_bit": self.link_energy_pj_per_bit,
            "link_count": self.link_count,
            "round_trip_latency_us": self.round_trip_latency_us,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntraServerFabricModel":
        data = dict(data)
        if "topology" in data and isinstance(data["topology"], str):
            data["topology"] = IntraServerTopology(data["topology"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "IntraServerFabricModel":
        return cls.from_dict(json.loads(Path(path).read_text()))
