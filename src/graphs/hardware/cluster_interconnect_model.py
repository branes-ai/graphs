"""
Cluster Interconnect Model (Layer 9)

Describes the message-passing fabric between chassis: Ethernet (100/200/
400/800 GbE), InfiniBand (NDR / XDR), RoCE, and optical interconnects
such as the TPU v7 optical-circuit-switched (OCS) fabric between pods.
Traffic is packetized over a switched network; latency and energy are
dominated by NIC, switch, and topology.

Layer 9 is distinct from Layer 8 (intra-server fabric): a Layer 9 hop
is a switch crossing, not a NoC hop or an NVLink.

See ``docs/plans/microarch-model-delivery-plan.md`` and
``docs/plans/bottom-up-microbenchmark-plan.md`` (Phase 9).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import json
from pathlib import Path


class FabricType(Enum):
    """Cluster interconnect fabric type."""
    ETHERNET_100 = "ethernet_100"
    ETHERNET_200 = "ethernet_200"
    ETHERNET_400 = "ethernet_400"
    ETHERNET_800 = "ethernet_800"
    IB_NDR = "ib_ndr"         # InfiniBand NDR (400 Gb/s per port)
    IB_XDR = "ib_xdr"         # InfiniBand XDR (800 Gb/s per port)
    ROCE = "roce"             # RDMA over Converged Ethernet
    OPTICAL_OCS = "optical_ocs"  # Optical circuit switching (TPU v7)
    UNKNOWN = "unknown"


class ClusterTopology(Enum):
    """Cluster-scale topology."""
    FAT_TREE = "fat_tree"
    DRAGONFLY = "dragonfly"
    TORUS = "torus"
    OPTICAL_OCS = "optical_ocs"
    FULL_MESH = "full_mesh"
    UNKNOWN = "unknown"


@dataclass
class ClusterInterconnectModel:
    """
    Layer 9 fabric model: message-passing, inter-server.

    Defaults leave the model unpopulated so M0 scaffolding round-trips.
    Per-SKU values land in a later phase; most v1 SKUs will remain
    THEORETICAL until a multi-node facility is available.

    Attributes:
        fabric_type: Fabric type enum.
        topology: Cluster topology enum.
        node_count: Number of nodes participating in the fabric.
        per_hop_latency_ns: Per-switch-hop latency, nanoseconds.
        pj_per_bit: Energy per bit traversing the fabric (end-to-end).
        nic_energy_fixed_pj: Fixed NIC overhead per message, picojoules.
        reconfig_latency_ms: Reconfiguration latency (OCS only), milliseconds.
        provenance: Source tag.
    """
    fabric_type: FabricType = FabricType.UNKNOWN
    topology: ClusterTopology = ClusterTopology.UNKNOWN
    node_count: int = 0
    per_hop_latency_ns: float = 0.0
    pj_per_bit: float = 0.0
    nic_energy_fixed_pj: float = 0.0
    reconfig_latency_ms: Optional[float] = None
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fabric_type": self.fabric_type.value,
            "topology": self.topology.value,
            "node_count": self.node_count,
            "per_hop_latency_ns": self.per_hop_latency_ns,
            "pj_per_bit": self.pj_per_bit,
            "nic_energy_fixed_pj": self.nic_energy_fixed_pj,
            "reconfig_latency_ms": self.reconfig_latency_ms,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterInterconnectModel":
        data = dict(data)
        if "fabric_type" in data and isinstance(data["fabric_type"], str):
            data["fabric_type"] = FabricType(data["fabric_type"])
        if "topology" in data and isinstance(data["topology"], str):
            data["topology"] = ClusterTopology(data["topology"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ClusterInterconnectModel":
        return cls.from_dict(json.loads(Path(path).read_text()))
