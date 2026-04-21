"""Tests for the three fabric dataclass stubs added in M0."""
from __future__ import annotations

import json
from pathlib import Path

from graphs.hardware.fabric_model import (
    SoCFabricModel,
    Topology,
)
from graphs.hardware.intra_server_fabric_model import (
    IntraServerFabricModel,
    IntraServerTopology,
)
from graphs.hardware.cluster_interconnect_model import (
    ClusterInterconnectModel,
    FabricType,
    ClusterTopology,
)


class TestSoCFabricModel:
    def test_defaults_unpopulated(self):
        m = SoCFabricModel()
        assert m.topology is Topology.UNKNOWN
        assert m.hop_latency_ns == 0.0
        assert m.pj_per_flit_per_hop == 0.0
        assert m.bisection_bandwidth_gbps == 0.0
        assert m.controller_count == 0
        assert m.flit_size_bytes == 0
        assert m.max_injection_gbps is None

    def test_roundtrip_dict(self):
        m = SoCFabricModel(
            topology=Topology.MESH_2D,
            hop_latency_ns=1.25,
            pj_per_flit_per_hop=0.18,
            bisection_bandwidth_gbps=900.0,
            controller_count=4,
            flit_size_bytes=64,
            max_injection_gbps=1200.0,
            provenance="test-source",
        )
        restored = SoCFabricModel.from_dict(m.to_dict())
        assert restored == m

    def test_roundtrip_file(self, tmp_path: Path):
        m = SoCFabricModel(topology=Topology.RING, hop_latency_ns=2.0)
        p = tmp_path / "soc.json"
        m.save(p)
        loaded = SoCFabricModel.load(p)
        assert loaded == m


class TestIntraServerFabricModel:
    def test_defaults_unpopulated(self):
        m = IntraServerFabricModel()
        assert m.topology is IntraServerTopology.UNKNOWN
        assert m.device_count == 0
        assert m.per_link_bandwidth_gbps == 0.0
        assert m.link_energy_pj_per_bit == 0.0

    def test_roundtrip_dict(self):
        m = IntraServerFabricModel(
            topology=IntraServerTopology.NVSWITCH_FULL_MESH,
            device_count=8,
            per_link_bandwidth_gbps=450.0,
            link_energy_pj_per_bit=1.2,
            link_count=64,
            round_trip_latency_us=1.8,
            provenance="dgx-h100-spec",
        )
        restored = IntraServerFabricModel.from_dict(m.to_dict())
        assert restored == m

    def test_roundtrip_file(self, tmp_path: Path):
        m = IntraServerFabricModel(
            topology=IntraServerTopology.PCIE_TREE, device_count=2)
        p = tmp_path / "intra.json"
        m.save(p)
        assert IntraServerFabricModel.load(p) == m


class TestClusterInterconnectModel:
    def test_defaults_unpopulated(self):
        m = ClusterInterconnectModel()
        assert m.fabric_type is FabricType.UNKNOWN
        assert m.topology is ClusterTopology.UNKNOWN
        assert m.node_count == 0

    def test_roundtrip_dict(self):
        m = ClusterInterconnectModel(
            fabric_type=FabricType.IB_NDR,
            topology=ClusterTopology.FAT_TREE,
            node_count=64,
            per_hop_latency_ns=90.0,
            pj_per_bit=30.0,
            nic_energy_fixed_pj=500.0,
            provenance="ib-ndr-datasheet",
        )
        restored = ClusterInterconnectModel.from_dict(m.to_dict())
        assert restored == m

    def test_roundtrip_file(self, tmp_path: Path):
        m = ClusterInterconnectModel(
            fabric_type=FabricType.ETHERNET_400,
            topology=ClusterTopology.DRAGONFLY,
            node_count=256,
        )
        p = tmp_path / "cluster.json"
        m.save(p)
        assert ClusterInterconnectModel.load(p) == m

    def test_json_is_valid(self, tmp_path: Path):
        """The saved file must be plain JSON (no pickling or similar)."""
        m = ClusterInterconnectModel(
            fabric_type=FabricType.OPTICAL_OCS,
            topology=ClusterTopology.OPTICAL_OCS,
            reconfig_latency_ms=15.0,
        )
        p = tmp_path / "ocs.json"
        m.save(p)
        # Should round-trip through the json module directly too.
        data = json.loads(p.read_text())
        assert data["fabric_type"] == "optical_ocs"
        assert data["reconfig_latency_ms"] == 15.0
