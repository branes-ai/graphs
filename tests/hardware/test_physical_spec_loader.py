"""Tests for cli/list_hardware_resources.py YAML loader (issue #136 Phase 3.5).

Locks in the contract:
- ``load_physical_spec(base_id)`` reads the embodied-schemas YAML and
  returns a populated PhysicalSpec.
- ``KNOWN_OVERRIDES`` corrects documented YAML bugs at load time.
- ``validate_physical_spec`` flags bandwidth-math inconsistencies (the
  bug class hit by embodied-schemas#8).
- Path resolution honors ``EMBODIED_SCHEMAS_DATA_DIR``, then falls back
  through installed package -> sibling clone.
"""

import os
from pathlib import Path

import pytest

from graphs.hardware.physical_spec import PhysicalSpec
from graphs.hardware.physical_spec_loader import (
    KNOWN_OVERRIDES,
    _format_process_node_name,
    _resolve_data_dir,
    load_physical_spec,
    validate_physical_spec,
)


class TestLoaderRoundTrip:
    """Each populated factory's base_id resolves and produces a sensible
    PhysicalSpec. Anchors against the values we expect post-loader."""

    def test_h100_loads_correctly(self):
        spec = load_physical_spec("nvidia_h100_sxm5_80gb_hbm3")
        assert spec.die_size_mm2 == 814.0
        assert spec.transistors_billion == 80.0
        assert spec.process_node_nm == 4
        assert spec.foundry == "tsmc"
        assert spec.architecture == "Hopper"
        assert spec.memory_type == "hbm3"
        assert spec.memory_bus_width_bits == 5120
        assert spec.launch_date == "2022-09-20"
        assert spec.launch_msrp_usd == 30000.0
        assert spec.source.startswith("embodied-schemas:")

    def test_orin_agx_loads_correctly(self):
        spec = load_physical_spec("nvidia_orin_gpu_64gb_lpddr5")
        assert spec.die_size_mm2 == 455.0
        assert spec.transistors_billion == 17.0
        assert spec.architecture == "Ampere"
        assert spec.memory_bus_width_bits == 256
        assert spec.memory_type == "lpddr5"

    def test_orin_nx_loads_correctly(self):
        spec = load_physical_spec("nvidia_orin_nx_gpu_16gb_lpddr5")
        assert spec.die_size_mm2 == 455.0
        assert spec.memory_bus_width_bits == 128

    def test_unknown_base_id_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            load_physical_spec("nvidia_definitely_not_a_real_chip_xyz")


class TestKnownOverridesApplied:
    """KNOWN_OVERRIDES captures field-level corrections for documented
    embodied-schemas YAML bugs (#8). The loader must apply them before
    returning so consumers see the correct values."""

    def test_thor_bus_width_corrected_from_yaml_bug(self):
        # Raw YAML lists 512-bit, but bandwidth math (273 GB/s /
        # 8.533 GT/s LPDDR5X) confirms 256-bit. The KNOWN_OVERRIDES
        # entry corrects this at load time.
        spec = load_physical_spec("nvidia_thor_gpu_128gb_lpddr5x")
        assert spec.memory_bus_width_bits == 256
        # Sanity: the override key is documented for this base_id.
        assert "nvidia_thor_gpu_128gb_lpddr5x" in KNOWN_OVERRIDES

    def test_orin_nano_bus_width_corrected_from_yaml_bug(self):
        # Raw YAML lists 64-bit, but bandwidth math (68 GB/s /
        # 4.267 GT/s LPDDR5-4267) confirms 128-bit.
        spec = load_physical_spec("nvidia_orin_nano_gpu_8gb_lpddr5")
        assert spec.memory_bus_width_bits == 128
        assert "nvidia_orin_nano_gpu_8gb_lpddr5" in KNOWN_OVERRIDES

    def test_known_overrides_list_is_minimal(self):
        # Documents the principle that overrides should be REMOVED when
        # the upstream YAML is fixed. If this assertion grows beyond a
        # handful of entries, that's a signal to escalate the upstream
        # bug fix rather than accumulate corrections downstream.
        # As of #139, only the two bugs from embodied-schemas#8.
        assert len(KNOWN_OVERRIDES) == 2


class TestProcessNodeNameComposition:
    """``_format_process_node_name`` composes ``"{Foundry} {process_name}"``
    with display-cased foundry. Documents the expected naming convention."""

    def test_tsmc_uppercased(self):
        assert _format_process_node_name("tsmc", "N4") == "TSMC N4"

    def test_samsung_titlecased(self):
        assert _format_process_node_name("samsung", "8LPP") == "Samsung 8LPP"

    def test_intel_titlecased(self):
        assert _format_process_node_name("intel", "7") == "Intel 7"

    def test_unknown_foundry_falls_through_to_title_case(self):
        assert _format_process_node_name("globalfoundries", "12LP") == "GlobalFoundries 12LP"

    def test_no_inputs_returns_none(self):
        assert _format_process_node_name(None, None) is None

    def test_only_foundry_returns_foundry(self):
        assert _format_process_node_name("tsmc", None) == "TSMC"

    def test_only_process_name_returns_process_name(self):
        assert _format_process_node_name(None, "N4") == "N4"


class TestBandwidthValidator:
    """validate_physical_spec catches the specific bug class that cost us
    a round of review on PR #139: the YAMLs internally contradicted
    themselves on memory_bus_bits vs memory_bandwidth_gbps."""

    def test_buggy_thor_512bit_caught(self):
        # Synthesize the buggy YAML state: 512-bit @ 8.533 GT/s would
        # imply 545 GB/s, not 273. 100% relative error.
        spec = PhysicalSpec(memory_bus_width_bits=512)
        warnings = validate_physical_spec(
            spec,
            peak_bandwidth_gbps=273.0,
            dram_rate_gtps=8.533,
            base_id="nvidia_thor_gpu_128gb_lpddr5x",
        )
        assert len(warnings) == 1
        assert "memory bandwidth math inconsistent" in warnings[0]

    def test_buggy_nano_64bit_caught(self):
        # The other half of embodied-schemas#8: 64-bit @ 4.267 GT/s
        # implies 34 GB/s, not 68. 50% relative error.
        spec = PhysicalSpec(memory_bus_width_bits=64)
        warnings = validate_physical_spec(
            spec,
            peak_bandwidth_gbps=68.0,
            dram_rate_gtps=4.267,
            base_id="nvidia_orin_nano_gpu_8gb_lpddr5",
        )
        assert len(warnings) == 1

    def test_corrected_thor_256bit_passes(self):
        spec = PhysicalSpec(memory_bus_width_bits=256)
        warnings = validate_physical_spec(
            spec,
            peak_bandwidth_gbps=273.0,
            dram_rate_gtps=8.533,
            base_id="nvidia_thor_gpu_128gb_lpddr5x",
        )
        assert warnings == []

    def test_corrected_nano_128bit_passes(self):
        spec = PhysicalSpec(memory_bus_width_bits=128)
        warnings = validate_physical_spec(
            spec,
            peak_bandwidth_gbps=68.0,
            dram_rate_gtps=4.267,
            base_id="nvidia_orin_nano_gpu_8gb_lpddr5",
        )
        assert warnings == []

    def test_within_5pct_tolerance_does_not_warn(self):
        # Real specs round; tolerate small deviations. A 4% error
        # (close to but under the 5% threshold) should pass clean.
        # 256-bit @ 8.5 GT/s = 272 GB/s (vs spec'd 273 = 0.4% err)
        spec = PhysicalSpec(memory_bus_width_bits=256)
        warnings = validate_physical_spec(
            spec,
            peak_bandwidth_gbps=273.0,
            dram_rate_gtps=8.5,  # slightly off from 8.533
            base_id="test:within-tolerance",
        )
        assert warnings == []

    def test_no_warning_when_inputs_incomplete(self):
        # Validator silently skips checks it doesn't have data for --
        # only fires when ALL three of bus_width, peak_bandwidth, and
        # dram_rate are supplied.
        spec = PhysicalSpec(memory_bus_width_bits=256)
        # Missing dram_rate_gtps -- no check possible.
        warnings = validate_physical_spec(spec, peak_bandwidth_gbps=273.0)
        assert warnings == []


class TestDataDirResolution:
    """Resolution priority: env var > installed pkg > sibling clone.
    Tests the env-var override path which is what CI / fixture tests
    rely on for deterministic data."""

    def test_env_var_override_works(self, tmp_path):
        # Set EMBODIED_SCHEMAS_DATA_DIR to a real directory; resolver
        # returns that path even if it doesn't actually contain YAMLs
        # (the dir-existence check is what matters here, not content).
        marker_dir = tmp_path / "schema_data"
        marker_dir.mkdir()
        old = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
        os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = str(marker_dir)
        try:
            resolved = _resolve_data_dir()
            assert resolved == marker_dir
        finally:
            if old is None:
                del os.environ["EMBODIED_SCHEMAS_DATA_DIR"]
            else:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old

    def test_env_var_pointing_nowhere_raises(self, tmp_path):
        bogus_path = tmp_path / "definitely_does_not_exist"
        old = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
        os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = str(bogus_path)
        try:
            with pytest.raises(FileNotFoundError) as exc:
                _resolve_data_dir()
            assert "EMBODIED_SCHEMAS_DATA_DIR" in str(exc.value)
        finally:
            if old is None:
                del os.environ["EMBODIED_SCHEMAS_DATA_DIR"]
            else:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old

    def test_default_resolution_finds_sibling_clone(self):
        # In this dev environment, the sibling clone exists at
        # ../embodied-schemas. The resolver should find it without an
        # explicit env var.
        # Clear any env override first.
        old = os.environ.pop("EMBODIED_SCHEMAS_DATA_DIR", None)
        try:
            resolved = _resolve_data_dir()
            assert resolved.exists()
            assert resolved.name == "data"
        finally:
            if old is not None:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old
