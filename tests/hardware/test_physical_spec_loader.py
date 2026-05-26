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

import pytest

from graphs.hardware.physical_spec import PhysicalSpec
from graphs.hardware.physical_spec_loader import (
    KNOWN_OVERRIDES,
    _format_process_node_name,
    _resolve_data_dir,
    load_physical_spec,
    load_physical_spec_or_none,
    validate_physical_spec,
)
import graphs.hardware.physical_spec_loader as _loader_module


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
    embodied-schemas YAML bugs. The loader applies them before returning
    so consumers see the correct values.

    The two original entries (Jetson Thor + Orin Nano memory_bus_bits
    from embodied-schemas#8) were removed once the upstream YAML fix
    landed in embodied-schemas#83. KNOWN_OVERRIDES is now empty, and
    the loader returns the correct values directly from the YAML.

    The tests below double-check that the values still resolve correctly
    -- they would fail if the upstream YAML regresses to the buggy
    values, or if the loader stops surfacing the field at all.
    """

    def test_thor_bus_width_is_256(self):
        # embodied-schemas#8 (closed by #83): Jetson Thor YAML correctly
        # lists 256-bit LPDDR5X. Bandwidth math (273 GB/s / 8.533 GT/s)
        # and NVIDIA's announcement blog both confirm 256.
        spec = load_physical_spec("nvidia_thor_gpu_128gb_lpddr5x")
        assert spec.memory_bus_width_bits == 256

    def test_orin_nano_bus_width_is_128(self):
        # embodied-schemas#8 (closed by #83): Orin Nano YAML correctly
        # lists 128-bit LPDDR5-4267. Bandwidth math (68 GB/s / 4.267
        # GT/s) and NVIDIA's datasheet both confirm 128.
        spec = load_physical_spec("nvidia_orin_nano_gpu_8gb_lpddr5")
        assert spec.memory_bus_width_bits == 128

    def test_known_overrides_list_is_empty(self):
        # Documents the principle that overrides should be REMOVED when
        # the upstream YAML is fixed. embodied-schemas#83 closed the
        # original #8 entries; the dict is now empty. If this assertion
        # grows back beyond zero, the new entry MUST cite the upstream
        # issue (see the docstring on KNOWN_OVERRIDES).
        assert len(KNOWN_OVERRIDES) == 0


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

    def test_env_var_pointing_at_file_not_dir_raises(self, tmp_path):
        # is_dir() check (not just exists()): if the env var points at
        # a regular file, fail fast with a clear error rather than
        # confusing downstream YAML lookup errors.
        marker_file = tmp_path / "not_a_dir.txt"
        marker_file.write_text("placeholder")
        old = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
        os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = str(marker_file)
        try:
            with pytest.raises(FileNotFoundError) as exc:
                _resolve_data_dir()
            assert "is not an existing directory" in str(exc.value)
        finally:
            if old is None:
                del os.environ["EMBODIED_SCHEMAS_DATA_DIR"]
            else:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old


class TestGracefulLoader:
    """``load_physical_spec_or_none`` returns None instead of raising
    when embodied-schemas data isn't reachable, so factories don't
    crash for users who pip-installed graphs without the [schemas]
    extra. Emits a one-time warning to stderr on first miss.
    """

    def test_present_data_returns_real_spec(self):
        # When data IS reachable, the helper returns the same value
        # that load_physical_spec would.
        spec = load_physical_spec_or_none("nvidia_h100_sxm5_80gb_hbm3")
        assert spec is not None
        assert spec.die_size_mm2 == 814.0

    def test_missing_data_returns_none(self, tmp_path, capsys):
        # Reset the module-level "warned once" flag so we can observe
        # the warning emit cleanly.
        _loader_module._warned_about_missing_data = False
        marker_file = tmp_path / "not_a_dir"  # doesn't exist
        old = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
        os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = str(marker_file)
        try:
            result = load_physical_spec_or_none("nvidia_h100_sxm5_80gb_hbm3")
            assert result is None
            captured = capsys.readouterr()
            assert "warning" in captured.err.lower()
            assert "embodied-schemas" in captured.err
        finally:
            if old is None:
                del os.environ["EMBODIED_SCHEMAS_DATA_DIR"]
            else:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old
            _loader_module._warned_about_missing_data = False

    def test_missing_data_warning_fires_once_per_process(self, tmp_path, capsys):
        # The warning is gated by a module-level flag so multiple
        # factory calls in the same process don't spam stderr.
        _loader_module._warned_about_missing_data = False
        marker_file = tmp_path / "still_does_not_exist"
        old = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
        os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = str(marker_file)
        try:
            # Fire 3 misses; only the first should produce stderr output.
            load_physical_spec_or_none("nvidia_h100_sxm5_80gb_hbm3")
            load_physical_spec_or_none("nvidia_orin_gpu_64gb_lpddr5")
            load_physical_spec_or_none("nvidia_thor_gpu_128gb_lpddr5x")
            captured = capsys.readouterr()
            # One "warning:" line; subsequent calls were silent.
            assert captured.err.count("warning:") == 1
        finally:
            if old is None:
                del os.environ["EMBODIED_SCHEMAS_DATA_DIR"]
            else:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old
            _loader_module._warned_about_missing_data = False

    def test_strict_load_still_raises(self, tmp_path):
        # The strict ``load_physical_spec`` keeps its raise-on-missing
        # contract for callers (like tests) that need to fail loudly.
        marker_file = tmp_path / "no_data_here"
        old = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
        os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = str(marker_file)
        try:
            with pytest.raises(FileNotFoundError):
                load_physical_spec("nvidia_h100_sxm5_80gb_hbm3")
        finally:
            if old is None:
                del os.environ["EMBODIED_SCHEMAS_DATA_DIR"]
            else:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old


# ---------------------------------------------------------------------------
# YAML propagation contract -- closes issue #132 exit criterion 4
# ---------------------------------------------------------------------------

class TestYamlPropagationContract:
    """End-to-end propagation: when an embodied-schemas YAML field
    changes on disk, the change must be observable through
    ``load_physical_spec``.

    This is issue #132's exit criterion 4: "test that updates an
    embodied-schemas YAML field and confirms the change propagates to
    graphs/ via the loader." The other exit criteria are satisfied by
    the loader-backed factory wiring that landed across sprints #234,
    #241, and #245 (43 of 47 registered mappers now source PhysicalSpec
    from embodied-schemas via this loader; the 4 unpopulated SKUs are
    bounded by data availability per
    ``DECISION-2026-05-21-001.yaml`` Question B).

    The tests build a synthetic YAML in ``tmp_path`` and point
    ``EMBODIED_SCHEMAS_DATA_DIR`` at it -- so the contract is verified
    against a controlled file, not the live catalog (whose values are
    pinned separately by ``TestLoaderRoundTrip``).
    """

    @staticmethod
    def _write_yaml(yaml_path, die_size_mm2, transistors_billion, bus_bits=256):
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            "id: test_synthetic_propagation_sku\n"
            "name: Test Synthetic Propagation SKU\n"
            "vendor: test\n"
            "die:\n"
            f"  die_size_mm2: {die_size_mm2}\n"
            f"  transistors_billion: {transistors_billion}\n"
            "  foundry: tsmc\n"
            "  process_nm: 7\n"
            "  process_name: N7\n"
            "  is_chiplet: false\n"
            "  num_dies: 1\n"
            "memory:\n"
            "  memory_size_gb: 8.0\n"
            "  memory_type: hbm3\n"
            f"  memory_bus_bits: {bus_bits}\n"
        )

    def test_yaml_value_propagates_to_loader_output(self, tmp_path):
        """Loader returns the value written to the YAML on disk."""
        yaml_path = (
            tmp_path / "gpus" / "test" / "test_synthetic_propagation_sku.yaml"
        )
        self._write_yaml(yaml_path, die_size_mm2=123.4, transistors_billion=5.6)

        old = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
        os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = str(tmp_path)
        try:
            spec = load_physical_spec("test_synthetic_propagation_sku")
            assert spec.die_size_mm2 == 123.4
            assert spec.transistors_billion == 5.6
            assert spec.memory_bus_width_bits == 256
            assert spec.process_node_nm == 7
        finally:
            if old is None:
                del os.environ["EMBODIED_SCHEMAS_DATA_DIR"]
            else:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old

    def test_yaml_edit_propagates_to_loader_output(self, tmp_path):
        """When the YAML is mutated, a second loader call returns the
        new value -- there's no stale-cache path between graphs and
        embodied-schemas."""
        yaml_path = (
            tmp_path / "gpus" / "test" / "test_synthetic_propagation_sku.yaml"
        )
        self._write_yaml(yaml_path, die_size_mm2=100.0, transistors_billion=1.0)

        old = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
        os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = str(tmp_path)
        try:
            first = load_physical_spec("test_synthetic_propagation_sku")
            assert first.die_size_mm2 == 100.0
            assert first.transistors_billion == 1.0

            # Mutate the YAML on disk
            self._write_yaml(
                yaml_path, die_size_mm2=200.0, transistors_billion=2.0
            )

            second = load_physical_spec("test_synthetic_propagation_sku")
            assert second.die_size_mm2 == 200.0
            assert second.transistors_billion == 2.0
        finally:
            if old is None:
                del os.environ["EMBODIED_SCHEMAS_DATA_DIR"]
            else:
                os.environ["EMBODIED_SCHEMAS_DATA_DIR"] = old


# ---------------------------------------------------------------------------
# Coverage pin -- closes issue #132 exit criteria 1+2
# ---------------------------------------------------------------------------

class TestPhysicalSpecCoveragePin:
    """Pins the current registry's physical_spec coverage state so a
    future refactor that accidentally drops physical_spec wiring on a
    previously-populated mapper is caught loudly.

    Coverage state is the closure of campaigns #130 + #234 + #241:
    43 of 47 registered mappers source PhysicalSpec from embodied-
    schemas; the 4 N/A SKUs are the data-availability tail documented
    in ``DECISION-2026-05-21-001.yaml`` Question B.

    Reopening criteria for any of the 4 N/A SKUs:
      - ARM disclosure (Mali) -- ARM is licensable IP; unlikely
      - Qualcomm disclosure (Snapdragon Ride) -- proprietary; unlikely
      - Third-party die-shot research landing for either of the above
      - The 1-core AmpereOne reference is synthetic (Ampere only ships
        full-die parts) -- can't have public die data
      - DFM-128 is a Stillwater research prototype
    """

    EXPECTED_UNPOPULATED = {
        "ARM-Mali-G78-MP20",
        "Ampere-AmpereOne-1core-ref",
        # Reference-IP resource model, no ComputeProduct YAML / die data (#176).
        "ARM-Neoverse-N2-1core",
        "Qualcomm-Snapdragon-Ride",
        "Stillwater-DFM-128",
    }

    def test_registry_physical_spec_coverage_state(self):
        """43 populated, 5 unpopulated -- post-#241 closure + N2 reference (#176)."""
        from graphs.hardware.mappers import list_all_mappers, get_mapper_by_name

        unpopulated = set()
        populated_count = 0
        for name in list_all_mappers():
            mapper = get_mapper_by_name(name)
            if mapper is None or not hasattr(mapper, "physical_spec"):
                continue
            if mapper.physical_spec is None:
                unpopulated.add(name)
            else:
                populated_count += 1

        assert unpopulated == self.EXPECTED_UNPOPULATED, (
            f"physical_spec coverage drift detected.\n"
            f"  Expected unpopulated: {sorted(self.EXPECTED_UNPOPULATED)}\n"
            f"  Actual unpopulated:   {sorted(unpopulated)}\n"
            f"  If a new mapper is N/A by data availability, add it to "
            f"EXPECTED_UNPOPULATED. If a previously-populated mapper "
            f"regressed, fix the factory wiring."
        )
        assert populated_count == 43, (
            f"populated mapper count drift: expected 43, got "
            f"{populated_count}. Cross-check against "
            f"DECISION-2026-05-21-001 Question B coverage closure."
        )
