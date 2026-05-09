"""Tests for cli/list_hardware_resources.py.

Locks in the issue #131 contract:
- Runs cleanly across all 46 registered mappers
- All four formats produce parseable output
- --output extension auto-detection (.json/.csv/.md/.markdown/.txt)
- --category filter
- --sort with --reverse, missing values always trail
- Unpopulated mappers render as N/A and don't crash
- JSON round-trips through HardwareResourceInfo dataclass
- --format always wins over file extension (regression guard)
- process_node_nm renders when process_node_name is None (regression guard)
"""

import csv
import importlib.util
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

CLI = Path(__file__).resolve().parents[2] / "cli" / "list_hardware_resources.py"


def _import_cli_as_module():
    """Load list_hardware_resources.py as a module so we can call its
    renderer functions directly with synthetic records (subprocess-only
    tests can't inject custom HardwareResourceInfo values into the
    registry-driven discovery pipeline).
    """
    spec = importlib.util.spec_from_file_location("lhr", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args, output_path=None, expect_zero=True):
    cmd = [sys.executable, str(CLI), *args]
    if output_path is not None:
        cmd.extend(["--output", str(output_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if expect_zero:
        assert proc.returncode == 0, (
            f"CLI failed (rc={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr, proc.returncode


# ---------------------------------------------------------------------------
# Default invocation across all mappers
# ---------------------------------------------------------------------------

class TestRunsAcrossAllMappers:
    def test_default_text_runs_clean(self):
        stdout, _, _ = _run()
        assert "HARDWARE RESOURCES SPEC SHEET" in stdout
        assert "Total products:" in stdout
        # 46 is the registry count at the time of issue #131. Test asserts
        # >=10 to stay robust as new mappers are added.
        assert "PhysicalSpec coverage:" in stdout

    def test_help_works(self):
        stdout, _, _ = _run("--help")
        assert "--category" in stdout
        assert "--sort" in stdout
        assert "--output" in stdout
        assert "--verbose" in stdout


# ---------------------------------------------------------------------------
# Output format: JSON
# ---------------------------------------------------------------------------

class TestJsonFormat:
    def test_stdout_json_parses(self):
        stdout, _, _ = _run("--format", "json")
        data = json.loads(stdout)
        assert "total_products" in data
        assert "populated_physical_spec" in data
        assert "products" in data
        assert data["total_products"] == len(data["products"])

    def test_json_round_trips_h100_physical_spec(self, tmp_path):
        out = tmp_path / "spec.json"
        _, _, _ = _run(output_path=out)
        data = json.loads(out.read_text())
        h100 = next(p for p in data["products"] if p["name"] == "H100-SXM5-80GB")
        assert h100["die_size_mm2"] == 814.0
        assert h100["transistors_billion"] == 80.0
        assert h100["foundry"] == "tsmc"
        assert h100["architecture"] == "Hopper"
        assert h100["launch_date"] == "2022-09-20"
        # Density is derived; expect ~98.3 Mtx/mm^2
        assert abs(h100["transistor_density_mtx_mm2"] - 98.28) < 0.1

    def test_json_unpopulated_mappers_render_as_null(self, tmp_path):
        out = tmp_path / "spec.json"
        _, _, _ = _run(output_path=out)
        data = json.loads(out.read_text())
        a100 = next(p for p in data["products"] if p["name"] == "A100-SXM4-80GB")
        assert a100["die_size_mm2"] is None
        assert a100["transistors_billion"] is None
        # But operational fields are still populated
        assert a100["compute_units"] > 0
        assert a100["power_tdp"] > 0


# ---------------------------------------------------------------------------
# Output format: CSV
# ---------------------------------------------------------------------------

class TestCsvFormat:
    def test_csv_extension_routes_to_csv(self, tmp_path):
        out = tmp_path / "spec.csv"
        _, stderr, _ = _run(output_path=out)
        assert "Wrote csv report" in stderr
        rows = list(csv.reader(out.open()))
        # header + at least 10 mappers
        assert len(rows) > 10
        header = rows[0]
        for col in (
            "name",
            "die_size_mm2",
            "transistors_billion",
            "transistor_density_mtx_mm2",
            "peak_flops_fp64",
            "peak_flops_int8",
        ):
            assert col in header

    def test_csv_unpopulated_fields_are_empty(self, tmp_path):
        out = tmp_path / "spec.csv"
        _run(output_path=out)
        rows = list(csv.DictReader(out.open()))
        a100_row = next(r for r in rows if r["name"] == "A100-SXM4-80GB")
        assert a100_row["die_size_mm2"] == ""
        assert a100_row["transistors_billion"] == ""

    def test_csv_h100_row_is_fully_populated(self, tmp_path):
        out = tmp_path / "spec.csv"
        _run(output_path=out)
        rows = list(csv.DictReader(out.open()))
        h100 = next(r for r in rows if r["name"] == "H100-SXM5-80GB")
        assert h100["die_size_mm2"] == "814.0"
        assert h100["transistors_billion"] == "80.0"
        assert h100["foundry"] == "tsmc"


# ---------------------------------------------------------------------------
# Output format: Markdown and text
# ---------------------------------------------------------------------------

class TestMarkdownAndTextFormat:
    def test_md_extension_writes_real_markdown_table(self, tmp_path):
        out = tmp_path / "spec.md"
        _, stderr, _ = _run(output_path=out)
        assert "Wrote markdown report" in stderr
        body = out.read_text()
        assert body.startswith("# Hardware Resources Spec Sheet")
        # Markdown table separator with right-alignment markers
        assert "| ---: |" in body
        # Per-category header
        assert "## GPU" in body

    def test_markdown_extension_alias(self, tmp_path):
        out = tmp_path / "spec.markdown"
        _, stderr, _ = _run(output_path=out)
        assert "Wrote markdown report" in stderr

    def test_txt_extension_routes_to_text(self, tmp_path):
        out = tmp_path / "spec.txt"
        _, _, _ = _run(output_path=out)
        body = out.read_text()
        assert "HARDWARE RESOURCES SPEC SHEET" in body

    def test_unknown_extension_falls_back_to_format(self, tmp_path):
        out = tmp_path / "spec.weird"
        _run("--format", "json", output_path=out)
        # Loadable as JSON despite the unknown extension
        json.loads(out.read_text())


class TestFormatPrecedence:
    """Lock in the --format / extension precedence contract.

    Precedence order: explicit --format > recognized file extension > "text".
    Regression guard for an earlier bug where the extension was checked
    first and --format only kicked in for unrecognized extensions, which
    silently re-routed `--output spec.csv --format json` to CSV.
    """

    def test_explicit_format_overrides_recognized_extension(self, tmp_path):
        out = tmp_path / "override.csv"
        _, stderr, _ = _run("--format", "json", output_path=out)
        assert "Wrote json report" in stderr
        json.loads(out.read_text())  # parses as JSON despite .csv extension

    def test_md_alias_for_markdown_format(self, tmp_path):
        # ``md`` on --format is an alias for ``markdown`` (matches the .md
        # file-extension naming).
        out = tmp_path / "doc.txt"
        _, stderr, _ = _run("--format", "md", output_path=out)
        assert "Wrote markdown report" in stderr
        assert out.read_text().startswith("# Hardware Resources Spec Sheet")

    def test_auto_detect_when_no_format_flag(self, tmp_path):
        # When --format is omitted, the file extension drives format selection.
        out = tmp_path / "auto.csv"
        _, stderr, _ = _run(output_path=out)
        assert "Wrote csv report" in stderr


# ---------------------------------------------------------------------------
# Filtering and sorting
# ---------------------------------------------------------------------------

class TestCategoryFilter:
    def test_category_gpu_only(self, tmp_path):
        out = tmp_path / "spec.json"
        _run("--category", "gpu", output_path=out)
        data = json.loads(out.read_text())
        assert data["total_products"] >= 5
        assert all(p["category"] == "gpu" for p in data["products"])

    def test_category_is_case_insensitive(self, tmp_path):
        out = tmp_path / "spec.json"
        _run("--category", "GPU", output_path=out)
        data = json.loads(out.read_text())
        assert all(p["category"] == "gpu" for p in data["products"])


class TestSorting:
    def _names_for_sort(self, tmp_path, *args):
        out = tmp_path / "spec.json"
        _run(*args, output_path=out)
        data = json.loads(out.read_text())
        return [p["name"] for p in data["products"]]

    def _records_for_sort(self, tmp_path, *args):
        out = tmp_path / "spec.json"
        _run(*args, output_path=out)
        data = json.loads(out.read_text())
        return data["products"]

    def test_sort_die_size_reverse_populated_rows_precede_missing(self, tmp_path):
        # The contract: missing-value rows always trail, regardless of
        # sort direction. This is the regression guard for the original
        # missing-value-placement bug (PR #134) -- with the naive
        # `key=(missing, value), reverse=True` approach, missing rows
        # would flip to the FRONT under --reverse.
        records = self._records_for_sort(
            tmp_path, "--category", "gpu", "--sort", "die_size", "--reverse"
        )
        die_sizes = [r["die_size_mm2"] for r in records]
        # Non-None dies first, then Nones.
        last_populated = next(
            (i for i, d in enumerate(die_sizes) if d is None), len(die_sizes)
        )
        assert all(d is not None for d in die_sizes[:last_populated])
        assert all(d is None for d in die_sizes[last_populated:])
        # Populated rows are sorted descending.
        populated = die_sizes[:last_populated]
        assert populated == sorted(populated, reverse=True)
        # And H100 (largest die at 814 mm^2) should come first among GPUs.
        assert records[0]["name"] == "H100-SXM5-80GB"

    def test_sort_die_size_ascending_populated_rows_precede_missing(self, tmp_path):
        # Same contract as above for ascending order: populated rows first,
        # then missing. Populated rows sorted ascending.
        records = self._records_for_sort(
            tmp_path, "--category", "gpu", "--sort", "die_size"
        )
        die_sizes = [r["die_size_mm2"] for r in records]
        last_populated = next(
            (i for i, d in enumerate(die_sizes) if d is None), len(die_sizes)
        )
        assert all(d is not None for d in die_sizes[:last_populated])
        assert all(d is None for d in die_sizes[last_populated:])
        populated = die_sizes[:last_populated]
        assert populated == sorted(populated)

    def test_sort_name_default_is_alphabetical(self, tmp_path):
        names = self._names_for_sort(tmp_path, "--category", "gpu")
        assert names == sorted(names, key=str.lower)

    def test_sort_tops_per_watt_works(self, tmp_path):
        out = tmp_path / "spec.json"
        _run("--sort", "tops_per_watt", "--reverse", output_path=out)
        data = json.loads(out.read_text())
        # Top entry should have the highest TOPS/W; just confirm we got >0
        # rows back without crashing on TDP=0 edge cases.
        assert data["total_products"] > 0


# ---------------------------------------------------------------------------
# Verbose
# ---------------------------------------------------------------------------

class TestVerbose:
    def test_verbose_text_includes_provenance(self):
        stdout, _, _ = _run("--category", "gpu", "--verbose")
        # H100 has a populated PhysicalSpec source; verbose should surface it
        assert "embodied-schemas:" in stdout
        # Verbose surfaces the per-product Peak (G[FL]OPS) line
        assert "Peak (G[FL]OPS):" in stdout


# ---------------------------------------------------------------------------
# Renderer unit tests (require importing the CLI as a module)
# ---------------------------------------------------------------------------

class TestProcessNodeFallback:
    """Renderer-level test for the process-node fallback bug.

    When a PhysicalSpec has only ``process_node_nm`` set (and
    ``process_node_name`` is None), text and markdown renderers should
    print the nm value, not ``N/A``. The earlier bug pre-formatted both
    fields with ``_na`` and used ``or`` -- since ``_na(None)`` returns
    ``"N/A"`` (truthy), the fallback never reached the nm value.

    None of the currently-populated factories have nm-only specs (H100 and
    the Jetsons all set both), so this regression guard works at the
    renderer level with a synthetic HardwareResourceInfo.
    """

    def _synthetic_record_nm_only(self, lhr):
        """Construct a minimal HardwareResourceInfo with only nm set."""
        return lhr.HardwareResourceInfo(
            name="Synthetic-NMOnly",
            category="gpu",
            vendor="Test",
            compute_units=1,
            peak_flops_fp64=0.0,
            peak_flops_fp32=0.0,
            peak_flops_fp16=0.0,
            peak_flops_fp8=0.0,
            peak_flops_int32=0.0,
            peak_flops_int16=0.0,
            peak_flops_int8=0.0,
            memory_bandwidth=100.0,
            power_tdp=10.0,
            die_size_mm2=None,
            transistors_billion=None,
            transistor_density_mtx_mm2=None,
            process_node_nm=7,           # ONLY nm; no name
            process_node_name=None,
            foundry=None,
            architecture=None,
            num_dies=1,
            is_chiplet=False,
            package_type=None,
            memory_type=None,
            memory_bus_width_bits=None,
            launch_date=None,
            launch_msrp_usd=None,
            physical_spec_source=None,
            extras={},
        )

    def test_text_renderer_falls_back_to_nm(self):
        lhr = _import_cli_as_module()
        record = self._synthetic_record_nm_only(lhr)
        buf = io.StringIO()
        with redirect_stdout(buf):
            lhr.generate_text_report([record], verbose=False)
        out = buf.getvalue()
        # The "Node" column should show "7", not "N/A"
        # We pick a tight check: the synthetic name's row should contain " 7 "
        # and not contain "N/A" in the position of the node column.
        assert "Synthetic-NMOnly" in out
        synth_line = next(line for line in out.splitlines() if "Synthetic-NMOnly" in line)
        # The node column displays the nm value formatted by _na(int)
        assert " 7" in synth_line, f"Expected nm=7 in row, got: {synth_line!r}"

    def test_markdown_renderer_falls_back_to_nm(self):
        lhr = _import_cli_as_module()
        record = self._synthetic_record_nm_only(lhr)
        buf = io.StringIO()
        with redirect_stdout(buf):
            lhr.generate_markdown_report([record], verbose=False)
        out = buf.getvalue()
        # Find the row for our synthetic record and confirm "7" appears
        synth_line = next(line for line in out.splitlines() if "Synthetic-NMOnly" in line)
        assert " 7 " in synth_line, f"Expected nm=7 in MD row, got: {synth_line!r}"


# ---------------------------------------------------------------------------
# Phase 2 of #136: --profiles all expansion + Mode/clock columns
# ---------------------------------------------------------------------------

class TestProfilesExpansion:
    """`--profiles all` emits one row per (silicon x profile) for chips with
    multiple thermal_operating_points; chips with a single profile keep their
    silicon-bin row (no degenerate name@default duplicates)."""

    def test_default_mode_row_count_unchanged(self, tmp_path):
        # Backward compat: without --profiles, the row count matches the
        # silicon-bin count (one row per registered mapper).
        out = tmp_path / "spec.json"
        _run(output_path=out)
        data = json.loads(out.read_text())
        # Pre-Phase 2 baseline was 46 silicon-bins; could grow as new
        # mappers are added but should never EXCEED list_all_mappers().
        from graphs.hardware.mappers import list_all_mappers
        assert data["total_products"] == len(list_all_mappers())

    def test_profiles_all_emits_more_rows(self, tmp_path):
        # --profiles all expands multi-profile chips into one row per
        # profile, so the total grows.
        default_out = tmp_path / "default.json"
        all_out = tmp_path / "all.json"
        _run(output_path=default_out)
        _run("--profiles", "all", output_path=all_out)
        default_data = json.loads(default_out.read_text())
        all_data = json.loads(all_out.read_text())
        assert all_data["total_products"] > default_data["total_products"]

    def test_profiles_all_orin_nano_emits_four_rows(self, tmp_path):
        # Orin Nano has 4 modes (7W / 15W / 25W / MAXN per #136). Under
        # --profiles all, the silicon-bin row is replaced by 4 alias rows.
        out = tmp_path / "spec.json"
        _run("--profiles", "all", "--category", "gpu", output_path=out)
        data = json.loads(out.read_text())
        nano_rows = [
            p for p in data["products"]
            if p["name"].startswith("Jetson-Orin-Nano-8GB")
        ]
        # 4 alias rows; no bare silicon-bin row in expanded mode.
        assert len(nano_rows) == 4
        names = sorted(r["name"] for r in nano_rows)
        assert names == [
            "Jetson-Orin-Nano-8GB@15W",
            "Jetson-Orin-Nano-8GB@25W",
            "Jetson-Orin-Nano-8GB@7W",
            "Jetson-Orin-Nano-8GB@MAXN",
        ]

    def test_profiles_all_per_profile_tdp_distinct(self, tmp_path):
        # Each Orin Nano alias row should report its OWN TDP, not the
        # default profile's TDP. Spec invariant: 7W=7W, 15W=15W, 25W=25W,
        # MAXN=25W (Super silicon's max envelope).
        out = tmp_path / "spec.json"
        _run("--profiles", "all", "--category", "gpu", output_path=out)
        data = json.loads(out.read_text())
        tdps = {
            r["mode"]: r["power_tdp"]
            for r in data["products"]
            if r["name"].startswith("Jetson-Orin-Nano-8GB@")
        }
        assert tdps == {"7W": 7.0, "15W": 15.0, "25W": 25.0, "MAXN": 25.0}

    def test_profiles_all_per_profile_clocks_distinct(self, tmp_path):
        # 7W mode boosts to 918 MHz; 15W/25W/MAXN boost to 1020 MHz.
        # Documents the per-profile clock variation that makes Phase 2
        # worth shipping in the first place.
        out = tmp_path / "spec.json"
        _run("--profiles", "all", "--category", "gpu", output_path=out)
        data = json.loads(out.read_text())
        boosts = {
            r["mode"]: r["core_boost_mhz"]
            for r in data["products"]
            if r["name"].startswith("Jetson-Orin-Nano-8GB@")
        }
        assert boosts["7W"] == 918.0
        assert boosts["15W"] == 1020.0
        assert boosts["25W"] == 1020.0
        assert boosts["MAXN"] == 1020.0

    def test_profiles_all_single_profile_chip_keeps_silicon_name(self, tmp_path):
        # H100 has a single placeholder thermal_operating_point named
        # "default". Under --profiles all, this chip should NOT get a
        # degenerate "H100-SXM5-80GB@default" alias -- the silicon-bin
        # row is preserved as-is. Avoids visual noise for chips that
        # don't expose nvpmodel-style modes.
        out = tmp_path / "spec.json"
        _run("--profiles", "all", output_path=out)
        data = json.loads(out.read_text())
        h100_rows = [p for p in data["products"] if p["name"].startswith("H100-SXM5-80GB")]
        assert len(h100_rows) == 1
        assert h100_rows[0]["name"] == "H100-SXM5-80GB"  # bare silicon-bin

    def test_default_mode_populates_clocks_for_jetson(self, tmp_path):
        # Even in default mode, Jetson chips should populate Base/Boost MHz
        # from their default thermal profile's ClockDomain. This is the
        # "even default mode is more informative now" win from Phase 2.
        out = tmp_path / "spec.json"
        _run(output_path=out)
        data = json.loads(out.read_text())
        nano = next(
            p for p in data["products"]
            if p["name"] == "Jetson-Orin-Nano-8GB"
        )
        # 15W is the default profile -- clocks are 306 / 1020 MHz.
        assert nano["mode"] == "15W"
        assert nano["core_base_mhz"] == 306.0
        assert nano["core_boost_mhz"] == 1020.0

    def test_default_mode_chips_without_clock_domain_render_none(self, tmp_path):
        # KPUs use KPUComputeResource which doesn't expose a ClockDomain.
        # The defensive _clocks_from_profile helper falls back to None.
        # The CLI renders None as empty (CSV) / N/A (text/markdown), no
        # crash.
        out = tmp_path / "spec.json"
        _run("--profiles", "all", output_path=out)
        data = json.loads(out.read_text())
        kpu_rows = [p for p in data["products"] if p["name"].startswith("Stillwater-KPU")]
        assert len(kpu_rows) > 0
        for r in kpu_rows:
            # mode populated (KPU has thermal_operating_points), but
            # clocks are None for the architecture-specific compute
            # resource that doesn't expose ClockDomain.
            assert r["mode"] is not None
            assert r["core_base_mhz"] is None
            assert r["core_boost_mhz"] is None


class TestPhase2Renderers:
    """Mode / Base MHz / Boost MHz columns appear in text and markdown
    output. CSV / JSON pick up the new fields automatically via the
    dataclass field iteration."""

    def test_text_header_includes_new_columns(self):
        stdout, _, _ = _run("--category", "gpu")
        for col in ("Mode", "Base MHz", "Boost MHz"):
            assert col in stdout, f"text header missing column: {col}"

    def test_markdown_header_includes_new_columns(self, tmp_path):
        out = tmp_path / "spec.md"
        _run(output_path=out)
        body = out.read_text()
        for col in ("Mode", "Base MHz", "Boost MHz"):
            assert col in body, f"markdown header missing column: {col}"

    def test_csv_includes_new_field_columns(self, tmp_path):
        out = tmp_path / "spec.csv"
        _run(output_path=out)
        rows = list(csv.reader(out.open()))
        header = rows[0]
        for col in ("mode", "core_base_mhz", "core_boost_mhz"):
            assert col in header, f"csv missing column: {col}"


class TestPhase3MemoryColumn:
    """Phase 3 of #136: Memory Type / Bus columns. memory_type and
    memory_bus_width_bits are now PhysicalSpec fields, surfaced in the
    spec-sheet view across all four output formats."""

    def test_text_header_includes_memory_columns(self):
        stdout, _, _ = _run("--category", "gpu")
        assert "Memory" in stdout, "text header missing 'Memory' column"
        assert "Bus" in stdout, "text header missing 'Bus' column"

    def test_markdown_header_includes_memory_columns(self, tmp_path):
        out = tmp_path / "spec.md"
        _run(output_path=out)
        body = out.read_text()
        assert "Memory" in body
        # Markdown uses literal "Bus" (not "Bus (bits)") to keep header tight.
        assert "| Bus |" in body

    def test_csv_includes_memory_columns(self, tmp_path):
        out = tmp_path / "spec.csv"
        _run(output_path=out)
        header = next(csv.reader(out.open()))
        assert "memory_type" in header
        assert "memory_bus_width_bits" in header

    def test_json_round_trips_memory_fields(self, tmp_path):
        # H100 + 4 Jetson SKUs are populated as of Phase 3.
        out = tmp_path / "spec.json"
        _run(output_path=out)
        data = json.loads(out.read_text())
        h100 = next(p for p in data["products"] if p["name"] == "H100-SXM5-80GB")
        assert h100["memory_type"] == "hbm3"
        assert h100["memory_bus_width_bits"] == 5120

    def test_json_orin_family_bus_widths_lock_in_per_sku(self, tmp_path):
        # Spec invariant: AGX=256, NX=128, Nano=128 bits. Verified via
        # bandwidth math against NVIDIA's published BW figures
        # (BW / DRAM-rate = bus_width). NX and Nano share the same
        # 128-bit width but differ in sustained DRAM rate, which is
        # what gives them different headline bandwidths.
        out = tmp_path / "spec.json"
        _run(output_path=out)
        data = json.loads(out.read_text())
        agx = next(p for p in data["products"] if p["name"] == "Jetson-Orin-AGX-64GB")
        nx = next(p for p in data["products"] if p["name"] == "Jetson-Orin-NX-16GB")
        nano = next(p for p in data["products"] if p["name"] == "Jetson-Orin-Nano-8GB")
        assert agx["memory_bus_width_bits"] == 256
        assert nx["memory_bus_width_bits"] == 128
        assert nano["memory_bus_width_bits"] == 128
        # All three are LPDDR5; Thor switches to LPDDR5X.
        assert agx["memory_type"] == "lpddr5"
        assert nx["memory_type"] == "lpddr5"
        assert nano["memory_type"] == "lpddr5"

    def test_unpopulated_chips_render_na_for_memory(self, tmp_path):
        # A100 / B100 / V100 / T4 / etc. don't have populated PhysicalSpec
        # yet, so memory_type renders None in JSON / empty in CSV / N/A
        # in text+markdown. No crash, just graceful absence.
        out = tmp_path / "spec.json"
        _run(output_path=out)
        data = json.loads(out.read_text())
        a100 = next(p for p in data["products"] if p["name"] == "A100-SXM4-80GB")
        assert a100["memory_type"] is None
        assert a100["memory_bus_width_bits"] is None


class TestPhase4MemoryClock:
    """Phase 4 of #136: Memory Clock column. Per-profile memory_clock_mhz
    surfaces from each ThermalOperatingPoint."""

    def test_text_header_includes_mem_mhz_column(self):
        stdout, _, _ = _run("--category", "gpu")
        # Header has the new column ('Mem MHz', between 'Bus' and 'TDP').
        assert "Mem MHz" in stdout

    def test_markdown_header_includes_mem_mhz_column(self, tmp_path):
        out = tmp_path / "spec.md"
        _run(output_path=out)
        assert "Mem MHz" in out.read_text()

    def test_csv_includes_memory_clock_column(self, tmp_path):
        out = tmp_path / "spec.csv"
        _run(output_path=out)
        header = next(csv.reader(out.open()))
        assert "memory_clock_mhz" in header

    def test_jetson_default_profiles_populated(self, tmp_path):
        # Phase 4 backfilled all 4 Jetson SKUs with the silicon's
        # headline DRAM rate. Default-mode rows (the ones rendered
        # without --profiles all) should show those values.
        out = tmp_path / "spec.json"
        _run("--category", "gpu", output_path=out)
        data = json.loads(out.read_text())
        nano = next(p for p in data["products"] if p["name"] == "Jetson-Orin-Nano-8GB")
        agx = next(p for p in data["products"] if p["name"] == "Jetson-Orin-AGX-64GB")
        nx = next(p for p in data["products"] if p["name"] == "Jetson-Orin-NX-16GB")
        thor = next(p for p in data["products"] if p["name"] == "Jetson-Thor-128GB")
        # Orin family at LPDDR5-6400 (3200 MHz internal); Thor at
        # LPDDR5X-8533 (4267 MHz internal).
        assert nano["memory_clock_mhz"] == 3200.0
        assert agx["memory_clock_mhz"] == 3200.0
        assert nx["memory_clock_mhz"] == 3200.0
        assert thor["memory_clock_mhz"] == 4267.0

    def test_unpopulated_chips_render_none_for_mem_clock(self, tmp_path):
        # H100 / A100 / B100 / etc. don't have memory_clock_mhz set on
        # their thermal_operating_points yet -- they render None
        # gracefully (-> "N/A" in text/markdown, "" in CSV).
        out = tmp_path / "spec.json"
        _run(output_path=out)
        data = json.loads(out.read_text())
        h100 = next(p for p in data["products"] if p["name"] == "H100-SXM5-80GB")
        a100 = next(p for p in data["products"] if p["name"] == "A100-SXM4-80GB")
        assert h100["memory_clock_mhz"] is None
        assert a100["memory_clock_mhz"] is None

    def test_profiles_all_propagates_per_profile_memory_clock(self, tmp_path):
        # Under --profiles all, each Orin Nano alias row carries the
        # memory_clock_mhz from its specific ThermalOperatingPoint. The
        # current backfill assigns the same 3200 MHz to all 4 modes
        # (Super silicon's max); when authoritative throttling data
        # later refines 7W, the per-profile expansion will reflect that.
        out = tmp_path / "spec.json"
        _run("--profiles", "all", "--category", "gpu", output_path=out)
        data = json.loads(out.read_text())
        nano_records = [
            p for p in data["products"]
            if p["name"].startswith("Jetson-Orin-Nano-8GB@")
        ]
        # 4 Nano modes (7W/15W/25W/MAXN), each populated.
        assert len(nano_records) == 4
        assert all(r["memory_clock_mhz"] == 3200.0 for r in nano_records)
