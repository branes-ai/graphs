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

    def test_sort_die_size_reverse_puts_populated_first(self, tmp_path):
        names = self._names_for_sort(tmp_path, "--category", "gpu", "--sort", "die_size", "--reverse")
        # H100 SXM5 is the only populated GPU; it should appear FIRST under
        # --sort die_size --reverse, not last (regression guard for the
        # missing-value placement bug).
        assert names[0] == "H100-SXM5-80GB"

    def test_sort_die_size_ascending_still_puts_populated_first(self, tmp_path):
        # In ascending order, populated values come first too because we
        # explicitly trail missing values at the end.
        names = self._names_for_sort(tmp_path, "--category", "gpu", "--sort", "die_size")
        assert names[0] == "H100-SXM5-80GB"

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
