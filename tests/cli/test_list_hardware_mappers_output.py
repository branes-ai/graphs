"""Tests for the --output / --format contract on cli/list_hardware_mappers.py.

Locks in:
- Format auto-detection from file extension (.json/.csv/.md/.txt)
- --format as override when extension is unrecognized
- Each format produces parseable / non-empty output
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

CLI = Path(__file__).resolve().parents[2] / "cli" / "list_hardware_mappers.py"


def _run(*args, output_path=None):
    """Invoke the CLI with the given args; return (stdout, stderr, returncode)."""
    cmd = [sys.executable, str(CLI), *args]
    if output_path is not None:
        cmd.extend(["--output", str(output_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return proc.stdout, proc.stderr, proc.returncode


class TestOutputFormatDetection:
    def test_json_extension_writes_valid_json(self, tmp_path):
        out = tmp_path / "mappers.json"
        _, stderr, rc = _run(output_path=out)
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "total_mappers" in data
        assert data["total_mappers"] > 0
        assert "Wrote json report" in stderr

    def test_csv_extension_writes_valid_csv(self, tmp_path):
        out = tmp_path / "mappers.csv"
        _, stderr, rc = _run(output_path=out)
        assert rc == 0
        rows = list(csv.reader(out.open()))
        # header + at least 10 mappers
        assert len(rows) > 10
        # Header includes the new precision fields
        header = rows[0]
        for col in ("peak_flops_fp64", "peak_flops_fp16", "peak_flops_fp8"):
            assert col in header

    def test_md_extension_routes_to_text(self, tmp_path):
        out = tmp_path / "mappers.md"
        _, stderr, rc = _run(output_path=out)
        assert rc == 0
        assert "Wrote text report" in stderr
        body = out.read_text()
        assert "HARDWARE MAPPER DISCOVERY REPORT" in body

    def test_txt_extension_routes_to_text(self, tmp_path):
        out = tmp_path / "mappers.txt"
        _, _, rc = _run(output_path=out)
        assert rc == 0
        assert "HARDWARE MAPPER DISCOVERY REPORT" in out.read_text()

    def test_unknown_extension_falls_back_to_format_flag(self, tmp_path):
        out = tmp_path / "mappers.weird"
        _, _, rc = _run("--format", "json", output_path=out)
        assert rc == 0
        # Format override resolved to JSON despite the unknown extension.
        json.loads(out.read_text())

    def test_no_output_prints_to_stdout(self):
        stdout, _, rc = _run("--category", "gpu", "--format", "json")
        assert rc == 0
        # JSON to stdout still parses
        json.loads(stdout)
