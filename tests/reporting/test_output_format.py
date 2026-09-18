"""The shared --output format helper (#269 item 1.3).

Sixteen CLIs carried their own copy of the same function and the copies had
drifted: most mapped ``.markdown`` to ``md``, the two newest did not. These
tests pin the behaviour the copies agreed on, the drift that is now gone, and
that no CLI has quietly grown a copy back.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from graphs.reporting.output_format import (
    EXTENSION_FORMATS,
    FORMATS,
    detect_format,
    write_report,
)

CLI_DIR = pathlib.Path(__file__).resolve().parents[2] / "cli"


@pytest.mark.parametrize("output, expected", [
    (None, "text"),
    ("", "text"),
    ("report.json", "json"),
    ("report.csv", "csv"),
    ("report.md", "md"),
    ("report.markdown", "md"),   # the drift: two CLIs used to call this "text"
    ("report.txt", "text"),
    ("report.out", "text"),      # unrecognized: print it, do not refuse it
    ("REPORT.JSON", "json"),     # extension case does not matter
    ("a.b/c.d/report.CSV", "csv"),
    ("no_extension", "text"),
    (pathlib.Path("report.md"), "md"),
])
def test_detect_format(output, expected):
    assert detect_format(output) == expected


def test_every_mapped_format_is_one_the_clis_emit():
    assert set(EXTENSION_FORMATS.values()) <= set(FORMATS)


@pytest.mark.parametrize("force, expected", [
    ("json", "json"),
    ("markdown", "md"),
    (".md", "md"),
    (None, "csv"),
])
def test_an_explicit_flag_wins_over_the_extension(force, expected):
    """``show_floorplan --json report.csv`` means JSON: the flag is explicit
    and the extension is a default."""
    assert detect_format("report.csv", force) == expected


def test_write_report_to_a_file(tmp_path, capsys):
    path = tmp_path / "report.md"
    write_report("# hello\n", str(path))
    assert path.read_text() == "# hello\n"
    assert f"wrote {path}" in capsys.readouterr().out


def test_write_report_can_stay_quiet(tmp_path, capsys):
    path = tmp_path / "report.md"
    write_report("x", str(path), announce=False)
    assert capsys.readouterr().out == ""


def test_write_report_to_stdout(capsys):
    write_report("line one\n")
    assert capsys.readouterr().out == "line one\n"
    write_report("no trailing newline")
    assert capsys.readouterr().out == "no trailing newline\n"


def test_no_cli_keeps_its_own_copy():
    """The point of the exercise: one implementation, not sixteen. A new CLI
    that defines ``_detect_format`` again fails this."""
    offenders = [
        p.name for p in sorted(CLI_DIR.glob("*.py"))
        if "def _detect_format" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_every_cli_that_detects_a_format_imports_the_shared_one():
    users = [
        p for p in sorted(CLI_DIR.glob("*.py"))
        if "detect_format(" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert len(users) >= 16
    for path in users:
        source = path.read_text(encoding="utf-8", errors="ignore")
        assert "from graphs.reporting.output_format import" in source, path.name


@pytest.mark.parametrize("name", ["show_pipeline_workload.py", "analyze_dies_on_workload.py"])
def test_a_migrated_cli_still_writes_each_format(tmp_path, name):
    """End to end, because the migration touched sixteen files: the CLI still
    routes --output through the shared helper and writes the right shape."""
    for extension, opener in ((".md", "#"), (".csv", "form_factor"), (".json", "{")):
        out = tmp_path / f"report{extension}"
        result = subprocess.run(
            [sys.executable, str(CLI_DIR / name), "--regimes", "--output", str(out)],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stderr[-400:]
        assert out.exists(), (name, extension)
        assert out.read_text().lstrip().startswith(opener), (name, extension)
