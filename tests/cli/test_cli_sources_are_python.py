"""Every file in cli/ with a .py extension must be Python.

`cli/model_registry_tv2dot7.py` was captured console output from
`cli/discover_models.py`, redirected into a file with a `.py` extension. The
tick marks in its summary table meant Python could not parse it, and the
`MODEL_REGISTRY` block it ended with was a snippet the tool prints for pasting
elsewhere, not a module -- so nothing imported it and nothing noticed for a
year. `cli/README.md` documented an import from it that never worked.

It now lives in `docs/logs/` as a `.txt`. This test is the guard: a transcript
saved as source fails here rather than sitting unparseable in a directory of
executables.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

CLI_DIR = pathlib.Path(__file__).resolve().parents[2] / "cli"


@pytest.mark.parametrize(
    "path", sorted(CLI_DIR.glob("*.py")), ids=lambda p: p.name
)
def test_cli_file_parses_as_python(path: pathlib.Path):
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(
            f"{path.name} is not Python ({exc.msg} at line {exc.lineno}). "
            f"Captured output belongs in docs/logs/ with a .txt extension."
        )
