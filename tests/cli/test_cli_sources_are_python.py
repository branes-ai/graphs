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
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CLI_DIR = REPO / "cli"


def minimum_python_minor() -> int:
    """The minor version pyproject declares, e.g. 10 for ``>=3.10``.

    Read rather than hardcoded so the guard cannot drift from the
    declaration. Pull-request CI runs one interpreter (3.12), so without
    this the guard would accept syntax that only the push build, on the full
    3.10 / 3.11 / 3.12 matrix, would reject.
    """
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"""requires-python\s*=\s*["']>=\s*3\.(\d+)""", text)
    assert match, "pyproject.toml does not declare requires-python >=3.x"
    return int(match.group(1))


MIN_MINOR = minimum_python_minor()


@pytest.mark.parametrize(
    "path", sorted(CLI_DIR.glob("*.py")), ids=lambda p: p.name
)
def test_cli_file_parses_as_python(path: pathlib.Path):
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path), feature_version=MIN_MINOR)
    except SyntaxError as exc:
        pytest.fail(
            f"{path.name} does not parse as Python 3.{MIN_MINOR} "
            f"({exc.msg} at line {exc.lineno}). Captured output belongs in "
            f"docs/logs/ with a .txt extension; newer syntax needs the "
            f"declared minimum raised."
        )


def test_the_declared_minimum_is_what_the_guard_enforces():
    """The point of reading pyproject: parsing at the interpreter's own
    grammar accepts syntax the declared minimum cannot run, and pull-request
    CI only runs 3.12. ``except*`` is 3.11, so at 3.10 it must be rejected --
    while the interpreter running this suite parses it happily."""
    assert MIN_MINOR == 10
    except_star = "try:\n    pass\nexcept* ValueError:\n    pass\n"
    if sys.version_info < (3, 11):
        pytest.skip("this interpreter cannot parse the 3.11 sample either")
    ast.parse(except_star)
    with pytest.raises(SyntaxError):
        ast.parse(except_star, feature_version=MIN_MINOR)
