"""Top-level pytest fixtures for the test suite.

The main thing this provides is ``cli_runner`` -- an in-process
replacement for the per-file ``subprocess.run([python, script, ...])``
helpers that were spending ~2 seconds per CLI test on torch import.

Importing each CLI script as a module exactly once means torch is
imported once for the whole suite (instead of once per test), and the
imported CLI modules stay cached in ``sys.modules`` between tests.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import pytest


# Cache of file_path -> imported module. Populated lazily on first
# call to cli_runner(script, ...). Surviving for the whole pytest
# session means torch + heavy deps load once across all CLI tests.
_CLI_MODULE_CACHE: dict[str, object] = {}


def _import_cli(script_path: Path):
    """Import a CLI script as a Python module (cached).

    The script's ``if __name__ == '__main__':`` guard prevents the
    import from running ``main()`` -- we only want the function
    definitions in scope so we can call ``main()`` directly with our
    own argv.
    """
    key = str(script_path.resolve())
    cached = _CLI_MODULE_CACHE.get(key)
    if cached is not None:
        return cached

    # Synthesize a unique module name. Real names like "analyze_batch"
    # would collide if the same module name was imported from a
    # different path elsewhere; the prefix avoids that.
    mod_name = f"_clitest_{script_path.stem}_{abs(hash(key)) & 0xffff:04x}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path} as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _CLI_MODULE_CACHE[key] = module
    return module


def _run_cli_inprocess(
    script: Path | str,
    args: list[str],
    timeout: float | None = None,  # accepted for signature compat; ignored
) -> tuple[int, str, str]:
    """In-process CLI invocation. Drop-in replacement for the existing
    ``subprocess.run(["python", script, *args])`` helpers.

    Returns ``(returncode, stdout, stderr)`` to match the subprocess
    helper. The returncode comes from ``main()``'s return value, or
    from a ``SystemExit`` if the CLI calls ``sys.exit(N)`` directly.

    The ``timeout`` parameter is accepted for signature compatibility
    but ignored -- in-process calls don't need a wall-clock guard
    (a hung test will fail under pytest's own collection timeout).
    """
    script_path = Path(script)
    module = _import_cli(script_path)
    if not hasattr(module, "main"):
        raise AttributeError(
            f"{script_path} has no main() function -- in-process runner needs one"
        )

    saved_argv = sys.argv
    sys.argv = [str(script_path), *args]
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            try:
                rc = module.main()
            except SystemExit as e:
                # Some CLIs call sys.exit(N) instead of returning N.
                # argparse error-paths also raise SystemExit(2).
                rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = saved_argv

    if rc is None:
        rc = 0
    return rc, out_buf.getvalue(), err_buf.getvalue()


@pytest.fixture
def cli_runner() -> Callable[..., tuple[int, str, str]]:
    """Pytest fixture exposing the in-process CLI runner.

    Usage in a test:

        def test_something(cli_runner):
            rc, stdout, stderr = cli_runner(SCRIPT_PATH, ["--flag", "v"])

    Drop-in replacement for the per-file ``run_cli`` helpers that
    previously spawned a subprocess. About ~2 seconds faster per call
    once torch is loaded into the cached module.
    """
    return _run_cli_inprocess
