"""KPU catalog CLIs only operate on products that carry a KPU block.

The unified catalog (``load_compute_products_unified``) also holds CPU / GPU /
NPU / DSP / ... products. ``list_kpus`` used to build a row for every one of
them and crashed on ``CPUBlock.tiles``; ``show_floorplan --list`` listed every
product id. Both now filter with ``has_kpu_block`` (graphs#268 A2).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from graphs.hardware.compute_product_loader import load_compute_products_unified
from graphs.hardware.kpu_access import has_kpu_block

_CLI = Path(__file__).resolve().parents[2] / "cli"
_NON_KPU = "amd_epyc_9654_sp5"


def _kpu_ids() -> list[str]:
    return sorted(k for k, v in load_compute_products_unified().items() if has_kpu_block(v))


def test_catalog_mixes_kpu_and_non_kpu_products():
    products = load_compute_products_unified()
    assert _NON_KPU in products and not has_kpu_block(products[_NON_KPU])
    assert 0 < len(_kpu_ids()) < len(products)


def test_list_kpus_lists_exactly_the_kpu_products(cli_runner):
    rc, out, err = cli_runner(_CLI / "list_kpus.py", [])
    assert rc == 0, err
    kpu_ids = _kpu_ids()
    assert f"{len(kpu_ids)} KPU SKU(s)" in out
    assert all(k in out for k in kpu_ids)
    assert _NON_KPU not in out


def test_show_floorplan_list_is_kpu_only(cli_runner):
    rc, out, err = cli_runner(_CLI / "show_floorplan.py", ["--list"])
    assert rc == 0, err
    assert out.split() == _kpu_ids()


def test_show_kpu_rejects_non_kpu_product(cli_runner):
    rc, _, err = cli_runner(_CLI / "show_kpu.py", [_NON_KPU])
    assert rc == 1
    assert "no KPU SKU" in err
    assert _NON_KPU not in err.split("Available:")[1]


# ---------------------------------------------------------------------------
# A product with two KPU blocks gets a clear error, not a traceback
# ---------------------------------------------------------------------------

_DUP = "dup_kpu_two_blocks"


def _load_cli(name: str):
    spec = importlib.util.spec_from_file_location(f"_kpu_cli_{name}", _CLI / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def catalog_with_duplicate():
    """The real unified catalog plus one product carrying two KPU dies."""
    products = dict(load_compute_products_unified())
    base = products["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
    die = base.dies[0]
    products[_DUP] = base.model_copy(update={
        "id": _DUP,
        "dies": [die, die.model_copy(update={"die_id": "second"})],
    })
    return products


def _run(mod, monkeypatch, catalog, argv):
    monkeypatch.setattr(mod, "load_compute_products_unified", lambda: catalog)
    monkeypatch.setattr(sys, "argv", [mod.__file__, *argv])
    return mod.main()


def test_list_kpus_reports_duplicate_and_lists_the_rest(monkeypatch, capsys, catalog_with_duplicate):
    rc = _run(_load_cli("list_kpus"), monkeypatch, catalog_with_duplicate, [])
    out, err = capsys.readouterr()
    assert rc == 1
    assert f"invalid KPU SKU {_DUP!r}" in err and "2 KPUBlocks" in err
    assert f"{len(_kpu_ids())} KPU SKU(s)" in out  # the valid SKUs are still listed
    assert _DUP not in out


@pytest.mark.parametrize("cli,expected_rc", [("show_kpu", 1), ("show_floorplan", 2)])
def test_show_clis_reject_duplicate_cleanly(monkeypatch, capsys, catalog_with_duplicate, cli, expected_rc):
    rc = _run(_load_cli(cli), monkeypatch, catalog_with_duplicate, [_DUP])
    _, err = capsys.readouterr()
    assert rc == expected_rc
    assert f"invalid KPU SKU {_DUP!r}" in err and "2 KPUBlocks" in err


def test_show_compute_product_rejects_non_kpu_products(cli_runner):
    rc, out, err = cli_runner(_CLI / "show_compute_product.py", [_NON_KPU])
    assert rc == 1
    assert "without a KPU block" in err and "Traceback" not in err
    rc, _, err = cli_runner(_CLI / "show_compute_product.py", ["no_such_sku"])
    assert rc == 1 and "no KPU SKU" in err and _NON_KPU not in err
