"""KPU catalog CLIs only operate on products that carry a KPU block.

The unified catalog (``load_compute_products_unified``) also holds CPU / GPU /
NPU / DSP / ... products. ``list_kpus`` used to build a row for every one of
them and crashed on ``CPUBlock.tiles``; ``show_floorplan --list`` listed every
product id. Both now filter with ``has_kpu_block`` (graphs#268 A2).
"""

from __future__ import annotations

from pathlib import Path

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
