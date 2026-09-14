"""Tests for graphs.hardware.kpu_access (graphs#268 A2).

KPU code used to assume the KPU block is ``cp.dies[0].blocks[0]``. The
lookup helpers find it by kind. These tests pin:

  * the lookup contract (one KPU block -> found; none / several -> error),
  * the domain error each caller raises (SiliconMathError, GeneratorError,
    ContextError, KPUYamlLoaderError),
  * positional independence end to end: a catalog SKU re-shaped so its KPU
    sits on the second die, behind a CPU die, and after a CPU block on its
    own die, produces the same silicon math, input spec, floorplan,
    resource model and validator findings as the original.
"""

from __future__ import annotations

import pytest

from embodied_schemas import KPUBlock

from graphs.hardware import kpu_golden as kg
from graphs.hardware.kpu_access import (
    KPUBlockLookupError,
    has_kpu_block,
    kpu_block_of,
    kpu_die_of,
)

KPU_SKU = "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"
CPU_SKU = "amd_epyc_9654_sp5"  # two dies: a CPU compute die + an IO die


@pytest.fixture(scope="module")
def catalogs():
    return kg.load_catalogs()


@pytest.fixture(scope="module")
def kpu_cp(catalogs):
    return catalogs["kpus"][KPU_SKU]


@pytest.fixture(scope="module")
def cpu_cp(catalogs):
    return catalogs["kpus"][CPU_SKU]


@pytest.fixture(scope="module")
def reshaped(kpu_cp, cpu_cp):
    """KPU_SKU with the KPU die moved to position 1 (after the CPU die) and a
    CPU block placed before the KPU block on the KPU die."""
    cpu_die = cpu_cp.dies[0]
    cpu_block = cpu_die.blocks[0]
    kpu_die = kpu_cp.dies[0]
    kpu_die_mixed = kpu_die.model_copy(update={"blocks": [cpu_block, *kpu_die.blocks]})
    return kpu_cp.model_copy(update={"dies": [cpu_die, kpu_die_mixed]})


# ---------------------------------------------------------------------------
# Lookup contract
# ---------------------------------------------------------------------------

def test_monolithic_kpu_is_found(kpu_cp):
    assert kpu_die_of(kpu_cp) is kpu_cp.dies[0]
    assert kpu_block_of(kpu_cp) is kpu_cp.dies[0].blocks[0]
    assert has_kpu_block(kpu_cp)


def test_kpu_found_by_kind_not_position(reshaped, kpu_cp):
    assert not isinstance(reshaped.dies[0].blocks[0], KPUBlock)
    assert not isinstance(reshaped.dies[1].blocks[0], KPUBlock)
    assert kpu_die_of(reshaped).die_id == kpu_cp.dies[0].die_id
    assert kpu_block_of(reshaped) is kpu_cp.dies[0].blocks[0]


def test_product_without_kpu_raises(cpu_cp):
    assert not has_kpu_block(cpu_cp)
    with pytest.raises(KPUBlockLookupError, match="has no KPUBlock") as exc:
        kpu_block_of(cpu_cp)
    # The message names each die's block kinds, to aid debugging.
    assert "cpu" in str(exc.value) and "io" in str(exc.value)
    with pytest.raises(KPUBlockLookupError):
        kpu_die_of(cpu_cp)


def test_multiple_kpu_blocks_are_ambiguous(kpu_cp):
    die = kpu_cp.dies[0]
    twice = kpu_cp.model_copy(update={"dies": [die, die.model_copy(update={"die_id": "second"})]})
    assert has_kpu_block(twice)
    with pytest.raises(KPUBlockLookupError, match="2 KPUBlocks") as exc:
        kpu_block_of(twice)
    # Like the no-KPU error, the message names each die's block kinds.
    assert f"{die.die_id}: [kpu]" in str(exc.value)
    assert "second: [kpu]" in str(exc.value)


def test_model_construct_without_dies_raises_lookup_error(kpu_cp):
    empty = kpu_cp.model_copy(update={"dies": []})
    assert not has_kpu_block(empty)
    with pytest.raises(KPUBlockLookupError, match="no dies"):
        kpu_die_of(empty)


# ---------------------------------------------------------------------------
# Each caller keeps its own error type
# ---------------------------------------------------------------------------

def test_callers_raise_their_domain_errors(cpu_cp, catalogs):
    from graphs.hardware.kpu_sku_generator import GeneratorError, input_spec_from_compute_product
    from graphs.hardware.models.accelerators.kpu_yaml_loader import (
        KPUYamlLoaderError,
        load_kpu_resource_model_from_yaml,
    )
    from graphs.hardware.sku_validators import build_context_for_kpu
    from graphs.hardware.sku_validators.context import ContextError
    from graphs.hardware.sku_validators.silicon_math import SiliconMathError, total_pe_count

    with pytest.raises(SiliconMathError, match="has no KPUBlock"):
        total_pe_count(cpu_cp)
    with pytest.raises(GeneratorError, match="has no KPUBlock"):
        input_spec_from_compute_product(cpu_cp)
    with pytest.raises(ContextError, match="has no KPUBlock"):
        build_context_for_kpu(
            CPU_SKU,
            kpus=catalogs["kpus"],
            process_nodes=catalogs["process_nodes"],
            cooling_solutions=catalogs["cooling_solutions"],
        )
    with pytest.raises(KPUYamlLoaderError, match="has no KPUBlock"):
        load_kpu_resource_model_from_yaml(
            CPU_SKU, kpus=catalogs["kpus"], process_nodes=catalogs["process_nodes"]
        )


# ---------------------------------------------------------------------------
# Positional independence, end to end
# ---------------------------------------------------------------------------

def test_silicon_math_is_position_independent(kpu_cp, reshaped, catalogs):
    from graphs.hardware.sku_validators import silicon_math as sm

    node = catalogs["process_nodes"][kpu_cp.dies[0].process_node_id]
    assert sm.total_pe_count(reshaped) == sm.total_pe_count(kpu_cp)
    assert sm.resolve_all_block_areas(reshaped, node) == sm.resolve_all_block_areas(kpu_cp, node)
    assert sm.total_chip_leakage_w(reshaped, node) == sm.total_chip_leakage_w(kpu_cp, node)


def test_input_spec_is_position_independent(kpu_cp, reshaped):
    from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product

    assert input_spec_from_compute_product(reshaped) == input_spec_from_compute_product(kpu_cp)


def test_floorplans_are_position_independent(kpu_cp, reshaped, catalogs):
    from graphs.hardware.silicon_floorplan import (
        derive_kpu_architectural_floorplan,
        derive_kpu_floorplan,
    )

    node = catalogs["process_nodes"][kpu_cp.dies[0].process_node_id]
    assert derive_kpu_floorplan(reshaped, node) == derive_kpu_floorplan(kpu_cp, node)
    assert derive_kpu_architectural_floorplan(reshaped, node) == (
        derive_kpu_architectural_floorplan(kpu_cp, node)
    )


def test_resource_model_is_position_independent(kpu_cp, reshaped, catalogs):
    from graphs.hardware.models.accelerators.kpu_yaml_loader import (
        load_kpu_resource_model_from_yaml,
    )

    def load(cp):
        return load_kpu_resource_model_from_yaml(
            KPU_SKU, kpus={KPU_SKU: cp}, process_nodes=catalogs["process_nodes"]
        )

    assert kg.to_jsonable(load(reshaped)) == kg.to_jsonable(load(kpu_cp))


def test_validator_findings_are_position_independent(kpu_cp, reshaped, catalogs):
    from graphs.hardware.sku_validators import (
        build_context_for_kpu,
        default_registry,
        load_validators,
    )

    load_validators()

    def findings(cp):
        ctx = build_context_for_kpu(
            KPU_SKU,
            kpus={KPU_SKU: cp},
            process_nodes=catalogs["process_nodes"],
            cooling_solutions=catalogs["cooling_solutions"],
        )
        return sorted(f.render_one_line() for f in default_registry.run_all(ctx))

    assert findings(reshaped) == findings(kpu_cp)
