"""Parity test: YAML-loaded Synopsys ARC EV7x matches the hand-coded factory.

PR 4-equivalent of the v9 DSP follow-up batch (graphs#223). Proves
``dsp_yaml_loader.load_dsp_resource_model_from_yaml`` produces a
``HardwareResourceModel`` whose key fields match the existing hand-
coded ``synopsys_arc_ev7x_resource_model()`` factory.

Once the hand-coded body is retired (in the same PR as this test
lands), both ``hand_coded`` and ``yaml_loaded`` fixtures resolve
through the same loader; the structural assertions become contract
tests on the loader's output.

**3 documented drifts** -- the YAML loader picks fabric[0] for
chip-level derived fields, but the hand-coded factory picked the
"primary" DNN fabric (fabric[1] in the YAML's ordering):

  - ``compute_units``: hand=128 (DNN units), yaml=4 (VPU units, fabric[0])
  - ``energy_per_flop_fp32``: hand=2.295 pJ (tensor_core baseline,
    from DNN), yaml=2.43 pJ (simd_packed baseline, from VPU fabric[0])
  - ``precision_profiles``: yaml includes INT32 (from VPU fabric),
    hand-coded omits it

Resolution candidates (v10+ follow-ups):
  - Reorder YAML fabrics so DNN comes first (atomically with a YAML
    bump + parity-test refresh)
  - Change loader convention to pick the highest-num_units fabric
    as the "primary"

For now: documented drifts; the cleanup wrapper relies on the loader
output as-is.

Same shape as ``tests/hardware/test_dsp_yaml_loader_cadence_q8_parity.py``.
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.models.ip_cores.synopsys_arc_ev7x import (
    synopsys_arc_ev7x_resource_model,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "synopsys_arc_ev7x"
LEGACY_NAME = "Synopsys-ARC-EV7x-4core"


@pytest.fixture(scope="module")
def hand_coded():
    return synopsys_arc_ev7x_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_dsp_resource_model_from_yaml(
        SKU_ID,
        name_override=LEGACY_NAME,
        fabric_type_overrides={
            DSPFabricKind.VECTOR_SIMD: "ev7x_vpu",
            DSPFabricKind.TENSOR_MATRIX: "ev7x_dnn_accelerator",
        },
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == LEGACY_NAME


def test_hardware_type_is_dsp(hand_coded, yaml_loaded):
    assert yaml_loaded.hardware_type == hand_coded.hardware_type == HardwareType.DSP


# ---------------------------------------------------------------------------
# Compute fabrics (multi-fabric: VPU + DNN)
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    """Synopsys EV7x ships 2 fabrics: VPU (vector) + DNN (tensor)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_vpu_fabric_present(hand_coded, yaml_loaded):
    """Both factories carry an 'ev7x_vpu' fabric with 4 units."""
    yl_types = {f.fabric_type for f in yaml_loaded.compute_fabrics}
    hc_types = {f.fabric_type for f in hand_coded.compute_fabrics}
    assert "ev7x_vpu" in yl_types
    assert "ev7x_vpu" in hc_types
    # 4 VPU cores
    yl_vpu = next(f for f in yaml_loaded.compute_fabrics if f.fabric_type == "ev7x_vpu")
    hc_vpu = next(f for f in hand_coded.compute_fabrics if f.fabric_type == "ev7x_vpu")
    assert yl_vpu.num_units == hc_vpu.num_units == 4


def test_dnn_fabric_present(hand_coded, yaml_loaded):
    """Both factories carry an 'ev7x_dnn_accelerator' fabric with 128 units."""
    yl_types = {f.fabric_type for f in yaml_loaded.compute_fabrics}
    hc_types = {f.fabric_type for f in hand_coded.compute_fabrics}
    assert "ev7x_dnn_accelerator" in yl_types
    assert "ev7x_dnn_accelerator" in hc_types
    yl_dnn = next(f for f in yaml_loaded.compute_fabrics if f.fabric_type == "ev7x_dnn_accelerator")
    hc_dnn = next(f for f in hand_coded.compute_fabrics if f.fabric_type == "ev7x_dnn_accelerator")
    assert yl_dnn.num_units == hc_dnn.num_units == 128


def test_dnn_fabric_int8_ops(hand_coded, yaml_loaded):
    """DNN INT8 ops/cycle/unit = 273 (yields ~35 TOPS @ 1 GHz)."""
    yl_dnn = next(f for f in yaml_loaded.compute_fabrics if f.fabric_type == "ev7x_dnn_accelerator")
    hc_dnn = next(f for f in hand_coded.compute_fabrics if f.fabric_type == "ev7x_dnn_accelerator")
    assert yl_dnn.ops_per_unit_per_clock[Precision.INT8] == \
        hc_dnn.ops_per_unit_per_clock[Precision.INT8] == 273


def test_vpu_fabric_fp32_ops(hand_coded, yaml_loaded):
    """VPU FP32 ops/cycle/unit = 2200 (yields ~8.8 GFLOPS for 4 cores @ 1 GHz)."""
    yl_vpu = next(f for f in yaml_loaded.compute_fabrics if f.fabric_type == "ev7x_vpu")
    hc_vpu = next(f for f in hand_coded.compute_fabrics if f.fabric_type == "ev7x_vpu")
    assert yl_vpu.ops_per_unit_per_clock[Precision.FP32] == \
        hc_vpu.ops_per_unit_per_clock[Precision.FP32] == 2200


# ---------------------------------------------------------------------------
# Precision profiles
# ---------------------------------------------------------------------------

def test_precision_profile_keys_match(hand_coded, yaml_loaded):
    """Post-cleanup: both fixtures resolve through the same loader, so
    the precision_profiles set is identical (includes INT32 from VPU)."""
    assert set(hand_coded.precision_profiles) == set(yaml_loaded.precision_profiles)
    assert Precision.INT32 in yaml_loaded.precision_profiles


def test_int8_peak_matches_marketed(hand_coded, yaml_loaded):
    """35 TOPS INT8 (Synopsys-marketed)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(35e12, rel=0.01)


def test_fp32_peak_matches_marketed(hand_coded, yaml_loaded):
    """8.8 GFLOPS FP32 (4 VPU cores * 2.2 GFLOPS)."""
    hc_peak = hand_coded.precision_profiles[Precision.FP32].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.FP32].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(8.8e9, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory subsystem
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    """60 GB/s typical-integration LPDDR4 bandwidth (automotive)."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(60e9)


def test_l1_cache_per_unit_matches(hand_coded, yaml_loaded):
    """32 KiB per unit."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 32 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """4 MiB shared L2."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 4 * 1024 * 1024


def test_main_memory_matches(hand_coded, yaml_loaded):
    """8 GiB typical automotive pairing."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 8 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes
# ---------------------------------------------------------------------------

def test_min_occupancy_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.70


def test_max_concurrent_kernels_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 8


def test_wave_quantization_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 4


# ---------------------------------------------------------------------------
# Documented drifts (3): compute_units, energy_per_flop_fp32, precision_profiles
# ---------------------------------------------------------------------------

def test_compute_units_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup the hand-coded factory hard-coded ``compute_units=128``
    (the DNN's num_units). Post-cleanup it resolves through the loader,
    which picks ``compute_units=4`` (fabric[0]=VPU's num_units, per the
    pick-first convention). Both fixtures now agree on 4. The DNN
    fabric's num_units stays 128 inside compute_fabrics; consumers
    needing the primary INT8 path iterate compute_fabrics."""
    assert hand_coded.compute_units == yaml_loaded.compute_units == 4
    yl_dnn = next(f for f in yaml_loaded.compute_fabrics if f.fabric_type == "ev7x_dnn_accelerator")
    assert yl_dnn.num_units == 128


def test_energy_per_flop_fp32_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Same root cause as compute_units (loader picks fabric[0]'s
    energy baseline). Post-cleanup both resolve through the loader."""
    assert hand_coded.energy_per_flop_fp32 == yaml_loaded.energy_per_flop_fp32
    assert hand_coded.energy_per_flop_fp32 == pytest.approx(2.43e-12, rel=0.01)


# ---------------------------------------------------------------------------
# Thermal operating points
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    assert set(yaml_loaded.thermal_operating_points) == set(hand_coded.thermal_operating_points)


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "5W"


def test_thermal_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["5W"]
    hc = hand_coded.thermal_operating_points["5W"]
    assert yl.tdp_watts == hc.tdp_watts == 5.0
