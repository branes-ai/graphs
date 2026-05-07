"""Per-shape predict + measure + assert orchestration.

Implements the V4-3 dev loop:

    for each shape in the sweep:
        predict (closed-form via op_footprint -> SubgraphDescriptor -> RooflineAnalyzer)
        measure (lookup cache; optionally call measurer to refresh)
        assert  (assertions.assert_record)
        emit    (ValidationRecord -> caller, report.py aggregates)

Design choices:

- The runner builds a SubgraphDescriptor by hand from the closed-form
  op_footprint rather than going through frontend.trace_and_partition.
  Two reasons:
    1. Single-input vs multi-input traceability inconsistency between
       linear (1 input) and matmul (2 inputs).
    2. Even after #64, isolating the prediction from the partitioner
       means a future partitioner regression doesn't masquerade as a
       harness bug.
  The roofline analyzer itself (the byte-traffic + bandwidth math) is
  reused exactly as the production analyzer uses it.

- Cache lookup is the default. If a measurement is missing and no
  measurer is provided (or refresh_measurements=False), the shape is
  recorded as "no baseline" -- the harness never silently moves the
  validation reference.

- Energy is best-effort: if the analyzer or the measurer can't produce
  a number, the assertion module treats that record as not-failing on
  energy (per the all_pass() rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from graphs.core.structures import (
    OperationType,
    SubgraphDescriptor,
    create_tensor_descriptor,
)
from graphs.estimation.energy import EnergyAnalyzer
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers import get_mapper_by_name
from graphs.hardware.resource_model import HardwareResourceModel, Precision

from validation.model_v4.ground_truth.base import Measurer
from validation.model_v4.ground_truth.cache import (
    CacheKey,
    DEFAULT_BASELINE_DIR,
    load_baseline,
    store,
)
from validation.model_v4.harness.assertions import (
    MeasurementContext,
    ValidationRecord,
    assert_record,
)
from validation.model_v4.sweeps.classify import (
    Regime,
    bytes_per_element,
    op_footprint,
)
from validation.model_v4.workloads.linear import build_linear
from validation.model_v4.workloads.matmul import build_matmul
from validation.model_v4.workloads.vector_add import build_vector_add


# Sweep-JSON keys -> mapper-registry keys. Sweep JSONs use a normalized
# form (lowercase, underscores) so they're stable across registry naming
# changes; the runner translates here. Must stay in sync with
# KNOWN_TARGETS in validation/model_v4/sweeps/_augment.py.
SWEEP_HW_TO_MAPPER: dict[str, str] = {
    "i7_12700k": "Intel-i7-12700K",
    "h100_sxm5_80gb": "H100-SXM5-80GB",
    "jetson_orin_nano_8gb": "Jetson-Orin-Nano-8GB",
    "jetson_orin_agx_64gb": "Jetson-Orin-AGX-64GB",
    "jetson_orin_nx_16gb": "Jetson-Orin-NX-16GB",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    sweep_path: Path
    hardware_key: str  # sweep-JSON key, e.g. "i7_12700k"
    measurer: Optional[Measurer] = None
    refresh_measurements: bool = False  # True: capture missing entries via measurer
    baseline_dir: Path = field(default=DEFAULT_BASELINE_DIR)
    launch_overhead_s: float = 5e-6
    # V5-4 opt-in: when True, the runner constructs RooflineAnalyzer
    # with use_tier_aware_memory=True so each ValidationRecord is
    # annotated with the binding tier. Default False keeps V4 floors
    # byte-identical to pre-V5-3b until V5-5 calibrates per-tier
    # achievable_fraction.
    use_tier_aware_memory: bool = False


@dataclass
class RunnerResult:
    records: List[ValidationRecord]
    skipped_no_baseline: List[Tuple[str, tuple, str]]  # (op, shape, dtype)
    skipped_unsupported: List[Tuple[str, tuple, str]]


def run_sweep(config: RunnerConfig) -> RunnerResult:
    """Walk the sweep, produce ValidationRecords (or note skips)."""
    sweep = json.loads(Path(config.sweep_path).read_text())
    op = sweep["op"]

    # Resolve hardware
    if config.hardware_key not in SWEEP_HW_TO_MAPPER:
        raise ValueError(
            f"Hardware key {config.hardware_key!r} has no mapper-registry "
            f"translation. Known: {sorted(SWEEP_HW_TO_MAPPER)}"
        )
    mapper = get_mapper_by_name(SWEEP_HW_TO_MAPPER[config.hardware_key])
    hw = mapper.resource_model

    # Pre-load the baseline cache once (one CSV read per (hw, op))
    cache = load_baseline(config.hardware_key, op, baseline_dir=config.baseline_dir)

    records: List[ValidationRecord] = []
    skipped_missing: List[Tuple[str, tuple, str]] = []
    skipped_unsupp: List[Tuple[str, tuple, str]] = []

    for entry in sweep["shapes"]:
        shape = tuple(entry["shape"])
        dtype = entry["dtype"]
        regime_str = entry["regime_per_hw"].get(config.hardware_key)
        if regime_str is None:
            continue
        regime_pred = Regime(regime_str)
        if regime_pred in (Regime.UNSUPPORTED, Regime.AMBIGUOUS):
            skipped_unsupp.append((op, shape, dtype))
            continue

        # Build the SubgraphDescriptor by hand from the closed-form footprint
        sg = _build_subgraph(op, shape, dtype)
        precision = _resolve_precision(dtype)

        latency_pred_s, binding_tier = _predict_latency_s(
            sg,
            hw,
            precision,
            use_tier_aware_memory=config.use_tier_aware_memory,
        )
        energy_pred_j = _predict_energy_j(sg, hw, precision, latency_pred_s)

        # Lookup or capture the measurement
        key = CacheKey(config.hardware_key, op, shape, dtype)
        meas = cache.get(key)
        if meas is None and config.refresh_measurements and config.measurer is not None:
            workload = _build_workload(op, shape, dtype)
            meas = config.measurer.measure(workload.model, workload.inputs)
            store(key, meas, baseline_dir=config.baseline_dir)
            cache[key] = meas
        if meas is None:
            skipped_missing.append((op, shape, dtype))
            continue

        # Build the validation record
        ctx = MeasurementContext(
            peak_flops=hw.get_peak_ops(precision),
            peak_dram_bandwidth_bps=hw.peak_bandwidth,
            launch_overhead_s=config.launch_overhead_s,
        )
        fp = op_footprint(op, shape, dtype)
        rec = assert_record(
            hardware=config.hardware_key,
            op=op,
            shape=shape,
            dtype=dtype,
            regime_predicted=regime_pred,
            latency_predicted_s=latency_pred_s,
            energy_predicted_j=energy_pred_j,
            flops=fp.flops,
            working_set_bytes=fp.working_set_bytes,
            measured_latency_s=meas.latency_s,
            measured_energy_j=meas.energy_j,
            ctx=ctx,
            binding_tier=binding_tier,
        )
        records.append(rec)

    return RunnerResult(
        records=records,
        skipped_no_baseline=skipped_missing,
        skipped_unsupported=skipped_unsupp,
    )


# ---------------------------------------------------------------------------
# Internals: prediction
# ---------------------------------------------------------------------------


def _resolve_precision(dtype: str) -> Precision:
    table = {
        "fp64": Precision.FP64,
        "fp32": Precision.FP32,
        "tf32": Precision.TF32,
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
        "fp8": Precision.FP8,
        "fp8_e4m3": Precision.FP8_E4M3,
        "fp8_e5m2": Precision.FP8_E5M2,
        "int64": Precision.INT64,
        "int32": Precision.INT32,
        "int16": Precision.INT16,
        "int8": Precision.INT8,
        "int4": Precision.INT4,
        "fp4": Precision.FP4,
    }
    if dtype.lower() not in table:
        raise ValueError(f"Unknown dtype {dtype!r}")
    return table[dtype.lower()]


def _build_subgraph(op: str, shape: tuple, dtype: str) -> SubgraphDescriptor:
    """Hand-construct a 1-op SubgraphDescriptor matching the workload.

    Bypasses the partitioner -- the v4 harness owns the closed-form
    footprint formula in classify.op_footprint, so a partitioner
    regression cannot silently shift the harness baseline.

    V5-4: also populates ``input_tensors`` / ``weight_tensors`` /
    ``output_tensors`` with the canonical shapes per op kind so the
    V5-3b roofline tier-aware path (when opted in) can extract
    (M, K, N) / (B, IN, OUT) / (N,) from the subgraph. Without these
    fields the V5-3b shape extractors return None and the analyzer
    silently falls back to the scalar path -- which would defeat
    V5-4's whole point of surfacing binding tiers in V4 reports.
    """
    fp = op_footprint(op, shape, dtype)
    bpe = bytes_per_element(dtype)
    # Use a dtype string that TensorDescriptor / RooflineAnalyzer
    # normalize_dtype both speak. The V4 dtype shortform ('fp32') is
    # passed through normalize_dtype since 'fp32' is already a valid
    # bytes_per_element key.
    td_dtype = dtype

    if op == "matmul":
        M, K, N = shape
        # Both args are activations; no parameter weight.
        input_bytes = int(round((M * K + K * N) * bpe))
        output_bytes = int(round(M * N * bpe))
        weight_bytes = 0
        op_type = OperationType.MATMUL
        input_tensors = [
            create_tensor_descriptor((M, K), td_dtype),
            create_tensor_descriptor((K, N), td_dtype),
        ]
        output_tensors = [create_tensor_descriptor((M, N), td_dtype)]
        weight_tensors: list = []
    elif op == "linear":
        B, IN, OUT = shape
        input_bytes = int(round(B * IN * bpe))
        output_bytes = int(round(B * OUT * bpe))
        # Weight matrix + bias (mirrors what build_linear() actually allocates)
        weight_bytes = int(round((IN * OUT + OUT) * bpe))
        op_type = OperationType.LINEAR
        input_tensors = [create_tensor_descriptor((B, IN), td_dtype)]
        output_tensors = [create_tensor_descriptor((B, OUT), td_dtype)]
        weight_tensors = [create_tensor_descriptor((OUT, IN), td_dtype)]
    elif op == "vector_add":
        # c[i] = a[i] + b[i]. Two N-element inputs, one N-element output,
        # no weights. The zero-reuse op the V5 plan uses for tier-BW
        # microbenchmarks.
        (N_elems,) = shape
        input_bytes = int(round(2 * N_elems * bpe))
        output_bytes = int(round(N_elems * bpe))
        weight_bytes = 0
        op_type = OperationType.ELEMENTWISE
        input_tensors = [
            create_tensor_descriptor((N_elems,), td_dtype),
            create_tensor_descriptor((N_elems,), td_dtype),
        ]
        output_tensors = [create_tensor_descriptor((N_elems,), td_dtype)]
        weight_tensors = []
    else:
        raise ValueError(f"Unsupported op {op!r} for v4 runner")

    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["v4_workload"],
        node_names=[f"{op}_{shape}_{dtype}"],
        operation_types=[op_type],
        fusion_pattern=op,
        total_flops=fp.flops,
        total_macs=fp.flops // 2,
        total_input_bytes=input_bytes,
        total_output_bytes=output_bytes,
        total_weight_bytes=weight_bytes,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        weight_tensors=weight_tensors,
    )


def _predict_latency_s(
    sg: SubgraphDescriptor,
    hw: HardwareResourceModel,
    precision: Precision,
    *,
    use_tier_aware_memory: bool = False,
) -> Tuple[float, Optional[str]]:
    """Run the subgraph through RooflineAnalyzer (the same code path
    UnifiedAnalyzer uses) and return ``(latency_s, binding_tier_name)``.

    ``binding_tier_name`` is populated only when ``use_tier_aware_memory``
    is True AND the V5-3b eligibility predicate passes (single-op
    MATMUL/LINEAR with a clean 2D shape on a >=2-tier hierarchy).
    Otherwise it's None and callers should treat that as "the tier-aware
    path didn't fire" rather than an error.
    """
    analyzer = RooflineAnalyzer(
        hw, precision=precision, use_tier_aware_memory=use_tier_aware_memory
    )
    lat = analyzer._analyze_subgraph(sg)
    binding_tier = (
        lat.memory_explanation.binding_tier_name
        if lat.memory_explanation is not None
        else None
    )
    return float(lat.actual_latency), binding_tier


def _predict_energy_j(
    sg: SubgraphDescriptor,
    hw: HardwareResourceModel,
    precision: Precision,
    latency_s: float,
) -> Optional[float]:
    """Predict energy via EnergyAnalyzer. Returns None if the analyzer
    raises (e.g., precision unsupported by the energy model)."""
    try:
        analyzer = EnergyAnalyzer(hw, precision=precision)
        report = analyzer.analyze(subgraphs=[sg], latencies=[latency_s])
        return float(
            report.compute_energy_j + report.memory_energy_j + report.static_energy_j
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internals: workload construction (only when refreshing measurements)
# ---------------------------------------------------------------------------


def _build_workload(op: str, shape: tuple, dtype: str):
    if op == "matmul":
        M, K, N = shape
        return build_matmul(M, K, N, dtype)
    if op == "linear":
        B, IN, OUT = shape
        return build_linear(B, IN, OUT, dtype)
    if op == "vector_add":
        (N,) = shape
        return build_vector_add(N, dtype)
    raise ValueError(f"Unsupported op {op!r}")
