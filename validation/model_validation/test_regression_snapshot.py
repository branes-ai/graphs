"""
Regression Snapshot Tests

Records model outputs for a reference workload set and fails when
any output diverges by more than 1% from the recorded baseline.
This prevents silent coefficient drift across PRs.

Workflow:
1. First run generates the snapshot file (if missing)
2. Subsequent runs compare against the snapshot
3. If a coefficient change is intentional, regenerate:
   python -m pytest validation/model_validation/test_regression_snapshot.py --snapshot-update

The snapshot file lives at validation/model_validation/snapshot.json
and is checked into source control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from graphs.hardware.resource_model import Precision

SNAPSHOT_PATH = Path(__file__).parent / "snapshot.json"
TOLERANCE = 0.01  # 1% relative tolerance

REFERENCE_SKUS = [
    "Intel-i7-12700K",
    "Intel-Xeon-Platinum-8490H",
    "H100-SXM5-80GB",
    "Jetson-Orin-AGX-64GB",
    "Google-TPU-v4",
    "Stillwater-KPU-T64",
    "Stillwater-KPU-T256",
    "Qualcomm-QRB5165",
]

WORKLOADS = {
    "tiny_matmul": {"ops": 2 * 64**3, "bytes": 3 * 64**2 * 4},
    "medium_matmul": {"ops": 2 * 1024**3, "bytes": 3 * 1024**2 * 4},
    "large_matmul": {"ops": 2 * 4096**3, "bytes": 3 * 4096**2 * 4},
    "small_conv": {"ops": 2 * 9 * 64 * 64 * 56 * 56, "bytes": 64 * 56 * 56 * 4 * 3},
    "bandwidth_only": {"ops": 0, "bytes": 4_000_000},
}

PRECISIONS = [Precision.FP32, Precision.INT8]


def _generate_snapshot() -> Dict:
    """Generate reference snapshot from current model outputs."""
    from graphs.hardware.mappers import get_mapper_by_name

    snapshot = {}
    for sku in REFERENCE_SKUS:
        try:
            mapper = get_mapper_by_name(sku)
        except Exception:
            continue
        if mapper is None:
            continue

        sku_data = {}
        for wl_name, wl in WORKLOADS.items():
            for prec in PRECISIONS:
                ops = wl["ops"]
                bytes_t = wl["bytes"]

                try:
                    compute_e, memory_e = mapper._calculate_energy(ops, bytes_t, prec)
                except Exception:
                    continue

                peak_ops = mapper.resource_model.get_peak_ops(prec)
                peak_bw = mapper.resource_model.peak_bandwidth
                compute_t = ops / peak_ops if peak_ops > 0 else 0
                memory_t = bytes_t / peak_bw if peak_bw > 0 else 0

                key = f"{wl_name}_{prec.value}"
                sku_data[key] = {
                    "compute_energy": compute_e,
                    "memory_energy": memory_e,
                    "compute_latency": compute_t,
                    "memory_latency": memory_t,
                }

        if sku_data:
            snapshot[sku] = sku_data

    return snapshot


def _load_snapshot() -> Dict:
    if not SNAPSHOT_PATH.exists():
        return {}
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)


def _save_snapshot(snapshot: Dict) -> None:
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)


def _check_relative(actual: float, expected: float, tol: float) -> bool:
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= tol


class TestRegressionSnapshot:
    """Compare current model outputs against recorded baseline."""

    @pytest.fixture(autouse=True)
    def _ensure_snapshot(self):
        """Generate snapshot if it doesn't exist."""
        if not SNAPSHOT_PATH.exists():
            snapshot = _generate_snapshot()
            _save_snapshot(snapshot)

    def test_energy_matches_snapshot(self):
        baseline = _load_snapshot()
        current = _generate_snapshot()
        failures = []

        for sku, sku_data in baseline.items():
            if sku not in current:
                continue
            for key, expected in sku_data.items():
                actual = current[sku].get(key)
                if actual is None:
                    continue
                for metric in ["compute_energy", "memory_energy"]:
                    exp_val = expected[metric]
                    act_val = actual[metric]
                    if not _check_relative(act_val, exp_val, TOLERANCE):
                        failures.append(
                            f"{sku}/{key}/{metric}: "
                            f"expected={exp_val:.6e}, got={act_val:.6e}, "
                            f"delta={abs(act_val-exp_val)/abs(exp_val)*100:.2f}%"
                        )

        if failures:
            msg = (
                f"{len(failures)} energy regression(s) detected "
                f"(tolerance={TOLERANCE*100:.0f}%):\n"
                + "\n".join(f"  {f}" for f in failures[:10])
            )
            if len(failures) > 10:
                msg += f"\n  ... and {len(failures)-10} more"
            msg += (
                "\n\nIf these changes are intentional, regenerate "
                "the snapshot:\n  python -c \"from validation.model_validation"
                ".test_regression_snapshot import _generate_snapshot, "
                "_save_snapshot; _save_snapshot(_generate_snapshot())\""
            )
            pytest.fail(msg)

    def test_latency_matches_snapshot(self):
        baseline = _load_snapshot()
        current = _generate_snapshot()
        failures = []

        for sku, sku_data in baseline.items():
            if sku not in current:
                continue
            for key, expected in sku_data.items():
                actual = current[sku].get(key)
                if actual is None:
                    continue
                for metric in ["compute_latency", "memory_latency"]:
                    exp_val = expected[metric]
                    act_val = actual[metric]
                    if not _check_relative(act_val, exp_val, TOLERANCE):
                        failures.append(
                            f"{sku}/{key}/{metric}: "
                            f"expected={exp_val:.6e}, got={act_val:.6e}"
                        )

        if failures:
            msg = (
                f"{len(failures)} latency regression(s) detected:\n"
                + "\n".join(f"  {f}" for f in failures[:10])
            )
            pytest.fail(msg)

    def test_snapshot_covers_all_reference_skus(self):
        baseline = _load_snapshot()
        missing = [sku for sku in REFERENCE_SKUS if sku not in baseline]
        assert not missing, f"Snapshot missing SKUs: {missing}"

    def test_snapshot_has_both_precisions(self):
        baseline = _load_snapshot()
        for sku, data in baseline.items():
            fp32_keys = [k for k in data if k.endswith("_fp32")]
            int8_keys = [k for k in data if k.endswith("_int8")]
            assert len(fp32_keys) > 0, f"{sku}: no FP32 entries"
            assert len(int8_keys) > 0, f"{sku}: no INT8 entries"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
