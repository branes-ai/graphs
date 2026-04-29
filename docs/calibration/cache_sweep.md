# Cache-sweep microbenchmark — Path B PR-1

## What it measures

A working-set-size sweep that streams an in-place kernel over a
log-spaced range of buffer sizes (default 8 KiB → 64 MiB) and
records the bandwidth-vs-size curve. The plateaus identify the
L1, L2, L3, and DRAM regions; the per-byte energy at each plateau
calibrates the per-cache-level energy coefficients on
`HardwareResourceModel`.

## Output schema

```json
{
  "schema_version": "1.0",
  "tool": "cli/run_cache_sweep.py",
  "hardware_id": "intel_core_i7_12700k",
  "timestamp": "2026-04-29T...",
  "config": {
    "min_bytes": 8192,
    "max_bytes": 67108864,
    "num_points": 24,
    "target_seconds_per_point": 0.5,
    "repeats_per_point": 3,
    "capture_energy": true,
    "rapl_available": true
  },
  "points": [
    {
      "bytes_resident": 8192,
      "iterations": 12345,
      "elapsed_seconds": 0.512,
      "bandwidth_gbps": 200.0,
      "energy_joules": 0.024,
      "energy_per_byte_pj": 0.18,
      "extra": {...}
    }, ...
  ]
}
```

## Hardware requirements

| SKU | Energy capture | Notes |
|---|---|---|
| Intel x86 (Linux) | RAPL via `/sys/class/powercap/intel-rapl:0/energy_uj` | Requires read access; usually available to the user, sometimes root-only after kernel patches |
| AMD x86 (Linux) | RAPL via `intel-rapl:0` (despite the name) | Same access pattern |
| Jetson Orin AGX | Not yet wired (PR-3 will add `tegrastats` integration) | Sweep runs time-only |
| Other | None | `--no-energy` recommended; expect time-only output |

## Running

```bash
# Default 24-point sweep on the host CPU, ~12 s + warmup:
python cli/run_cache_sweep.py \
    --hardware intel_core_i7_12700k \
    --output sweeps/i7_12700k.json

# Tighter sweep for quick iteration:
python cli/run_cache_sweep.py \
    --hardware intel_core_i7_12700k \
    --output sweeps/i7_quick.json \
    --num-points 16 --seconds-per-point 0.3 --repeats 2

# Force time-only (no RAPL):
python cli/run_cache_sweep.py \
    --hardware intel_core_i7_12700k \
    --output sweeps/i7_time_only.json \
    --no-energy
```

## Interpreting the bandwidth curve

A clean run on a 12-core x86 CPU (i7-12700K) shows plateaus at
roughly:

```
working_set     bandwidth     plateau
8 - 32 KiB      ~rising       (call-overhead amortization region)
50 KiB - 1 MiB  ~75 GB/s      (L1 + L2 - kernel is bandwidth-bound)
2 - 11 MiB      ~33 GB/s      (L3)
20 - 64 MiB     ~20 GB/s      (DRAM)
```

The L1-vs-L2 transition is hard to see because the in-place XOR
kernel saturates both cache levels at roughly the same bandwidth
on modern x86 CPUs (the L2 prefetcher kicks in early). A future
PR may add a second kernel variant that's more sensitive to L1
boundaries; the L2-L3 and L3-DRAM transitions are clear enough
to back the calibration we want today.

## Applying the result

```python
import json
from graphs.benchmarks.cache_sweep import WorkingSetPoint
from graphs.calibration.fitters.cache_sweep_fitter import CacheSweepFitter
from graphs.hardware.models.edge.intel_core_i7_12700k import (
    intel_core_i7_12700k_resource_model,
)

data = json.load(open("sweeps/i7_12700k.json"))
points = [WorkingSetPoint(
    bytes_resident=p["bytes_resident"],
    iterations=p["iterations"],
    elapsed_seconds=p["elapsed_seconds"],
    bandwidth_gbps=p["bandwidth_gbps"],
    energy_joules=p["energy_joules"],
    energy_per_byte_pj=p["energy_per_byte_pj"],
) for p in data["points"]]

model = intel_core_i7_12700k_resource_model()
fitter = CacheSweepFitter()
fit = fitter.fit(points, sku_name="intel_core_i7_12700k")
fitter.apply_to_model(model, fit)

# model.field_provenance now carries CALIBRATED tags for the
# detected levels:
for k, v in model.field_provenance.items():
    if "cache_energy" in k:
        print(f"{k}: {v}")
```

## Future work

- **Kernel variants.** An additional pointer-chase kernel would
  isolate L1 latency more cleanly and would not be amortized away
  by the L2 streaming prefetcher.
- **Jetson energy.** PR-3 wires `tegrastats` for Orin AGX so the
  same sweep harness produces calibrated coefficients on Jetson.
- **Apply CLI.** A dedicated `cli/apply_calibration.py` runner that
  reads a sweep JSON, locates the matching SKU resource model, and
  persists the fit result back into the in-tree model files (or a
  side calibration cache).
- **Cross-validation.** Compare measured DRAM bandwidth against
  the M7 `peak_bandwidth` field; surface the delta in the Layer 7
  panel.
