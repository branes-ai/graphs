# Package-RAPL amortization — why raw cache-sweep energy is not intrinsic energy

## Observation

A `cli/benchmark_cache_sweep.py` run on i7-12700K with package-RAPL
read access (`/sys/class/powercap/intel-rapl:0/energy_uj`) produces
energy/byte numbers in the 500–1800 pJ/B range across the working-set
sweep. Inspecting the relationship between bandwidth and energy/byte
at every plateau:

| working set | bw (GB/s) | energy (pJ/B) | bw × pJ/B (pW) |
|---|---|---|---|
| 402 KiB | 76.6 | 508  | **38,900** |
| 1.3 MiB | 38.2 | 963  | **36,800** |
| 6.1 MiB | 33.6 | 1148 | **38,500** |
| 64 MiB  | 20.0 | 1767 | **35,300** |

The product is roughly constant at **~36 W**. That number is the
i7-12700K package power floor — every bit of energy/byte we measured
is the *whole SoC's* idle/active power amortized over the bytes the
streaming kernel touches: 12 cores' worth of leakage, ring bus,
uncore, integrated GPU, IO ring. It is not the cache subsystem's
intrinsic per-byte transfer cost.

For reference, intrinsic L1 read energy on a 10 nm Alder Lake part
is expected to be in the **0.5–2 pJ/B** range — two to three orders
of magnitude below what package-RAPL reports through the current
kernel.

## Implication

`graphs.calibration.fitters.cache_sweep_fitter.CacheSweepFitter` as
shipped in PR #45 reads `WorkingSetPoint.energy_per_byte_pj` directly
and writes it into `HardwareResourceModel.{l1,l2,l3}_cache_energy_per_byte`
with `ConfidenceLevel.CALIBRATED` provenance. If the input is the raw
package-RAPL number, the output is a **calibrated-but-wrong**
coefficient, 100–1000× too high. The confidence ladder would
incorrectly promote a known-bad number from THEORETICAL to CALIBRATED.

This blocks the planned PR-2 (`cli/apply_calibration.py`) — applying
the current sweep results to the in-tree resource models would
poison the Layer 3/4/5 panels with measurements that look authoritative
and aren't.

## Mitigations

Two complementary fixes; either one is sufficient to unblock PR-2.

### Option A: baseline subtraction

Add a no-memory-traffic kernel (e.g., busy-loop on a register, or
`for _ in range(N): pass`) calibrated to the same wallclock duration
as each sweep point. Sample RAPL across the no-op kernel; that's
the package idle/baseline energy. Subtract from the streaming-kernel
measurement to get the marginal energy attributable to the cache
traffic, then divide by bytes streamed.

Pros: works on any RAPL-capable system. Self-contained in the
benchmark module. No new sysfs dependency.

Cons: the baseline kernel needs to do enough work to keep the CPU
out of deeper C-states, otherwise the residual is dominated by
package-state-transition noise.

### Option B: per-core (PP0) RAPL domain

Read `/sys/class/powercap/intel-rapl:0:0/energy_uj` (PP0 / "core"
domain) instead of the whole package. PP0 reports just the core
complex's energy, excluding uncore + iGPU + ring bus.

Pros: closer to intrinsic-energy semantics; no kernel change needed.

Cons: not exposed on every Intel SKU; AMD parts label the domain
differently or don't expose it at all; per-core RAPL is gated on
microcode and BIOS settings on some platforms.

The right move is probably both — PP0 when available (better
attribution), package + baseline subtraction otherwise (universal
fallback).

## Status

- PR #45 (Path B PR-1) ships the sweep + plateau detector + fitter.
  The infrastructure is correct; the issue is in *what we feed* the
  fitter.
- PR-2 (`cli/apply_calibration.py`) is **blocked** until a
  baseline-subtraction or PP0 path lands. Without that, applying the
  sweep would write wrong numbers to the model with high confidence.
- Tracking issues: see GitHub issues filed on the repo with the
  `calibration` label. The dependency chain is:

    [issue: baseline subtraction] -+-> [issue: apply_calibration]
    [issue: PP0 RAPL domain]      -+

  Either upstream issue unblocks the downstream one.

## What the current data IS useful for

Workload-level total-SoC energy modeling. The 36 W floor we
measured is real; multiplying it by wallclock for a given workload
gives a defensible "joules per inference" number for the i7-12700K
under load. That's a different (and useful) modeling target from
the per-cache-level intrinsic coefficients on `HardwareResourceModel`.

## References

- PR #45: introduces the sweep + fitter
- `docs/calibration/cache_sweep.md`: the sweep operator's guide
- `src/graphs/calibration/fitters/cache_sweep_fitter.py`: the fitter
  whose output semantics this document constrains
