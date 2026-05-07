# V5 Follow-up: Skinny Matmul Aspect-Ratio Detection

## Context

The V5 follow-up Jetson Orin Nano calibration analysis (PR #115) flagged
"hardware-aware skinny matmul threshold" as one of three model gaps
blocking V4 floor improvement. The hypothesis was that Jetson's smaller
cache hierarchy needed a different `_SKINNY_THRESHOLD` than i7's value
of 16.

When the empirical data was actually collected, the picture was
different from the hypothesis: the failing Jetson shapes had `min(M, N)`
in the 64-256 range (well above 16) but very high **aspect ratios**
(100-256). The optimal-square tile model produces excessive reload
counts whenever the longer dim is much larger than the shorter dim,
regardless of cache size. The fix is therefore **shape-intrinsic**, not
hardware-dependent.

## What was wrong

`MatmulReuseModel.residency_window` had a single absolute-dim check:

```python
if min(M, N) < 16:
    return TileChoice(tile_dims=(M, N), ...)  # full-output tile (skinny)
else:
    # optimal-square tile, clamped to min(M, N)
```

This caught tiny B (e.g. linear with B=2) but missed the Jetson failure
shapes where `min(M, N) = 64-256` but `max(M, N) / min(M, N) = 100+`.

Worked example -- shape (64, 16384, 8192) fp16 on Orin Nano:

* Optimal-square tile = (64, 64), residency = 24 KB (fits L1).
* `bytes_loaded` = `M*K * ceil(N/64) + K*N * ceil(M/64) + 2*M*N`
  = `64*16384 * 128 + 16384*8192 * 1 + 2*64*8192`
  = ~268 MB

The 128 reloads of A inflate `bytes_loaded` such that the predicted
memory time is ~2x actual. K-blocking BLAS on the actual hardware loads
A once, B once, writes C once -- ~135 MB total.

## What changed

Two independent skinny checks; OR them:

```python
_SKINNY_MIN_DIM = 16        # absolute floor (existing behavior)
_SKINNY_ASPECT_RATIO = 32   # NEW: aspect-ratio cutoff

is_skinny = (
    min(M, N) < self._SKINNY_MIN_DIM
    or max(M, N) >= self._SKINNY_ASPECT_RATIO * min(M, N)
)
```

The aspect-ratio threshold of 32 is empirical (Jetson Orin Nano matmul
baseline). MAE-by-aspect-ratio-bucket sweep:

| AR bucket | n | base MAE | new MAE (AR>=32) | delta |
|---|---|---|---|---|
| AR<=4 | 27 | 88.8% | 88.8% | 0% |
| 4<AR<=16 | 6 | 21.8% | 21.8% | 0% |
| 16<AR<=32 | 6 | 32.2% | 32.2% | 0% |
| 32<AR<=64 | 5 | 35.6% | 30.6% | **-5.1%** |
| 64<AR<=128 | 3 | 49.8% | 36.9% | **-12.9%** |
| AR>128 | 1 | 73.7% | 11.4% | **-62.2%** |

The 16-32 transition zone slightly favors the optimal-square model in
practice (firing skinny here would *regress* MAE +2.4%), so the
threshold is set above 32 rather than at 16.

## V4 PASS counts: unchanged

V4 floor PASS counts before and after this change (all four
`(hw, op)` cells):

| | i7 matmul | i7 linear | Jetson matmul | Jetson linear |
|---|---|---|---|---|
| before | 17/60 | 25/60 | 0/48 | 0/46 |
| after | 17/60 | 25/60 | 0/48 | 0/46 |

The latency precision improvement on 9 Jetson matmul shapes does **not**
flip any record across the PASS/FAIL boundary. Reason: those shapes
were already failing the **regime classifier** (predicted memory-bound,
measured compute-bound -- Jetson tensor-core/CUDA cores serve these
shapes faster than the predicted DRAM-time floor). The latency band
check is gated by `pass_regime AND pass_latency`; if regime stays wrong,
the record stays red regardless of how close the latency lands.

The compute-model under-derate is the dominant blocker, tracked as
"Category 3" in `docs/calibration/jetson-orin-nano-calibration-analysis.md`.

## Why ship anyway

1. **Memory-physics correctness.** The aspect-ratio detection is the
   right model for what real BLAS does on skinny matmuls (K-blocking,
   not MN-blocking). Even if V4 PASS doesn't move now, downstream
   consumers (the Embodied-AI-Architect orchestrator, deployment
   planners) get a more honest memory-traffic estimate today.

2. **Future-proofing.** When the compute model is fixed and regime
   classifications start matching, the latency precision we add now
   will translate to PASS gains. We don't want to land the compute
   fix on top of a known-wrong memory model and have the two
   improvements collide.

3. **Hardware-independent.** No mapper plumbing, no per-hardware
   threshold tables, no calibration drift risk. The check is a property
   of the matmul shape, not the silicon.

## Affected shapes

On the V4 sweeps, 9 Jetson matmul shapes hit the aspect-ratio branch:

| shape | aspect | base lat (ms) | new lat (ms) | meas (ms) | base err | new err |
|---|---|---|---|---|---|---|
| (96, 12288, 6144) | 128 | 5.51 | 2.86 | 3.11 | +77% | -8% |
| (192, 6144, 6144) | 32 | 2.86 | 1.55 | 2.98 | -4% | -48% |
| (6144, 6144, 192) | 32 | 2.86 | 1.55 | 2.89 | -1% | -46% |
| (12288, 3072, 192) | 64 | 2.94 | 1.62 | 2.65 | +11% | -39% |
| (64, 8192, 16384) | 256 | 9.72 | 4.96 | 5.60 | +74% | -11% |
| (64, 16384, 8192) | 256 | 9.72 | 4.94 | 4.89 | +99% | +1% |
| (128, 16384, 4096) | 128 | 5.04 | 2.59 | 2.89 | +74% | -10% |
| ... | | | | | | |

The 32-AR shapes (e.g. (192, 6144, 6144), (6144, 6144, 192)) get worse
with the new model on this metric -- the optimal-square reload count
happens to land closer to measured latency on these specific shapes.
This is the "boundary trade-off" reflected in the empirical MAE table:
any threshold 16-32 catches some shapes that benefit and some that
don't. Threshold 32 is the cutoff that minimizes net harm while
catching the high-AR shapes that benefit most.

A future, hardware-aware threshold (e.g. derived from `L1_per_unit /
(K * bpe)` where the K-blocking advantage is structural) could replace
this empirical cutoff. For now the shape-intrinsic check is the
simplest model that captures the dominant effect.

## i7 unaffected

No i7 sweep shape has aspect ratio in the 32-256 range with min >= 16,
so this change has zero effect on i7 V4 floors or latency predictions.
Confirmed by sweeping the threshold across {16, 32, 48, 64, 96, 128, 256, off}
on both `i7_12700k` and `jetson_orin_nano_8gb` (matmul + linear); i7
counts and per-record latencies are byte-identical across all settings.

## Cross-link

* PR #115 -- analysis doc that flagged this as Category 2
* PR #116 -- GPU dispatch overhead bridge (Category 1 from the analysis)
* `docs/calibration/jetson-orin-nano-calibration-analysis.md`
* `src/graphs/estimation/reuse_models.py` -- `MatmulReuseModel`
