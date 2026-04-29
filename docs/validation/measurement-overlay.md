# Measurement-overlay validation panel — Path A

## What it is

An optional layer panel appended to the micro-architectural report
that compares the M1-M7 *analytical* latency predictions against
*measured* model-level latencies in `calibration_data/`. Tagged
`LayerTag.COMPOSITE` so it renders alongside the seven layer panels
without changing the layer enum.

Path A answers a single question: *"Does the analytical chain we
built layer-by-layer in M1-M7 actually agree with the silicon?"* It
is the bridge between the bottom-up theoretical model and the
top-down measurement data, and is the only way an SKU's aggregate
confidence can promote from `THEORETICAL` to `INTERPOLATED` today.

For per-cache-level energy calibration (a *finer-grained* upgrade
that touches `field_provenance` rather than the aggregate
confidence), see [docs/calibration/cache_sweep.md](../calibration/cache_sweep.md)
— that's Path B.

## Running

The validation panel is **opt-in**: each validated SKU adds ~60 s
to the report build because every measurement triggers a
`UnifiedAnalyzer` prediction. Pass `--with-validation`:

```bash
./cli/microarch_validation_report.py \
    --hardware intel_core_i7_12700k jetson_orin_agx_64gb \
    --with-validation \
    --output reports/microarch_model/with_validation
```

Without the flag, the renderer's `LayerTag.COMPOSITE` gate
suppresses the cross-SKU validation table in `compare.html` and the
per-SKU panel never gets appended — the report builds in seconds
and looks identical to a pre-Path-A run.

## Where it lands

| File | What appears with `--with-validation` |
|---|---|
| `hardware/<sku>.html` | "Validation: predicted vs measured" panel after Layers 1-7. Status badge = `INTERPOLATED` (within tolerance) or `THEORETICAL` (drift surfaced). |
| `compare.html` | Cross-SKU table with one row per validated SKU: models validated, median MAPE, tolerance badge. |
| `index.html` | Per-SKU confidence column reflects the promotion when median MAPE is within tolerance. |
| `data/<sku>.json` | Same data as the panel, machine-readable, including top-3 best / top-3 worst per-model MAPEs. |

## Promotion rule

Encoded in `validation_panel.build_validation_panel`:

- `median_mape_pct <= 30%` -> panel status `interpolated`,
  `report.overall_confidence` promoted from `THEORETICAL` to
  `INTERPOLATED`.
- `median_mape_pct >  30%` -> panel status `theoretical`, numeric
  MAPE surfaced for inspection. The drift is reported per-model so
  a Path B microbenchmark campaign can target the layers most
  responsible.

The `_populate_validation` hook (`cli/microarch_validation_report.py`)
only promotes when the layer-level confidence is still `THEORETICAL`
— it never demotes.

## SKU coverage today

Coverage is gated by what's in `calibration_data/` plus an explicit
mapping from microarch-report SKU id to calibration directory and
to `UnifiedAnalyzer` mapper name. The mapping table lives in
`src/graphs/reporting/validation/sku_id_resolution.py`:

| Microarch SKU id | calibration_data/ dir | UnifiedAnalyzer mapper | Status |
|---|---|---|---|
| `intel_core_i7_12700k` | `intel_core_i7_12700k` | `i7-12700k` | ~5% MAPE on 10 fp32 models -> promoted INTERPOLATED |
| `jetson_orin_agx_64gb` | `jetson_orin_agx_30w` (30W matches the M1-M7 baseline) | `jetson-orin-agx-64gb` | ~60% MAPE -> stays THEORETICAL; drift surfaced for Path B follow-up |
| `ryzen_9_8945hs` | `ryzen_9_8945hs` | `ryzen` | Mapping registered; outcome depends on measurement set present |
| `kpu_t64`, `kpu_t128`, `kpu_t256`, `coral_edge_tpu`, `hailo8`, `hailo10h` | -- | -- | No measurement data; panel renders the empty-state explanation |

Adding a SKU is a three-row edit in `sku_id_resolution.py`
(`_SKU_TO_CALIBRATION_DIR`, `_SKU_TO_MAPPER_NAME`, optionally
`_SKU_TO_THERMAL_PROFILE`) plus a measurement directory under
`calibration_data/`. No panel-side changes are required.

## Empty-state behaviour

A SKU without measurement data is not an error. The panel renders
with status `theoretical` and an explanatory summary:

> "No measurement data registered for `<sku>`; the SKU's aggregate
> confidence stays at THEORETICAL until a calibration campaign
> lands."

A SKU registered but with zero matching measurements (the
calibration directory exists but every model file failed to load)
also renders cleanly with a pointer to check `calibration_data/`
contents. Neither case suppresses the panel — silent empty
sections are worse than visible empty sections.

## Data flow

```text
calibration_data/<dir>/measurements/<prec>/<model>_b<batch>.json
                         |
                         v
              GroundTruthLoader.load() / .list_models()
                         |
                         v   (per data-hygiene rule —
                              never json.loads directly)
                         |
                MeasurementSummary record
                         |
                         v
       UnifiedAnalyzer.analyze_model(<mapper>, <model>, ...)
                         |
                         v
                 PredictionRecord (analytical latency)
                         |
                         v
                  ValidationResult (MAPE)
                         |
                         v
                SKUValidationSummary (median, mean, n_within_tolerance)
                         |
                         v
                LayerPanel(LayerTag.COMPOSITE)
                         |
                         v
                  hardware/<sku>.html  +  compare.html  +  data/<sku>.json
```

Predictions are cached at module level (`_CACHE` in
`validation_panel.py`) so a re-render within a single process
doesn't re-pay the `UnifiedAnalyzer` cost. Tests can drop the cache
via `clear_validation_cache()`.

## Why this is opt-in

The `UnifiedAnalyzer` import alone pulls in PyTorch, FX, and the
full mapper registry. The pre-existing micro-arch report build
runs in single-digit seconds and is used for cross-SKU tables that
do not need predictions. Forcing every report build to pay
~60 s/SKU for validation would make the M0/M1-M7 layer panels
visibly slower for users who don't need the empirical overlay.
The render-time gate in `microarch_html_template.py` is intentional:
no `LayerTag.COMPOSITE` in any report's layers -> no validation
section anywhere in the bundle.

## Relationship to Path B

Path A validates *the model as a whole*: it tells you whether the
end-to-end analytical prediction matches measured latency. When it
fails (Orin AGX today), it does not tell you *which layer* is
responsible for the drift.

Path B (the cache-sweep work in PR #45 and beyond) calibrates
*individual layer coefficients* — per-cache-level read energy,
DRAM bandwidth, NoC ping-pong cost. Where Path A flips the
aggregate confidence badge, Path B flips per-field provenance
entries on `HardwareResourceModel` from `THEORETICAL` to
`CALIBRATED`. The two paths are complementary: Path A surfaces a
failed prediction, Path B narrows down the layer to fix.
