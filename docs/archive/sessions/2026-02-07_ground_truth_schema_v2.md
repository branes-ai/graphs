# Session Summary: Ground-Truth Measurement Schema v2.0

**Date**: 2026-02-07
**Duration**: ~4 hours (across 2 context windows)
**Phase**: Calibration Data Infrastructure
**Status**: Complete

---

## Goals for This Session

1. Implement ground-truth data schema v2.0 that decouples hardware measurement (run once on physical hardware) from estimator validation (run repeatedly on dev machine)
2. Create a canonical storage layout and loader for measurement data
3. Migrate all existing v1.0 measurement data to the new layout
4. Update all CLI query tools to use the new loader
5. Enforce that no new code reads measurement JSON directly

---

## What We Accomplished

### 1. Schema v2.0 (`src/graphs/calibration/ground_truth.py`)

Created the core schema file with dataclasses extending v1.0:

- **MeasurementRecord**: top-level record with new fields `batch_size`, `input_shape`, `model_summary`, `system_state`, `run_id`, `run_index`
- **ModelSummary**: whole-model aggregates computed from subgraphs (total FLOPs, bytes, latency, throughput)
- **SystemState**: hardware conditions during measurement (GPU/CPU clocks, temperature, power mode, CPU governor)
- **GroundTruthLoader**: query interface with `list_hardware()`, `list_models()`, `list_configurations()`, `load()`, `load_model_summary()`, `load_all()`, `rebuild_manifest()`
- **Manifest / ManifestEntry**: auto-generated per-hardware index for fast lookups without reading every JSON file
- Full `to_dict()` / `from_dict()` / `save()` / `load()` serialization following existing codebase patterns
- Backward-compatible: v1.0 files loaded transparently (batch_size inferred from filename, model_summary computed on the fly)

### 2. Canonical Storage Layout

Consolidated from two scattered locations into one:

```
calibration_data/
  <hardware_id>/
    manifest.json
    measurements/
      <precision>/
        <model>_b<batch>.json
```

Previously data lived in both `measurements/<hw>/<prec>/` and `calibration_data/<hw>/<prec>/measurements/`.

### 3. Migration (`cli/migrate_measurements.py`)

Created migration script that:
- Scans both `measurements/` and `calibration_data/` source directories
- Converts each v1.0 file to v2.0 (adds `model_summary`, bumps `schema_version`)
- Writes to canonical layout
- Generates `manifest.json` per hardware_id
- Supports `--dry-run`, `--verbose`, `--manifests-only`

Migration result: 661 files migrated, 0 failures, 11 hardware targets, 0 duplicates.

### 4. Updated `cli/measure_efficiency.py`

- Removed local dataclass definitions (now imports from `ground_truth.py`)
- Added `capture_system_state()` to query GPU/CPU clocks at measurement time
- Output now uses `MeasurementRecord.save()` with all v2.0 fields
- Added `uuid` for `run_id` tracking

### 5. Updated Query Tools

**`cli/query_calibration_data.py`**: Rewritten to use `GroundTruthLoader` for measurement queries. Efficiency curves (different data type) still read directly.

**`cli/aggregate_efficiency.py`**: `--hardware-id` now uses `GroundTruthLoader` by default. Old `--input`/`--input-dir` flags deprecated with warnings.

**`cli/run_full_calibration.py`**: Writes to canonical layout, rebuilds manifest after aggregation.

### 6. Enforced Legacy Code Removal

- Removed old `measurements/` top-level directory (445 duplicate JSON files)
- Created `ci/check_no_legacy_json_reads.sh` -- CI script that fails if:
  - `load_measurement_file()` appears outside safe-listed files
  - `json.load()` appears near measurement-related context
  - Glob patterns scan measurement directories
  - Old `measurements/` directory still exists
- Added `data-hygiene` job to `.github/workflows/ci.yml`

---

## Key Technical Decisions

### Filename Parsing Ambiguity

Model names like `efficientnet_b0`, `efficientnet_b1`, `vit_b_16` contain `_b<N>` which conflicts with the batch suffix `_b<N>`. For example, `efficientnet_b1_b4.json` could be:
- model=`efficientnet_b1`, batch=4 (correct)
- model=`efficientnet`, batch=1 (wrong -- `_b4` lost entirely)

**Solution**: `_is_valid_model_name()` heuristic that recognizes known model family prefixes. `efficientnet` alone is a bare prefix (incomplete), so `_b1` stays part of the model name. `resnet18` is a complete model name (prefix + digits), so `_b4` is treated as batch suffix.

### What Reads Measurement JSON vs. Efficiency Curves

Two distinct data types in `calibration_data/`:
1. **Measurements** (per-model, per-subgraph) -- must go through `GroundTruthLoader`
2. **Efficiency curves** (aggregated by operation type) -- read directly by `gpu_calibration.py`, `query_calibration_data.py`, etc.

The CI check only enforces the loader for measurement data, not efficiency curves.

### Deprecation Strategy

Old `--input`/`--input-dir` flags in `aggregate_efficiency.py` emit `DeprecationWarning` and print a visible banner. The legacy glob path still works but will be removed in a future version.

---

## Files Created

| File | Description |
|------|-------------|
| `src/graphs/calibration/ground_truth.py` | Schema v2.0 dataclasses + GroundTruthLoader |
| `cli/migrate_measurements.py` | Migration script for v1.0 -> v2.0 + canonical layout |
| `ci/check_no_legacy_json_reads.sh` | CI lint check for direct JSON reads |

## Files Modified

| File | Description |
|------|-------------|
| `src/graphs/calibration/__init__.py` | Export ground-truth types |
| `cli/measure_efficiency.py` | Emit v2.0 via MeasurementRecord, import from ground_truth |
| `cli/run_full_calibration.py` | Canonical paths + manifest rebuild |
| `cli/query_calibration_data.py` | Rewritten to use GroundTruthLoader |
| `cli/aggregate_efficiency.py` | Default to GroundTruthLoader, deprecate old flags |
| `.github/workflows/ci.yml` | Added data-hygiene job |

## Files Removed

| File | Description |
|------|-------------|
| `measurements/` (445 files) | Old top-level directory, all data migrated to `calibration_data/` |

---

## Commits

1. `a3da892` - Ground-truth measurement schema v2.0 with migration and query tools
2. `5555279` - Remove legacy measurements/ directory and enforce GroundTruthLoader usage

---

## Verification

- 661 files migrated successfully, 0 failures
- 11 manifests generated (one per hardware target)
- All 556 unit tests pass
- CI data-hygiene check passes clean
- `aggregate_efficiency.py --hardware-id jetson_orin_agx_maxn` loads 15 records via GroundTruthLoader

---

## Next Steps

- Run `measure_efficiency.py` on physical hardware to verify v2.0 output end-to-end
- Consider adding `system_state` capture for CPU targets (currently GPU-focused)
- Eventually remove deprecated `--input`/`--input-dir` flags from `aggregate_efficiency.py`
- Build estimator validation framework on top of GroundTruthLoader (Task 2 from the calibration workflow plan)
