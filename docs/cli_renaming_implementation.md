# CLI Tools Renaming - Implementation Plan

**Total tools**: 49
**Date**: 2026-01-17

## Legend

- **Action**: `rename` | `keep` | `move` | `remove`
- **Priority**: `P1` (primary tools) | `P2` (common tools) | `P3` (specialized) | `P4` (internal/rare)

---

## 1. Primary Estimation Tools (P1)

These are the main user-facing analysis tools - highest priority.

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 1 | `analyze_comprehensive.py` | `estimate.py` | rename | Main analysis tool |
| 2 | `analyze_batch.py` | `estimate_batch.py` | rename | Batch size sweep |
| 3 | `profile_graph.py` | `estimate_graph.py` | rename | Graph profiling |

---

## 2. Graph Exploration Tools (P2)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 4 | `partition_analyzer.py` | `explore_partitions.py` | rename | Partition analysis |
| 5 | `graph_explorer.py` | `explore_graph.py` | rename | Interactive graph view |

---

## 3. Energy Exploration Tools (P2)

Architecture-specific energy breakdown tools for educational purposes.

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 6 | `analyze_gpu_energy.py` | `explore_energy_gpu.py` | rename | GPU energy breakdown |
| 7 | `analyze_cpu_energy.py` | `explore_energy_cpu.py` | rename | CPU energy breakdown |
| 8 | `analyze_tpu_energy.py` | `explore_energy_tpu.py` | rename | TPU energy breakdown |
| 9 | `analyze_kpu_energy.py` | `explore_energy_kpu.py` | rename | KPU energy breakdown |
| 10 | `analyze_dpu_energy.py` | `explore_energy_dpu.py` | rename | DPU energy breakdown |
| 11 | `analyze_dsp_energy.py` | `explore_energy_dsp.py` | rename | DSP energy breakdown |
| 12 | `analyze_cgra_energy.py` | `explore_energy_cgra.py` | rename | CGRA energy breakdown |
| 13 | `sm_energy_breakdown.py` | `explore_energy_sm.py` | rename | SM-level energy |
| 14 | `spm_architectural_energy.py` | `explore_energy_spm.py` | rename | SPM architecture energy |
| 15 | `energy_step_by_step.py` | `explore_energy_tutorial.py` | rename | Energy tutorial |
| 16 | `energy_walkthrough.py` | `explore_energy_walkthrough.py` | rename | Step-by-step comparison |
| 17 | `compare_architecture_energy.py` | `explore_energy_theory.py` | rename | Theoretical cycle-level |

---

## 4. Comparison Tools (P1-P2)

### Primary Comparison Tools (P1)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 18 | `compare_architectures.py` | `compare_hardware.py` | rename | Primary hardware comparison |
| 19 | `compare_models.py` | `compare_models.py` | keep | Already well-named |
| 20 | `compare_architectures_energy.py` | `compare_energy_architectures.py` | rename | Vertical energy comparison |

### Secondary Comparison Tools (P2)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 21 | `compare_graph_mappings.py` | `compare_mappings.py` | rename | Graph mapping comparison |
| 22 | `compare_power_profiles.py` | `compare_power.py` | rename | Power profile comparison |
| 23 | `compare_tdp_registry.py` | `compare_tdp.py` | rename | TDP vs registry |

### Platform Comparison Tools (P2)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 24 | `compare_edge_ai_platforms.py` | `compare_edge_platforms.py` | rename | Edge AI platforms |
| 25 | `compare_automotive_adas.py` | `compare_automotive_platforms.py` | rename | ADAS platforms |
| 26 | `compare_datacenter_cpus.py` | `compare_datacenter_cpus.py` | keep | Already well-named |
| 27 | `compare_ip_cores.py` | `compare_ip_cores.py` | keep | Already well-named |
| 28 | `automotive_hardware_comparison.py` | `compare_automotive_hardware.py` | rename | Consistency with prefix |

### Specialized Comparison Tools (P4)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 29 | `compare_i7_12700k_mappers.py` | - | remove | Too specific, internal only |

---

## 5. Calibration Tools (P1-P2)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 30 | `calibrate.py` | `calibrate.py` | keep | Primary calibration tool |
| 31 | `calibrate_hardware.py` | `calibrate_full.py` | rename | Full/advanced calibration |
| 32 | `benchmark_sweep.py` | `benchmark.py` | rename | Primary benchmark tool |
| 33 | `calibration_coverage.py` | `show_calibration.py` | rename | Display calibration status |
| 34 | `show_calibration_efficiency.py` | `show_efficiency.py` | rename | Display efficiency metrics |
| 35 | `show_tops_per_watt.py` | `show_tops_per_watt.py` | keep | Already well-named |
| 36 | `migrate_calibrations.py` | `migrate_calibrations.py` | keep | Internal utility |

---

## 6. Discovery Tools (P1)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 37 | `discover_models.py` | `discover_models.py` | keep | Already well-named |
| 38 | `discover_transformers.py` | `discover_transformers.py` | keep | Already well-named |
| 39 | `list_hardware_mappers.py` | `discover_hardware.py` | rename | Consistency with prefix |
| 40 | `device_detection.py` | `discover_device.py` | rename | Consistency with prefix |

---

## 7. Show/Display Tools (P2-P3)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 41 | `analyze_graph_mapping.py` | `show_mapping.py` | rename | Display mapping details |
| 42 | `profile_graph_with_fvcore.py` | `show_fvcore.py` | rename | FVCore reference display |

---

## 8. Utility/Specialized Tools (P3)

| # | Current | New | Action | Notes |
|---|---------|-----|--------|-------|
| 43 | `estimate_tdp.py` | `estimate_tdp.py` | keep | Already well-named |
| 44 | `measure_model_efficiency.py` | `benchmark_efficiency.py` | rename | Efficiency benchmarking |
| 45 | `analyze_product_trajectory.py` | `explore_trajectory.py` | rename | Product trajectory |
| 46 | `profile_torchvision_graph.py` | `estimate_torchvision.py` | rename | Torchvision estimation |

---

## 9. Move to Library (not CLI tools)

| # | Current | Destination | Action | Notes |
|---|---------|-------------|--------|-------|
| 47 | `model_factory.py` | `src/graphs/frontends/model_factory.py` | move | Library code |
| 48 | `energy_breakdown_utils.py` | `src/graphs/estimation/energy_utils.py` | move | Library code |
| 49 | `model_registry_tv2dot7.py` | `src/graphs/frontends/model_registry.py` | move | Data/registry |

---

## Summary by Action

| Action | Count | Tools |
|--------|-------|-------|
| **rename** | 33 | Primary work |
| **keep** | 12 | Already well-named |
| **move** | 3 | Library code |
| **remove** | 1 | Too specific |
| **Total** | 49 | |

---

## Implementation Order

### Phase A: Primary Tools (P1) - Do First
1. `analyze_comprehensive.py` -> `estimate.py`
2. `analyze_batch.py` -> `estimate_batch.py`
3. `profile_graph.py` -> `estimate_graph.py`
4. `compare_architectures.py` -> `compare_hardware.py`
5. `compare_architectures_energy.py` -> `compare_energy_architectures.py`
6. `list_hardware_mappers.py` -> `discover_hardware.py`
7. `device_detection.py` -> `discover_device.py`

### Phase B: Energy Exploration Tools
8-19. All `analyze_*_energy.py` -> `explore_energy_*.py` (12 files)

### Phase C: Remaining Comparison Tools
20-28. Remaining comparison tool renames (9 files)

### Phase D: Calibration & Show Tools
29-35. Calibration and show tool renames (7 files)

### Phase E: Utility Tools
36-38. Remaining utility renames (3 files)

### Phase F: Library Migration
39-41. Move 3 files to library

### Phase G: Cleanup
42. Remove `compare_i7_12700k_mappers.py`
43. Update documentation references
44. Update cross-tool imports

---

## Deprecation Wrapper Template

For each renamed file, the old file becomes a thin wrapper:

```python
#!/usr/bin/env python3
"""
DEPRECATED: Use {new_name}.py instead.
This file will be removed in version 1.0.
"""
import warnings
import sys
from pathlib import Path

warnings.warn(
    "{old_name}.py is deprecated. Use {new_name}.py instead. "
    "This script will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=1
)

# Import and run the new tool
from {new_name} import main

if __name__ == "__main__":
    sys.exit(main())
```

---

## Documentation Updates Required

1. `CLAUDE.md` - Update CLI section
2. `docs/guides/guided_tour.md` - Update all CLI examples
3. `docs/guides/getting_started.md` - Update CLI examples
4. `cli/README.md` - Update tool list
5. Any other docs referencing CLI tools

---

## Questions for Review

1. Should we implement all phases at once or incrementally?
2. Should deprecated wrappers be kept for one release cycle or longer?
3. Any tools I should reconsider the naming for?
