# CLI Tools Renaming Plan

**Date**: 2026-01-17
**Status**: Approved for implementation

## Naming Convention

| Prefix | Purpose | Maps to Package |
|--------|---------|-----------------|
| `estimate_` | Performance/energy/memory estimation | `estimation/` |
| `compare_` | Multi-target comparisons | `hardware/` |
| `calibrate_` | Hardware calibration | `calibration/` |
| `benchmark_` | Run benchmarks | `benchmarks/` |
| `discover_` | Find models/hardware | `frontends/` + `hardware/` |
| `explore_` | Educational/interactive | (documentation) |
| `show_` | Display data/reports | (reporting) |

## Renaming Table

### Primary Estimation Tools

| Current | New | Purpose |
|---------|-----|---------|
| `analyze_comprehensive.py` | `estimate.py` | Main analysis tool |
| `analyze_batch.py` | `estimate_batch.py` | Batch size sweep |
| `profile_graph.py` | `estimate_graph.py` | Graph profiling |
| `partition_analyzer.py` | `explore_partitions.py` | Partition details |
| `graph_explorer.py` | `explore_graph.py` | Interactive graph view |

### Energy Exploration Tools

| Current | New | Purpose |
|---------|-----|---------|
| `analyze_gpu_energy.py` | `explore_energy_gpu.py` | GPU energy breakdown |
| `analyze_cpu_energy.py` | `explore_energy_cpu.py` | CPU energy breakdown |
| `analyze_tpu_energy.py` | `explore_energy_tpu.py` | TPU energy breakdown |
| `analyze_kpu_energy.py` | `explore_energy_kpu.py` | KPU energy breakdown |
| `analyze_dpu_energy.py` | `explore_energy_dpu.py` | DPU energy breakdown |
| `analyze_dsp_energy.py` | `explore_energy_dsp.py` | DSP energy breakdown |
| `analyze_cgra_energy.py` | `explore_energy_cgra.py` | CGRA energy breakdown |
| `sm_energy_breakdown.py` | `explore_energy_sm.py` | SM-level energy detail |
| `spm_architectural_energy.py` | `explore_energy_spm.py` | SPM architecture energy |
| `energy_step_by_step.py` | `explore_energy_tutorial.py` | Energy walkthrough |
| `energy_walkthrough.py` | `explore_energy_walkthrough.py` | Step-by-step comparison |
| `compare_architecture_energy.py` | `explore_energy_theory.py` | Theoretical cycle-level energy |

### Comparison Tools

| Current | New | Purpose |
|---------|-----|---------|
| `compare_architectures.py` | `compare_hardware.py` | Compare across hardware |
| `compare_models.py` | `compare_models.py` | (keep) Compare models |
| `compare_architectures_energy.py` | `compare_energy_architectures.py` | Vertical energy comparison |
| `compare_graph_mappings.py` | `compare_mappings.py` | Graph mapping comparison |
| `compare_power_profiles.py` | `compare_power.py` | Power profile comparison |
| `compare_tdp_registry.py` | `compare_tdp.py` | TDP vs registry |
| `compare_edge_ai_platforms.py` | `compare_edge_platforms.py` | Edge AI platforms |
| `compare_automotive_adas.py` | `compare_automotive_platforms.py` | ADAS platforms |
| `compare_datacenter_cpus.py` | `compare_datacenter_cpus.py` | (keep) Datacenter CPUs |
| `compare_ip_cores.py` | `compare_ip_cores.py` | (keep) IP cores |
| `automotive_hardware_comparison.py` | `compare_automotive_hardware.py` | Automotive hardware |

### Calibration Tools

| Current | New | Purpose |
|---------|-----|---------|
| `calibrate.py` | `calibrate.py` | (keep) Primary calibration |
| `calibrate_hardware.py` | `calibrate_full.py` | Full calibration |
| `benchmark_sweep.py` | `benchmark.py` | Primary benchmark tool |
| `calibration_coverage.py` | `show_calibration.py` | Calibration coverage |
| `show_calibration_efficiency.py` | `show_efficiency.py` | Hardware efficiency |
| `show_tops_per_watt.py` | `show_tops_per_watt.py` | (keep) TOPS/W metrics |
| `migrate_calibrations.py` | `migrate_calibrations.py` | (keep - internal) Migration utility |

### Discovery Tools

| Current | New | Purpose |
|---------|-----|---------|
| `discover_models.py` | `discover_models.py` | (keep) Find torchvision models |
| `discover_transformers.py` | `discover_transformers.py` | (keep) Find transformer models |
| `list_hardware_mappers.py` | `discover_hardware.py` | List hardware mappers |
| `device_detection.py` | `discover_device.py` | Detect local hardware |

### Utility Tools

| Current | New | Purpose |
|---------|-----|---------|
| `estimate_tdp.py` | `estimate_tdp.py` | (keep) TDP estimation |
| `measure_model_efficiency.py` | `benchmark_efficiency.py` | Efficiency measurement |
| `analyze_graph_mapping.py` | `show_mapping.py` | Graph mapping details |
| `analyze_product_trajectory.py` | `explore_trajectory.py` | Product trajectory |
| `profile_graph_with_fvcore.py` | `show_fvcore.py` | FVCore reference |
| `profile_torchvision_graph.py` | `estimate_torchvision.py` | Torchvision profiling |

### Move to Library (not CLI tools)

| Current | Action | Reason |
|---------|--------|--------|
| `model_factory.py` | Move to `src/graphs/frontends/` | Library, not CLI |
| `energy_breakdown_utils.py` | Move to `src/graphs/estimation/` | Library, not CLI |
| `model_registry_tv2dot7.py` | Move to `src/graphs/frontends/` | Data file |

### Remove

| Current | Reason |
|---------|--------|
| `compare_i7_12700k_mappers.py` | Too specific, internal testing only |

## Summary

| Category | Count |
|----------|-------|
| Primary estimation | 5 |
| Energy exploration | 12 |
| Comparison | 11 |
| Calibration | 7 |
| Discovery | 4 |
| Utility | 6 |
| Move to library | 3 |
| Remove | 1 |
| **Total** | **49** -> **45 CLI tools** |

## Migration Strategy

1. Create new files with new names
2. Update old files to be thin wrappers that:
   - Import from new file
   - Emit deprecation warning
   - Call main() from new file
3. Update documentation (CLAUDE.md, guided_tour.md, getting_started.md)
4. Update any cross-references between CLI tools
5. After one release cycle, remove old wrapper files
