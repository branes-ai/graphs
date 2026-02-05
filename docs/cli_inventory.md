# CLI Tools Inventory and Naming Proposal

## Current State Analysis

The current CLI tools use inconsistent naming:
- `analyze_*` - 12 tools (mixed purposes)
- `compare_*` - 13 tools (comparisons)
- `calibrate*` - 2 tools
- `profile_*` - 3 tools
- `discover_*` - 2 tools
- Various others with no prefix pattern

## Proposed Naming Convention

Based on the new foundation structure, I propose these prefixes:

| Prefix | Purpose | Maps to Package |
|--------|---------|-----------------|
| `estimate` | Performance/energy/memory estimation | `estimation/` |
| `compare` | Multi-target comparisons | `estimation/` + `hardware/` |
| `calibrate` | Hardware calibration | `calibration/` |
| `benchmark` | Run benchmarks | `benchmarks/` |
| `discover` | Find models/hardware | `frontends/` + `hardware/` |
| `explore` | Educational/interactive tools | (documentation) |
| `show` | Display data/reports | (reporting) |

---

## Category 1: Primary Estimation Tools

These are the main user-facing analysis tools.

| Current Name | Purpose | Proposed Name | Notes |
|--------------|---------|---------------|-------|
| `analyze_comprehensive.py` | Full model analysis | `estimate.py` | Primary tool |
| `analyze_batch.py` | Batch size sweep | `estimate-batch.py` | Batch analysis |
| `profile_graph.py` | Graph profiling | `estimate-graph.py` | Or merge into estimate.py |
| `partition_analyzer.py` | Partition analysis | `estimate-partitions.py` | Partition details |
| `graph_explorer.py` | Interactive graph view | `explore-graph.py` | Educational |

## Category 2: Architecture-Specific Energy Tools

These provide detailed energy breakdowns per architecture type.

| Current Name | Purpose | Proposed Name | Notes |
|--------------|---------|---------------|-------|
| `analyze_gpu_energy.py` | GPU energy breakdown | `explore-energy-gpu.py` | Educational |
| `analyze_cpu_energy.py` | CPU energy breakdown | `explore-energy-cpu.py` | Educational |
| `analyze_tpu_energy.py` | TPU energy breakdown | `explore-energy-tpu.py` | Educational |
| `analyze_kpu_energy.py` | KPU energy breakdown | `explore-energy-kpu.py` | Educational |
| `analyze_dpu_energy.py` | DPU energy breakdown | `explore-energy-dpu.py` | Educational |
| `analyze_dsp_energy.py` | DSP energy breakdown | `explore-energy-dsp.py` | Educational |
| `analyze_cgra_energy.py` | CGRA energy breakdown | `explore-energy-cgra.py` | Educational |
| `sm_energy_breakdown.py` | SM-level energy detail | `explore-energy-sm.py` | Educational |
| `spm_architectural_energy.py` | SPM architecture energy | `explore-energy-spm.py` | Educational |
| `energy_step_by_step.py` | Energy walkthrough | `explore-energy-tutorial.py` | Educational |
| `energy_walkthrough.py` | Step-by-step comparison | `explore-energy-walkthrough.py` | Educational |

**Alternative**: Consolidate all `explore-energy-*.py` into single `explore-energy.py --arch gpu|cpu|tpu|...`

## Category 3: Comparison Tools

Multi-target comparison tools.

| Current Name | Purpose | Proposed Name | Notes |
|--------------|---------|---------------|-------|
| `compare_architectures.py` | Compare across hardware | `compare-hardware.py` | Primary comparison |
| `compare_models.py` | Compare models on hardware | `compare-models.py` | Keep as-is |
| `compare_architectures_energy.py` | Energy comparison | `compare-energy.py` | Consolidate |
| `compare_architecture_energy.py` | Energy comparison (dup?) | (remove - duplicate?) | |
| `compare_graph_mappings.py` | Graph mapping comparison | `compare-mappings.py` | |
| `compare_power_profiles.py` | Power profile comparison | `compare-power.py` | |
| `compare_tdp_registry.py` | TDP vs registry | `compare-tdp.py` | |

### Platform-Specific Comparisons (consider consolidating)

| Current Name | Purpose | Proposed Name | Notes |
|--------------|---------|---------------|-------|
| `compare_edge_ai_platforms.py` | Edge AI platforms | `compare-platforms.py --category edge` | Consolidate |
| `compare_automotive_adas.py` | ADAS platforms | `compare-platforms.py --category automotive` | Consolidate |
| `compare_datacenter_cpus.py` | Datacenter CPUs | `compare-platforms.py --category datacenter` | Consolidate |
| `automotive_hardware_comparison.py` | Automotive hardware | (merge with above) | Duplicate? |
| `compare_ip_cores.py` | IP cores for SoC | `compare-platforms.py --category ip-cores` | Consolidate |
| `compare_i7_12700k_mappers.py` | Specific CPU mappers | (remove or internal) | Too specific |

## Category 4: Calibration Tools

Hardware calibration and benchmarking.

| Current Name | Purpose | Proposed Name | Notes |
|--------------|---------|---------------|-------|
| `calibrate.py` | Simplified calibration | `calibrate.py` | Keep as primary |
| `calibrate_hardware.py` | Full calibration | `calibrate-full.py` | Or merge |
| `benchmark_sweep.py` | Run benchmark suite | `benchmark.py` | Primary benchmark |
| `calibration_coverage.py` | Show calibration coverage | `show-calibration.py` | Reporting |
| `show_calibration_efficiency.py` | Show efficiency | `show-efficiency.py` | Reporting |
| `show_tops_per_watt.py` | Show TOPS/W | `show-tops-per-watt.py` | Reporting |
| `migrate_calibrations.py` | Migrate old files | (keep internal) | Utility |

## Category 5: Discovery Tools

Find available models and hardware.

| Current Name | Purpose | Proposed Name | Notes |
|--------------|---------|---------------|-------|
| `discover_models.py` | Find torchvision models | `discover-models.py` | Keep |
| `discover_transformers.py` | Find transformer models | `discover-transformers.py` | Keep |
| `list_hardware_mappers.py` | List hardware | `discover-hardware.py` | Rename for consistency |
| `device_detection.py` | Detect local hardware | `discover-device.py` | Rename |

## Category 6: Utility/Internal Tools

| Current Name | Purpose | Proposed Name | Notes |
|--------------|---------|---------------|-------|
| `model_factory.py` | Model instantiation | (internal - no CLI) | Library, not CLI |
| `model_registry_tv2dot7.py` | Registry data | (internal - no CLI) | Data file |
| `energy_breakdown_utils.py` | Utilities | (internal - no CLI) | Library |
| `estimate_tdp.py` | TDP estimation | `estimate-tdp.py` | Or merge |
| `measure_model_efficiency.py` | Efficiency measurement | `benchmark-efficiency.py` | |
| `analyze_graph_mapping.py` | Graph mapping | `show-mapping.py` | |
| `analyze_product_trajectory.py` | Product trajectory | `explore-trajectory.py` | Specialized |
| `profile_graph_with_fvcore.py` | FVCore reference | `show-fvcore.py` | Reference tool |
| `profile_torchvision_graph.py` | Torchvision profiling | (merge into estimate.py) | |

---

## Recommended Final Structure

### Primary User Tools (6 tools)
```
estimate.py              # Main analysis tool (was analyze_comprehensive.py)
estimate-batch.py        # Batch size analysis
compare-hardware.py      # Compare across hardware targets
compare-models.py        # Compare models on same hardware
calibrate.py             # Hardware calibration
benchmark.py             # Run calibration benchmarks
```

### Discovery Tools (4 tools)
```
discover-models.py       # Find FX-traceable models
discover-transformers.py # Find transformer models
discover-hardware.py     # List available hardware mappers
discover-device.py       # Detect local hardware
```

### Exploration/Educational Tools (3-4 tools)
```
explore-graph.py         # Interactive graph exploration
explore-energy.py        # Energy breakdown by architecture (--arch flag)
explore-partitions.py    # Partition analysis details
```

### Reporting Tools (4 tools)
```
show-calibration.py      # Calibration coverage/status
show-efficiency.py       # Hardware efficiency metrics
show-tops-per-watt.py    # TOPS/W from calibration
show-mapping.py          # Graph-to-hardware mapping details
```

### Platform Comparison (1 consolidated tool)
```
compare-platforms.py     # --category edge|automotive|datacenter|ip-cores
```

---

## Migration Notes

1. **Keep backward compatibility**: Create wrapper scripts that emit deprecation warnings
2. **Consolidate duplicates**: Several tools appear to do similar things
3. **Move utilities to library**: `model_factory.py`, `energy_breakdown_utils.py` should not be CLI tools
4. **Consider subcommands**: Could use `graphs estimate`, `graphs compare`, etc. as a single entry point

## Questions for Review

1. Should we use hyphens (`estimate-batch.py`) or underscores (`estimate_batch.py`)?
2. Should `explore-energy-*.py` be consolidated into one tool with `--arch` flag?
3. Should platform comparisons be consolidated into `compare-platforms.py`?
4. Should we create a single `graphs` entry point with subcommands?
