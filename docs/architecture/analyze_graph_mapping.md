# Hardware-Aware Graph Analysis Tool Design

**Tool**: `cli/analyze_graph_mapping.py`
**Author**: Design Session 2025-11-15
**Status**: Implementation Ready

## Overview

This document describes the design for a new CLI tool that performs hardware-aware subgraph analysis with comprehensive visualization. The tool addresses limitations in the existing `compare_architectures_energy.py` by providing detailed per-subgraph hardware mapping, latency, and energy estimates for any PyTorch model.

### Problem Statement

The current `compare_architectures_energy.py` tool:
- Only supports MLP workloads
- Focuses on architecture comparison rather than deep-dive analysis
- Doesn't show per-subgraph hardware allocation details
- Lacks visualization of the mapping from graph nodes → subgraphs → hardware resources

### Solution

Create a new tool that:
1. Works with any PyTorch model (torchvision or synthetic)
2. Performs graph partitioning into fused subgraphs
3. Maps each subgraph to specific hardware resources
4. Computes per-subgraph latency and energy estimates
5. Visualizes the complete pipeline in a three-column format

## Architecture

### Three-Column Visualization

```
Column 1: Raw FX Graph          Column 2: Fused Subgraphs       Column 3: Hardware Allocation
─────────────────────────────   ─────────────────────────────   ─────────────────────────────
Individual nodes                Grouped operations              Resource allocation
Execution order                 Workload characteristics        Latency & energy estimates
```

**Example Output**:

```
=================================================================================================
HARDWARE-AWARE GRAPH ANALYSIS
Model: resnet18 (batch=1, FP32) | Hardware: H100 SXM5 80GB @ 700W TDP
=================================================================================================

FX Graph (Execution Order)         Fused Subgraphs                    Hardware Allocation
----------------------------        --------------------------------   ---------------------------------

1. [placeholder] x
   Shape: [1, 3, 224, 224]
   Memory: 602.1 KB

2. [call_module] conv1              ╔═══ SUBGRAPH 0 ═══════════════   ┌─ ALLOCATION ──────────────────┐
   Conv2d(3→64, k=7×7, s=2, p=3)    ║ Conv2d + BatchNorm + ReLU       │ SMs: 12 / 132 (9.1%)           │
   Shape: [1, 64, 112, 112]         ║ FLOPs: 118.0 MFLOPs             │ Warps/SM: 16                   │
   FLOPs: 118.0M                    ║ Memory: 3.2 MB                  │ Waves: 1.0                     │
                                    ║ Arithmetic Intensity: 36.9       │ Occupancy: 50.0%               │
3. [call_module] bn1                ║ Bottleneck: COMPUTE_BOUND       │ ──────────────────────────────│
   BatchNorm2d(64)                  ╚═════════════════════════════   │ Latency: 0.142 ms              │
   Shape: [1, 64, 112, 112]                                           │ Energy: 1.83 mJ                │
                                                                      │ Efficiency: 69.4%              │
4. [call_module] relu                                                 └────────────────────────────────┘
   ReLU(inplace=True)
   Shape: [1, 64, 112, 112]

5. [call_module] maxpool            ╔═══ SUBGRAPH 1 ═══════════════   ┌─ ALLOCATION ──────────────────┐
   MaxPool2d(k=3×3, s=2, p=1)       ║ MaxPool2d                       │ SMs: 4 / 132 (3.0%)            │
   Shape: [1, 64, 56, 56]           ║ FLOPs: 0.0 MFLOPs               │ Warps/SM: 8                    │
                                    ║ Memory: 1.6 MB                  │ Waves: 0.5                     │
                                    ║ Arithmetic Intensity: 0.0        │ Occupancy: 25.0%               │
                                    ║ Bottleneck: MEMORY_BOUND        │ ──────────────────────────────│
                                    ╚═════════════════════════════   │ Latency: 0.089 ms              │
                                                                      │ Energy: 0.45 mJ                │
                                                                      │ Efficiency: 12.8%              │
                                                                      └────────────────────────────────┘
```

### Analysis Pipeline

```python
# 1. FX Trace + Shape Propagation
traced = symbolic_trace(model)
ShapeProp(traced).propagate(example_input)

# 2. Partition into fused subgraphs
partitioner = FusionBasedPartitioner()
partition_report = partitioner.partition(traced)

# 3. Create execution stages (sequential or parallel)
execution_stages = [[i] for i in range(len(partition_report.fused_subgraphs))]

# 4. Map to hardware
mapper = create_hardware_mapper(hardware_name, thermal_profile)
hw_allocation = mapper.map_graph(
    partition_report, execution_stages, batch_size, precision
)

# 5. Extract per-subgraph metrics
for i, subgraph in enumerate(partition_report.fused_subgraphs):
    metrics = {
        'subgraph_id': i,
        'flops': subgraph.total_flops,
        'memory_bytes': subgraph.total_memory_bytes,
        'arithmetic_intensity': subgraph.arithmetic_intensity,
        'bottleneck': subgraph.bottleneck_type,
        'hardware_allocation': hw_allocation.subgraph_allocations[i],
        'latency_ms': compute_subgraph_latency(subgraph, hw_allocation, mapper),
        'energy_mj': compute_subgraph_energy(subgraph, hw_allocation, mapper),
    }
```

## CLI Interface

### Command-Line Arguments

```bash
# Basic usage
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100

# Custom batch size and precision
./cli/analyze_graph_mapping.py --model mobilenet_v2 --hardware KPU-T256 \
    --batch-size 8 --precision FP16

# Edge device with power budget
./cli/analyze_graph_mapping.py --model resnet18 --hardware Jetson-Orin \
    --thermal-profile 15W

# Output formats
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --output report.txt
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --output report.json
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --output report.md
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --output report.csv

# Detailed mode (show all subgraph nodes)
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --detailed

# Focus on specific subgraph range
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --start 10 --end 20

# Force color output
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --color
```

### Argument Specification

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | Required | Model name (torchvision or synthetic) |
| `--hardware` | str | Required | Hardware target (H100, KPU-T256, etc.) |
| `--batch-size` | int | 1 | Batch size for inference |
| `--precision` | str | FP32 | Numerical precision (FP32, FP16, INT8) |
| `--thermal-profile` | str | default | Power budget (15W, 30W, 60W, default) |
| `--output` | str | None | Output file (format auto-detected) |
| `--detailed` | flag | False | Show all node-level details |
| `--start` | int | 0 | Starting subgraph index |
| `--end` | int | None | Ending subgraph index |
| `--color` | flag | False | Force color output |

## Per-Subgraph Metrics

### Latency Estimation

Using the roofline model with hardware-specific scaling:

```python
def compute_subgraph_latency(subgraph, hw_allocation, mapper):
    """Compute latency for single subgraph using roofline model"""

    # Hardware peak performance
    peak_flops = mapper.resource_model.peak_tflops * 1e12
    peak_bandwidth = mapper.resource_model.memory_bandwidth_gbps * 1e9

    # Roofline model: max of compute-bound and memory-bound
    compute_time = subgraph.total_flops / peak_flops
    memory_time = subgraph.total_memory_bytes / peak_bandwidth
    roofline_latency = max(compute_time, memory_time)

    # Apply hardware-specific utilization scaling
    # (accounts for SM allocation, wave quantization, etc.)
    actual_utilization = hw_allocation.resource_utilization
    scaled_latency = roofline_latency / actual_utilization

    return scaled_latency * 1000  # Convert to ms
```

### Energy Estimation

Using the three-component energy model:

```python
def compute_subgraph_energy(subgraph, hw_allocation, mapper):
    """Compute energy for single subgraph"""

    resource_model = mapper.resource_model

    # Component 1: Compute energy (FLOPs × energy/FLOP)
    compute_energy = (
        subgraph.total_flops * resource_model.energy_per_flop_fp32
    )

    # Component 2: Memory energy (bytes × energy/byte)
    memory_energy = (
        subgraph.total_memory_bytes * resource_model.energy_per_byte_dram
    )

    # Component 3: Leakage energy (static power × time)
    latency_sec = compute_subgraph_latency(subgraph, hw_allocation, mapper) / 1000
    leakage_energy = resource_model.leakage_power_w * latency_sec

    # Component 4: Architecture-specific overhead
    # (instruction fetch, coherence, control logic)
    arch_overhead = mapper.get_architectural_overhead(subgraph, hw_allocation)

    total_energy = (
        compute_energy + memory_energy + leakage_energy + arch_overhead
    )

    return total_energy * 1000  # Convert to mJ
```

## Output Formats

### Text Format (Default)

Three-column layout with:
- Left: FX graph nodes in execution order
- Center: Fused subgraphs with workload characteristics
- Right: Hardware allocation with latency/energy

### JSON Format

Structured data for programmatic access:

```json
{
  "metadata": {
    "model": "resnet18",
    "hardware": "H100",
    "batch_size": 1,
    "precision": "FP32",
    "thermal_profile": "default"
  },
  "summary": {
    "total_fx_nodes": 57,
    "fused_subgraphs": 21,
    "fusion_factor": 2.7,
    "total_flops": 1820000000,
    "total_memory_bytes": 47534080,
    "overall_arithmetic_intensity": 40.2,
    "total_latency_ms": 2.14,
    "total_energy_mj": 28.5,
    "throughput_inferences_per_sec": 467,
    "energy_per_inference_mj": 28.5
  },
  "subgraphs": [
    {
      "subgraph_id": 0,
      "node_ids": ["conv1", "bn1", "relu"],
      "num_operators": 3,
      "flops": 118000000,
      "memory_bytes": 3200000,
      "arithmetic_intensity": 36.9,
      "bottleneck": "COMPUTE_BOUND",
      "hardware_allocation": {
        "sms_allocated": 12,
        "total_sms": 132,
        "sm_utilization_percent": 9.1,
        "warps_per_sm": 16,
        "waves": 1.0,
        "occupancy": 0.5
      },
      "latency_ms": 0.142,
      "energy_mj": 1.83,
      "compute_efficiency_percent": 69.4
    }
  ]
}
```

### CSV Format

For spreadsheet analysis:

```csv
subgraph_id,node_ids,num_ops,flops,memory_bytes,arithmetic_intensity,bottleneck,sms_allocated,occupancy,latency_ms,energy_mj
0,"conv1|bn1|relu",3,118000000,3200000,36.9,COMPUTE_BOUND,12,0.5,0.142,1.83
1,maxpool,1,0,1600000,0.0,MEMORY_BOUND,4,0.25,0.089,0.45
...
```

### Markdown Format

For documentation and reports:

```markdown
# Hardware-Aware Graph Analysis

**Model**: resnet18 (batch=1, FP32)
**Hardware**: H100 SXM5 80GB @ 700W TDP

## Summary

- Total FX Nodes: 57
- Fused Subgraphs: 21
- Fusion Factor: 2.7×

...
```

## Reusable Infrastructure

This tool leverages existing components from the codebase:

### Core Components

| Component | Location | Usage |
|-----------|----------|-------|
| `FusionBasedPartitioner` | `transform/partitioning/fusion_partitioner.py` | Graph partitioning |
| Hardware Mappers | `hardware/mappers/*.py` | Resource allocation |
| `SubgraphDescriptor` | `ir/structures.py` | Subgraph representation |
| `PartitionReport` | `ir/structures.py` | Partition results |
| `RooflineAnalyzer` | `analysis/roofline_analyzer.py` | Latency estimation |
| `EnergyAnalyzer` | `analysis/energy_analyzer.py` | Energy calculation |
| `visualize_partitioning()` | `fusion_partitioner.py` | Two-column template |

### Hardware Mapper Factories

Available hardware targets (20+ configurations):

**GPUs**:
- `create_h100_sxm5_80gb_mapper()` - NVIDIA H100 datacenter
- `create_a100_sxm4_80gb_mapper()` - NVIDIA A100
- `create_jetson_orin_agx_64gb_mapper()` - NVIDIA Jetson Orin AGX
- `create_jetson_orin_nano_8gb_mapper()` - NVIDIA Jetson Orin Nano
- `create_jetson_thor_128gb_mapper()` - NVIDIA Jetson Thor

**CPUs**:
- `create_xeon_8480_mapper()` - Intel Xeon Sapphire Rapids
- `create_epyc_9654_mapper()` - AMD EPYC Genoa
- `create_ampere_one_mapper()` - Ampere One
- `create_i7_12700k_mapper()` - Intel Core i7

**Accelerators**:
- `create_tpu_v4_mapper()` - Google TPU v4
- `create_tpu_edge_pro_mapper()` - Google Coral Edge TPU
- `create_kpu_t256_mapper()` - Kendryte KPU T256
- `create_kpu_t768_mapper()` - Kendryte KPU T768
- `create_dpu_xilinx_vck5000_mapper()` - Xilinx VCK5000 DPU
- `create_cgra_plasticine_mapper()` - CGRA (Plasticine-style)
- `create_hailo8_mapper()` - Hailo-8 NPU

**DSPs**:
- `create_hexagon_v73_mapper()` - Qualcomm Hexagon V73
- `create_ti_c7x_mapper()` - Texas Instruments C7x

## Implementation Plan

### Phase 1: Core Structure
1. Create `cli/analyze_graph_mapping.py` with argument parser
2. Implement `GraphMappingAnalyzer` class for orchestration
3. Add model loading and FX tracing logic

### Phase 2: Analysis Functions
4. Implement `compute_subgraph_latency()` helper
5. Implement `compute_subgraph_energy()` helper
6. Create `SubgraphMetrics` dataclass for results

### Phase 3: Visualization
7. Implement three-column text formatter
8. Extend existing `visualize_partitioning()` format
9. Add summary statistics section

### Phase 4: Output Formats
10. Add JSON output support
11. Add CSV output support
12. Add Markdown output support

### Phase 5: Testing & Documentation
13. Test with ResNet18 on H100
14. Test with MobileNetV2 on KPU-T256
15. Test with MLP on TPU Edge Pro
16. Document in `cli/README.md`

## Comparison with Existing Tools

### vs. `compare_architectures_energy.py`

| Feature | compare_architectures_energy.py | analyze_graph_mapping.py (NEW) |
|---------|--------------------------------|-------------------------------|
| **Models** | MLP-only | Any PyTorch model |
| **Visualization** | Hierarchical energy breakdown | Three-column graph→subgraph→hardware |
| **Focus** | Architecture comparison (4 archs) | Single hardware deep-dive |
| **Per-subgraph metrics** | ❌ No | ✅ Yes (latency + energy) |
| **Hardware mapping** | ✅ Yes (via map_graph) | ✅ Enhanced with allocation details |
| **Graph partitioning** | ✅ Yes (FusionBasedPartitioner) | ✅ Yes (same) |
| **Output formats** | Text, JSON, CSV | Text, JSON, CSV, Markdown |

### vs. `analyze_comprehensive.py`

| Feature | analyze_comprehensive.py | analyze_graph_mapping.py (NEW) |
|---------|-------------------------|-------------------------------|
| **Scope** | Complete model analysis | Subgraph-level analysis |
| **Reports** | Roofline, energy, memory, concurrency | Per-subgraph latency + energy |
| **Visualization** | Summary tables | Three-column graph mapping |
| **Hardware details** | Overall allocation | Per-subgraph allocation |
| **Use case** | Model comparison | Hardware mapping debugging |

## Example Use Cases

### Use Case 1: Debugging Hardware Utilization
```bash
# Why is my ResNet18 only using 15% of my H100?
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --detailed
```

**Answer**: The visualization shows that batch=1 creates small subgraphs that don't saturate the 132 SMs. Most subgraphs only need 4-12 SMs, leading to low utilization.

### Use Case 2: Edge Device Power Budget
```bash
# Can I run MobileNetV2 at 15W on Jetson Orin?
./cli/analyze_graph_mapping.py --model mobilenet_v2 --hardware Jetson-Orin \
    --thermal-profile 15W --output analysis.json
```

**Answer**: The JSON output shows total energy per inference. Multiply by inference rate to get average power. Compare to 15W budget.

### Use Case 3: Accelerator Comparison
```bash
# Which subgraphs benefit from KPU's tile engine?
./cli/analyze_graph_mapping.py --model resnet18 --hardware KPU-T256 \
    --precision INT8 --output kpu.txt
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 \
    --precision INT8 --output gpu.txt
diff kpu.txt gpu.txt
```

**Answer**: Side-by-side comparison shows which subgraphs run faster on KPU (large convolutions with good tile reuse) vs GPU (everything else).

### Use Case 4: Batch Size Scaling
```bash
# How does latency scale with batch size?
for bs in 1 4 8 16 32; do
    ./cli/analyze_graph_mapping.py --model resnet50 --hardware H100 \
        --batch-size $bs --output resnet50_bs${bs}.json
done
python analyze_batch_scaling.py resnet50_bs*.json
```

**Answer**: Per-subgraph latency shows which layers become memory-bound at higher batch sizes.

## Success Criteria

1. ✅ Works with any torchvision model
2. ✅ Supports all 20+ hardware mappers
3. ✅ Produces three-column visualization
4. ✅ Computes per-subgraph latency and energy
5. ✅ Outputs to text, JSON, CSV, and Markdown
6. ✅ Handles batch size 1-64
7. ✅ Supports FP32, FP16, INT8 precisions
8. ✅ Documented in cli/README.md

## Future Enhancements

### Phase 2 Enhancements
- Interactive mode with subgraph selection
- Graphviz DOT output with hardware annotations
- Comparison mode (multiple hardware targets side-by-side)
- Bottleneck mitigation suggestions

### Phase 3 Enhancements
- Custom model support (user-provided PyTorch modules)
- Multi-GPU mapping (model parallelism)
- Pipeline parallelism visualization
- Tensor parallelism support

### Phase 4 Enhancements
- Web UI for visualization
- Time-series analysis (memory timeline, SM utilization over time)
- Hardware recommendation engine
- Cost analysis (cloud pricing integration)

## References

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [cli/README.md](../../cli/README.md) - CLI tools documentation
- [docs/unified_framework_api.md](../unified_framework_api.md) - UnifiedAnalyzer API
- [docs/hardware/architecture_taxonomy.md](../hardware/architecture_taxonomy.md) - Hardware execution models
- [src/graphs/transform/partitioning/fusion_partitioner.py](../../src/graphs/transform/partitioning/fusion_partitioner.py) - Partitioning implementation
- [src/graphs/hardware/mappers/](../../src/graphs/hardware/mappers/) - Hardware mapper implementations
