# Architecture-Specific Energy CLI Tools - Session Summary

**Date**: 2025-11-13
**Session**: Building CLI tools for detailed DNN energy analysis per architecture

## Objective

Create 4 CLI tools for architecture-specific energy analysis:
1. `cli/analyze_cpu_energy.py` - CPU energy breakdown
2. `cli/analyze_gpu_energy.py` - GPU energy breakdown
3. `cli/analyze_tpu_energy.py` - TPU energy breakdown
4. `cli/analyze_kpu_energy.py` - KPU energy breakdown

Each tool analyzes DNN models on specific hardware with detailed hierarchical energy breakdowns.

---

## Progress Summary

### ✅ Completed

1. **Shared Model Factory** (`cli/model_factory.py`)
   - Loads 30+ built-in torchvision models (ResNet, MobileNet, EfficientNet, ViT, etc.)
   - Supports custom PyTorch models from file paths
   - Automatic Dynamo tracing and shape propagation
   - Fusion-based partitioning
   - Proper error handling and input shape management
   - **Status**: Tested and working

2. **Basic CPU Analysis Pipeline** (`cli/analyze_cpu_energy.py`)
   - 9 CPU configurations (Jetson Orin, Intel Xeon, AMD EPYC, Ampere)
   - Model loading → tracing → partitioning → mapping → metrics
   - Command-line interface with `--list-cpus`, `--list-models`
   - Basic metrics: latency, throughput, total energy, utilization
   - **Status**: Functional, tested on multiple CPUs and models

3. **CPU Hierarchical Energy Breakdown**
   - ✅ Shared utilities created (`cli/energy_breakdown_utils.py`)
   - ✅ Integrated into `analyze_cpu_energy.py`
   - ✅ Tested with multiple CPUs and models
   - **Status**: Complete and working

---

## What We Learned: Architectural Energy System

### Architecture Overview

The energy analysis system has multiple layers:

```
User Tool (analyze_cpu_energy.py)
    ↓
CPUMapper.map_graph()
    ↓
GraphHardwareAllocation (basic metrics: latency, total_energy)
    ↓ [architectural details lost here]
StoredProgramEnergyModel.compute_architectural_energy()
    ↓
ArchitecturalEnergyBreakdown (detailed events in extra_details)
```

### Key Components

#### 1. StoredProgramEnergyModel
**Location**: `src/graphs/hardware/architectural_energy.py`

**Purpose**: Models CPU-specific energy events

**Energy Events Tracked**:
```python
extra_details = {
    # Instruction Pipeline
    'instruction_fetch_energy': float,
    'instruction_decode_energy': float,
    'instruction_dispatch_energy': float,
    'num_instructions': int,

    # Register File
    'register_read_energy': float,
    'register_write_energy': float,
    'num_register_reads': int,
    'num_register_writes': int,

    # Memory Hierarchy (4-stage)
    'l1_cache_energy': float,
    'l2_cache_energy': float,
    'l3_cache_energy': float,
    'dram_energy': float,
    'l1_accesses': int,
    'l2_accesses': int,
    'l3_accesses': int,
    'dram_accesses': int,

    # ALU Operations
    'alu_energy': float,

    # Branch Prediction
    'branch_energy': float,
    'num_branches': int,
    'num_mispredicted_branches': int,
    'branch_prediction_success_rate': float,
}
```

#### 2. Integration Pattern

**How CPUs Get Architectural Energy**:
```python
# In create_*_cpu_mapper() functions
model.architecture_energy_model = StoredProgramEnergyModel(
    instruction_fetch_energy=2.0e-12,
    instruction_decode_energy=0.9e-12,
    # ... other parameters
)

# During mapping (in HardwareMapper base class)
if self.resource_model.architecture_energy_model:
    arch_breakdown = self.resource_model.architecture_energy_model.compute_architectural_energy(
        ops=ops,
        bytes_transferred=bytes_transferred,
        compute_energy_baseline=compute_energy,
        memory_energy_baseline=memory_energy,
        execution_context=execution_context
    )
    # Returns ArchitecturalEnergyBreakdown with extra_details
```

#### 3. Current Challenge

**Problem**: `GraphHardwareAllocation` doesn't store architectural breakdowns

```python
# What we get from map_graph()
class GraphHardwareAllocation:
    total_latency: float
    total_energy: float  # Just a number
    peak_compute_units_used: int
    average_utilization: float
    # ... no arch_breakdown field!
```

**What we need**: Access to per-subgraph `ArchitecturalEnergyBreakdown` objects

#### 4. Workaround Used in compare_architectures_energy.py

**Current approach** (lines 300-415):
1. Manually create MLP models (not using real DNNs)
2. Call mapper with execution_context
3. Extract arch_breakdown from return value
4. Store in custom `ArchitecturalEnergyBreakdown` dataclass
5. Print with `_print_cpu_hierarchical_breakdown()`

**Limitation**: Only works for simple single-layer models, not real DNNs

---

## Integration Challenges

### Challenge 1: GraphHardwareAllocation Doesn't Store Breakdowns

The `map_graph()` method aggregates energy but doesn't preserve per-subgraph architectural breakdowns.

**Potential Solutions**:
1. **Modify GraphHardwareAllocation** to include `List[ArchitecturalEnergyBreakdown]`
2. **Post-process**: Re-run energy calculation on each subgraph to extract breakdowns
3. **Aggregate**: Sum arch events across all subgraphs (lose per-layer granularity)

### Challenge 2: Not All CPUs Have Architectural Models

Currently only these CPUs have `StoredProgramEnergyModel` attached:
- ✅ Intel Xeon Platinum 8490H (Emerald Rapids)
- ✅ Jetson Orin AGX CPU
- ❌ AMD EPYC (missing)
- ❌ Ampere (missing)
- ❌ Others (missing)

**Solution**: Add models for missing CPUs or provide sensible defaults

### Challenge 3: Hierarchical Print Logic is Coupled

The `_print_cpu_hierarchical_breakdown()` function in `compare_architectures_energy.py` is:
- 95 lines long
- Tightly coupled to `ArchitecturalEnergyBreakdown` dataclass (different from `GraphHardwareAllocation`)
- Has hardcoded assumptions about event names

**Solution**: Extract to shared utility module

---

## Files Created

### 1. cli/model_factory.py (312 lines)

**Purpose**: Unified model loading for all 4 CLI tools

**Key Functions**:
```python
load_and_prepare_model(
    model_name: str,
    model_path: str,
    model_class: str,
    batch_size: int,
    verbose: bool
) -> Tuple[nn.Module, GraphModule, PartitionReport, input_shape]
```

**Features**:
- 30+ built-in models
- Custom model support
- Automatic tracing and partitioning
- Error handling

**Usage**:
```bash
python cli/model_factory.py --list
python cli/model_factory.py --model resnet18 --batch-size 4
```

### 2. cli/analyze_cpu_energy.py (310 lines)

**Purpose**: CPU-specific energy analysis

**Key Features**:
- 9 CPU configurations
- Load → Trace → Partition → Map → Analyze
- Basic metrics output
- Ready for hierarchical breakdown integration

**Usage**:
```bash
# List available CPUs
python cli/analyze_cpu_energy.py --list-cpus

# Analyze model
python cli/analyze_cpu_energy.py --cpu jetson_orin_agx_cpu --model mobilenet_v2
python cli/analyze_cpu_energy.py --cpu epyc_genoa --model resnet18 --batch-size 8
```

**Current Output**:
```
================================================================================
CPU ENERGY BREAKDOWN ANALYSIS
================================================================================
CPU: ARM Cortex-A78AE (12 cores, 8nm, 30W)
Model: mobilenet_v2
Batch Size: 1
Precision: FP32
================================================================================

[1/4] Loading and tracing model...
✓ Model loaded: 66 subgraphs

[2/4] Creating CPU mapper for jetson_orin_agx_cpu...
✓ CPU mapper created: Jetson-Orin-AGX-CPU
  Cores: 12
  Peak Performance (FP32): 0.21 TFLOPS

[3/4] Mapping model to CPU...
✓ Model mapped
  Peak cores used: 12 / 12
  Avg utilization: 94.6%
  Estimated latency: 1.49 ms
  Total energy: 36.39 mJ

[4/4] Energy Breakdown
================================================================================

Total Energy: 36.389 mJ
Energy per Inference: 36.389 mJ
Latency: 1.49 ms
Throughput: 669.67 inferences/sec

================================================================================
Analysis complete!
```

---

## Phase 1A: CPU Hierarchical Breakdown Integration (COMPLETED)

### Approach Taken

**Decision**: Implemented Option C (Aggregate events) - manual architectural energy calculation

**Why this approach**:
- Simplest to implement without modifying core data structures
- Reuses existing `compute_architectural_energy()` method
- Provides complete breakdown for the entire model
- Can be applied to other architectures (GPU, TPU, KPU) using same pattern

**Implementation**:
1. Created `cli/energy_breakdown_utils.py`:
   - `print_cpu_hierarchical_breakdown()` - Extracted from compare_architectures_energy.py
   - `aggregate_subgraph_events()` - For future per-subgraph aggregation (if needed)
   - Clean separation of printing logic from analysis logic

2. Modified `cli/analyze_cpu_energy.py` (lines 293-359):
   - After mapping completes, aggregate ops and bytes across all subgraphs
   - Call `architecture_energy_model.compute_architectural_energy()` with aggregate values
   - Extract `extra_details` dict containing architectural events
   - Call `print_cpu_hierarchical_breakdown()` with events
   - Graceful fallback for CPUs without architectural models

### Test Results

**Test 1**: Jetson Orin AGX CPU + MobileNetV2 (batch 1)
- ✅ Hierarchical breakdown printed successfully
- Total energy: 36.4 mJ
- Architectural overhead: 6.8 mJ (18.7% of total)
- Breakdown: Pipeline (0.9 mJ), Register File (2.6 mJ), Memory (2.6 mJ), ALU (0.7 mJ), Branch (0.003 mJ)

**Test 2**: Intel Xeon Emerald Rapids + ResNet18 (batch 4)
- ✅ Hierarchical breakdown printed successfully
- Total energy: 351.7 mJ
- Architectural overhead: 138.3 mJ (39.3% of total)
- Scaled correctly with larger model and batch size

**Test 3**: AMD EPYC Genoa + MobileNetV2 (batch 1)
- ✅ Fallback works correctly (no architectural model)
- Shows basic energy metrics with explanation

### Files Created/Modified

**New Files**:
1. `cli/energy_breakdown_utils.py` (178 lines)
   - Reusable energy breakdown printing utilities
   - Can be extended for GPU, TPU, KPU in future phases

**Modified Files**:
1. `cli/analyze_cpu_energy.py` (lines 293-359)
   - Added architectural energy extraction
   - Integrated hierarchical breakdown printing
   - Graceful fallback for CPUs without models

### Limitations and Future Work

**Known Limitations**:
- Only CPUs with `StoredProgramEnergyModel` configured get detailed breakdown
- AMD EPYC and Ampere CPUs missing architectural models (can be added later)
- Aggregate energy only (not per-subgraph)
- No JSON/CSV export yet

**Future Enhancements**:
- Add StoredProgramEnergyModel to remaining CPUs (AMD EPYC, Ampere)
- Implement JSON/CSV export with hierarchical events
- Optional per-subgraph breakdown (if needed)

---

## Next Steps (Remaining Work)

### Phase 1B: CPU Tool - JSON/CSV Export (Optional)

**Tasks**:
1. Create output formatters in `energy_breakdown_utils.py`
2. Support `--output file.json` and `--output file.csv` in `analyze_cpu_energy.py`
3. Include all hierarchical events in exports
4. Test with various output formats

**Time Estimate**: 1-2 hours

### Phase 1C: Add Missing CPU Architectural Models (Optional)

**CPUs Missing StoredProgramEnergyModel**:
- AMD EPYC 9654 (Genoa)
- AMD EPYC 9754 (Bergamo)
- AMD EPYC Turin
- Ampere One M192
- Ampere Altra Max M128

**Tasks**:
1. Create StoredProgramEnergyModel instances for each
2. Configure energy parameters based on process node
3. Attach to resource models in mapper factory functions
4. Test hierarchical breakdown with each

**Time Estimate**: 2-3 hours

### Phase 2: GPU Energy Analysis Tool

**Tasks**:
1. Create `cli/analyze_gpu_energy.py` using same pattern as CPU tool
2. Extract GPU hierarchical breakdown function to `energy_breakdown_utils.py`
3. Integrate DataParallelEnergyModel (already exists)
4. Test with H100, Jetson, A100

**GPU-Specific Events**:
- Tensor Core / CUDA Core operations
- Register file access
- Shared Memory / L1 unified cache
- L2 cache, DRAM
- SIMT control (warp divergence, coalescing, barriers, coherence)

**Time Estimate**: 2-3 hours (already have blueprint from CPU)

### Phase 3: TPU Energy Analysis Tool

**Tasks**:
1. Create `cli/analyze_tpu_energy.py`
2. Extract TPU hierarchical breakdown function
3. Integrate SystolicArrayEnergyModel
4. Test with TPU v4 configurations

**TPU-Specific Events**:
- Systolic array operations
- Vector/scalar units
- Weight FIFO
- Unified buffer
- Control unit

**Time Estimate**: 2-3 hours

### Phase 4: KPU Energy Analysis Tool

**Tasks**:
1. Create `cli/analyze_kpu_energy.py`
2. Extract KPU hierarchical breakdown function
3. Integrate DomainFlowEnergyModel
4. Test with KPU-T256 and other KPU variants

**KPU-Specific Events**:
- Tile compute energy
- Tile memory (local SRAM)
- Shared memory
- Network-on-Chip (NoC)
- Control and synchronization

**Time Estimate**: 2-3 hours

---

## Architecture-Specific Event Summary

From `compare_architectures_energy.py`, we know each architecture tracks:

### CPU (Stored-Program)
- 5 categories: Pipeline, Register File, Memory (4-stage), ALU, Branch Prediction
- 20+ individual events

### GPU (Data-Parallel SIMT)
- 7 categories: Compute Units, Register File, Pipeline, Memory (3-stage), SIMT Control
- 30+ individual events
- GPU-specific: warp divergence, memory coalescing, barriers, coherence

### TPU (Systolic Array)
- 6 categories: Systolic Array, Vector/Scalar Units, Buffers, Control
- 20+ individual events
- TPU-specific: systolic array operations, weight FIFO, unified buffer

### KPU (Domain-Flow)
- 8 categories: Tile Compute, Tile Memory, Shared Memory, NoC, Control
- 25+ individual events
- KPU-specific: tile-to-tile routing, token injection/extraction, barriers

---

## Code Quality Notes

### Good Patterns Established
1. ✅ Incremental development (foundation first)
2. ✅ Shared utilities (model_factory.py)
3. ✅ Clear separation of concerns
4. ✅ Comprehensive error handling
5. ✅ Consistent CLI interface

### Technical Debt to Address
1. ⚠️ Architectural breakdown storage (not in GraphHardwareAllocation)
2. ⚠️ Missing StoredProgramEnergyModel for some CPUs
3. ⚠️ Print logic tightly coupled to compare_architectures_energy.py
4. ⚠️ Need shared formatting utilities for all 4 tools

---

## Testing Status

### Model Factory
- ✅ Tested: resnet18, mobilenet_v2, efficientnet_b4
- ✅ Batch sizes: 1, 4, 8
- ✅ Error handling: invalid model names
- ✅ Custom input shapes: efficientnet_b4 (380x380)

### CPU Analysis Tool
- ✅ Tested CPUs: jetson_orin_agx_cpu, epyc_genoa
- ✅ Tested models: mobilenet_v2, resnet18
- ✅ Batch sizes: 1, 8
- ✅ List functionality: --list-cpus, --list-models
- ⏳ Pending: Hierarchical breakdown
- ⏳ Pending: JSON/CSV export
- ⏳ Pending: All 9 CPUs

---

## References

### Key Files
- `src/graphs/hardware/architectural_energy.py` - Energy models (StoredProgramEnergyModel, DataParallelEnergyModel, etc.)
- `src/graphs/hardware/resource_model.py` - HardwareResourceModel, HardwareMapper base
- `src/graphs/hardware/mappers/cpu.py` - CPUMapper implementation
- `cli/compare_architectures_energy.py` - Reference implementation for hierarchical printing

### Hierarchical Print Functions (to extract)
- `_print_cpu_hierarchical_breakdown()` - Lines 1266-1360 (95 lines)
- `_print_gpu_hierarchical_breakdown()` - Lines 898-988 (91 lines)
- `_print_tpu_hierarchical_breakdown()` - Lines 990-1128 (139 lines)
- `_print_kpu_hierarchical_breakdown()` - Lines 1130-1264 (135 lines)

---

## Decisions Made

### 1. Integration Approach
**Decision**: Aggregate events (Option C)
- Manual call to `architecture_energy_model.compute_architectural_energy()` after mapping
- Doesn't modify core data structures
- Provides complete model-level breakdown
- Can be replicated for GPU/TPU/KPU

### 2. Missing CPU Models
**Decision**: Graceful fallback, add models as needed
- CPUs without StoredProgramEnergyModel show basic metrics with explanation
- Can add models for AMD EPYC and Ampere later (Phase 1C)

### 3. Shared Utilities
**Decision**: Extract to `cli/energy_breakdown_utils.py`
- Reusable across all 4 CLI tools
- Clean separation of concerns
- Easy to extend for GPU/TPU/KPU

### 4. Granularity
**Decision**: Aggregate only (model-level)
- Simpler implementation
- Sufficient for most use cases
- Can add per-subgraph detail later if needed

---

## Session Statistics

### Phase 1 (Initial Session - 2025-11-13 Morning)
- **Time**: ~2 hours
- **Files created**: 2 (model_factory.py, analyze_cpu_energy.py)
- **Lines of code**: ~622 lines
- **Tests run**: 6+ manual tests
- **Models tested**: 3 (resnet18, mobilenet_v2, efficientnet_b4)
- **CPUs tested**: 2 (Jetson Orin AGX, AMD EPYC Genoa)

### Phase 1A (Hierarchical Breakdown - 2025-11-13 Afternoon)
- **Time**: ~1.5 hours
- **Files created**: 1 (energy_breakdown_utils.py, 178 lines)
- **Files modified**: 1 (analyze_cpu_energy.py, +67 lines)
- **Tests run**: 3 (Jetson Orin + MobileNetV2, Xeon + ResNet18, EPYC + MobileNetV2)
- **Features added**:
  - Hierarchical energy breakdown printing
  - Architectural event extraction
  - Graceful fallback for CPUs without models

### Combined Statistics
- **Total time**: ~3.5 hours
- **Total files created**: 3 (800+ lines)
- **Total tests**: 9+
- **Phase 1A Status**: ✅ COMPLETE

---

**Next Session**: Phase 1B (JSON/CSV export) or Phase 2 (GPU tool)
