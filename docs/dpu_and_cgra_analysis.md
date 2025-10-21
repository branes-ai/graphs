# DPU & CGRA Architecture Analysis

**Date**: 2025-10-21
**Purpose**: Analyze how to add Deep Processing Units (DPU) and Coarse-Grained Reconfigurable Architectures (CGRA) to our hardware modeling framework

---

## Executive Summary

Both DPU and CGRA represent fundamentally different architectural paradigms that will extend our modeling capabilities:

- **DPU (e.g., Xilinx Vitis AI)**: 2D array of integer ALUs, similar to TPU but FPGA-based and INT8-focused
- **CGRA (e.g., UT Austin TRIPS, Stanford Plasticine)**: Spatially programmable fabric with reconfigurable dataflow

**Key Finding**: Our current framework can support both with new mapper classes, but CGRA will require the most significant extensions due to spatial mapping and reconfiguration overhead.

---

## 1. Deep Processing Unit (DPU) Analysis

### 1.1 Architecture Characteristics

**Xilinx Vitis AI DPU Example**:
```
┌─────────────────────────────────────┐
│  2D ALU Array (e.g., 64×64 = 4096)  │
│  ┌──┬──┬──┬──┐                      │
│  │PE│PE│PE│PE│ ... INT8 ALUs        │
│  ├──┼──┼──┼──┤                      │
│  │PE│PE│PE│PE│ ...                  │
│  └──┴──┴──┴──┘                      │
├─────────────────────────────────────┤
│  Scratchpad Memory (e.g., 512KB)    │
├─────────────────────────────────────┤
│  Instruction Sequencer              │
└─────────────────────────────────────┘
```

**Key Properties**:
| Property | Value | Notes |
|----------|-------|-------|
| **Compute Units** | 2D array (e.g., 64×64) | Configurable size |
| **Precision** | INT8 native, INT16 accumulator | Fixed-point only |
| **Memory** | On-chip scratchpad (128KB-1MB) | Similar to KPU |
| **Operations** | Conv, Pool, ElementWise, Concat | Limited op set |
| **Performance** | ~1-4 TOPS (INT8) | Depends on array size |
| **Power** | 2-10W | Much lower than GPU |
| **Reconfigurability** | FPGA-based | Can reprogram |
| **Pipeline Depth** | 10-50 cycles | Operation dependent |

### 1.2 Similarities to Existing Architectures

**Like KPU**:
- Tile-based processing (due to scratchpad constraints)
- Limited on-chip memory forces tiling
- Optimized for quantized inference

**Like TPU**:
- 2D array structure
- Matrix operations use full array
- Pipeline execution model

**Like CPU**:
- Integer arithmetic (INT8/INT16)
- Instruction-based (not data-driven)

### 1.3 Unique Characteristics

1. **FPGA Reconfigurability**:
   - Can change array size/topology
   - Reconfiguration time: ~10ms-1s
   - Trade-off: flexibility vs fixed hardware

2. **Fixed-Point Arithmetic**:
   - No floating-point support
   - Quantization is mandatory, not optional
   - INT8 is native precision (no overhead)

3. **Instruction Set Architecture**:
   - Specialized DNN instructions (CONV, POOL, etc.)
   - Not Turing-complete
   - Limited to supported operations

4. **Low Power**:
   - 5-10× more energy efficient than GPU
   - Suitable for edge/embedded deployment
   - Similar to KPU in power profile

### 1.4 Mapping Strategy for DPU

**Algorithm**:
```python
def map_subgraph_to_dpu(subgraph):
    # 1. Check if operation is supported
    if subgraph.op not in ['conv2d', 'pool', 'eltwise']:
        return fallback_to_cpu()

    # 2. Analyze 2D array utilization
    array_utilization = calculate_array_usage(
        op_size=subgraph.output_shape,
        array_size=(64, 64)  # DPU array dimensions
    )

    # 3. Calculate tiling (like KPU)
    tiles = calculate_tiles(
        data_size=subgraph.total_bytes,
        scratchpad_size=512 * 1024  # 512KB
    )

    # 4. Calculate pipeline cycles
    pipeline_cycles = estimate_pipeline_depth(subgraph.op)
    compute_cycles = (subgraph.total_ops / array_size) + pipeline_cycles

    # 5. Calculate latency
    clock_freq = 300e6  # 300 MHz typical
    latency = compute_cycles / clock_freq

    return latency, array_utilization
```

**Key Metrics**:
- **Array Utilization**: % of 2D array actively computing
- **Tiling Overhead**: Scratchpad constraints force tiling
- **Instruction Pipeline**: Fixed overhead per operation
- **INT8 Native**: No quantization overhead

### 1.5 DPU Mapper Implementation Plan

**New Class**: `DPUMapper(HardwareMapper)`

**Resources**:
```python
@dataclass
class DPUResourceModel(HardwareResourceModel):
    hardware_type: HardwareType.DPU
    array_rows: int = 64
    array_cols: int = 64
    scratchpad_size: int = 512 * 1024  # 512KB
    supported_ops: List[str] = ['conv2d', 'pool2d', 'eltwise', 'concat']
    pipeline_depth: Dict[str, int] = {
        'conv2d': 30,
        'pool2d': 10,
        'eltwise': 5,
    }
    native_precision: Precision = Precision.INT8
    accumulator_precision: Precision = Precision.INT16
```

**Complexity**: Medium
- Similar to KPU (tiling) + TPU (2D array)
- ~300-400 lines of code

---

## 2. Coarse-Grained Reconfigurable Architecture (CGRA) Analysis

### 2.1 Architecture Characteristics

**Stanford Plasticine / UT Austin TRIPS Example**:
```
┌─────────────────────────────────────────────┐
│  PE Grid (e.g., 8×8 = 64 Processing Elements) │
│  ┌────┬────┬────┬────┐                      │
│  │PE  │PE  │PE  │PE  │ ... ALU + Register   │
│  │ ↓→ │ ↓→ │ ↓→ │ ↓→ │     File             │
│  ├────┼────┼────┼────┤                      │
│  │PE  │PE  │PE  │PE  │ ...                  │
│  │ ↓→ │ ↓→ │ ↓→ │ ↓→ │                      │
│  └────┴────┴────┴────┘                      │
│  Reconfigurable Interconnect (routing)      │
├─────────────────────────────────────────────┤
│  Configuration Memory (stores PE configs)   │
│  Context 1: Conv  Context 2: ReLU  ...      │
└─────────────────────────────────────────────┘
```

**Key Properties**:
| Property | Value | Notes |
|----------|-------|-------|
| **PE Grid** | 8×8 to 16×16 typical | Limited resources |
| **PE Capabilities** | ALU, Reg File, Local Mem | RISC-like |
| **Interconnect** | Reconfigurable mesh/torus | Routing constraints |
| **Precision** | FP32, INT8, INT16 | Depends on PE design |
| **Configuration** | Spatial dataflow graph | Entire subgraph mapped |
| **Reconfiguration Time** | 10-1000 cycles | Context switch cost |
| **Performance** | 10-100 GFLOPS | Depends on grid size |
| **Power** | 0.5-5W | Very energy efficient |
| **Programming Model** | Spatial (not temporal) | Fundamentally different |

### 2.2 Fundamentally Different Execution Model

**Traditional (GPU/CPU/TPU)**:
```
Instructions → Execute → Store Results
(Temporal execution, instruction-driven)
```

**CGRA (Spatial)**:
```
Configure Fabric → Stream Data → Results Flow Out
(Spatial execution, dataflow-driven)
```

**Key Difference**:
- Traditional: Operations executed sequentially by fetching instructions
- CGRA: Operations "baked into" fabric, data flows through configured circuit

### 2.3 Unique Characteristics

1. **Spatial Mapping**:
   - Entire computation graph mapped to PE grid
   - Each PE performs one operation
   - Data flows through interconnect

2. **Reconfiguration Overhead**:
   - Switching between operations requires reconfiguration
   - Cost: 10-1000 cycles (vs 1 cycle for instruction fetch)
   - Must minimize context switches

3. **Routing Constraints**:
   - Limited interconnect bandwidth
   - Not all PE pairs can communicate directly
   - Place-and-route problem (NP-hard!)

4. **Resource Constraints**:
   - Limited PEs (64-256 typical)
   - Must fit subgraph onto grid
   - If doesn't fit, must partition

5. **Context Switching**:
   - Multiple configurations stored on-chip
   - Fast context switch (10-100 cycles)
   - But limited contexts (4-16 typical)

### 2.4 Mapping Strategy for CGRA

**Algorithm** (much more complex than others):
```python
def map_subgraph_to_cgra(subgraph):
    # 1. SPATIAL ALLOCATION - map ops to PEs
    pe_mapping = spatial_place_and_route(
        subgraph_ops=subgraph.operations,
        pe_grid_size=(8, 8),
        routing_topology='mesh'
    )

    if pe_mapping is None:
        # Subgraph too large, must partition
        sub_subgraphs = partition_for_cgra(subgraph, max_pes=64)
        return map_partitioned_subgraphs(sub_subgraphs)

    # 2. ROUTING ANALYSIS - can data flow?
    routing_feasible, routing_hops = analyze_routing(
        pe_mapping, interconnect_topology='mesh'
    )

    if not routing_feasible:
        # Routing infeasible, must remap
        pe_mapping = remap_with_routing_constraints(pe_mapping)

    # 3. RECONFIGURATION COST
    config_id = check_if_config_exists(subgraph)
    if config_id is None:
        # Need new configuration
        reconfig_cost = reconfiguration_latency  # 100 cycles
        contexts_used += 1
    else:
        # Can reuse existing configuration
        reconfig_cost = context_switch_latency  # 10 cycles

    # 4. DATAFLOW EXECUTION
    pipeline_depth = max(routing_hops.values())  # Critical path
    compute_latency = pipeline_depth + subgraph.total_ops / num_active_pes

    # 5. TOTAL LATENCY
    total_latency = reconfig_cost + compute_latency

    return total_latency, pe_mapping, routing_hops
```

**Key Challenges**:
1. **Spatial place-and-route**: NP-hard problem
2. **Routing constraints**: Limited interconnect
3. **Configuration management**: Limited context slots
4. **Partitioning**: May need to split subgraphs

### 2.5 CGRA Mapper Implementation Plan

**New Class**: `CGRAMapper(HardwareMapper)`

**Resources**:
```python
@dataclass
class CGRAResourceModel(HardwareResourceModel):
    hardware_type: HardwareType.CGRA
    pe_grid_rows: int = 8
    pe_grid_cols: int = 8
    pe_operations: List[str] = ['add', 'mul', 'relu', 'max']
    interconnect_topology: str = 'mesh'  # or 'torus'
    max_routing_hops: int = 4
    num_contexts: int = 8  # Stored configurations
    reconfiguration_cycles: int = 100
    context_switch_cycles: int = 10
    register_file_size: int = 32  # per PE
```

**Additional Complexity**:
- **Place-and-route algorithm**: Spatial allocation
- **Routing analysis**: Check interconnect feasibility
- **Configuration management**: Track used contexts
- **Partitioning**: Split large subgraphs

**Complexity**: High
- Most complex mapper yet
- Requires spatial algorithms (place-and-route)
- ~500-700 lines of code

---

## 3. Integration Strategy

### 3.1 Extended Architecture Hierarchy

```python
from enum import Enum

class HardwareType(Enum):
    GPU = "gpu"      # NVIDIA H100, etc.
    CPU = "cpu"      # Intel/AMD x86
    TPU = "tpu"      # Google TPU v4
    KPU = "kpu"      # Edge accelerators
    DPU = "dpu"      # Xilinx Vitis AI (NEW)
    CGRA = "cgra"    # Plasticine, TRIPS (NEW)
```

### 3.2 Mapper Class Hierarchy

```
HardwareMapper (base)
├── GPUMapper           (SM-based, wave quantization)
├── CPUMapper           (core-based, SIMD)
├── KPUMapper           (tile-based, scratchpad)
├── TPUMapper           (systolic array)
├── DPUMapper           (2D ALU array, INT8) [NEW]
└── CGRAMapper          (spatial fabric, reconfigurable) [NEW]
```

### 3.3 Common Abstractions

All mappers share:
- `map_subgraph()` - Map single fused subgraph
- `map_graph()` - Map entire computation graph
- `_calculate_latency()` - Roofline model
- `_calculate_energy()` - Energy estimation

**DPU-specific additions**:
- `_analyze_array_utilization()` - 2D array usage
- `_calculate_tiling()` - Scratchpad constraints
- `_check_operation_support()` - DPU instruction set

**CGRA-specific additions**:
- `_spatial_place_and_route()` - Map ops to PE grid
- `_analyze_routing()` - Check interconnect feasibility
- `_estimate_reconfiguration_cost()` - Context management
- `_partition_subgraph()` - Split large subgraphs

### 3.4 Implementation Priority

**Phase 1: DPU Mapper** (Recommended Next)
- Complexity: Medium
- Reuses concepts from KPU + TPU
- ~300-400 lines of code
- 2-3 days implementation + testing

**Phase 2: CGRA Mapper** (After DPU)
- Complexity: High
- Requires new spatial algorithms
- ~500-700 lines of code
- 4-5 days implementation + testing

---

## 4. Expected Results

### 4.1 DPU Performance Estimates

**Xilinx Vitis AI DPU (4096 INT8 ALUs @ 300 MHz)**:
```
ResNet-18 (INT8, Batch=1):
- Latency: ~2-5 ms
- Throughput: ~200-500 inferences/sec
- Energy: ~0.01 J/inference
- Speedup vs CPU: ~5-10×
- Position: Between KPU and CPU
```

**Strengths**:
- Very power efficient (FPGA-based)
- INT8 native (no quantization overhead)
- Reconfigurable (can optimize for specific models)

**Weaknesses**:
- Slower than GPU/TPU (smaller array)
- Limited to supported operations
- Reconfiguration overhead

### 4.2 CGRA Performance Estimates

**Stanford Plasticine-like (64 PEs @ 500 MHz)**:
```
ResNet-18 (FP32, Batch=1):
- Latency: ~10-50 ms (highly variable)
- Depends on: Reconfiguration overhead
- Throughput: ~20-100 inferences/sec
- Energy: ~0.001-0.005 J/inference
- Speedup vs CPU: ~1-5×
```

**Strengths**:
- Extremely energy efficient
- Spatial execution (no instruction fetch)
- Flexible (can map arbitrary dataflow)

**Weaknesses**:
- Limited resources (64-256 PEs)
- Reconfiguration overhead significant
- Place-and-route complexity
- May not fit large models

### 4.3 Updated Performance Rankings

**Predicted 6-Way Comparison (ResNet-18, INT8, Batch=1)**:
1. **GPU (H100)**: 0.024 ms → 41,556 inf/sec ⚡ (champion)
2. **TPU (v4)**: 0.040 ms → 24,934 inf/sec 🔷
3. **KPU (T100)**: 0.050 ms → 20,014 inf/sec 📱
4. **DPU (Vitis AI)**: ~3-5 ms → ~200-300 inf/sec 🔧 (predicted)
5. **CGRA (Plasticine)**: ~20-50 ms → ~20-50 inf/sec ⚙️ (predicted, highly variable)
6. **CPU (Intel)**: 0.602 ms → 1,662 inf/sec 💻

**Energy Efficiency Rankings**:
1. **CGRA**: ~0.001 J (spatial execution, no instruction overhead)
2. **KPU**: 0.001 J
3. **TPU**: 0.001 J
4. **DPU**: ~0.01 J (FPGA overhead)
5. **GPU**: 0.001 J
6. **CPU**: 0.002 J

---

## 5. Recommendations

### 5.1 Implementation Roadmap

**Immediate (This Week)**:
1. ✅ Review this analysis
2. Create `DPUResourceModel` in `hardware_mapper.py`
3. Implement `DPUMapper` class
4. Create `test_dpu_simple.py` for validation

**Short Term (Next Week)**:
1. Add DPU to 6-way comparison
2. Validate DPU mapper on ResNet-18, MobileNet-V2
3. Document DPU mapper design decisions

**Medium Term (Following Week)**:
1. Research CGRA place-and-route algorithms
2. Create `CGRAResourceModel`
3. Implement `CGRAMapper` class (most complex)
4. Create `test_cgra_simple.py`

**Long Term**:
1. Complete 6-way comparison (GPU/TPU/KPU/CPU/DPU/CGRA)
2. Write research paper on comprehensive hardware modeling
3. Validate against real hardware (if available)

### 5.2 Research Questions

**For DPU**:
- How does scratchpad size affect tiling overhead?
- What's the optimal array size for different models?
- How much does FPGA reconfiguration impact latency?

**For CGRA**:
- Can we use greedy place-and-route or need exact solver?
- How many contexts are needed for typical models?
- What's the trade-off between PE count and interconnect?
- Can we partition smartly to minimize reconfiguration?

### 5.3 Validation Strategy

**DPU Validation**:
- Compare against Xilinx Vitis AI documented performance
- Test on: ResNet-18, MobileNet-V2, YOLOv3
- Validate INT8 performance claims

**CGRA Validation**:
- Compare against Stanford Plasticine papers
- Analyze reconfiguration overhead sensitivity
- Test spatial mapping quality (utilization)

---

## 6. Conclusion

**DPU**: Straightforward extension
- Combines KPU (tiling) + TPU (2D array) concepts
- Medium complexity, ~300-400 lines
- Good match for our framework

**CGRA**: Significant research challenge
- Requires new spatial mapping algorithms
- High complexity, ~500-700 lines
- Will push our framework to new limits

**Both architectures will make our modeling framework comprehensive**, covering:
- General purpose: CPU
- Cloud accelerators: GPU, TPU
- Edge accelerators: KPU, DPU
- Research architectures: CGRA

**Total coverage**: 6 major hardware paradigms for deep learning!

---

## Appendix A: Reference Architectures

### A.1 DPU Examples

**Xilinx Vitis AI DPU**:
- 4096 INT8 ALUs (64×64 array)
- 512KB on-chip memory
- ~1.2 TOPS @ 300 MHz
- Supports: Conv, Pool, ElementWise
- Power: ~5W

**Intel Neural Compute Stick (Myriad X)**:
- 16 SHAVE processors (similar to DPU)
- ~1 TOPS INT8
- Power: ~1W

### A.2 CGRA Examples

**Stanford Plasticine**:
- 32 Pattern Compute Units (PCUs)
- 32 Pattern Memory Units (PMUs)
- 16-32 PEs per PCU
- ~40 GFLOPS FP32
- Power: ~2W

**UT Austin TRIPS**:
- 16×16 ALU grid
- Explicit Data Graph Execution (EDGE)
- 64 functional units
- Research prototype

**MIT Eyeriss (spatial architecture)**:
- 168 PEs in 12×14 array
- Optimized for CNNs
- ~35 GOPS FP32
- Power: ~0.3W

---

## Appendix B: Implementation Hints

### B.1 DPU Array Utilization

```python
def calculate_dpu_array_utilization(output_shape, array_shape):
    """
    Calculate what % of DPU array is actively used.

    For Conv2D with output (B, C, H, W):
    - Can parallelize across C (output channels)
    - Can parallelize across H, W (spatial)

    Array shape: (rows, cols) e.g., (64, 64)
    """
    batch, channels, height, width = output_shape
    array_rows, array_cols = array_shape

    # Strategy: Map channels to rows, spatial to cols
    active_rows = min(channels, array_rows)
    active_cols = min(height * width, array_cols)

    utilization = (active_rows * active_cols) / (array_rows * array_cols)
    return utilization
```

### B.2 CGRA Place-and-Route Heuristic

```python
def greedy_place_and_route(ops, pe_grid):
    """
    Greedy algorithm for mapping ops to PE grid.

    Not optimal, but fast and reasonable.
    """
    placed_ops = {}

    # 1. Sort ops by topological order (data dependencies)
    sorted_ops = topological_sort(ops)

    # 2. Place ops greedily
    for op in sorted_ops:
        # Find PE close to op's inputs
        best_pe = find_closest_pe(
            op.inputs,
            placed_ops,
            pe_grid
        )

        if best_pe is None:
            return None  # No feasible placement

        placed_ops[op] = best_pe

    # 3. Check routing feasibility
    for op in ops:
        for input_op in op.inputs:
            src_pe = placed_ops[input_op]
            dst_pe = placed_ops[op]

            path = find_route(src_pe, dst_pe, pe_grid)
            if path is None:
                return None  # Routing infeasible

    return placed_ops
```

---

**End of Analysis**
