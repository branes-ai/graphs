# Graph Partitioning, Hardware Mapping & Visualization Infrastructure Analysis

## Overview
The codebase contains a sophisticated pipeline for analyzing Deep Neural Networks (DNNs) across different hardware architectures. The pipeline is organized into logical phases, with clear data structures and reusable components.

---

## 1. GRAPH PARTITIONING INFRASTRUCTURE

### 1.1 Data Structures (src/graphs/ir/structures.py)

**Core Classes:**
- **SubgraphDescriptor**: Unified representation supporting both single-op and multi-op (fused) subgraphs
  - Stores: node IDs, operation types, FLOPs, memory (input/output/weight/internal), arithmetic intensity
  - Supports: fusion metadata, parallelism info, bottleneck classification, partition reasoning
  - Backward compatible: Properties like `node_id`, `operation_type`, `flops` for single-op access

- **PartitionReport**: Complete graph partitioning results
  - Contains: list of SubgraphDescriptor, aggregate statistics, fusion metrics
  - Metrics: total FLOPs/memory, data_movement_reduction, operation_type_counts, bottleneck_distribution
  - Concurrency info: Optional ConcurrencyDescriptor for graph-level analysis

- **ParallelismDescriptor**: Available parallelism dimensions
  - Tracks: batch, channels, spatial parallelism
  - Flags: is_depthwise, is_grouped, can_split_batch/spatial/channels
  - Vectorization info: vectorizable_dim, vector_width

- **TensorDescriptor**: Shape and memory metadata
  - Storage: shape, dtype, size_bytes, layout (NCHW, NHWC, etc.)

- **BottleneckType Enum**: Classification (COMPUTE_BOUND, MEMORY_BOUND, BANDWIDTH_BOUND, BALANCED)

- **PartitionReason Enum**: Why a partition boundary was created
  - Options: OPERATION_BOUNDARY, MEMORY_LIMIT_EXCEEDED, COMPUTE_THRESHOLD_EXCEEDED, FUSION_INCOMPATIBLE, DATA_DEPENDENCY, FUSION_OPPORTUNITY, PARALLEL_SCHEDULE_DIVERGENCE

### 1.2 Fusion Partitioner (src/graphs/transform/partitioning/fusion_partitioner.py)

**FusionBasedPartitioner Class:**
- **Inputs**: PyTorch FX GraphModule with shape propagation via ShapeProp
- **Algorithm**: Greedy fusion along sequential paths, breaks at fork/join points
- **Returns**: PartitionReport with unified SubgraphDescriptor entries

**Key Methods:**
1. `partition(fx_graph)` - Main entry point
   - Extracts FX nodes → builds dependency graphs → fuses operators → generates report
   - Caches results for visualization: `self.fused_subgraphs`, `self.fx_graph_cached`

2. `_fuse_operators()` - Core fusion logic
   - Topological traversal, greedy forward fusion
   - Stops at fork (multiple consumers), join (multiple producers), incompatible ops

3. `_is_fusible(node1, node2)` - Pattern matching
   - Supports: Conv→BN, BN→ReLU, Conv→ReLU, Linear→GELU
   - SE blocks: AdaptiveAvgPool→Conv→SiLU→Conv→Sigmoid→mul
   - Transformer patterns: LayerNorm→MultiheadAttention, GELU→Linear
   - Element-wise chains: ReLU→Dropout, etc.

4. `_create_fused_subgraph()` - Subgraph descriptor generation
   - Computes: FLOPs via `_compute_flops()`, memory via `_compute_memory()`
   - Distinguishes: external (cross-boundary) vs internal (fusion-saved) bytes
   - Calculates: arithmetic_intensity, bottleneck classification

5. Specialized FLOP computations:
   - `_compute_flops_module()`: For symbolic_trace (call_module nodes)
   - `_compute_flops_aten()`: For Dynamo export (call_function with ATen ops)
   - Handles: Conv2d, Linear, BatchNorm, LayerNorm, Pooling, SDPA (attention), element-wise

**Output (PartitionReport):**
```python
{
  subgraphs: [SubgraphDescriptor, ...],
  total_subgraphs: int,
  total_flops: int,
  total_memory_traffic: int,
  
  # Fusion metrics
  original_operators: int,
  total_memory_traffic_unfused: int,
  data_movement_reduction: float,
  avg_fusion_size: float,
  max_fusion_size: int,
  
  # Statistics
  operation_type_counts: Dict[str, int],
  fusion_pattern_counts: Dict[str, int],
  bottleneck_distribution: Dict[str, int],
  concurrency: Optional[ConcurrencyDescriptor]
}
```

### 1.3 Analysis Methods on FusionBasedPartitioner

**Visualization Methods:**
- `visualize_partitioning(fx_graph, start=None, end=None)` - Side-by-side FX graph + fused subgraphs
- `visualize_partitioning_colored(fx_graph, ...)` - Color-coded by bottleneck type (ANSI colors)
- `export_to_graphviz(fx_graph, output_file)` - Graphviz DOT format for visual graphs

**Analysis Methods:**
- `analyze_balance()` - Comprehensive quality analysis
  - Fusion size distribution, histogram, categorized single-op analysis
  - Detects overly large fusions (>10, >20 operators)
  - Identifies missed fusion opportunities
  - Compares to sequential fusion baseline
  - Provides recommendations

- `_categorize_single_ops()` - Classifies single-op subgraphs
  - Structural: getitem, reshape, transpose, placeholders
  - Fusible: compute-heavy operations that could be fused

- `_detect_fusion_opportunities()` - Finds adjacent fusible operations

- `_calculate_sequential_fusion_baseline()` - Conservative baseline for comparison

---

## 2. HARDWARE MAPPING INFRASTRUCTURE

### 2.1 Resource Models (src/graphs/hardware/resource_model.py)

**Core Classes:**
- **HardwareType Enum**: GPU, CPU, TPU, KPU, DPU, CGRA, DSP
- **Precision Enum**: FP64, FP32, FP16, FP8, BF16, INT32, INT16, INT8, INT4

**Physics-Based Energy Model:**
- `PROCESS_NODE_ENERGY`: Per-operation energy by process node (3nm: 1.2pJ, 28nm: 4.0pJ)
- `CIRCUIT_TYPE_MULTIPLIER`: Circuit efficiency (standard_cell: 1.0×, tensor_core: 0.85×, simd_packed: 0.90×)
- `get_base_alu_energy(process_node_nm, circuit_type)` - Calculates FP32 energy per op

**ComputeFabric:**
- Represents specific compute unit type (CUDA core, Tensor Core, INT8 tile, etc.)
- Tracks: circuit_type, num_units, ops_per_unit_per_clock, core_frequency_hz, process_node_nm
- Methods: get_energy_per_op(precision), get_peak_ops_per_sec(precision), get_peak_power(precision)

**HardwareResourceModel:**
- Defines hardware capabilities: peak FLOPS, memory bandwidth, cache hierarchy
- Per-unit resources: l1_cache_per_unit, threads_per_unit, warps_per_unit
- Wave quantization: Minimum allocation granularity (e.g., GPU allocates in groups of 4 SMs)
- Thermal profiles: Multiple TDP points for power modeling

**HardwareAllocation:**
- Result of mapping a single subgraph to hardware
- Contains: compute_units_allocated, occupancy, utilization, compute/memory time, energy
- Tracks: execution_stage (for pipelining), is_parallel flag

**GraphHardwareAllocation:**
- Complete allocation of entire graph across hardware
- Aggregates all subgraph allocations
- Computes: total_latency, total_energy, average_utilization

### 2.2 Hardware Mappers (src/graphs/hardware/mappers/)

**Base Class: HardwareMapper**
- Abstract methods: map_subgraph(), should_use_sequential_execution()
- Utilities: roofline model for latency, energy calculation, thermal profile management

**GPU Mapper (gpu.py):**
- Maps threads → warps (32 threads/warp) → SMs (64 warps/SM typical)
- Wave quantization: Allocates SMs in groups (e.g., 4 SMs at a time)
- Occupancy calculation: warps_used / max_warps_possible
- Models: Sequential execution (batch=1, small parallelism), concurrent kernels
- Roofline: Compute vs memory bandwidth bottleneck

**CPU Mapper (cpu.py):**
- Maps threads → cores → SIMD units (8-lane AVX2, 16-lane AVX-512)
- Considers: NUMA topology, cache hierarchy (L1/L2/L3)
- Precision scaling: INT8 ops/sec >> FP32 ops/sec
- Dynamic frequency scaling awareness

**KPU Mapper (accelerators/kpu.py):**
- Maps threads → tiles (64/256/768 tiles typical)
- Scratchpad constraints: 256KB per tile (must tile large operations)
- Tiling overhead calculation for operations that exceed scratchpad
- Specialization: Better for INT8/INT4 quantized models
- Thermal profiles: Multiple power points (3W, 6W, 10W, 15W, 30W, 50W)

**TPU Mapper (accelerators/tpu.py):**
- Systolic array model: Weight-stationary dataflow
- Maps tensors to 128×128 matrix multiply units
- Loop tiling for large operations
- Energy: Very efficient for large batch inference

**DPU Mapper (accelerators/dpu.py):**
- Xilinx Vitis AI reconfigurable FPGA tiles
- Flexible mapping, configuration overhead modeling

**CGRA Mapper (accelerators/cgra.py):**
- Spatial dataflow (Plasticine-style)
- Reconfiguration overhead, route congestion

**DSP Mapper (dsp.py):**
- Vector/tensor units (Qualcomm Hexagon, TI C7x)
- VLIW instruction scheduling

### 2.3 Hardware Mapper Creation Pattern

All mappers follow consistent factory functions:
```python
def create_h100_sxm5_80gb_mapper(thermal_profile: str = None) -> GPUMapper:
    resource_model = HardwareResourceModel(...)
    return GPUMapper(resource_model, thermal_profile=thermal_profile)
```

Available in `unified_analyzer.py`:
- GPU: H100 (SXM5, PCIe), A100, V100, Jetson Orin (AGX, Nano), Jetson Thor
- TPU: v4, Coral Edge
- KPU: T64, T256, T768
- CPU: AMD EPYC, Intel Xeon, Ampere Ampere-One, i7-12700K
- DSP: QRB5165, TI TDA4VM, Snapdragon Ride, SA8775P
- Accelerators: DPU (Xilinx), CGRA (Plasticine-v2), Hailo10H, Hailo8

---

## 3. VISUALIZATION INFRASTRUCTURE

### 3.1 Visualization Module (src/graphs/transform/visualization.py)

**Terminal Capability Detection:**
- `detect_terminal_capability()` - Returns BASIC, UTF8, COLOR, or TRUECOLOR
- Checks: isatty(), NO_COLOR env var, TERM env var, encoding

**Color Utilities:**
- `ANSIColor`: ANSI color codes (regular, bright, background, styles)
- `get_bottleneck_color(bottleneck_type, capability)` - Returns (start, end) color codes
  - Compute-bound: GREEN, Memory-bound: YELLOW, Bandwidth-bound: RED, Balanced: CYAN
- `colorize(text, color, capability)` - Applies color if supported

**Box Drawing:**
- `BoxChars.UTF8` - UTF-8 box characters (┌, ─, │, └, etc.)
- `BoxChars.ASCII` - ASCII fallback (+, -, |, etc.)
- `get_box_chars(capability)` - Selects appropriate set

**Output Methods:**
- `create_legend(capability)` - Generates color legend
- `export_to_dot(fused_subgraphs, fx_graph, output_file)` - Graphviz DOT format
  - Creates nodes for subgraphs (colored by bottleneck)
  - Creates edges based on data dependencies
  - Includes legend with color meanings
  - Suitable for `dot -Tpng file.dot -o file.png`

- `format_metric_with_color()` - Formats single metrics with color thresholds

### 3.2 Partitioner Visualization Methods

**visualize_partitioning():**
- Side-by-side: FX graph (execution order) vs Fused subgraphs
- Shows: Node names, operation types, fusion grouping
- Metrics: FLOPs, memory, AI, bottleneck type
- Output: Plain text with box borders

**visualize_partitioning_colored():**
- Same layout as above, but:
  - Color-codes subgraph headers by bottleneck
  - Dimmed text for structural operations
  - Optional range selection (start, end)
- Smart terminal detection for color fallback

**Format Methods (internal):**
- `_format_fx_node()` - Left column: FX node details
- `_format_fused_subgraph_header()` - Subgraph opening box with pattern, operator count
- `_format_fused_operator()` - Individual operator within subgraph
- `_format_fused_subgraph_footer()` - Subgraph closing box with metrics
- `_format_not_fused()` - Placeholder nodes (input, output, structural)

---

## 4. ANALYSIS PIPELINE

### 4.1 Unified Analyzer (src/graphs/analysis/unified_analyzer.py)

**AnalysisConfig:**
```python
{
  run_roofline: bool,           # Latency estimation
  run_energy: bool,             # Energy consumption
  run_memory: bool,             # Peak memory usage
  run_concurrency: bool,        # Graph-level parallelism
  run_hardware_mapping: bool,   # Use hardware mappers
  use_fusion_partitioning: bool,
  power_gating_enabled: bool,   # Model power gating of unused units
  run_operator_edp: bool,       # Operator-level EDP breakdown
  validate_consistency: bool
}
```

**UnifiedAnalyzer.analyze_model(model_name, hardware_name, batch_size, precision, config):**

1. **Model Creation**: Load torchvision model (ResNet, MobileNet, etc.)
2. **FX Tracing**: `torch.fx.symbolic_trace()` with shape propagation
3. **Partitioning**: FusionBasedPartitioner → PartitionReport
4. **Hardware Mapping** (if enabled):
   - Create mapper from factory function
   - Map each subgraph: `mapper.map_subgraph(subgraph, stage, concurrent_count, precision)`
5. **Analysis** (based on config):
   - **Roofline**: Bottleneck detection, latency estimation
   - **Energy**: Compute + memory energy, power models
   - **Memory**: Peak memory, activation/weight breakdown
   - **Concurrency**: Graph-level parallelism analysis
6. **Return**: UnifiedAnalysisResult (single data structure with all results)

**UnifiedAnalysisResult:**
- metadata: model_name, hardware_name, batch_size, precision, timestamp
- reports: roofline_report, energy_report, memory_report, concurrency_report
- hardware_allocation: GraphHardwareAllocation
- partition_report: PartitionReport
- derived_metrics: latency, throughput, energy_per_inference, peak_memory_mb

### 4.2 Individual Analyzers

**RooflineAnalyzer (roofline.py):**
- Input: PartitionReport, HardwareResourceModel
- Output: RooflineReport with per-subgraph latency analysis
- Computes: compute_time vs memory_time, bottleneck classification
- Metrics: FLOP/bandwidth utilization

**EnergyAnalyzer (energy.py):**
- Three-component model: compute energy, memory energy, static/leakage energy
- Input: PartitionReport, HardwareResourceModel
- Output: EnergyReport with per-subgraph energy breakdown
- Precision-aware: Energy scales with precision

**MemoryEstimator (memory.py):**
- Peak memory usage (activations + weights)
- Memory timeline (live tensors over execution)
- Activation vs weight breakdown
- Hardware fit analysis (vs L2 cache, total memory)

**ConcurrencyAnalyzer (concurrency.py):**
- Graph-level concurrency (independent subgraphs)
- Critical path analysis (longest sequential path)
- Parallelism efficiency metrics

### 4.3 Report Generator (src/graphs/reporting/report_generator.py)

**Output Formats:**
- Text (human-readable console)
- JSON (structured data)
- CSV (tabular, suitable for post-processing)
- Markdown (GitHub-compatible reports)
- HTML (web-ready reports)

**Features:**
- Auto-format detection from file extension
- Selective sections (executive, performance, energy, memory, recommendations)
- Optional diagrams (Mermaid for markdown/HTML)
- Comparison reports (batch sweep, model/hardware comparison)
- Pretty-printing with proper units and formatting

---

## 5. TYPICAL ANALYSIS FLOW

### 5.1 End-to-End Example

```python
# Step 1: Create analyzer
analyzer = UnifiedAnalyzer(verbose=True)

# Step 2: Analyze model on hardware
result = analyzer.analyze_model(
    model_name='resnet18',
    hardware_name='H100',
    batch_size=1,
    precision=Precision.FP32
)

# Step 3: Access results
print(f"Latency: {result.total_latency_ms:.2f} ms")
print(f"Energy: {result.total_energy_mj:.2f} mJ")
print(f"Peak Memory: {result.peak_memory_mb:.2f} MB")

# Step 4: Generate reports
from graphs.reporting import ReportGenerator
generator = ReportGenerator()
text_report = generator.generate_text_report(result)
json_report = generator.generate_json_report(result)
```

### 5.2 CLI Pattern

All CLI tools follow this pattern:
1. Argument parsing (model, hardware, precision, batch size, output format)
2. Model creation and tracing
3. Analysis (via UnifiedAnalyzer or direct use of analyzers)
4. Result formatting (via ReportGenerator)
5. Output (stdout or file)

Example: `analyze_comprehensive.py`
- Supports: All model/hardware combinations
- Options: FP32/FP16/INT8 precision, batch size, power gating, hardware mapping toggle
- Outputs: Text, JSON, CSV, Markdown formats

---

## 6. DATA STRUCTURE RELATIONSHIPS

```
FX Graph (PyTorch model traced with symbolic_trace)
    ↓
    [FusionBasedPartitioner]
    ↓
PartitionReport
├─ subgraphs: [SubgraphDescriptor, ...]
├─ total_subgraphs: int
├─ total_flops: int
├─ fusion_metrics: (data_movement_reduction, avg_fusion_size, etc.)
└─ statistics: (operation_type_counts, bottleneck_distribution, etc.)
    ↓
    [HardwareMapper (GPU/CPU/KPU/etc.)]
    ↓
GraphHardwareAllocation
├─ subgraph_allocations: [HardwareAllocation, ...]
├─ total_latency: float
├─ total_energy: float
└─ average_utilization: float
    ↓
    [RooflineAnalyzer, EnergyAnalyzer, MemoryEstimator]
    ↓
UnifiedAnalysisResult
├─ roofline_report: RooflineReport
├─ energy_report: EnergyReport
├─ memory_report: MemoryReport
├─ partition_report: PartitionReport
└─ hardware_allocation: GraphHardwareAllocation
    ↓
    [ReportGenerator]
    ↓
Text/JSON/CSV/Markdown Report
```

---

## 7. KEY REUSABLE COMPONENTS

### 7.1 Data Structures (for new CLI tools)
- **SubgraphDescriptor**: Core unit of computation (single-op or fused)
- **PartitionReport**: Complete graph partitioning with statistics
- **HardwareAllocation**: Subgraph → hardware mapping result
- **GraphHardwareAllocation**: Complete graph mapping

### 7.2 Partitioning
- **FusionBasedPartitioner**: Turn FX graph into fused subgraphs
- Requires: FX graph with shape propagation (ShapeProp)
- Returns: PartitionReport

### 7.3 Hardware Mapping
- **HardwareMapper factory functions**: Create configured mappers
- Available: GPU, CPU, KPU, TPU, DPU, CGRA, DSP (7 hardware types)
- Thermal profiles: Multiple power points per hardware
- Usage: `mapper.map_graph(fusion_report, execution_stages, batch_size, precision)`

### 7.4 Analysis
- **UnifiedAnalyzer**: Single call for complete analysis
- **Individual Analyzers**: RooflineAnalyzer, EnergyAnalyzer, MemoryEstimator
- All thread-safe and composable

### 7.5 Visualization
- **FusionBasedPartitioner.visualize_partitioning()**: Terminal display
- **FusionBasedPartitioner.export_to_graphviz()**: Graphviz DOT export
- **ReportGenerator**: Multi-format output
- **Terminal utilities**: Color detection, ANSI codes, box drawing

### 7.6 CLI Utilities
- **AnalysisConfig**: Configurable analysis options
- **ReportGenerator**: Format agnostic report generation
- **Model factory**: Load torchvision models with consistent interface
- **Hardware registry**: All available hardware mappers indexed by name

---

## 8. IDENTIFIED GAPS

### 8.1 Minor Gaps
1. **GraphPartitioner** (single-op unfused): Exists but less frequently used than FusionBasedPartitioner
2. **Operator-level FLOP validation**: Only tested on Conv2D/Linear; some operations (attention, normalization) have complex FLOP formulas
3. **Custom model support**: Currently expects torchvision models; custom architectures require manual FX tracing
4. **Tiling strategies**: KPU mapper has basic tiling; no advanced tile scheduling algorithms
5. **Multi-batch fusion**: Fusion assumes single batch; multi-batch scenarios need separate handling

### 8.2 Design Patterns Established
- **Factory pattern**: Hardware mappers created via factory functions (not direct instantiation)
- **Dataclass-driven**: Immutable result objects (SubgraphDescriptor, HardwareAllocation, etc.)
- **Visitor pattern**: Analyzers visit PartitionReport subgraphs to compute metrics
- **Strategy pattern**: Pluggable hardware mappers for different architectures
- **Template method**: UnifiedAnalyzer orchestrates analyzers in correct order

---

## 9. RECOMMENDED ARCHITECTURE FOR NEW CLI TOOLS

### Structure:
1. **Input**: Model name + hardware name + optional config
2. **Processing**:
   - Use **UnifiedAnalyzer** as main orchestrator
   - Pass **AnalysisConfig** to customize behavior
   - Leverage **HardwareMapper** factories for hardware support
3. **Output**:
   - Use **ReportGenerator** for multi-format support
   - Use **FusionBasedPartitioner.visualize_*()** for graph visualization
4. **Reuse**:
   - Don't duplicate partitioning logic
   - Don't create new hardware mappers (use factories)
   - Don't reimplements analyzers (they're composable)

### Example New Tool Pattern:
```python
# Compare performance across hardware types
def compare_hardware(model_name, precisions, batch_sizes):
    analyzer = UnifiedAnalyzer()
    results = {}
    
    for hw in ['H100', 'Jetson-Orin-AGX', 'KPU-T256']:
        for precision in precisions:
            for batch_size in batch_sizes:
                result = analyzer.analyze_model(
                    model_name, hw, batch_size, precision
                )
                results[(hw, precision, batch_size)] = result
    
    # Generate comparison report
    generator = ReportGenerator()
    return generator.generate_comparison_report(results)
```

