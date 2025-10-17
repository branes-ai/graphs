# Realistic Performance Modeling: Implementation Plan

**Date**: October 17, 2025
**Goal**: Transform characterization from peak theoretical to realistic hardware performance modeling
**Key Principle**: Explainability - every step produces verifiable statistics

---

## Problem Statement

### Current Limitations

1. **Assumes full hardware utilization**: All 132 SMs on H100, entire systolic array on TPU
2. **Ignores memory bottlenecks**: Assumes all operations are compute-bound
3. **No concurrency analysis**: Doesn't model how graphs map to parallel hardware
4. **Missing CPU vector units**: Doesn't account for AVX-512, AMX on modern CPUs

### Example of Current Error

**EfficientNet-B0 on H100**:
- Current model: 1.88 ms latency (assumes 750 TFLOPS utilization)
- Reality: Likely 10-50 ms (only 5-20% utilization due to limited parallelism)
- **Error magnitude**: 5-25× too optimistic

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Computation Graph (FX)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GraphPartitioner                             │
│  • Decompose into subgraphs (kernels)                           │
│  • Analyze parallelism dimensions                               │
│  • Compute data dependencies                                    │
│  • Calculate ingress/egress bandwidth                           │
│  Output: List[SubgraphDescriptor]                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HardwareMapper                               │
│  • Map subgraph → hardware resources                            │
│  • Estimate resource utilization                                │
│  • Identify memory vs compute bottleneck                        │
│  • Model concurrent execution                                   │
│  Output: List[MappingDescriptor]                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PerformanceEstimator                         │
│  • Analyze available concurrency in subgraphs                   │
│  • Compute realistic latency per subgraph                       │
│  • Apply roofline model (min(compute, memory) time)             │
│  • Add kernel launch overhead                                   │
│  • Aggregate with critical path analysis                        │
│  Output: PerformanceReport + ConcurrencyAnalysis                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├──────────────┬─────────────────┐
                             ▼              ▼                 ▼
                    ┌─────────────┐  ┌──────────┐  ┌──────────────┐
                    │   Energy    │  │  Memory  │  │ Explainability│
                    │  Estimator  │  │Estimator │  │     Data      │
                    └─────────────┘  └──────────┘  └──────────────┘
```

---

## Phase 1: Foundation - Graph Partitioning & Statistics

**Goal**: Decompose computation graph into hardware-executable subgraphs with full statistics

### 1.1 SubgraphDescriptor Data Structure

```python
@dataclass
class SubgraphDescriptor:
    """Complete description of a computational subgraph"""

    # Identity
    node_id: str
    node_name: str
    operation_type: str  # 'conv2d', 'matmul', 'depthwise_conv', 'elementwise'
    fusion_pattern: str  # 'conv_bn_relu', 'conv_relu6', etc.

    # Computation
    flops: int
    macs: int  # multiply-accumulate operations

    # Memory
    input_tensors: List[TensorDescriptor]
    output_tensors: List[TensorDescriptor]
    weight_tensors: List[TensorDescriptor]
    total_input_bytes: int
    total_output_bytes: int
    total_weight_bytes: int
    arithmetic_intensity: float  # FLOPs / total_bytes

    # Parallelism dimensions
    parallelism: ParallelismDescriptor

    # Data dependencies
    depends_on: List[str]  # node_ids of dependencies
    dependency_type: str  # 'sequential', 'independent', 'partial'

    # Hardware hints
    recommended_mapping: str  # 'compute_bound', 'memory_bound', 'bandwidth_bound'
```

```python
@dataclass
class TensorDescriptor:
    """Tensor shape and memory info"""
    shape: Tuple[int, ...]
    dtype: str  # 'float32', 'bfloat16', 'int8'
    size_bytes: int
    layout: str  # 'NCHW', 'NHWC', etc.

@dataclass
class ParallelismDescriptor:
    """Available parallelism dimensions"""
    batch: int
    channels: int  # or output features
    spatial: int  # height × width for convs
    total_threads: int  # batch × channels × spatial

    # Special cases
    is_depthwise: bool  # limited channel parallelism
    is_grouped: bool
    num_groups: int

    # Vectorization potential (for CPU)
    vectorizable_dim: str  # which dimension can use SIMD
    vector_width: int  # e.g., 8 for AVX-256 on float32
```

### 1.2 GraphPartitioner Implementation

```python
class GraphPartitioner:
    """Partition computation graph with full statistics"""

    def partition(self, fx_graph: GraphModule) -> PartitionReport:
        """
        Partition graph into subgraphs

        Returns:
            PartitionReport with:
            - List[SubgraphDescriptor]
            - Statistics summary
            - Dependency graph
            - Critical path analysis
        """
        subgraphs = []

        for node in fx_graph.graph.nodes:
            if node.op == 'call_module':
                subgraph = self._analyze_node(node, fx_graph)
                subgraphs.append(subgraph)

        # Build dependency graph
        dep_graph = self._build_dependency_graph(subgraphs)

        # Critical path analysis
        critical_path = self._find_critical_path(subgraphs, dep_graph)

        return PartitionReport(
            subgraphs=subgraphs,
            total_subgraphs=len(subgraphs),
            total_flops=sum(s.flops for s in subgraphs),
            total_memory_traffic=sum(s.total_input_bytes + s.total_output_bytes
                                     for s in subgraphs),
            dependency_graph=dep_graph,
            critical_path=critical_path,
            parallelism_distribution=self._analyze_parallelism_distribution(subgraphs)
        )

    def _analyze_node(self, node, graph) -> SubgraphDescriptor:
        """Analyze single node/operation"""
        meta = node.meta.get('tensor_meta')
        module = graph.get_submodule(node.target)

        # Compute parallelism
        parallelism = self._compute_parallelism(node, meta, module)

        # Memory analysis
        input_bytes, output_bytes, weight_bytes = self._analyze_memory(node, meta, module)

        # FLOPs calculation
        flops, macs = self._compute_flops(node, meta, module)

        # Arithmetic intensity
        total_bytes = input_bytes + output_bytes + weight_bytes
        arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0

        return SubgraphDescriptor(
            node_id=str(id(node)),
            node_name=node.name,
            operation_type=self._classify_operation(module),
            flops=flops,
            macs=macs,
            total_input_bytes=input_bytes,
            total_output_bytes=output_bytes,
            total_weight_bytes=weight_bytes,
            arithmetic_intensity=arithmetic_intensity,
            parallelism=parallelism,
            depends_on=self._find_dependencies(node),
            recommended_mapping=self._recommend_mapping(arithmetic_intensity)
        )

    def _compute_parallelism(self, node, meta, module) -> ParallelismDescriptor:
        """Analyze available parallelism"""
        if isinstance(module, nn.Conv2d):
            B, C_in, H, W = meta.shape
            C_out = module.out_channels
            K_h, K_w = module.kernel_size
            S_h, S_w = module.stride
            P = module.padding

            H_out = (H + 2*P - K_h) // S_h + 1
            W_out = (W + 2*P - K_w) // S_w + 1

            is_depthwise = (module.groups == C_in == C_out)

            return ParallelismDescriptor(
                batch=B,
                channels=C_out if not is_depthwise else C_in,
                spatial=H_out * W_out,
                total_threads=B * C_out * H_out * W_out,
                is_depthwise=is_depthwise,
                is_grouped=(module.groups > 1),
                num_groups=module.groups,
                vectorizable_dim='channels',
                vector_width=8  # example for AVX-256
            )

        # Similar for Linear, etc.
        return ParallelismDescriptor(...)

    def _recommend_mapping(self, arithmetic_intensity: float) -> str:
        """Recommend execution strategy based on arithmetic intensity"""
        if arithmetic_intensity > 50:
            return 'compute_bound'
        elif arithmetic_intensity > 10:
            return 'balanced'
        else:
            return 'memory_bound'
```

### 1.3 Statistics Report Example

```
Graph Partition Report: EfficientNet-B0
========================================

Total Subgraphs: 152
Total FLOPs: 2.35 G
Total Memory Traffic: 845 MB (input + output + weights)
Average Arithmetic Intensity: 2.78 FLOPs/byte

Subgraph Distribution:
  Conv2d (standard):        12 (8%)
  Conv2d (depthwise):       48 (32%)
  Conv2d (pointwise 1×1):   64 (42%)
  BatchNorm2d:              80 (53%) [fused with conv]
  ReLU6/Hardswish:          72 (47%)
  Squeeze-Excite:           16 (11%)

Parallelism Distribution:
  < 1K threads:      24 subgraphs (16%)  [low parallelism - CPU/small GPU]
  1K - 10K threads:  48 subgraphs (32%)  [moderate - needs 2-8 SMs]
  10K - 100K:        62 subgraphs (41%)  [good - needs 20-50 SMs]
  > 100K:            18 subgraphs (12%)  [excellent - can saturate GPU]

Arithmetic Intensity Distribution:
  < 1 FLOPs/byte:    28 subgraphs (18%) [memory-bound - BW limited]
  1-10:              86 subgraphs (57%) [balanced]
  > 10:              38 subgraphs (25%) [compute-bound]

Critical Path:
  Length: 152 sequential operations
  Parallelizable stages: 8 (Conv blocks can run independently with batching)
  Estimated critical path FLOPs: 2.35 G (no parallelism with batch=1)
```

---

## Phase 2: Hardware Mapping with CPU Support

**Goal**: Map subgraphs to specific hardware resources with utilization estimates

### 2.1 Hardware Resource Models

#### CPU Resource Model

```python
@dataclass
class CPUResourceModel:
    """Modern CPU with vector/matrix units"""

    # Core structure
    num_cores: int  # physical cores
    num_threads_per_core: int  # SMT/hyperthreading

    # SIMD/Vector units
    vector_unit: str  # 'AVX2', 'AVX-512', 'NEON', 'SVE'
    vector_width_bits: int  # 256, 512, etc.
    vector_lanes_fp32: int  # 8 for AVX2, 16 for AVX-512

    # Matrix units (if available)
    has_matrix_unit: bool  # Intel AMX, Apple AMX
    matrix_unit_name: str  # 'AMX', 'Apple_AMX', None
    matrix_unit_tiles: int  # 8 for Intel AMX
    matrix_tile_size: Tuple[int, int]  # (16, 64) for AMX

    # Performance
    peak_flops_scalar: float  # single-threaded scalar
    peak_flops_vector: float  # all cores, vectorized
    peak_flops_matrix: float  # with matrix unit (if available)

    # Memory
    l1_cache_per_core: int  # 32-48 KB typical
    l2_cache_per_core: int  # 256-512 KB typical
    l3_cache_shared: int  # 8-32 MB typical
    mem_bandwidth: float

# Example: Intel Core i7-13700K
intel_i7_13700k = CPUResourceModel(
    num_cores=16,  # 8P + 8E cores
    num_threads_per_core=2,
    vector_unit='AVX2',
    vector_width_bits=256,
    vector_lanes_fp32=8,
    has_matrix_unit=True,
    matrix_unit_name='AMX',
    matrix_unit_tiles=8,
    matrix_tile_size=(16, 64),
    peak_flops_scalar=0.1e12,  # ~100 GFLOPS scalar
    peak_flops_vector=1.5e12,  # ~1.5 TFLOPS vectorized
    peak_flops_matrix=3.0e12,  # ~3 TFLOPS with AMX (BF16)
    l1_cache_per_core=48 * 1024,
    l2_cache_per_core=512 * 1024,
    l3_cache_shared=30 * 1024**2,
    mem_bandwidth=80e9  # DDR5
)
```

#### GPU Resource Model

```python
@dataclass
class GPUResourceModel:
    """NVIDIA GPU (CUDA model)"""

    # SM structure
    num_sms: int
    cores_per_sm: int
    threads_per_sm: int  # max resident threads
    tensor_cores_per_sm: int

    # Occupancy
    max_blocks_per_sm: int  # typically 16-32
    max_threads_per_block: int  # 1024
    min_occupancy_for_efficiency: float  # 0.25-0.5

    # Memory hierarchy
    registers_per_sm: int
    shared_memory_per_sm: int  # 48-164 KB
    l1_cache_per_sm: int
    l2_cache_shared: int
    hbm_bandwidth: float

    # Performance
    peak_flops_fp32: float
    peak_flops_tensor_core: float  # BF16/FP16

    # Kernel launch overhead
    kernel_launch_us: float  # 5-20 microseconds typical

# Example: H100-PCIe
h100_pcie = GPUResourceModel(
    num_sms=132,
    cores_per_sm=128,
    threads_per_sm=2048,
    tensor_cores_per_sm=4,
    max_blocks_per_sm=32,
    max_threads_per_block=1024,
    min_occupancy_for_efficiency=0.25,
    shared_memory_per_sm=228 * 1024,
    l2_cache_shared=50 * 1024**2,
    hbm_bandwidth=2000e9,
    peak_flops_fp32=60e12,  # 60 TFLOPS FP32
    peak_flops_tensor_core=750e12,  # 750 TFLOPS BF16
    kernel_launch_us=10.0
)
```

#### TPU Resource Model

```python
@dataclass
class TPUResourceModel:
    """Google TPU (systolic array model)"""

    # Systolic array
    array_dimension: int  # 128 for v4
    macs_per_cycle: int  # 128×128 = 16,384

    # Memory
    hbm_capacity: int
    hbm_bandwidth: float
    vector_memory: int  # for non-matmul ops

    # Performance
    peak_tops: float  # trillion ops/sec (INT8)
    peak_tflops_bf16: float

    # Efficiency characteristics
    min_tensor_size_for_efficiency: Tuple[int, int]  # (32, 32)
    optimal_tensor_size: Tuple[int, int]  # (128, 128) or larger

# Example: TPU v4
tpu_v4 = TPUResourceModel(
    array_dimension=128,
    macs_per_cycle=16384,
    hbm_capacity=32 * 1024**3,
    hbm_bandwidth=1200e9,
    vector_memory=16 * 1024**2,
    peak_tops=275e12,
    peak_tflops_bf16=275e12,
    min_tensor_size_for_efficiency=(32, 32),
    optimal_tensor_size=(128, 128)
)
```

#### KPU Resource Model

```python
@dataclass
class KPUResourceModel:
    """KPU wavefront/tile architecture"""

    # Tile engines
    num_tile_engines: int
    tile_memory_per_engine: int  # local SRAM

    # Compute
    peak_tops: float  # INT8
    peak_tflops_fp16: float

    # Memory
    shared_memory: int
    dram_bandwidth: float

    # Tiling
    optimal_tile_size: Tuple[int, int]  # for efficiency

# Example: KPU-T100
kpu_t100 = KPUResourceModel(
    num_tile_engines=16,
    tile_memory_per_engine=4 * 1024**2,
    peak_tops=200e12,
    peak_tflops_fp16=100e12,
    shared_memory=64 * 1024**2,
    dram_bandwidth=1000e9,
    optimal_tile_size=(64, 64)
)
```

### 2.2 HardwareMapper Implementation

```python
class HardwareMapper:
    """Map subgraphs to hardware with utilization estimates"""

    def __init__(self, resource_model: Union[CPUResourceModel, GPUResourceModel, ...]):
        self.resource_model = resource_model
        self.mapper = self._get_mapper()

    def _get_mapper(self):
        """Select appropriate mapper"""
        if isinstance(self.resource_model, CPUResourceModel):
            return CPUMapper(self.resource_model)
        elif isinstance(self.resource_model, GPUResourceModel):
            return GPUMapper(self.resource_model)
        elif isinstance(self.resource_model, TPUResourceModel):
            return TPUMapper(self.resource_model)
        elif isinstance(self.resource_model, KPUResourceModel):
            return KPUMapper(self.resource_model)

    def map_all(self, subgraphs: List[SubgraphDescriptor]) -> MappingReport:
        """Map all subgraphs and generate statistics"""
        mappings = []

        for subgraph in subgraphs:
            mapping = self.mapper.map_subgraph(subgraph)
            mappings.append(mapping)

        return MappingReport(
            mappings=mappings,
            average_utilization=np.mean([m.utilization for m in mappings]),
            utilization_distribution=self._compute_distribution(mappings),
            bottleneck_analysis=self._analyze_bottlenecks(mappings)
        )
```

#### CPU Mapper

```python
class CPUMapper:
    """Map subgraphs to CPU resources"""

    def map_subgraph(self, subgraph: SubgraphDescriptor) -> MappingDescriptor:
        """
        CPU mapping strategy:
        1. Check if operation benefits from vector units
        2. Check if operation can use matrix units (matmul/conv)
        3. Estimate core utilization
        4. Compute effective FLOPS
        """

        # Determine execution mode
        if self._can_use_matrix_unit(subgraph):
            exec_mode = 'matrix_unit'
            effective_flops = self.resource_model.peak_flops_matrix
            utilization = self._estimate_matrix_utilization(subgraph)
        elif self._can_vectorize(subgraph):
            exec_mode = 'vector'
            effective_flops = self.resource_model.peak_flops_vector
            utilization = self._estimate_vector_utilization(subgraph)
        else:
            exec_mode = 'scalar'
            effective_flops = self.resource_model.peak_flops_scalar
            utilization = 1.0  # assume full utilization for scalar

        # Core allocation
        cores_needed = self._estimate_cores(subgraph)
        cores_allocated = min(cores_needed, self.resource_model.num_cores)

        # Memory bottleneck check
        is_memory_bound = self._check_memory_bound(subgraph)

        return MappingDescriptor(
            subgraph_id=subgraph.node_id,
            hardware_type='cpu',
            execution_mode=exec_mode,
            cores_allocated=cores_allocated,
            utilization=utilization,
            effective_flops=effective_flops * utilization,
            is_memory_bound=is_memory_bound,
            bottleneck='memory' if is_memory_bound else 'compute',
            explanation=self._explain_mapping(subgraph, exec_mode, utilization)
        )

    def _can_use_matrix_unit(self, subgraph) -> bool:
        """Check if AMX/matrix unit applicable"""
        if not self.resource_model.has_matrix_unit:
            return False

        # Matrix units good for: matmul, large convs
        if subgraph.operation_type in ['matmul', 'conv2d']:
            # Need sufficiently large tensors
            if subgraph.parallelism.total_threads > 1024:
                return True

        return False

    def _estimate_vector_utilization(self, subgraph) -> float:
        """Estimate how well operation vectorizes"""
        # Depthwise convs vectorize poorly (limited channel parallelism)
        if subgraph.parallelism.is_depthwise:
            return 0.3  # 30% efficiency

        # Standard convs/matmuls vectorize well
        if subgraph.operation_type in ['conv2d', 'matmul']:
            return 0.7  # 70% efficiency (typical for well-tuned code)

        # Elementwise ops vectorize perfectly
        if subgraph.operation_type == 'elementwise':
            return 0.95

        return 0.5  # default

    def _explain_mapping(self, subgraph, exec_mode, utilization) -> str:
        """Human-readable explanation"""
        op_name = subgraph.node_name
        op_type = subgraph.operation_type
        threads = subgraph.parallelism.total_threads

        if exec_mode == 'matrix_unit':
            return (f"{op_name} ({op_type}): {threads:,} threads → "
                   f"AMX matrix unit, {utilization*100:.0f}% efficiency")
        elif exec_mode == 'vector':
            lanes = self.resource_model.vector_lanes_fp32
            return (f"{op_name} ({op_type}): {threads:,} threads → "
                   f"AVX vectorized ({lanes} lanes), {utilization*100:.0f}% efficiency")
        else:
            return (f"{op_name} ({op_type}): {threads:,} threads → "
                   f"scalar execution, {utilization*100:.0f}% efficiency")
```

#### GPU Mapper

```python
class GPUMapper:
    """Map subgraphs to GPU (CUDA model)"""

    def map_subgraph(self, subgraph: SubgraphDescriptor) -> MappingDescriptor:
        """
        GPU mapping:
        1. Compute grid dimensions (blocks × threads)
        2. Estimate SM allocation
        3. Calculate occupancy and utilization
        4. Check memory bandwidth bottleneck
        """

        # Grid configuration
        threads_per_block = self._optimal_block_size(subgraph)
        blocks_needed = (subgraph.parallelism.total_threads + threads_per_block - 1) // threads_per_block

        # SM allocation (consider waves)
        blocks_per_sm = self.resource_model.max_blocks_per_sm
        max_concurrent_blocks = self.resource_model.num_sms * blocks_per_sm

        # Compute waves
        num_waves = (blocks_needed + max_concurrent_blocks - 1) // max_concurrent_blocks
        blocks_per_wave = min(blocks_needed, max_concurrent_blocks)

        # SM utilization
        sms_used = (blocks_per_wave + blocks_per_sm - 1) // blocks_per_sm
        sm_utilization = sms_used / self.resource_model.num_sms

        # Occupancy (threads per SM)
        threads_per_sm = threads_per_block * min(blocks_per_sm, blocks_needed)
        occupancy = threads_per_sm / self.resource_model.threads_per_sm

        # Overall utilization (consider occupancy and multi-wave inefficiency)
        utilization = sm_utilization * min(1.0, occupancy / self.resource_model.min_occupancy_for_efficiency)

        # Tensor core usage
        use_tensor_cores = self._can_use_tensor_cores(subgraph)
        effective_flops = (self.resource_model.peak_flops_tensor_core if use_tensor_cores
                          else self.resource_model.peak_flops_fp32) * utilization

        # Memory bottleneck
        is_memory_bound = self._check_memory_bound(subgraph)

        return MappingDescriptor(
            subgraph_id=subgraph.node_id,
            hardware_type='gpu',
            execution_mode='tensor_core' if use_tensor_cores else 'cuda_core',
            sms_allocated=sms_used,
            blocks_total=blocks_needed,
            blocks_per_wave=blocks_per_wave,
            num_waves=num_waves,
            threads_per_block=threads_per_block,
            occupancy=occupancy,
            sm_utilization=sm_utilization,
            utilization=utilization,
            effective_flops=effective_flops,
            is_memory_bound=is_memory_bound,
            bottleneck='memory' if is_memory_bound else 'compute',
            kernel_launch_overhead_us=self.resource_model.kernel_launch_us,
            explanation=self._explain_mapping(subgraph, sms_used, num_waves, utilization)
        )

    def _explain_mapping(self, subgraph, sms_used, num_waves, utilization) -> str:
        """Detailed explanation of GPU mapping"""
        op_name = subgraph.node_name
        threads = subgraph.parallelism.total_threads
        total_sms = self.resource_model.num_sms

        return (f"{op_name}: {threads:,} threads → "
               f"{sms_used}/{total_sms} SMs ({sms_used/total_sms*100:.0f}%), "
               f"{num_waves} wave{'s' if num_waves > 1 else ''}, "
               f"{utilization*100:.0f}% efficiency")
```

### 2.3 Mapping Report Example

```
Hardware Mapping Report: EfficientNet-B0 on H100-PCIe
======================================================

Overall Statistics:
  Total subgraphs: 152
  Average utilization: 18.3%
  GPU occupancy: 22.1% (29/132 SMs active on average)

Utilization Distribution:
  < 10%:   48 subgraphs (32%) [very inefficient - small kernels]
  10-25%:  72 subgraphs (47%) [poor - limited parallelism]
  25-50%:  28 subgraphs (18%) [moderate]
  > 50%:    4 subgraphs (3%)  [good utilization]

Bottleneck Analysis:
  Compute-bound:  38 subgraphs (25%)
  Memory-bound:   86 subgraphs (57%)
  Bandwidth-bound: 28 subgraphs (18%)

Example Subgraph Mappings:

[1] features_1_conv_0_0 (depthwise 3×3, 32ch, 112×112):
    Total threads: 401,408
    Grid: 1,568 blocks × 256 threads
    SM allocation: 49/132 SMs (37%)
    Waves: 3 (blocks don't fit in one wave)
    Occupancy: 51% (1,024/2,048 threads per SM)
    Utilization: 12.3% (low due to depthwise + multi-wave)
    Bottleneck: Memory (arithmetic intensity: 1.2 FLOPs/byte)
    Effective FLOPS: 92 TFLOPS (vs 750 peak)

[2] features_3_conv_1 (pointwise 1×1, 144→24, 56×56):
    Total threads: 75,264
    Grid: 294 blocks × 256 threads
    SM allocation: 9/132 SMs (7%)
    Waves: 1
    Occupancy: 82%
    Utilization: 5.8% (SM under-utilization dominates)
    Bottleneck: Compute (too small to saturate)
    Effective FLOPS: 44 TFLOPS

[3] classifier_1 (linear 1280→1000):
    Total threads: 1,000
    Grid: 4 blocks × 256 threads
    SM allocation: 1/132 SMs (0.8%)
    Utilization: 0.8%
    Bottleneck: Compute (tiny layer)
    Effective FLOPS: 6 TFLOPS
```

---

## Phase 3: Performance, Energy, and Memory Estimation

**Goal**: Compute realistic latency, energy consumption, and memory footprint with full explainability

### 3.0 Concurrency Analysis

Before computing performance, we need to understand the available concurrency in the computation graph. This is critical for validating our performance estimates.

```python
@dataclass
class ConcurrencyDescriptor:
    """Analysis of available concurrency in computation graph"""

    # Graph-level concurrency
    total_subgraphs: int
    independent_subgraphs: int  # can execute in parallel
    sequential_subgraphs: int  # must execute serially

    # Critical path analysis
    critical_path_length: int  # number of sequential ops
    critical_path_flops: int  # FLOPs on critical path
    parallelizable_flops: int  # FLOPs that can run concurrently

    # Batching potential
    batch_size: int
    batch_parallelism: int  # independent samples in batch
    max_theoretical_speedup: float  # with infinite batch

    # Layer-level concurrency
    subgraph_concurrency: Dict[str, SubgraphConcurrency]

    # Data dependencies
    dependency_graph: nx.DiGraph
    stages: List[List[str]]  # groups of subgraphs that can run in parallel

    # Validation metrics
    concurrency_utilization: float  # actual / theoretical concurrency
    parallelism_efficiency: float  # how well we can use hardware

@dataclass
class SubgraphConcurrency:
    """Concurrency available within a single subgraph"""

    # Thread-level parallelism
    total_threads: int  # batch × channels × spatial
    independent_threads: int  # truly independent (no sharing)

    # Instruction-level parallelism
    independent_operations: int  # ops with no dependencies
    dependency_chains: int  # longest dependency chain

    # Data parallelism
    can_split_batch: bool
    can_split_spatial: bool
    can_split_channels: bool

    # Hardware requirements
    min_hardware_units: int  # minimum cores/SMs needed
    optimal_hardware_units: int  # for best efficiency
    max_hardware_units: int  # diminishing returns beyond this

class ConcurrencyAnalyzer:
    """Analyze available concurrency for performance validation"""

    def analyze(self, subgraphs: List[SubgraphDescriptor],
                partition_report: PartitionReport) -> ConcurrencyDescriptor:
        """
        Analyze concurrency at both graph and subgraph levels

        This analysis helps validate performance estimates:
        - If we have low concurrency but high speedup → model is wrong
        - If we have high concurrency but low speedup → hardware bottleneck
        """

        # Build dependency graph
        dep_graph = partition_report.dependency_graph

        # Find stages (groups of independent subgraphs)
        stages = self._compute_stages(dep_graph)

        # Critical path (longest dependency chain)
        critical_path = self._find_critical_path(dep_graph, subgraphs)

        # Analyze each subgraph's internal concurrency
        subgraph_concurrency = {}
        for sg in subgraphs:
            subgraph_concurrency[sg.node_id] = self._analyze_subgraph_concurrency(sg)

        # Compute theoretical concurrency limits
        max_concurrent_flops = sum(sg.flops for stage in stages
                                   for sg_id in stage
                                   for sg in subgraphs if sg.node_id == sg_id)

        return ConcurrencyDescriptor(
            total_subgraphs=len(subgraphs),
            independent_subgraphs=sum(len(stage) for stage in stages if len(stage) > 1),
            sequential_subgraphs=len([s for s in stages if len(s) == 1]),
            critical_path_length=len(critical_path),
            critical_path_flops=sum(sg.flops for sg in critical_path),
            parallelizable_flops=max_concurrent_flops,
            batch_size=subgraphs[0].parallelism.batch if subgraphs else 1,
            batch_parallelism=self._estimate_batch_parallelism(subgraphs),
            stages=stages,
            dependency_graph=dep_graph,
            subgraph_concurrency=subgraph_concurrency,
            concurrency_utilization=self._compute_utilization(stages, subgraphs),
            explanation=self._explain_concurrency(stages, critical_path)
        )

    def _analyze_subgraph_concurrency(self, subgraph: SubgraphDescriptor) -> SubgraphConcurrency:
        """Analyze concurrency within a single subgraph"""

        if subgraph.operation_type == 'conv2d':
            # Conv2d: independent output pixels
            B = subgraph.parallelism.batch
            C_out = subgraph.parallelism.channels
            spatial = subgraph.parallelism.spatial
            total_threads = B * C_out * spatial

            # Depthwise has limited channel parallelism
            if subgraph.parallelism.is_depthwise:
                independent_threads = B * spatial  # channels are sequential-ish
                optimal_hardware_units = min(32, B * spatial)  # limited by depthwise
            else:
                independent_threads = total_threads
                optimal_hardware_units = min(128, total_threads // 256)  # assume 256 threads/unit

            return SubgraphConcurrency(
                total_threads=total_threads,
                independent_threads=independent_threads,
                independent_operations=total_threads,  # each output pixel independent
                dependency_chains=1,  # convs are embarrassingly parallel
                can_split_batch=True,
                can_split_spatial=True,
                can_split_channels=not subgraph.parallelism.is_depthwise,
                min_hardware_units=1,
                optimal_hardware_units=optimal_hardware_units,
                max_hardware_units=optimal_hardware_units * 2  # diminishing returns after
            )

        # Similar for matmul, elementwise, etc.
        return SubgraphConcurrency(...)

    def _compute_stages(self, dep_graph: nx.DiGraph) -> List[List[str]]:
        """
        Compute stages: groups of nodes that can execute in parallel

        This is critical for understanding graph-level parallelism:
        - Stage 1: [input_conv]
        - Stage 2: [block1_conv1, block2_conv1]  # can run in parallel
        - Stage 3: [block1_conv2, block2_conv2]
        - ...
        """
        stages = []
        remaining_nodes = set(dep_graph.nodes())
        completed_nodes = set()

        while remaining_nodes:
            # Find nodes whose dependencies are all completed
            ready_nodes = []
            for node in remaining_nodes:
                deps = set(dep_graph.predecessors(node))
                if deps.issubset(completed_nodes):
                    ready_nodes.append(node)

            if not ready_nodes:
                break  # circular dependency or error

            stages.append(ready_nodes)
            completed_nodes.update(ready_nodes)
            remaining_nodes -= set(ready_nodes)

        return stages

    def _explain_concurrency(self, stages: List[List[str]], critical_path: List) -> str:
        """Human-readable concurrency explanation"""
        num_stages = len(stages)
        max_parallel = max(len(stage) for stage in stages)

        explanation = f"""
Concurrency Analysis:
  Total stages: {num_stages}
  Max parallel ops per stage: {max_parallel}
  Critical path: {len(critical_path)} sequential operations

  Parallelism potential:
    - Graph-level: {max_parallel}× (max ops in parallel)
    - Batch-level: {critical_path[0].parallelism.batch}× (independent samples)
    - Thread-level: {critical_path[0].parallelism.total_threads:,} (within each op)

  Validation checks:
    ✓ If batch=1: speedup limited to {max_parallel}× by graph structure
    ✓ If batch=32: speedup could reach {max_parallel * 32}× theoretically
    ✓ Hardware with <{max_parallel} units will be under-utilized
"""
        return explanation
```

### 3.1 Roofline Model & Realistic Latency

**Goal**: Compute realistic latency using roofline model (min of compute and memory time)

### 3.1 Roofline Model

```python
class RooflineModel:
    """Compute vs memory bandwidth bottleneck analysis"""

    def compute_latency(self, subgraph: SubgraphDescriptor,
                       mapping: MappingDescriptor,
                       arch_profile: ArchitectureProfile) -> LatencyDescriptor:
        """
        Apply roofline model:
        Latency = max(compute_time, memory_time) + overhead
        """

        # Compute time
        compute_time = subgraph.flops / mapping.effective_flops

        # Memory time (including all traffic: inputs + outputs + weights)
        total_memory_bytes = (subgraph.total_input_bytes +
                             subgraph.total_output_bytes +
                             subgraph.total_weight_bytes)
        memory_time = total_memory_bytes / arch_profile.mem_bandwidth

        # Actual latency is the max (bottleneck)
        compute_latency = compute_time
        memory_latency = memory_time

        bottleneck = 'compute' if compute_time > memory_time else 'memory'
        actual_latency = max(compute_time, memory_time)

        # Add overheads
        if mapping.hardware_type == 'gpu':
            kernel_launch = mapping.kernel_launch_overhead_us * 1e-6
            actual_latency += kernel_launch

        return LatencyDescriptor(
            subgraph_id=subgraph.node_id,
            compute_time=compute_time,
            memory_time=memory_time,
            actual_latency=actual_latency,
            bottleneck=bottleneck,
            overhead=kernel_launch if mapping.hardware_type == 'gpu' else 0,
            explanation=self._explain_latency(subgraph, compute_time, memory_time, bottleneck)
        )

    def _explain_latency(self, subgraph, compute_time, memory_time, bottleneck) -> str:
        """Explain which factor limits performance"""
        op_name = subgraph.node_name

        if bottleneck == 'memory':
            ratio = memory_time / compute_time
            return (f"{op_name}: Memory-bound (BW limit) - "
                   f"memory time {memory_time*1e6:.1f}μs vs "
                   f"compute time {compute_time*1e6:.1f}μs ({ratio:.1f}× slower)")
        else:
            ratio = compute_time / memory_time
            return (f"{op_name}: Compute-bound - "
                   f"compute time {compute_time*1e6:.1f}μs vs "
                   f"memory time {memory_time*1e6:.1f}μs ({ratio:.1f}× slower)")
```

### 3.2 Energy Estimator

**Goal**: Realistic energy consumption modeling based on actual hardware utilization

```python
@dataclass
class EnergyDescriptor:
    """Energy consumption breakdown for a subgraph"""

    subgraph_id: str

    # Component-level energy
    compute_energy_j: float  # energy for FLOPs
    memory_energy_j: float  # energy for data movement
    static_energy_j: float  # idle power during execution
    total_energy_j: float

    # Detailed breakdown
    dram_accesses: int
    dram_energy_j: float
    cache_accesses: int
    cache_energy_j: float
    compute_ops: int
    compute_energy_j: float

    # Utilization impact
    utilization: float
    wasted_energy_j: float  # energy spent on idle resources

    # Comparison
    peak_energy_j: float  # if 100% utilized
    efficiency: float  # actual / peak

    explanation: str

@dataclass
class EnergyReport:
    """Complete energy analysis for model"""

    total_energy_j: float
    total_energy_mj: float  # millijoules
    energy_per_inference_j: float

    # Breakdown
    compute_energy_j: float
    memory_energy_j: float
    static_energy_j: float

    # Efficiency
    average_efficiency: float
    wasted_energy_j: float
    wasted_energy_percent: float

    # Power analysis
    average_power_w: float
    peak_power_w: float

    # Top contributors
    top_energy_consumers: List[Tuple[str, float]]  # (subgraph_id, energy_j)

    # Optimization suggestions
    optimization_opportunities: List[str]

class EnergyEstimator:
    """Estimate energy consumption with hardware-aware modeling"""

    def __init__(self, arch_profile: ArchitectureProfile,
                 resource_model: Union[CPUResourceModel, GPUResourceModel, ...]):
        self.arch = arch_profile
        self.resource_model = resource_model
        self._load_energy_coefficients()

    def _load_energy_coefficients(self):
        """
        Load hardware-specific energy coefficients

        Based on real measurements and published data:
        - DRAM access: ~10-20 pJ/bit
        - L2 cache: ~1-2 pJ/bit
        - L1 cache: ~0.1-0.2 pJ/bit
        - FP32 FLOP: ~0.5-2 pJ (depends on architecture)
        - INT8 operation: ~0.05-0.2 pJ
        - Idle power: varies by architecture
        """
        if isinstance(self.resource_model, GPUResourceModel):
            # H100-PCIe example
            self.energy_per_dram_byte = 30e-12  # 30 pJ/byte (HBM2e)
            self.energy_per_l2_byte = 5e-12  # 5 pJ/byte
            self.energy_per_flop = 0.5e-12  # 0.5 pJ/FLOP (tensor core, BF16)
            self.tdp_watts = 350  # H100-PCIe TDP
            self.idle_power_watts = 50  # ~15% idle power

        elif isinstance(self.resource_model, CPUResourceModel):
            # Intel i7 example
            self.energy_per_dram_byte = 100e-12  # 100 pJ/byte (DDR5, higher than HBM)
            self.energy_per_l3_byte = 10e-12
            self.energy_per_l2_byte = 2e-12
            self.energy_per_l1_byte = 0.5e-12
            self.energy_per_flop = 1.0e-12  # 1 pJ/FLOP (AVX2)
            self.tdp_watts = 125
            self.idle_power_watts = 15

        elif isinstance(self.resource_model, KPUResourceModel):
            # KPU optimized for energy efficiency
            self.energy_per_dram_byte = 20e-12  # efficient memory
            self.energy_per_sram_byte = 0.5e-12  # local SRAM
            self.energy_per_flop = 0.1e-12  # 0.1 pJ/FLOP (INT8, highly efficient)
            self.tdp_watts = 50  # KPU-T100
            self.idle_power_watts = 5

    def estimate_energy(self, subgraph: SubgraphDescriptor,
                       mapping: MappingDescriptor,
                       latency: LatencyDescriptor) -> EnergyDescriptor:
        """
        Estimate energy for a subgraph

        Energy = Compute Energy + Memory Energy + Static Energy
        """

        # Compute energy
        compute_energy = subgraph.flops * self.energy_per_flop

        # Memory energy (detailed hierarchy analysis)
        dram_energy, cache_energy = self._estimate_memory_energy(subgraph, mapping)

        # Static energy (idle power × execution time)
        # Account for under-utilization: unused SMs/cores still consume idle power
        active_power = self.tdp_watts * mapping.utilization
        idle_power = self.idle_power_watts * (1.0 - mapping.utilization)
        static_energy = (active_power + idle_power) * latency.actual_latency

        total_energy = compute_energy + dram_energy + cache_energy + static_energy

        # Wasted energy (energy spent on idle resources)
        peak_energy = self.tdp_watts * latency.actual_latency
        wasted_energy = peak_energy - total_energy

        return EnergyDescriptor(
            subgraph_id=subgraph.node_id,
            compute_energy_j=compute_energy,
            memory_energy_j=dram_energy + cache_energy,
            static_energy_j=static_energy,
            total_energy_j=total_energy,
            dram_energy_j=dram_energy,
            cache_energy_j=cache_energy,
            utilization=mapping.utilization,
            wasted_energy_j=wasted_energy,
            peak_energy_j=peak_energy,
            efficiency=total_energy / peak_energy if peak_energy > 0 else 0,
            explanation=self._explain_energy(subgraph, compute_energy, dram_energy, static_energy)
        )

    def _estimate_memory_energy(self, subgraph, mapping):
        """
        Estimate memory hierarchy energy

        Models cache hits/misses and data movement
        """
        total_bytes = (subgraph.total_input_bytes +
                      subgraph.total_output_bytes +
                      subgraph.total_weight_bytes)

        if isinstance(self.resource_model, GPUResourceModel):
            # GPU: L2 cache hit rate depends on reuse
            l2_hit_rate = self._estimate_cache_hit_rate(subgraph, 'L2')
            l2_bytes = total_bytes * l2_hit_rate
            dram_bytes = total_bytes * (1 - l2_hit_rate)

            cache_energy = l2_bytes * self.energy_per_l2_byte
            dram_energy = dram_bytes * self.energy_per_dram_byte

        elif isinstance(self.resource_model, CPUResourceModel):
            # CPU: L1 → L2 → L3 → DRAM hierarchy
            l1_hit_rate = 0.8  # assume 80% L1 hit for convs
            l2_hit_rate = 0.15
            l3_hit_rate = 0.04

            l1_energy = total_bytes * l1_hit_rate * self.energy_per_l1_byte
            l2_energy = total_bytes * l2_hit_rate * self.energy_per_l2_byte
            l3_energy = total_bytes * l3_hit_rate * self.energy_per_l3_byte
            dram_energy = total_bytes * (1 - l1_hit_rate - l2_hit_rate - l3_hit_rate) * self.energy_per_dram_byte

            cache_energy = l1_energy + l2_energy + l3_energy

        elif isinstance(self.resource_model, KPUResourceModel):
            # KPU: tile-based, most data in local SRAM
            sram_hit_rate = 0.9  # 90% data in local SRAM
            dram_bytes = total_bytes * (1 - sram_hit_rate)
            sram_bytes = total_bytes * sram_hit_rate

            cache_energy = sram_bytes * self.energy_per_sram_byte
            dram_energy = dram_bytes * self.energy_per_dram_byte

        return dram_energy, cache_energy

    def _explain_energy(self, subgraph, compute_energy, dram_energy, static_energy) -> str:
        """Human-readable energy breakdown"""
        total = compute_energy + dram_energy + static_energy
        op_name = subgraph.node_name

        return f"""
{op_name} Energy Breakdown:
  Compute: {compute_energy*1e3:.2f} mJ ({compute_energy/total*100:.0f}%)
  Memory:  {dram_energy*1e3:.2f} mJ ({dram_energy/total*100:.0f}%)
  Static:  {static_energy*1e3:.2f} mJ ({static_energy/total*100:.0f}%)
  Total:   {total*1e3:.2f} mJ

  Efficiency: {(compute_energy/(compute_energy+dram_energy))*100:.0f}% (compute/total_dynamic)
"""

    def generate_report(self, energy_descriptors: List[EnergyDescriptor],
                       latency_report: PerformanceReport) -> EnergyReport:
        """Generate comprehensive energy report"""

        total_energy = sum(e.total_energy_j for e in energy_descriptors)
        compute_energy = sum(e.compute_energy_j for e in energy_descriptors)
        memory_energy = sum(e.memory_energy_j for e in energy_descriptors)
        static_energy = sum(e.static_energy_j for e in energy_descriptors)

        # Top contributors
        sorted_energy = sorted(energy_descriptors, key=lambda e: e.total_energy_j, reverse=True)
        top_consumers = [(e.subgraph_id, e.total_energy_j) for e in sorted_energy[:10]]

        # Optimization opportunities
        optimizations = self._suggest_optimizations(energy_descriptors)

        return EnergyReport(
            total_energy_j=total_energy,
            total_energy_mj=total_energy * 1e3,
            energy_per_inference_j=total_energy,
            compute_energy_j=compute_energy,
            memory_energy_j=memory_energy,
            static_energy_j=static_energy,
            average_efficiency=np.mean([e.efficiency for e in energy_descriptors]),
            wasted_energy_j=sum(e.wasted_energy_j for e in energy_descriptors),
            wasted_energy_percent=(sum(e.wasted_energy_j for e in energy_descriptors) / total_energy * 100),
            average_power_w=total_energy / latency_report.total_latency,
            peak_power_w=self.tdp_watts,
            top_energy_consumers=top_consumers,
            optimization_opportunities=optimizations
        )
```

### 3.3 Memory Estimator

**Goal**: Accurate memory footprint analysis for deployment planning

```python
@dataclass
class MemoryDescriptor:
    """Memory footprint for a subgraph"""

    subgraph_id: str

    # Peak memory (concurrent allocations)
    peak_memory_bytes: int

    # Component breakdown
    activation_memory_bytes: int  # input + output activations
    weight_memory_bytes: int  # model parameters
    workspace_memory_bytes: int  # intermediate buffers

    # Memory access patterns
    read_bytes: int  # total bytes read
    write_bytes: int  # total bytes written
    reuse_factor: float  # how many times data is reused

    # Optimization potential
    can_checkpoint: bool  # can trade memory for compute
    can_quantize: bool  # can reduce precision
    quantization_savings_bytes: int  # FP32 → INT8 savings

    explanation: str

@dataclass
class MemoryReport:
    """Complete memory analysis"""

    # Peak memory (maximum at any point)
    peak_memory_bytes: int
    peak_memory_mb: float
    peak_memory_gb: float

    # Breakdown
    activation_memory_bytes: int
    weight_memory_bytes: int
    workspace_memory_bytes: int

    # Memory efficiency
    average_memory_utilization: float  # actual / peak
    memory_reuse_factor: float  # higher is better

    # Deployment constraints
    fits_in_l2_cache: bool
    fits_in_shared_memory: bool  # for GPU
    fits_on_device: bool  # total memory available

    # Optimization opportunities
    checkpointing_savings_bytes: int
    quantization_savings_bytes: int
    optimization_suggestions: List[str]

    # Memory timeline
    memory_timeline: List[Tuple[str, int]]  # (subgraph_id, memory_at_point)

class MemoryEstimator:
    """Estimate memory footprint with allocation tracking"""

    def __init__(self, resource_model):
        self.resource_model = resource_model

    def estimate_memory(self, subgraphs: List[SubgraphDescriptor],
                       dependency_graph: nx.DiGraph) -> MemoryReport:
        """
        Estimate peak memory by simulating execution

        Key insight: Memory = max over time of concurrent allocations
        Not just sum of all tensors!
        """

        # Simulate execution and track live tensors
        memory_timeline = []
        live_tensors = {}  # tensor_id → size_bytes
        peak_memory = 0

        for subgraph in self._execution_order(subgraphs, dependency_graph):
            # Allocate outputs
            output_size = subgraph.total_output_bytes
            live_tensors[f"{subgraph.node_id}_output"] = output_size

            # Allocate workspace if needed
            workspace = self._estimate_workspace(subgraph)
            if workspace > 0:
                live_tensors[f"{subgraph.node_id}_workspace"] = workspace

            # Current memory
            current_memory = sum(live_tensors.values())
            memory_timeline.append((subgraph.node_id, current_memory))

            if current_memory > peak_memory:
                peak_memory = current_memory

            # Free inputs if no longer needed
            for dep_id in subgraph.depends_on:
                if self._can_free(dep_id, subgraph, subgraphs, dependency_graph):
                    live_tensors.pop(f"{dep_id}_output", None)

        # Add weight memory (persistent)
        total_weight_memory = sum(sg.total_weight_bytes for sg in subgraphs)
        peak_memory += total_weight_memory

        # Analysis
        activation_memory = peak_memory - total_weight_memory
        workspace_memory = max(self._estimate_workspace(sg) for sg in subgraphs)

        # Optimization potential
        checkpointing_savings = self._estimate_checkpointing_savings(subgraphs)
        quantization_savings = self._estimate_quantization_savings(subgraphs)

        return MemoryReport(
            peak_memory_bytes=peak_memory,
            peak_memory_mb=peak_memory / 1024**2,
            peak_memory_gb=peak_memory / 1024**3,
            activation_memory_bytes=activation_memory,
            weight_memory_bytes=total_weight_memory,
            workspace_memory_bytes=workspace_memory,
            average_memory_utilization=np.mean([m for _, m in memory_timeline]) / peak_memory,
            fits_on_device=self._check_fits_on_device(peak_memory),
            checkpointing_savings_bytes=checkpointing_savings,
            quantization_savings_bytes=quantization_savings,
            memory_timeline=memory_timeline,
            optimization_suggestions=self._suggest_memory_optimizations(peak_memory, subgraphs)
        )

    def _estimate_workspace(self, subgraph: SubgraphDescriptor) -> int:
        """
        Estimate workspace memory (e.g., im2col buffers for convs)

        Convolution often requires im2col transformation:
        - Input: N×C_in×H×W
        - Im2col: N×(C_in×K×K)×(H_out×W_out)
        """
        if subgraph.operation_type == 'conv2d':
            # Simplified: assume im2col buffer ~= input size × kernel size
            return subgraph.total_input_bytes * 2  # rough estimate
        return 0

    def _suggest_memory_optimizations(self, peak_memory: int, subgraphs: List) -> List[str]:
        """Suggest concrete memory optimizations"""
        suggestions = []

        # Check if activation checkpointing would help
        activation_memory = sum(sg.total_output_bytes for sg in subgraphs)
        if activation_memory > peak_memory * 0.5:
            suggestions.append(
                f"Activation checkpointing: Could reduce memory by {activation_memory//1024**2} MB "
                f"at cost of 33% more compute"
            )

        # Check quantization potential
        weight_memory = sum(sg.total_weight_bytes for sg in subgraphs)
        if weight_memory > peak_memory * 0.3:
            savings = weight_memory * 0.75  # FP32 → INT8 = 4× reduction
            suggestions.append(
                f"INT8 quantization: Could reduce weight memory by {savings//1024**2} MB"
            )

        return suggestions
```

### 3.4 Performance Report (Updated)

```
Complete Performance Report: EfficientNet-B0 on H100-PCIe
==========================================================

CONCURRENCY ANALYSIS:
  Total stages: 152
  Max parallel ops: 1 (sequential execution with batch=1)
  Critical path: 152 sequential operations
  Thread-level parallelism: 401,408 threads (varies per layer)

  Validation: ✓ Low graph-level parallelism limits speedup at batch=1

PERFORMANCE:
  Total Latency: 45.2 ms (vs 1.88 ms peak = 24× more realistic)
  Average Power: 27.3 W (vs 350W TDP = 7.8% utilization)

  Latency Breakdown:
    Compute:  8.3 ms (18%)
    Memory:   34.7 ms (77%)
    Overhead: 2.2 ms (5%)

  Bottlenecks: 75% memory-bound (114/152 layers)

ENERGY:
  Total Energy: 1.23 J per inference
  Energy Breakdown:
    Compute:  220 mJ (18%)
    Memory:   850 mJ (69%)
    Static:   160 mJ (13%)

  Efficiency: 23% (wasted 950 mJ due to under-utilization)

MEMORY:
  Peak Memory: 114.5 MB
  Breakdown:
    Activations: 92.3 MB (81%)
    Weights:     21.2 MB (19%)
    Workspace:   1.0 MB (1%)

  Fits on device: ✓ (H100 has 80 GB)

OPTIMIZATION OPPORTUNITIES:
  1. Batch=8: 18.2 ms (2.5× faster), 42% utilization
  2. INT8 quantization: 15.1 ms, 4× less memory traffic
  3. Kernel fusion: Save 1.2 ms overhead (28 fusable layers)
  4. Activation checkpointing: Save 70 MB memory

VALIDATION CHECKS:
  ✓ Low utilization (7.8%) consistent with low parallelism
  ✓ Memory-bound (77%) consistent with arithmetic intensity < 3
  ✓ Energy efficiency (23%) matches utilization
```

---

## Phase 4: Validation & Verification

**Goal**: Validate model against known benchmarks

### 4.1 Validation Methodology

1. **Collect real benchmarks**:
   - Use PyTorch profiler on real H100 hardware
   - MLPerf inference benchmarks
   - Vendor-published performance data

2. **Compare predictions**:
   - For each model × hardware combination
   - Compute error: `|predicted - actual| / actual`
   - Target: < 30% error (5-10× better than current peak model)

3. **Iterative refinement**:
   - If error > 30%, analyze which component is wrong
   - Adjust utilization models, overhead estimates
   - Re-validate

### 4.2 Validation Test Suite

```python
# tests/validation/test_realistic_modeling.py

def test_resnet18_h100():
    """Validate ResNet-18 on H100 against known benchmark"""
    # Known: ResNet-18 @ batch=1 achieves ~800 FPS on H100 = 1.25 ms/image
    actual_latency_ms = 1.25

    model = models.resnet18()
    predicted_latency = characterize_with_realistic_model(model, h100_pcie_profile)

    error = abs(predicted_latency - actual_latency_ms) / actual_latency_ms
    assert error < 0.30, f"Error {error*100:.1f}% exceeds 30% threshold"

def test_efficientnet_b0_h100():
    """Validate EfficientNet-B0"""
    # Known: ~2,000 FPS on H100 @ batch=1 = 0.5 ms/image
    actual_latency_ms = 0.5

    model = models.efficientnet_b0()
    predicted_latency = characterize_with_realistic_model(model, h100_pcie_profile)

    error = abs(predicted_latency - actual_latency_ms) / actual_latency_ms
    assert error < 0.30

# Similar tests for CPU, TPU, KPU
```

---

## Implementation Timeline

### Week 1-2: Foundation (Graph Partitioning)
- [ ] Implement SubgraphDescriptor and data structures
- [ ] Implement GraphPartitioner with statistics
- [ ] Implement ConcurrencyAnalyzer
- [ ] Create partition reports with concurrency analysis
- [ ] Validate partition statistics manually
- [ ] **Deliverable**: Partition report for ResNet-18 with concurrency metrics

### Week 3-4: Hardware Mapping
- [ ] Implement CPUResourceModel and CPUMapper (with vector/matrix units)
- [ ] Implement GPUResourceModel and GPUMapper (with wave analysis)
- [ ] Implement TPUResourceModel and TPUMapper (systolic array)
- [ ] Implement KPUResourceModel and KPUMapper (tile-based)
- [ ] Generate mapping reports with utilization breakdowns
- [ ] **Deliverable**: Mapping report for EfficientNet-B0 on all architectures

### Week 5: Performance, Energy, Memory Estimation
- [ ] Implement RooflineModel for realistic latency
- [ ] Implement EnergyEstimator with component-level breakdown
- [ ] Implement MemoryEstimator with allocation tracking
- [ ] Integrate all three estimators
- [ ] Generate unified performance/energy/memory reports
- [ ] **Deliverable**: Complete report for MobileNet-V2 with all metrics

### Week 6: Validation & Refinement
- [ ] Collect real benchmark data (H100, CPU, TPU if available)
- [ ] Implement validation test suite
- [ ] Compare predictions vs reality for 5+ models
- [ ] Iterate on utilization models to achieve < 30% error
- [ ] Document limitations and accuracy bounds
- [ ] **Deliverable**: Validation report with error analysis

---

## Success Criteria

1. **Explainability**: Every prediction (latency, energy, memory) has detailed breakdown
2. **Accuracy**: < 30% error vs real benchmarks (vs current 500-2000% error)
3. **Coverage**: Works for CPU (with vector/matrix units), GPU, TPU, KPU architectures
4. **Statistics**: Rich reports showing:
   - Concurrency analysis (validation of parallelism assumptions)
   - Hardware utilization (why resources are under-utilized)
   - Bottleneck identification (compute vs memory)
   - Energy breakdown (compute vs memory vs static)
   - Memory timeline (peak allocation analysis)
5. **Actionable**: Reports suggest concrete optimizations:
   - Batching strategies (quantified speedup)
   - Kernel fusion opportunities (overhead savings)
   - Quantization impact (memory/energy/latency trade-offs)
   - Memory optimizations (checkpointing, recomputation)
6. **Validation**: Concurrency metrics enable validation:
   - If low concurrency → speedup should be limited
   - If high utilization → energy efficiency should be high
   - If memory-bound → latency insensitive to compute upgrades

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Validation data unavailable | Can't verify accuracy | Start with PyTorch profiler on available hardware |
| Model too complex | Hard to maintain | Start simple (Phase 1), iterate |
| Hardware models inaccurate | Wrong predictions | Use conservative estimates, document assumptions |
| Overhead dominates small models | Poor accuracy | Model kernel launch explicitly |

---

## Future Extensions

- Dynamic batching analysis (how does latency scale with batch size?)
- Kernel fusion optimizer (which fusions provide best speedup?)
- Mixed precision analysis (FP32 vs BF16 vs INT8)
- Multi-GPU modeling (pipeline and tensor parallelism)
- Cost modeling ($ per inference on cloud)

---

## Questions for Review

1. Is the phased approach reasonable (6 weeks)?
2. Should we prioritize certain hardware (GPU first)?
3. What validation benchmarks do we have access to?
4. Should we model kernel fusion optimization in Phase 1?
