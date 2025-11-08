# KPU Energy Modeling Implementation Plan

**Date**: 2025-11-08
**Objective**: Develop accurate tile-based energy model for KPU to enable fair comparison with CPU, GPU, and TPU architectures

---

## Executive Summary

The KPU (Knowledge Processing Unit) is a **Domain Flow Architecture (DFA)** that differs fundamentally from CPU (MIMD stored program), GPU (SIMT data parallel), and TPU (weight-stationary systolic array). The KPU is a **programmable spatial dataflow machine** that executes Systems of Uniform Recurrence Equations (SUREs) through a hierarchical memory system feeding a systolic compute fabric.

**Key Differentiators**:
- **vs CPU**: Spatial dataflow eliminates instruction fetch/decode overhead per operation
- **vs GPU**: Push-based execution eliminates warp scheduling and occupancy management
- **vs TPU**: Programmable fabric supports any linear algebra operator, not just GEMM

**Energy Modeling Challenge**: The KPU's energy profile is dominated by:
1. **Memory hierarchy traversal** (L3 → L2 → L1 → Fabric)
2. **Data movement engines** (DMA, BlockMover, Streamer)
3. **Distributed token routing** in systolic fabric
4. **Token signature matching** (distributed CAM)

This plan details implementation of a comprehensive KPU energy model comparable to the existing TPU tile energy model.

---

## Part 1: Architectural Analysis

### 1.1 KPU vs TPU: Fundamental Differences

| Aspect | TPU | KPU |
|--------|-----|-----|
| **Execution Model** | Weight-stationary systolic | Token-based spatial dataflow |
| **Programming** | Fixed GEMM hardware | Programmable SURE execution |
| **Memory Hierarchy** | Weight Memory → FIFO → Array → Accumulators → Unified Buffer | L3 → L2 → L1 → Fabric (3-level scratchpad) |
| **Data Movement** | Single DMA + Weight FIFO | DMA → BlockMover → Streamer (3 engines) |
| **Control Overhead** | Minimal (fixed schedule) | SURE program loading per operator |
| **Operator Coverage** | GEMM optimized | All linear algebra (BLAS, solvers, etc.) |
| **Token Matching** | Not applicable | Distributed CAM at each PE |
| **Fusion** | Limited (edge operations) | Automatic via dataflow |

### 1.2 KPU Memory Hierarchy Energy Profile

**Critical Insight**: Unlike TPU's 2-stage memory (DRAM → Unified Buffer), KPU has **4-stage hierarchy**:

```
DRAM (off-chip)
  ↓ DMA Engine (system schedule)
L3 Scratchpad (1-8 MB tiles, distributed)
  ↓ BlockMover Engine (block schedule)
L2 Scratchpad (16-64 KB tiles, high concurrency)
  ↓ Streamer (vector → token stream)
L1 Stream Buffers (immediate fabric access)
  ↓ Systolic Fabric (token-based execution)
```

**Energy Implications**:
- **4× more memory stages** than TPU (more opportunities for energy optimization)
- **3× more data movement engines** (DMA, BlockMover, Streamer vs TPU's single DMA)
- **Distributed scratchpad** reduces routing distance but adds complexity

### 1.3 KPU Compute Fabric Energy Profile

**Token-Based Execution** (unique to KPU):

```
Every token carries:
  - Payload: computational data
  - Signature: (spatial_tag, recurrence_variable)

PE Operation Sequence:
  1. Token arrives at PE → signature matching (CAM-like)
  2. Match found → execute SURE instruction
  3. Result → form new token with updated signature
  4. Route token to next PE in spatial structure
```

**Energy Components**:
1. **Token signature matching** (every token arrival, every PE)
2. **Signature comparison** (spatial tag + recurrence variable)
3. **Distributed CAM** (replaces centralized dataflow matching)
4. **Token formation** (result packaging + signature update)
5. **Inter-PE routing** (token carries routing information)

**Contrast with TPU**:
- TPU: Data values flow, no signature matching
- KPU: Tokens flow, every PE performs matching

### 1.4 KPU Product Families

| Family | Compute Tiles | PE Count | Process Node | Target Market | Power Budget |
|--------|---------------|----------|--------------|---------------|--------------|
| **KPU-T64** | 64 | 16K-64K | 22nm | Edge/Drone | 5-15W |
| **KPU-T256** | 256 | 256K-1M | 16nm/7nm | Robotics | 25-75W |
| **KPU-T768** | 768 | 3M-12M | 16nm/7nm/4nm | Automotive | 75-250W |

**Modeling Implication**: Need parameterized model that scales across 3 families.

---

## Part 2: Energy Event Catalog (from Operations Manual)

### 2.1 Memory Subsystem Energy Events

#### DRAM Operations
```python
# Energy events from Section 3.1.1
DRAM_READ = {
    'base_energy_per_byte': 10e-12,  # 10 pJ/byte (HBM2e baseline)
    'row_activation': 100e-12,       # 100 pJ per row
    'column_access': 5e-12,          # 5 pJ per column
    'io_driver': 2e-12,              # 2 pJ per byte
    'burst_bonus': 0.7,              # 30% reduction for bursts
}

DRAM_WRITE = {
    'base_energy_per_byte': 12e-12,  # 12 pJ/byte (write > read)
    'row_activation': 100e-12,
    'column_access': 5e-12,
    'io_driver': 2e-12,
    'write_buffer': 3e-12,
    'burst_bonus': 0.7,
}
```

#### L3 Scratchpad (1-8 MB tiles, distributed)
```python
L3_SCRATCHPAD = {
    'read_energy_per_byte': 2.0e-12,    # 2 pJ/byte (large SRAM)
    'write_energy_per_byte': 2.5e-12,   # 2.5 pJ/byte
    'address_decode': 0.5e-12,          # Per access
    'distance_factor': 1.2,             # Distributed tiles (variable distance)
}
```

#### L2 Scratchpad (16-64 KB tiles, high concurrency)
```python
L2_SCRATCHPAD = {
    'read_energy_per_byte': 0.8e-12,    # 0.8 pJ/byte (medium SRAM)
    'write_energy_per_byte': 1.0e-12,   # 1.0 pJ/byte
    'port_arbitration': 0.3e-12,        # Concurrent access overhead
    'buffer_swap': 0.1e-12,             # Double/triple buffering
}
```

#### L1 Stream Buffers (immediate fabric access)
```python
L1_STREAM_BUFFER = {
    'read_energy_per_byte': 0.3e-12,    # 0.3 pJ/byte (small, fast SRAM)
    'write_energy_per_byte': 0.4e-12,   # 0.4 pJ/byte
    'single_cycle': True,               # No decode overhead
}
```

### 2.2 Data Movement Engine Energy Events

#### DMA Engine (DRAM ↔ L3)
```python
DMA_ENGINE = {
    'transfer_setup': 50e-12,               # 50 pJ per transfer
    'address_generation': 5e-12,            # Per burst
    'on_chip_interconnect_per_byte': 1.5e-12,  # NoC traversal
    'descriptor_fetch': 20e-12,             # Per DMA descriptor
    'completion_signal': 10e-12,            # Per transfer
}
```

#### BlockMover Engine (L3 ↔ L2)
```python
BLOCKMOVER_ENGINE = {
    'transfer_setup': 20e-12,               # 20 pJ per block transfer
    'address_calculation': 2e-12,           # Per block
    'on_chip_routing_per_byte': 0.8e-12,    # Shorter distance than DMA
    'block_descriptor': 10e-12,             # Per block
}
```

#### Streamer (L2 ↔ L1, token stream generation)
```python
STREAMER = {
    'vector_read': 5e-12,                   # Per vector from L2
    'vector_write': 5e-12,                  # Per vector to L2
    'token_formation': 0.5e-12,             # Per token (data + signature)
    'stream_sequencing': 0.3e-12,           # Per token (row/col/diag pattern)
    'transposition': 1.0e-12,               # Per element (on-the-fly)
    'signature_generation': 0.4e-12,        # Spatial + recurrence tag
}
```

### 2.3 Compute Fabric Energy Events

#### Token Routing (KPU-specific!)
```python
TOKEN_ROUTING = {
    'inter_pe_transfer': 0.2e-12,           # Per token, nearest-neighbor
    'wire_capacitance': 0.1e-12,            # Proportional to distance
    'router_logic': 0.05e-12,               # Per hop
    'fabric_edge_injection': 0.3e-12,       # Into fabric
    'fabric_edge_collection': 0.3e-12,      # From fabric
}
```

#### Processing Element (PE) Operations
```python
PE_OPERATIONS = {
    # Token signature matching (UNIQUE TO KPU!)
    'signature_matching': 0.6e-12,          # Per token arrival (CAM-like)
    'spatial_tag_compare': 0.3e-12,         # Tag width dependent
    'recurrence_var_compare': 0.3e-12,      # Variable matching

    # Arithmetic (similar to TPU, but token-wrapped)
    'mac_int8': 0.2e-12,                    # INT8 MAC
    'mac_fp16': 0.4e-12,                    # FP16 MAC
    'mac_bf16': 0.25e-12,                   # BF16 MAC
    'mac_fp32': 0.8e-12,                    # FP32 MAC

    # Token formation
    'result_token_formation': 0.4e-12,      # Package result + signature
    'signature_update': 0.2e-12,            # Update for next PE
    'routing_tag_generation': 0.1e-12,      # Where to send token

    # Instruction decode (amortized across lock-step)
    'instruction_decode': 0.3e-12,          # Per cycle (shared)
}
```

#### SURE Program Loading (KPU-specific!)
```python
PROGRAM_LOADING = {
    'program_broadcast_per_pe': 50e-12,     # One-time per operator
    'instruction_memory_write': 10e-12,     # Per instruction
    'broadcast_network': 20e-12,            # Fanout energy
}
```

### 2.4 Optimization Features Energy

#### Operator Fusion (Hardware-driven)
```python
OPERATOR_FUSION = {
    'bias_addition': 0.3e-12,               # Per element at fabric edge
    'activation_sfu': {
        'relu': 0.1e-12,                    # Trivial
        'sigmoid': 1.5e-12,                 # Lookup table
        'tanh': 1.5e-12,
        'gelu': 2.0e-12,                    # Polynomial approx
    },
    'avoided_l2_write': -1.0e-12,           # Energy SAVED
    'avoided_l2_read': -0.8e-12,            # Energy SAVED
}
```

#### Quantization (Just-in-time)
```python
QUANTIZATION = {
    'fp32_to_fp16': 0.5e-12,                # Per element
    'fp16_to_int8': 0.3e-12,
    'int8_to_fp16': 0.4e-12,                # Dequantization
    'scale_factor_fetch': 0.2e-12,          # Per tensor
}
```

#### Sparsity (Zero-value gating)
```python
SPARSITY = {
    'zero_detection': 0.1e-12,              # Per element
    'clock_gate_save': -0.5e-12,            # Energy SAVED per zero
    'sparse_encoding': 0.8e-12,             # Per block (CSR/COO)
    'sparse_decoding': 0.6e-12,             # Per block
}
```

---

## Part 3: Implementation Plan

### 3.1 Phase 1: Core KPU Energy Model (Week 1-2)

**Objective**: Implement basic tile-based energy model for GEMM operator

#### 3.1.1 Create KPUTileEnergyModel Class

**File**: `src/graphs/hardware/architectural_energy.py`

```python
@dataclass
class KPUTileEnergyModel:
    """
    Tile-based energy model for KPU Domain Flow Architecture.

    Captures 8 major energy components:
    1. DRAM access (via DMA engines)
    2. L3 scratchpad operations (distributed tiles)
    3. BlockMover data movement (L3 → L2)
    4. L2 scratchpad operations (high concurrency)
    5. Streamer operations (L2 → L1, token formation)
    6. L1 stream buffer access
    7. Token routing in systolic fabric
    8. PE computation + token signature matching
    """

    # ===== Product Family Configuration =====
    product_family: str  # "T64", "T256", "T768"
    process_node: int    # 22, 16, 7, 4 (nm)

    # ===== Memory Hierarchy Parameters =====
    # L3 Scratchpad (distributed tiles)
    l3_total_size: int                      # Total L3 across all tiles
    l3_tile_size: int                       # Individual tile size
    l3_num_tiles: int                       # Number of L3 tiles
    l3_read_energy_per_byte: float
    l3_write_energy_per_byte: float
    l3_distance_factor: float               # Distributed routing overhead

    # L2 Scratchpad (concurrent access)
    l2_total_size: int                      # Total L2 across compute tiles
    l2_tile_size: int                       # Per compute tile
    l2_num_tiles: int                       # Number of L2 tiles
    l2_read_energy_per_byte: float
    l2_write_energy_per_byte: float
    l2_port_arbitration_energy: float       # Concurrent access overhead

    # L1 Stream Buffers
    l1_total_size: int
    l1_read_energy_per_byte: float
    l1_write_energy_per_byte: float

    # ===== Data Movement Engine Parameters =====
    # DMA Engine (DRAM ↔ L3)
    dma_num_engines: int
    dma_transfer_setup_energy: float
    dma_address_gen_energy: float
    dma_noc_energy_per_byte: float          # NoC traversal

    # BlockMover Engine (L3 ↔ L2)
    blockmover_num_engines: int
    blockmover_setup_energy: float
    blockmover_routing_energy_per_byte: float

    # Streamer (L2 → L1, token stream generation)
    streamer_num_units: int
    streamer_vector_read_energy: float
    streamer_vector_write_energy: float
    streamer_token_formation_energy: float  # KPU-specific!
    streamer_signature_gen_energy: float    # KPU-specific!
    streamer_transpose_energy_per_element: float

    # ===== Compute Fabric Parameters =====
    # Fabric dimensions
    fabric_width: int                       # PEs per row
    fabric_height: int                      # PEs per column
    num_fabrics: int                        # Number of compute tiles

    # Token routing (KPU-specific!)
    token_inter_pe_energy: float            # Per token hop
    token_edge_injection_energy: float      # Into fabric
    token_edge_collection_energy: float     # From fabric

    # PE operations
    pe_signature_matching_energy: float     # UNIQUE TO KPU!
    pe_mac_energy: Dict[str, float]         # By precision
    pe_token_formation_energy: float        # Result packaging
    pe_instruction_decode_energy: float     # Amortized

    # SURE program loading (KPU-specific!)
    program_broadcast_energy_per_pe: float
    program_load_frequency: str             # "per_operator"

    # ===== Clock and Frequency =====
    fabric_clock_frequency_hz: float
    memory_clock_frequency_hz: float

    # ===== Optimization Features =====
    # Operator fusion
    fusion_bias_add_energy: float
    fusion_sfu_energy: Dict[str, float]     # By activation type
    fusion_l2_write_save: float             # Energy avoided
    fusion_l2_read_save: float              # Energy avoided

    # Quantization (just-in-time)
    quantization_energy: Dict[str, float]   # By conversion type

    # Sparsity (zero-value gating)
    sparsity_zero_detect_energy: float
    sparsity_clock_gate_save: float         # Energy saved per zero
    sparsity_encoding_energy: float         # Sparse format conversion


    def compute_gemm_energy(
        self,
        M: int,              # Matrix dimensions
        N: int,
        K: int,
        batch_size: int = 1,
        precision: str = "BF16",
        activation: str = None,      # Optional fused activation
        sparsity_ratio: float = 0.0,  # Weight sparsity
    ) -> Dict[str, float]:
        """
        Compute energy breakdown for GEMM operation.

        Returns detailed energy breakdown across all subsystems.
        """

        # 1. Calculate total operations
        total_ops = 2 * M * N * K * batch_size  # MACs × 2 for FLOPs

        # 2. Memory hierarchy energy
        #    - DRAM → L3 (weights, inputs)
        #    - L3 → L2 (BlockMover staging)
        #    - L2 → L1 (Streamer token generation)
        #    - L1 → Fabric (stream injection)

        # 3. Data movement engine energy
        #    - DMA transfers
        #    - BlockMover transfers
        #    - Streamer operations + token formation

        # 4. Compute fabric energy
        #    - Token routing through PEs
        #    - Signature matching (EVERY token!)
        #    - MAC operations
        #    - Result token formation

        # 5. SURE program loading (if first operator or operator change)

        # 6. Optimization features
        #    - Fused activation (if specified)
        #    - Sparsity savings (if applicable)

        # Return comprehensive breakdown
        return {
            # Memory hierarchy
            'dram_read_energy_j': ...,
            'dram_write_energy_j': ...,
            'l3_read_energy_j': ...,
            'l3_write_energy_j': ...,
            'l2_read_energy_j': ...,
            'l2_write_energy_j': ...,
            'l1_read_energy_j': ...,
            'l1_write_energy_j': ...,

            # Data movement engines
            'dma_transfer_energy_j': ...,
            'blockmover_energy_j': ...,
            'streamer_energy_j': ...,
            'streamer_token_formation_j': ...,  # KPU-specific

            # Compute fabric
            'token_routing_energy_j': ...,       # KPU-specific
            'signature_matching_energy_j': ...,  # KPU-specific!
            'mac_energy_j': ...,
            'token_formation_energy_j': ...,     # KPU-specific

            # Control overhead
            'program_loading_energy_j': ...,     # KPU-specific

            # Optimizations
            'fusion_energy_j': ...,              # If activation fused
            'fusion_savings_j': ...,             # Memory writes avoided
            'sparsity_savings_j': ...,           # If sparse

            # Totals
            'total_compute_energy_j': ...,
            'total_memory_energy_j': ...,
            'total_control_energy_j': ...,
            'total_energy_j': ...,
        }
```

#### 3.1.2 Create KPU Resource Models

**File**: `src/graphs/hardware/models/edge/kpu_t64.py`

```python
def kpu_t64_resource_model() -> HardwareResourceModel:
    """
    KPU-T64: Edge and Drone Applications

    Configuration:
    - 64 compute tiles (8×8 checkerboard)
    - 16×16 to 32×32 PE fabric per tile
    - 22nm process (cost-optimized)
    - 5-15W power budget
    """

    # Tile energy model
    tile_energy_model = KPUTileEnergyModel(
        # Product family
        product_family="T64",
        process_node=22,

        # Memory hierarchy (conservative for 22nm)
        l3_total_size=16 * 1024 * 1024,      # 16 MB total
        l3_tile_size=512 * 1024,             # 512 KB per tile
        l3_num_tiles=32,                     # 32 L3 tiles
        l3_read_energy_per_byte=2.5e-12,     # 2.5 pJ/byte (22nm SRAM)
        l3_write_energy_per_byte=3.0e-12,
        l3_distance_factor=1.3,              # Distributed routing

        l2_total_size=8 * 1024 * 1024,       # 8 MB total (128 KB × 64 tiles)
        l2_tile_size=128 * 1024,
        l2_num_tiles=64,
        l2_read_energy_per_byte=1.0e-12,     # 1.0 pJ/byte
        l2_write_energy_per_byte=1.2e-12,
        l2_port_arbitration_energy=0.4e-12,

        l1_total_size=1 * 1024 * 1024,       # 1 MB total (16 KB × 64 tiles)
        l1_read_energy_per_byte=0.4e-12,
        l1_write_energy_per_byte=0.5e-12,

        # Data movement engines
        dma_num_engines=8,
        dma_transfer_setup_energy=60e-12,    # 22nm overhead
        dma_address_gen_energy=6e-12,
        dma_noc_energy_per_byte=1.8e-12,

        blockmover_num_engines=64,           # One per compute tile
        blockmover_setup_energy=25e-12,
        blockmover_routing_energy_per_byte=1.0e-12,

        streamer_num_units=64,               # One per compute tile
        streamer_vector_read_energy=6e-12,
        streamer_vector_write_energy=6e-12,
        streamer_token_formation_energy=0.6e-12,
        streamer_signature_gen_energy=0.5e-12,
        streamer_transpose_energy_per_element=1.2e-12,

        # Compute fabric (assume 32×32 per tile for T64)
        fabric_width=32,
        fabric_height=32,
        num_fabrics=64,

        # Token routing (22nm)
        token_inter_pe_energy=0.3e-12,
        token_edge_injection_energy=0.4e-12,
        token_edge_collection_energy=0.4e-12,

        # PE operations (22nm)
        pe_signature_matching_energy=0.8e-12,  # Higher at 22nm
        pe_mac_energy={
            'INT8': 0.25e-12,
            'FP16': 0.5e-12,
            'BF16': 0.3e-12,
            'FP32': 1.0e-12,
        },
        pe_token_formation_energy=0.5e-12,
        pe_instruction_decode_energy=0.4e-12,

        # SURE program loading
        program_broadcast_energy_per_pe=60e-12,
        program_load_frequency="per_operator",

        # Clocks
        fabric_clock_frequency_hz=1.0e9,     # 1.0 GHz (conservative for 22nm)
        memory_clock_frequency_hz=800e6,     # 800 MHz

        # Optimization features
        fusion_bias_add_energy=0.4e-12,
        fusion_sfu_energy={
            'relu': 0.15e-12,
            'sigmoid': 2.0e-12,
            'gelu': 2.5e-12,
        },
        fusion_l2_write_save=-1.2e-12,
        fusion_l2_read_save=-1.0e-12,

        quantization_energy={
            'fp32_to_fp16': 0.6e-12,
            'fp16_to_int8': 0.4e-12,
        },

        sparsity_zero_detect_energy=0.15e-12,
        sparsity_clock_gate_save=-0.6e-12,
        sparsity_encoding_energy=1.0e-12,
    )

    # Create resource model
    model = HardwareResourceModel(
        name="KPU-T64",
        hardware_type=HardwareType.KPU,  # NEW TYPE!
        compute_units=64,                # 64 compute tiles
        threads_per_unit=1024,           # 32×32 PEs per tile
        warps_per_unit=32,               # Rows
        warp_size=32,                    # Columns

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=4e12,   # 4 TOPS (conservative)
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=2e12,   # 2 TFLOPS
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=50e9,             # 50 GB/s LPDDR4
        l1_cache_per_unit=16 * 1024,     # L1 per tile
        l2_cache_total=8 * 1024 * 1024,  # L2 total
        main_memory=4 * 1024**3,         # 4 GB LPDDR4

        energy_per_flop_fp32=0.5e-12,    # Conservative for 22nm
        energy_per_byte=10e-12,          # LPDDR4 energy

        min_occupancy=0.7,               # Spatial dataflow has good utilization
        max_concurrent_kernels=1,        # Single operator at a time
        wave_quantization=1,
    )

    # Attach tile energy model
    model.tile_energy_model = tile_energy_model

    return model
```

Similarly create:
- `src/graphs/hardware/models/robotics/kpu_t256.py` (16nm/7nm, 25-75W)
- `src/graphs/hardware/models/automotive/kpu_t768.py` (16nm/7nm/4nm, 75-250W)

#### 3.1.3 Create KPU Mapper

**File**: `src/graphs/hardware/mappers/accelerators/kpu.py`

```python
class KPUMapper(HardwareMapper):
    """
    Hardware mapper for KPU Domain Flow Architecture.

    Maps computation graphs to KPU's programmable spatial dataflow fabric.
    Handles memory hierarchy traversal and SURE program generation.
    """

    def __init__(self, resource_model: HardwareResourceModel, batch_size: int = 1):
        super().__init__(resource_model, batch_size)
        self.tile_energy_model = resource_model.tile_energy_model

    def _calculate_energy(
        self,
        ops: int,
        bytes_transferred: int,
        precision: Precision
    ) -> Tuple[float, float]:
        """
        Calculate energy using KPU tile-based model.

        Unlike TPU (2-stage: weights → compute), KPU has 4-stage hierarchy:
        DRAM → L3 → L2 → L1 → Fabric

        Plus KPU-specific costs:
        - Token signature matching
        - SURE program loading
        - Multi-engine data movement
        """

        if self.tile_energy_model is None:
            # Fallback to base model
            return super()._calculate_energy(ops, bytes_transferred, precision)

        # Estimate GEMM dimensions from ops and bytes
        # (This is a simplification; real implementation would use graph analysis)
        M, N, K = self._estimate_gemm_dimensions(ops, bytes_transferred, precision)

        # Use tile energy model
        energy_breakdown = self.tile_energy_model.compute_gemm_energy(
            M=M,
            N=N,
            K=K,
            batch_size=self.batch_size,
            precision=precision.name,
        )

        # Aggregate compute vs memory
        compute_energy = (
            energy_breakdown['mac_energy_j'] +
            energy_breakdown['signature_matching_energy_j'] +  # KPU-specific!
            energy_breakdown['token_formation_energy_j'] +     # KPU-specific!
            energy_breakdown['token_routing_energy_j']         # KPU-specific!
        )

        memory_energy = (
            energy_breakdown['dram_read_energy_j'] +
            energy_breakdown['dram_write_energy_j'] +
            energy_breakdown['l3_read_energy_j'] +
            energy_breakdown['l3_write_energy_j'] +
            energy_breakdown['l2_read_energy_j'] +
            energy_breakdown['l2_write_energy_j'] +
            energy_breakdown['l1_read_energy_j'] +
            energy_breakdown['l1_write_energy_j'] +
            energy_breakdown['dma_transfer_energy_j'] +
            energy_breakdown['blockmover_energy_j'] +
            energy_breakdown['streamer_energy_j']
        )

        return compute_energy, memory_energy
```

### 3.2 Phase 2: Validation Testing (Week 3)

#### 3.2.1 Create KPU Unit Tests

**File**: `tests/hardware/test_kpu_tile_energy.py`

Tests:
1. KPU-T64 configuration validation
2. GEMM energy calculation (compare with analytical model)
3. Memory hierarchy energy breakdown
4. Token signature matching overhead quantification
5. SURE program loading amortization
6. Operator fusion energy savings

#### 3.2.2 Create KPU Comparison Tests

**File**: `tests/hardware/test_kpu_vs_tpu_comparison.py`

Compare KPU vs TPU for:
1. GEMM (same workload, different architectures)
2. Conv2D (KPU programmable advantage)
3. Attention (multi-operator fusion benefits)

Expected results:
- **GEMM**: TPU slightly more energy-efficient (specialized hardware)
- **Conv2D**: KPU competitive (programmable fabric adapts)
- **Attention**: KPU wins (automatic fusion of Q, K, V, attention, output)

### 3.3 Phase 3: Multi-Architecture Comparison (Week 4)

#### 3.3.1 Create 4-Way Comparison Test

**File**: `tests/hardware/test_cpu_gpu_tpu_kpu_comparison.py`

**Test Configuration**:
```python
# Same workload (BERT-Large, seq=512, batch=64)
# Different architectures:
architectures = [
    ('Intel-Xeon-8380', CPU),       # 40 cores, AVX-512
    ('NVIDIA-H100', GPU),           # 16896 CUDA cores, 528 Tensor Cores
    ('TPU-v4', TPU),                # 2× 128×128 systolic arrays
    ('KPU-T256', KPU),              # 256 compute tiles, 256K-1M PEs
]
```

**Metrics to Compare**:
```
Energy Breakdown:
  - Compute energy (MAC operations)
  - Memory energy (hierarchy traversal)
  - Control energy (instruction fetch, scheduling, etc.)
  - Total energy per inference

Energy per MAC:
  - CPU: 5-10 pJ/MAC (instruction overhead dominates)
  - GPU: 1-2 pJ/MAC (warp scheduling + occupancy overhead)
  - TPU: 0.8-1.2 pJ/MAC (weight-stationary efficiency)
  - KPU: 1.0-1.5 pJ/MAC (token matching + routing overhead)

Compute %:
  - CPU: 40-60% (memory-bound due to cache misses)
  - GPU: 70-85% (SM scheduling overhead, bank conflicts)
  - TPU: 95-98% (systolic array efficiency)
  - KPU: 90-95% (spatial dataflow, token matching overhead)

Memory Hierarchy Complexity:
  - CPU: L1 → L2 → L3 → DRAM (4 levels, cache coherence overhead)
  - GPU: Register → L1 → L2 → HBM (4 levels, broadcast/coalesce)
  - TPU: Weight Memory → Unified Buffer → DRAM (2 levels, simple)
  - KPU: L3 → L2 → L1 → Fabric → DRAM (4 levels, distributed)

Control Overhead:
  - CPU: HIGH (instruction fetch/decode per operation)
  - GPU: MEDIUM (warp scheduling, divergence handling)
  - TPU: LOW (fixed systolic schedule, weight-stationary)
  - KPU: LOW-MEDIUM (SURE program loading, token matching)
```

**Expected Results**:

| Architecture | Energy/Inference | Energy/MAC | Compute% | Memory% | Control% |
|--------------|------------------|------------|----------|---------|----------|
| **CPU** | ~500 mJ | ~5 pJ | 50% | 40% | 10% |
| **GPU** | ~150 mJ | ~1.5 pJ | 75% | 20% | 5% |
| **TPU** | ~70 mJ | ~0.8 pJ | 97% | 3% | <1% |
| **KPU** | ~90 mJ | ~1.1 pJ | 92% | 7% | 1% |

**Interpretation**:
- **TPU wins on pure GEMM** (specialized weight-stationary)
- **KPU competitive** despite programmability (~15% overhead vs TPU)
- **KPU wins on multi-operator graphs** (automatic fusion)
- **GPU better than CPU** but still 2× worse than TPU/KPU
- **CPU worst** (general-purpose overhead)

---

## Part 4: Key Modeling Challenges

### 4.1 Token Signature Matching Energy

**Challenge**: Every token arrival at every PE triggers signature matching

**Impact**:
```python
# BERT-Large (seq=512, batch=64)
num_tokens = batch_size * seq_len * hidden_size * num_layers
            = 64 * 512 * 1024 * 24
            = 805,306,368 tokens

signature_matching_events = num_tokens * avg_hops_per_token
                          = 805M * 3  # Assume 3 hops avg
                          = 2.4 billion matching operations

energy = 2.4e9 * 0.6e-12 = 1.44 mJ (just for signature matching!)
```

**This is unique to KPU** - TPU has no signature matching overhead.

**Solution**: Model explicitly, validate with RTL simulation data (if available)

### 4.2 SURE Program Loading Amortization

**Challenge**: Program loading is per-operator, but amortized over many data items

**Example**:
```python
# BERT-Large: 24 layers × 4 operators per layer (QKV, Attn, FFN1, FFN2)
num_operators = 24 * 4 = 96 operators

program_load_energy_per_operator = 64K PEs * 60e-12 = 3.84 mJ
total_program_load_energy = 96 * 3.84 mJ = 368.6 mJ

# But amortized over batch=64:
program_load_per_inference = 368.6 / 64 = 5.76 mJ per inference
```

**This is ~8% of total energy** for BERT at batch=64!

**TPU has no program loading** (fixed hardware).

**Solution**: Track operator changes in graph, amortize by batch size

### 4.3 Multi-Engine Data Movement

**Challenge**: KPU has 3 data movement engines vs TPU's 1 DMA

**Energy path**:
```
TPU (2 stages):
  DRAM --DMA--> Weight Memory/Unified Buffer --feed--> Systolic Array

KPU (4 stages):
  DRAM --DMA--> L3 --BlockMover--> L2 --Streamer--> L1 --inject--> Fabric
```

**Each engine adds setup and routing energy:**
- DMA setup: 60 pJ
- BlockMover setup: 25 pJ
- Streamer operation: 6 pJ per vector + token formation

**Total engine overhead**: ~91 pJ per tile transfer (vs TPU's single DMA)

**Trade-off**: More stages = more overhead, but better locality and concurrency

**Solution**: Model each engine explicitly with setup + transfer energy

### 4.4 Distributed L3 Scratchpad

**Challenge**: L3 is distributed across die, variable routing distance

**Energy impact**:
```python
# Average distance from L3 tile to compute tile
avg_distance = sqrt(die_area / num_tiles) * distance_factor

# Routing energy scales with distance
l3_routing_energy = base_energy * (1 + distance_factor)
                  = 2.0e-12 * 1.3 = 2.6e-12 pJ/byte
```

**TPU has centralized Unified Buffer** (fixed distance).

**Solution**: Use `l3_distance_factor` parameter in energy model

### 4.5 Operator Fusion Benefits

**Challenge**: Quantify energy savings from automatic hardware fusion

**KPU advantage example** (Linear + LayerNorm + GELU):
```python
# Without fusion (3 operators):
energy_without_fusion = (
    linear_compute + linear_l2_write +
    layernorm_l2_read + layernorm_compute + layernorm_l2_write +
    gelu_l2_read + gelu_compute + gelu_l2_write
)

# With fusion (automatic via dataflow):
energy_with_fusion = (
    linear_compute +                      # Same
    layernorm_compute +                   # Same
    gelu_compute +                        # Same
    final_l2_write                        # Only 1 write!
)

savings = 4 * l2_write + 2 * l2_read
        = 4 * 1.0e-12 + 2 * 0.8e-12
        = 5.6e-12 pJ per element

# For BERT-Large (64 * 512 * 1024 elements):
total_savings = 33.5M elements * 5.6e-12 = 187.6 μJ per layer
```

**TPU has limited fusion** (only at fabric edge).

**Solution**: Model fusion explicitly, compare fused vs unfused energy

---

## Part 5: Validation Strategy

### 5.1 Analytical Validation

**Sanity checks**:
1. Energy per MAC should be 1-2 pJ/MAC (BF16) for advanced process nodes
2. Memory energy should dominate at small batch sizes
3. Compute energy should dominate at large batch sizes
4. Total energy should scale linearly with ops (at same batch size)

### 5.2 Cross-Architecture Validation

**Consistency checks**:
1. TPU should be most efficient for pure GEMM (specialized)
2. KPU should be 10-20% less efficient than TPU for GEMM (programmability overhead)
3. KPU should beat TPU on multi-operator graphs (fusion advantage)
4. GPU should be 2× worse than TPU/KPU (general-purpose overhead)
5. CPU should be worst (instruction fetch/decode overhead)

### 5.3 Technology Scaling Validation

**Process node impact**:
```python
# Energy scaling rules (approximate)
energy_22nm = baseline
energy_16nm = baseline * 0.7    # 30% reduction
energy_7nm = baseline * 0.4     # 60% reduction
energy_4nm = baseline * 0.25    # 75% reduction

# Frequency scaling
freq_22nm = 1.0 GHz (baseline)
freq_16nm = 1.5 GHz (+50%)
freq_7nm = 2.0 GHz (+100%)
freq_4nm = 2.5 GHz (+150%)
```

**Validation**: KPU-T64 (22nm) should consume ~2.5× energy per MAC vs KPU-T768 (4nm)

### 5.4 Literature Comparison

**Published data points** (if available):
- Domain-specific accelerators: 0.5-2 pJ/MAC (typical range)
- Spatial dataflow machines: 1-3 pJ/MAC (with routing overhead)
- GPU Tensor Cores: 1-2 pJ/MAC (NVIDIA claims)
- TPU: 0.2-0.5 pJ/MAC (Google claims, likely optimistic)

**Our models should fall within these ranges.**

---

## Part 6: Documentation and Reporting

### 6.1 Energy Model Documentation

**File**: `docs/hardware/kpu_energy_model.md`

Sections:
1. KPU Architecture Overview (DFA, SURE, token-based execution)
2. Memory Hierarchy Energy Breakdown
3. Data Movement Engine Energy
4. Compute Fabric Energy (token routing, signature matching, MACs)
5. SURE Program Loading Amortization
6. Operator Fusion Modeling
7. Technology Scaling Parameters
8. Validation Results

### 6.2 Comparison Report

**File**: `docs/analysis/cpu_gpu_tpu_kpu_energy_comparison.md`

Sections:
1. Architectural Comparison (execution models)
2. Energy Breakdown by Architecture
3. Memory Hierarchy Analysis
4. Control Overhead Quantification
5. Workload-Specific Analysis (GEMM, Conv2D, Attention, Full BERT)
6. When to Use Each Architecture
7. Future Trends (process scaling, 3D stacking, etc.)

### 6.3 Self-Documenting Tests

All tests should include:
- Clear console output with energy breakdown
- Formatted summary tables
- Validation against expected ranges
- Comparison with other architectures (where applicable)

---

## Part 7: Implementation Timeline

### Week 1: Core Implementation
- [ ] Create `KPUTileEnergyModel` class
- [ ] Implement `compute_gemm_energy()` method
- [ ] Create KPU-T64 resource model
- [ ] Create KPU mapper with tile energy integration

### Week 2: Additional Configurations
- [ ] Create KPU-T256 resource model (robotics)
- [ ] Create KPU-T768 resource model (automotive)
- [ ] Implement technology scaling (22nm, 16nm, 7nm, 4nm)
- [ ] Add operator fusion modeling

### Week 3: Testing and Validation
- [ ] Unit tests for KPU tile energy model
- [ ] GEMM energy validation tests
- [ ] ResNet/BERT validation tests
- [ ] KPU vs TPU comparison tests

### Week 4: Multi-Architecture Comparison
- [ ] CPU/GPU/TPU/KPU comparison test
- [ ] Generate comparison report
- [ ] Document energy model
- [ ] Validate against literature

---

## Part 8: Success Criteria

### 8.1 Correctness

✅ Energy per MAC falls within expected range (1-2 pJ for BF16 at 7nm)
✅ Memory dominates at batch=1, compute dominates at batch=64
✅ KPU is 10-20% less efficient than TPU for pure GEMM
✅ KPU beats TPU on multi-operator fused graphs
✅ Technology scaling follows expected trends

### 8.2 Completeness

✅ All 8 energy components modeled (memory hierarchy, engines, fabric, control)
✅ All 3 product families (T64, T256, T768) supported
✅ All 4 process nodes (22nm, 16nm, 7nm, 4nm) parameterized
✅ Operator fusion benefits quantified
✅ Sparsity and quantization modeled

### 8.3 Usability

✅ Easy to instantiate KPU models for different configurations
✅ Self-documenting test output
✅ Clear energy breakdown by subsystem
✅ Direct comparison with CPU/GPU/TPU

---

## Part 9: Open Questions and Future Work

### 9.1 Questions Requiring Clarification

1. **PE fabric size per compute tile**: 16×16, 32×32, 64×64, or 128×128?
   - *Impact*: Affects token routing distance and signature matching overhead
   - *Recommendation*: Model all variants, use 32×32 as baseline

2. **Streamer bandwidth**: How many vectors/cycle can Streamer inject?
   - *Impact*: Affects fabric utilization and L2 bandwidth requirements
   - *Recommendation*: Assume matched to fabric ingress (32 elements/cycle for 32×32)

3. **SURE program size**: How many instructions per PE?
   - *Impact*: Affects program loading energy
   - *Recommendation*: Assume 64-256 instructions (similar to GPU warp instruction count)

4. **Token signature width**: How many bits for spatial tag + recurrence variable?
   - *Impact*: Affects signature matching energy
   - *Recommendation*: Assume 32-64 bits total

### 9.2 Future Enhancements

1. **Multi-operator graph energy modeling**
   - Model entire BERT transformer block as single fused operator
   - Quantify fusion savings across all operators

2. **Sparsity-aware energy modeling**
   - Dynamic zero-gating savings
   - Sparse encoding/decoding overhead
   - Net energy benefit vs dense computation

3. **Mixed-precision energy modeling**
   - FP32 activations, INT8 weights
   - Just-in-time quantization energy

4. **3D stacking energy benefits**
   - HBM3 stacked on compute die
   - Reduced DMA routing distance
   - Through-silicon via (TSV) energy

5. **Multi-chiplet energy modeling**
   - KPU-T768 as multi-chiplet system
   - Inter-chiplet communication energy
   - Die-to-die bandwidth constraints

---

## Conclusion

This plan provides a comprehensive roadmap for implementing an accurate KPU energy model that can be directly compared with CPU, GPU, and TPU architectures. The key innovation is modeling the **8 unique energy components** of the KPU's Domain Flow Architecture:

1. **4-stage memory hierarchy** (vs TPU's 2-stage)
2. **3 data movement engines** (vs TPU's 1 DMA)
3. **Token signature matching** (unique to KPU!)
4. **SURE program loading** (per-operator overhead)
5. **Distributed L3 scratchpad** (variable routing distance)
6. **Automatic operator fusion** (hardware-driven)
7. **Token routing overhead** (spatial dataflow cost)
8. **Multi-engine coordination** (DMA + BlockMover + Streamer)

The model will enable **fair, apples-to-apples comparison** of energy efficiency across all four architectures, highlighting the **trade-offs** between:
- **TPU**: Maximum efficiency for GEMM (specialized, fixed-function)
- **KPU**: High efficiency with flexibility (programmable, automatic fusion)
- **GPU**: General-purpose flexibility with moderate efficiency
- **CPU**: Maximum flexibility with lowest efficiency

**Estimated effort**: 4 weeks for complete implementation, testing, validation, and documentation.

**Expected outcome**: Comprehensive understanding of when to use each architecture based on workload characteristics, energy constraints, and deployment requirements.
