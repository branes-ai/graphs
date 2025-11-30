"""
Architectural Energy Models - Resource Contention Management Energy Events

This module defines architecture-specific energy models that capture the
fundamental differences in how different hardware architectures manage
resource contention. Each architecture class has unique energy events
that determine its efficiency.

The goal is to quantify WHY one architecture is more energy efficient
than another, and to provide educational explanations for audiences
learning about hardware architecture trade-offs.

Architecture Classes:
- STORED_PROGRAM: CPU, DSP, GPU (instruction stream control)
- SYSTOLIC_ARRAY: Google TPU (fixed spatial schedule)
- DOMAIN_FLOW: Stillwater KPU (programmable spatial scheduling)
- DATA_FLOW_MACHINE: Token-based execution (reference architecture)
- SPATIAL_PARTITION: Cerebras, Hailo (graph partitioning on mesh)
- ADAPTIVE_DATAPATH: FPGA, CGRA, DPU (reconfigurable fabric)
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional


class ArchitectureClass(Enum):
    """
    Resource contention management mechanism classification.

    Each architecture class represents a fundamentally different approach
    to managing resource contention, with distinct energy characteristics.

    Progression shows evolution toward energy efficiency:
    CPU (sequential) → GPU (data parallel) → TPU (fixed spatial) → KPU (programmable spatial)
    """

    STORED_PROGRAM = "stored_program"
    """
    Traditional von Neumann architectures (CPU, DSP) - Sequential/Modest Parallelism.

    Resource Contention: Sequential instruction stream with modest parallelism
    - Explicit instruction fetch (request/reply from memory)
    - Explicit operand loads (load/store architecture)
    - Pipeline control (decode, dispatch, ordering, writeback)
    - Modest SIMD parallelism (8-16 wide: AVX-512, NEON)
    - Out-of-order execution for ILP
    - Complex branch prediction

    Energy Characteristics: 1.0x (baseline for sequential)
    - Instruction fetch per operation (~2 pJ)
    - Memory request overhead (~10 pJ per load/store)
    - Pipeline control overhead
    - Moderate parallelism (8-16 cores typical)

    Examples: Intel Xeon, AMD EPYC, Ampere Altra, Qualcomm Hexagon DSP
    """

    DATA_PARALLEL = "data_parallel"
    """
    SIMT (Single Instruction Multiple Thread) architectures (GPU) - Massive Data Parallelism.

    Resource Contention: SIMT execution with massive thread parallelism
    - Same instruction stream controls thousands of threads (SIMT)
    - Lockstep execution within warps (32 threads)
    - Warp divergence penalties (control flow branching)
    - Massive coherence machinery (manage thousands of concurrent memory requests)
    - Thread scheduling across thousands of threads
    - Memory coalescing requirements

    Energy Characteristics: 2.5-3.0x stored program baseline
    - All CPU overheads PLUS:
    - Coherence machinery: ~5 pJ per memory request x thousands of concurrent requests
    - Thread scheduling: ~1 pJ per thread x thousands of threads
    - Warp divergence penalties
    - Synchronization barriers

    Key Difference vs CPU:
    - CPU: 8-16 cores, complex per-core control
    - GPU: 132 SMs x 2048 threads/SM = 270K threads, simple per-thread control
    - GPU hides memory latency with parallelism, but burns energy on coherence

    Why More Energy Than CPU: Coherence machinery dominates for small batches

    Examples: NVIDIA H100, AMD MI300, NVIDIA Jetson
    """

    SYSTOLIC_ARRAY = "systolic_array"
    """
    Fixed-function systolic array processors (Google TPU).

    Resource Contention: Pre-designed 2D spatial schedule (NO contention)
    - Schedule preloaded at kernel launch (one-time setup)
    - Data flows through predetermined paths (spatial separation)
    - No instruction fetch during execution
    - No dynamic resource arbitration
    - Full concurrency with zero contention overhead

    Energy Characteristics: 0.10-0.20x stored program
    - Minimal control overhead (schedule setup is amortized)
    - Low memory overhead (direct injection into array)
    - No coherence/ordering machinery needed
    - 5-10x more efficient than stored program (Google TPU data)

    Limitations: Fixed function, cannot change behavior

    Examples: Google TPU v4, Google Coral Edge TPU
    """

    DOMAIN_FLOW = "domain_flow"
    """
    Programmable systolic processors with domain tracking (Stillwater KPU).
    Domain Flow Architecture (DFA).

    Resource Contention: Domain-based spatial scheduling with adaptivity
    - SURE/SARE network overlay models variable dependencies
    - Domain checks track computation through N-dimensional abstract space
    - Execution wavefronts can change dynamically
    - Systolic array behavior is programmable
    - Spatial mapping with dynamic control

    Energy Characteristics: 0.25-0.40x stored program
    - Domain tracking overhead (more than fixed systolic, less than stored program)
    - Network overlay management energy (data token routing)
    - Wavefront control energy
    - No instruction fetch per operation, domain flow program loaded once
    - Programmability cost << stored program overhead

    Key Distinction: Systolic array is special case of DFA (all dynamic control
    reduced to one function). KPU maintains flexibility while preserving spatial
    efficiency.

    Historical Context: Evolved from Data Flow Machine concepts

    Examples: Stillwater KPU-T64, KPU-T256, KPU-T768
    """

    DATA_FLOW_MACHINE = "data_flow_machine"
    """
    Token-based data flow processors (Jack Dennis, Petri net execution).

    Resource Contention: Data availability triggers execution (no program counter)
    - CAM (Content Addressable Memory) holds instruction tokens
    - Token matching determines ready operations
    - Sea of processing elements execute ready tokens
    - Dataflow graph traversal
    - No sequential instruction stream

    Energy Characteristics: 0.30-0.50x stored program (CAM-limited)
    - CAM energy (associative search per cycle)
    - Token queue management
    - Matching logic overhead
    - Potentially massive parallelism when data available
    - No wasted cycles waiting for instructions

    Limitations: CAM is energy-intensive

    Historical Examples: MIT Tagged Token Dataflow, Manchester Dataflow Machine

    Reference Architecture: No commercial form factor, but influences modern designs.
    Modern usage: x86 micro-architectures use DFM in execution core (register renaming).

    Note: Included as reference architecture for AI workload mapping hypothesis.
    """

    SPATIAL_PARTITION = "spatial_partition"
    """
    Mesh of processing elements with graph partitioning (Cerebras, Hailo).

    Resource Contention: Computational graph decomposed into spatial partitions

    Architecture:
    - Mesh/grid of programmable processing elements
    - Computational graph partitioned into sub-graphs
    - Sub-graphs mapped to physical mesh topology
    - Efficient ingress/egress flow management between partitions
    - Data locality optimization (minimize off-chip access)

    Energy Characteristics: 0.15-0.30x stored program
    - Graph partitioning (compile-time, amortized)
    - Inter-partition communication (mesh links)
    - Partition-local computation (very efficient)
    - Ingress/egress buffer management
    - Synchronization barriers

    Key Efficiency Factor: Decomposition quality + matching ingress/egress flows
    - Good partitioning → high data locality → low inter-partition traffic
    - Poor partitioning → high communication → energy approaches stored program

    Examples:
    - Cerebras WSE: 850K cores on wafer, 2D mesh interconnect
    - Hailo-8: Structured array with graph mapping
    """

    ADAPTIVE_DATAPATH = "adaptive_datapath"
    """
    Reconfigurable logic and datapath architectures (FPGA, CGRA, DPU).

    Resource Contention: Spatial routing through reconfigurable fabric
    - Algorithm mapped to arrangement of fixed datapath elements
    - Reconfiguration infrastructure (LUTs, routing matrix, hard macros)
    - High reconfiguration cost (energy + time)
    - Once configured, operates like spatial dataflow

    Energy Characteristics: 0.15-0.30x runtime (+ reconfiguration penalty)
    - Reconfiguration energy (very high, amortized over many runs)
    - Routing overhead (multiplexers, interconnect)
    - Hard macro activation (DSP blocks, BRAMs)
    - Poor for general purpose (need to quantify why)

    Subcategories:
    - FPGA: Fine-grained LUT-based reconfigurable logic
    - CGRA: Coarse-grained reconfigurable arrays (word-level datapath)
    - DPU: FPGA with pre-defined 2D hard macros (Xilinx Vitis AI)
           Optimized for simple DNNs (Conv, Linear)
           Struggles with irregular ops (SOFTMAX, LayerNorm - need FP)

    Limitation for AI: Reconfiguration cost not amortized for dynamic workloads

    Examples: Xilinx Vitis AI DPU, Stanford Plasticine CGRA, Intel Stratix 10 FPGA
    """


@dataclass
class ArchitecturalEnergyBreakdown:
    """
    Result of architectural energy calculation with MAC/FLOP/IntOp separation.

    Contains energy overheads (positive = cost, negative = savings)
    and human-readable explanation.

    New Design (Phase 1):
    - Separates MAC/FLOP/IntOp energy for accurate hardware mapping
    - MACs → Tensor Cores, Systolic Arrays, AMX
    - FLOPs → CUDA cores, SIMD units, SFUs
    - IntOps → Integer ALUs, Indexing, Quantization
    """
    compute_overhead: float  # Additional compute energy (Joules)
    data_movement_overhead: float   # Energy cost of moving data through memory hierarchy (Joules)
    control_overhead: float  # Control/coordination energy (Joules)

    # ============================================================
    # MAC/FLOP/INTOP ENERGY BREAKDOWN (NEW - Phase 1)
    # ============================================================

    # MAC energy (matrix/convolution operations)
    mac_energy: float = 0.0
    """Energy from MAC operations (matmul, conv) in Joules"""

    # FLOP energy (non-MAC floating-point operations)
    flop_energy: float = 0.0
    """Energy from FLOP operations (bias, activation, elementwise) in Joules"""

    # IntOp energy (integer operations)
    intop_energy: float = 0.0
    """Energy from integer operations (quantization, indexing) in Joules"""

    # ============================================================
    # OPERATION COUNTS (NEW - Phase 1)
    # ============================================================

    # MAC operation counts
    mac_ops_executed: int = 0
    """Number of MACs executed"""

    # FLOP operation counts
    flop_ops_executed: int = 0
    """Number of FLOPs executed"""

    # IntOp operation counts
    intop_ops_executed: int = 0
    """Number of integer ops executed"""

    # Additional details for specific architectures
    extra_details: Dict[str, float] = field(default_factory=dict)

    # Human-readable explanation
    explanation: str = ""

    @property
    def total_overhead(self) -> float:
        """Total architectural overhead"""
        return self.compute_overhead + self.data_movement_overhead + self.control_overhead

    @property
    def total_compute_energy(self) -> float:
        """Total compute energy (MAC + FLOP + IntOp)"""
        return self.mac_energy + self.flop_energy + self.intop_energy

    @property
    def total_ops_executed(self) -> int:
        """Total operations executed (MAC + FLOP + IntOp)"""
        return self.mac_ops_executed + self.flop_ops_executed + self.intop_ops_executed

# ============================================================
# Architectural Energy Model Base Class
# ============================================================
class ArchitecturalEnergyModel(ABC):
    """
    Base class for architecture-specific energy modeling.

    Captures energy events unique to different resource contention
    management strategies.
    """

    @abstractmethod
    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        data_movement_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:
        """
        Compute architectural energy overhead.

        Args:
            ops: Number of operations (FLOPs, MACs, etc.)
            bytes_transferred: Total bytes read/written
            compute_energy_baseline: Baseline compute energy (ops x energy_per_op)
            data_movement_energy_baseline: Baseline data movement energy (bytes x energy_per_byte)
            execution_context: Additional context (threads, batch size, etc.)

        Returns:
            ArchitecturalEnergyBreakdown with overheads and explanation
        """
        pass


@dataclass
class StoredProgramEnergyModel(ArchitecturalEnergyModel):
    """
    Enhanced energy model for stored program architectures (CPU, DSP).

    Key energy events:
    1. Instruction Pipeline (Fetch → Decode → Execute)
    2. Register File Operations (2 reads + 1 write per instruction)
    3. 4-Stage Memory Hierarchy (L1 → L2 → L3 → DRAM)
    4. ALU Operations
    5. Branch Prediction

    Classic Pipeline Stages:

    Fetch → Decode → Dispatch → Execute → Writeback
        ↑       ↑         ↑          ↑         ↑
    I-cache  Logic   Control    ALU ops   Register
    read             signals   (sep.)     writeback
    1.5 pJ   0.8 pJ   0.5 pJ    4.0 pJ    3.0 pJ

    The memory wall: Request/reply latency is significant AND energy intensive.
    Register file energy is comparable to ALU energy (~same for read/write as for compute).
    """

    # ============================================================
    # Instruction Pipeline Energy (Fetch → Decode → Dispatch)
    # ============================================================
    # Note: Dispatch writes control signals to datapath registers.
    # Actual execution (ALU ops) is tracked separately in alu_energy_per_op.
    instruction_fetch_energy: float = 1.5e-12      # ~1.5 pJ per instruction (I-cache read)
    instruction_decode_energy: float = 0.8e-12     # ~0.8 pJ per instruction (decode logic)
    instruction_dispatch_energy: float = 0.5e-12   # ~0.5 pJ per instruction (control signal dispatch)

    # ============================================================
    # Register File Energy (CRITICAL: ~same as ALU energy!)
    # HIGH FREQUENCY CPUs (2-4 GHz) need much more energy than low-freq accelerators
    # ============================================================
    register_file_read_energy: float = 2.5e-12     # ~2.5 pJ per read (high-freq, complex CPU)
    register_file_write_energy: float = 3.0e-12    # ~3.0 pJ per write (write is more expensive)
    register_ops_per_instruction: int = 3          # 2 reads + 1 write per instruction

    # ============================================================
    # 4-Stage Memory Hierarchy Energy
    # HIGH FREQUENCY = higher energy per access
    # ============================================================
    l1_cache_energy_per_byte: float = 1.0e-12      # ~1.0 pJ (32 KB, per-core, high freq)
    l2_cache_energy_per_byte: float = 2.5e-12      # ~2.5 pJ (256 KB-1 MB, per-core or shared)
    l3_cache_energy_per_byte: float = 5.0e-12      # ~5.0 pJ (8-64 MB, shared LLC)
    dram_energy_per_byte: float = 20.0e-12         # ~20 pJ (DDR4/DDR5, off-chip)

    # Cache hit rates (conservative for AI workloads with streaming data)
    l1_hit_rate: float = 0.85                      # 85% L1 hits
    l2_hit_rate: float = 0.90                      # 90% L2 hits (of L1 misses)
    l3_hit_rate: float = 0.95                      # 95% L3 hits (of L2 misses)
    # DRAM gets remaining misses

    # ============================================================
    # ALU Energy (HIGH FREQUENCY = 3-5× more than low-freq accelerators!)
    # ============================================================
    alu_energy_per_op: float = 4.0e-12             # ~4.0 pJ per FP operation (2.2 GHz, complex OOO design)

    # ============================================================
    # Instruction Mix (for AI workloads)
    # ============================================================
    instructions_per_op: float = 2.0               # ~2 instructions per FLOP (load, compute, store, etc.)
    branches_per_1000_ops: int = 50                # ~50 branches per 1000 ops
    branch_prediction_overhead: float = 0.4e-12    # ~0.4 pJ per branch

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        data_movement_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # ============================================================
        # 1. Instruction Pipeline (Fetch → Decode → Dispatch)
        # ============================================================
        # Note: This models the instruction pipeline up to dispatch.
        # Actual execution (ALU operations) is modeled separately below.
        num_instructions = int(ops * self.instructions_per_op)

        instruction_fetch_energy_total = num_instructions * self.instruction_fetch_energy
        instruction_decode_energy_total = num_instructions * self.instruction_decode_energy
        instruction_dispatch_energy_total = num_instructions * self.instruction_dispatch_energy

        pipeline_energy_total = (
            instruction_fetch_energy_total +
            instruction_decode_energy_total +
            instruction_dispatch_energy_total
        )

        # ============================================================
        # 2. Register File Operations (2 reads + 1 write per instruction)
        # ============================================================
        num_register_reads = num_instructions * 2   # 2 source operands
        num_register_writes = num_instructions * 1  # 1 destination operand

        register_read_energy = num_register_reads * self.register_file_read_energy
        register_write_energy = num_register_writes * self.register_file_write_energy
        register_file_energy_total = register_read_energy + register_write_energy

        # ============================================================
        # 3. 4-Stage Memory Hierarchy (L1 → L2 → L3 → DRAM)
        # ============================================================
        # MEMORY MODEL: Option 1 - Pure Cold Start (Fair Comparison)
        #
        # For single-inference analysis, we model cold start where every cache line
        # must traverse the entire hierarchy on first access. This matches the
        # GPU/TPU/KPU models which account for all data movement.
        #
        # Example for 16×16 MLP (batch=1):
        # - Weights: 1,024 bytes = 16 cache lines
        # - Input:   64 bytes = 1 cache line
        # - Output:  64 bytes = 1 cache line
        #
        # Data movement for weights (16 lines):
        # - DRAM → L3: 16 reads from DRAM, 16 writes to L3
        # - L3 → L2:   16 reads from L3, 16 writes to L2
        # - L2 → L1:   16 reads from L2, 16 writes to L1
        # - L1 → PE:   16 reads from L1 (for computation)
        #
        # FUTURE: Option 3 hooks allow execution_context to specify cache_state
        # FUTURE: Option 2 (3C Model) for actual DNN graphs:
        #   - Compulsory misses: First touch of each cache line (modeled here)
        #   - Capacity misses: Working set > cache capacity
        #   - Conflict misses: Mapping conflicts in set-associative caches
        #   The 3C model will be essential for analyzing large DNN graphs where
        #   working sets exceed cache capacities and temporal reuse patterns matter.

        cache_line_size = 64  # bytes (typical for modern CPUs)
        total_bytes = bytes_transferred

        # Check for cache_state in execution context (Option 3 hook)
        cache_state = 'cold'  # Default to cold start for fair comparison
        if execution_context:
            cache_state = execution_context.get('cache_state', 'cold')

        if cache_state == 'cold':
            # ========================================================
            # COLD START MODEL (Option 1)
            # ========================================================
            # Every cache line must traverse the complete hierarchy.
            # Calculate total cache lines needed
            total_cache_lines = int((total_bytes + cache_line_size - 1) // cache_line_size)

            # DRAM → L3 transfer
            # Every line starts in DRAM and must be fetched
            dram_reads = total_cache_lines
            dram_bytes = dram_reads * cache_line_size
            dram_energy = dram_bytes * self.dram_energy_per_byte

            # L3 operations: write from DRAM, then read for L2
            l3_writes = dram_reads
            l3_reads = dram_reads
            l3_bytes = (l3_writes + l3_reads) * cache_line_size
            l3_energy = l3_bytes * self.l3_cache_energy_per_byte

            # L2 operations: write from L3, then read for L1
            l2_writes = l3_reads
            l2_reads = l3_reads
            l2_bytes = (l2_writes + l2_reads) * cache_line_size
            l2_energy = l2_bytes * self.l2_cache_energy_per_byte

            # L1 operations: write from L2, then read for computation
            # Note: For matmul, weights may be read multiple times from L1,
            # but we only count the fetch cost here (not reuse)
            l1_writes = l2_reads
            l1_reads = l2_reads  # Simplified: one read per line fetch
            l1_bytes = (l1_writes + l1_reads) * cache_line_size
            l1_energy = l1_bytes * self.l1_cache_energy_per_byte

            # Access counts for reporting
            dram_accesses = dram_reads
            l3_accesses = l3_reads + l3_writes
            l2_accesses = l2_reads + l2_writes
            l1_accesses = l1_reads + l1_writes

        else:
            # ========================================================
            # WARM CACHE MODEL (Option 3 - Future)
            # ========================================================
            # Use hit rate model for steady-state after warmup
            # TODO: Implement when needed for batch analysis
            raise NotImplementedError(f"cache_state='{cache_state}' not yet implemented. Only 'cold' is supported.")

        memory_hierarchy_energy_total = l1_energy + l2_energy + l3_energy + dram_energy

        # ============================================================
        # 4. ALU Operations
        # ============================================================
        alu_energy_total = ops * self.alu_energy_per_op

        # ============================================================
        # 5. Branch Prediction
        # ============================================================
        # More accurate branch estimation based on loop structure
        # For dense linear algebra (matmul, vector ops):
        # - Assume 3-nested loops for matmul (M, N, K) + bias loop + activation loop
        # - Branch frequency ≈ ops^(1/3) for outer loop, ops^(2/3) for middle loop, ops for inner
        # Simplified: branches ≈ 3 * sqrt(ops) for typical loop nests

        # Use a more realistic model: ~1 branch per 10 operations for dense compute
        # This accounts for loop iterations, conditions, and function calls
        num_branches = max(1, int(ops / 10))

        # Branch prediction success rate (95% for predictable loops)
        branch_prediction_success = 0.95
        num_mispredicted_branches = int(num_branches * (1 - branch_prediction_success))

        # Only mispredicted branches pay the full overhead
        # Correctly predicted branches have minimal cost (already in instruction fetch)
        branch_energy = num_mispredicted_branches * self.branch_prediction_overhead

        # ============================================================
        # Categorize into overhead components
        # ============================================================
        # Control overhead: instruction pipeline + branch prediction
        control_overhead = pipeline_energy_total + branch_energy

        # Compute overhead: register file + ALU
        compute_overhead = register_file_energy_total + alu_energy_total

        # Data movement overhead: cache hierarchy
        data_movement_overhead = memory_hierarchy_energy_total

        # ============================================================
        # Create detailed breakdown
        # ============================================================
        extra_details = {
            # Instruction Pipeline (Fetch → Decode → Dispatch)
            'instruction_fetch_energy': instruction_fetch_energy_total,
            'instruction_decode_energy': instruction_decode_energy_total,
            'instruction_dispatch_energy': instruction_dispatch_energy_total,
            'num_instructions': num_instructions,

            # Register File
            'register_read_energy': register_read_energy,
            'register_write_energy': register_write_energy,
            'num_register_reads': num_register_reads,
            'num_register_writes': num_register_writes,

            # Memory Hierarchy
            'l1_cache_energy': l1_energy,
            'l2_cache_energy': l2_energy,
            'l3_cache_energy': l3_energy,
            'dram_energy': dram_energy,
            'l1_bytes': l1_bytes,
            'l2_bytes': l2_bytes,
            'l3_bytes': l3_bytes,
            'dram_bytes': dram_bytes,
            'l1_accesses': l1_accesses,
            'l2_accesses': l2_accesses,
            'l3_accesses': l3_accesses,
            'dram_accesses': dram_accesses,

            # ALU (for arithmetic intensity calculation)
            'alu_energy': alu_energy_total,
            'alu_ops': ops,  # Total operations (for arithmetic intensity)
            'fpu_ops': 0,    # FPU ops (not separately tracked in this model)

            # Workload data movement (for consistent AI calculation across architectures)
            'bytes_transferred': bytes_transferred,

            # Branch Prediction
            'branch_energy': branch_energy,
            'num_branches': num_branches,
            'num_mispredicted_branches': num_mispredicted_branches,
            'branch_prediction_success_rate': branch_prediction_success,
        }

        explanation = (
            f"Stored Program Architecture (CPU) Energy Events:\n"
            f"  1. Instruction Pipeline: {pipeline_energy_total*1e6:.3f} μJ "
            f"({num_instructions:,} instructions)\n"
            f"  2. Register File: {register_file_energy_total*1e6:.3f} μJ "
            f"({num_register_reads:,} reads + {num_register_writes:,} writes)\n"
            f"  3. Memory Hierarchy: {memory_hierarchy_energy_total*1e6:.3f} μJ\n"
            f"     - L1: {l1_energy*1e6:.3f} μJ ({l1_bytes/1024:.1f} KB)\n"
            f"     - L2: {l2_energy*1e6:.3f} μJ ({l2_bytes/1024:.1f} KB)\n"
            f"     - L3: {l3_energy*1e6:.3f} μJ ({l3_bytes/1024:.1f} KB)\n"
            f"     - DRAM: {dram_energy*1e6:.3f} μJ ({dram_bytes/1024:.1f} KB)\n"
            f"  4. ALU Operations: {alu_energy_total*1e6:.3f} μJ ({ops:,} ops)\n"
            f"  5. Branch Prediction: {branch_energy*1e6:.3f} μJ ({num_branches:,} branches)\n"
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead,
            data_movement_overhead=data_movement_overhead,
            control_overhead=control_overhead,
            extra_details=extra_details,
            explanation=explanation
        )


@dataclass
class DataParallelEnergyModel(ArchitecturalEnergyModel):
    """
    Energy model for data parallel SIMT architectures (GPU).

    Key energy events:
    - All CPU energy events (instruction fetch, operand fetch, pipeline control)
    - SIMT-specific: Warp divergence penalties
    - Massive coherence machinery (thousands of concurrent memory requests)
    - Thread scheduling across thousands of threads
    - Memory coalescing overhead
    - Synchronization barriers

    Why GPUs use more energy than CPUs for small batches:
    The coherence machinery needed to manage thousands of concurrent memory
    requests dominates energy consumption. This overhead is only amortized
    at large batch sizes.
    """

    # Inherit CPU overheads
    instruction_fetch_energy: float = 2.0e-12      # ~2 pJ per instruction
    operand_fetch_overhead: float = 10.0e-12       # ~10 pJ per memory operation
    instructions_per_op: float = 0.1               # ~1 instruction per 10 ops

    # GPU-specific SIMT overheads
    coherence_energy_per_request: float = 5.0e-12  # ~5 pJ per memory request
    thread_scheduling_overhead: float = 1.0e-12    # ~1 pJ per thread
    warp_divergence_penalty: float = 3.0e-12       # ~3 pJ per divergent branch
    memory_coalescing_overhead: float = 2.0e-12    # ~2 pJ per uncoalesced access
    barrier_sync_energy: float = 10.0e-12          # ~10 pJ per barrier

    # SIMT characteristics
    warp_divergence_rate: float = 0.05             # 5% of ops cause divergence
    uncoalesced_access_rate: float = 0.10          # 10% of memory accesses uncoalesced
    barriers_per_1000_ops: int = 5                 # ~5 barriers per 1000 ops

    # NEW: Compute unit breakdown (CUDA cores vs Tensor Cores)
    cuda_core_mac_energy: float = 0.8e-12          # ~0.8 pJ per MAC (FP32 on CUDA cores)
    cuda_core_flop_energy: float = 0.8e-12         # ~0.8 pJ per FLOP (FP32 on CUDA cores)
    int_alu_energy: float = 0.1e-12                # ~0.1 pJ per IntOp (integer operations)
    tensor_core_mac_energy: float = 0.3e-12        # ~0.3 pJ per MAC (FP16/BF16/INT8 on Tensor Cores)
    tensor_core_utilization: float = 0.80          # 80% of MACs can use tensor cores (GEMM-like)
    register_file_energy_per_access: float = 0.6e-12  # ~0.6 pJ per register access (similar to ALU energy)

    # Tensor Core characteristics
    TENSOR_CORE_MACS_PER_OP: int = 64              # 4×4×4 FP16 matmul per Tensor Core operation

    # NEW: Memory hierarchy breakdown (NVIDIA Ampere nomenclature)
    # Level 1: Register File (64K regs/SM, separate from cache)
    # Level 2: Shared Memory / L1 Data Cache (UNIFIED, 128-192 KB configurable)
    # Level 3: L2 Cache (shared across all SMs)
    # Level 4: DRAM
    shared_memory_l1_unified_energy_per_byte: float = 0.25e-12  # ~0.25 pJ (Shared Mem/L1 unified, on-chip)
    l2_cache_energy_per_byte: float = 0.8e-12                   # ~0.8 pJ (shared L2, 4-40 MB)
    dram_energy_per_byte: float = 10.0e-12                      # ~10 pJ (HBM2e or LPDDR5)

    # NEW: Memory access patterns (conservative estimates)
    shared_mem_l1_hit_rate: float = 0.95                        # 95% hit in Shared Mem/L1
    l2_hit_rate: float = 0.90                                   # 90% L2 hits (of Shared/L1 misses)
    shared_mem_explicit_usage: float = 0.40                     # 40% explicitly use shared memory in kernel code

    # NEW: Instruction pipeline stages
    instruction_decode_energy: float = 0.5e-12         # ~0.5 pJ per instruction decode
    instruction_execute_energy: float = 0.3e-12        # ~0.3 pJ per instruction execute

    def compute_architectural_energy(
        self,
        ops: Optional[int] = None,
        bytes_transferred: int = 0,
        compute_energy_baseline: float = 0.0,
        data_movement_energy_baseline: float = 0.0,
        execution_context: Optional[Dict] = None,
        workload: Optional['WorkloadCharacterization'] = None  # NEW: Phase 3
    ) -> ArchitecturalEnergyBreakdown:
        """
        Compute architectural energy with MAC/FLOP/IntOp separation.

        Args:
            ops: (DEPRECATED) Total operations. Use workload parameter instead.
            bytes_transferred: Total bytes read/written
            compute_energy_baseline: Baseline compute energy
            data_movement_energy_baseline: Baseline memory energy
            execution_context: Additional context (threads, batch size, etc.)
            workload: (NEW) WorkloadCharacterization with MAC/FLOP/IntOp breakdown

        Returns:
            ArchitecturalEnergyBreakdown with MAC/FLOP/IntOp separation
        """
        if execution_context is None:
            execution_context = {}

        # ============================================================
        # Extract MAC/FLOP/IntOp counts (Phase 3)
        # ============================================================
        if workload is not None:
            # NEW path: use WorkloadCharacterization
            macs = workload.macs
            flops = workload.flops
            intops = workload.intops
            total_ops = workload.total_ops()
        else:
            # LEGACY path: assume all ops are MACs for backward compatibility
            macs = ops or 0
            flops = 0
            intops = 0
            total_ops = ops or 0

        # ============================================================
        # 1. COMPUTE UNIT BREAKDOWN (CUDA cores vs Tensor Cores)
        # ============================================================

        # MACs → Tensor Cores (80% utilization for GEMM-like operations)
        tensor_core_macs = int(macs * self.tensor_core_utilization)
        cuda_core_macs = macs - tensor_core_macs

        # Calculate actual Tensor Core operations (each does 64 MACs: 4×4×4)
        tensor_core_ops = tensor_core_macs // self.TENSOR_CORE_MACS_PER_OP

        # MAC energy
        tensor_core_mac_energy = tensor_core_macs * self.tensor_core_mac_energy
        cuda_core_mac_energy = cuda_core_macs * self.cuda_core_mac_energy

        # FLOP energy (bias, activation, elementwise → CUDA cores)
        cuda_core_flop_energy = flops * self.cuda_core_flop_energy

        # IntOp energy (quantization, indexing → Integer ALUs)
        int_alu_energy = intops * self.int_alu_energy

        # Total compute energy by type
        total_mac_energy = tensor_core_mac_energy + cuda_core_mac_energy
        total_flop_energy = cuda_core_flop_energy
        total_intop_energy = int_alu_energy

        # Register file accesses (2 per op: read operands + write result)
        num_register_accesses = total_ops * 2
        register_file_energy = num_register_accesses * self.register_file_energy_per_access

        # ============================================================
        # 2. INSTRUCTION PIPELINE (fetch, decode, execute)
        # ============================================================

        num_instructions = int(total_ops * self.instructions_per_op)
        instruction_fetch_energy_total = num_instructions * self.instruction_fetch_energy
        instruction_decode_energy_total = num_instructions * self.instruction_decode_energy
        instruction_execute_energy_total = num_instructions * self.instruction_execute_energy

        # ============================================================
        # 3. MEMORY HIERARCHY BREAKDOWN (NVIDIA Ampere Nomenclature)
        # Register File → Shared Memory/L1 (unified) → L2 → DRAM
        # ============================================================

        cache_line_size = execution_context.get('cache_line_size', 128)  # H100 uses 128B
        num_memory_accesses = max(1, int(bytes_transferred / 4))  # Assume 4-byte elements

        # Shared Memory / L1 unified (95% hit rate)
        # This is a single hardware structure with configurable carveout
        shared_mem_l1_accesses = int(num_memory_accesses * self.shared_mem_l1_hit_rate)
        shared_mem_l1_energy = shared_mem_l1_accesses * self.shared_memory_l1_unified_energy_per_byte * 4

        # L2 cache hits (90% of Shared/L1 misses)
        shared_l1_misses = num_memory_accesses - shared_mem_l1_accesses
        l2_accesses = int(shared_l1_misses * self.l2_hit_rate)
        l2_energy = l2_accesses * self.l2_cache_energy_per_byte * 4

        # DRAM accesses (remaining L2 misses)
        l2_misses = shared_l1_misses - l2_accesses
        dram_accesses = l2_misses
        dram_energy = dram_accesses * self.dram_energy_per_byte * 4

        # ============================================================
        # 4. GPU-SPECIFIC SIMT OVERHEADS
        # ============================================================

        # Coherence machinery (THE CRITICAL ENERGY COMPONENT)
        concurrent_threads = execution_context.get('concurrent_threads', 200_000)
        warp_size = execution_context.get('warp_size', 32)
        num_concurrent_warps = max(1, concurrent_threads // warp_size)

        num_memory_ops = max(1, int(bytes_transferred / cache_line_size))
        coherence_energy = num_concurrent_warps * self.coherence_energy_per_request * num_memory_ops

        # Thread scheduling overhead
        scheduling_energy = concurrent_threads * self.thread_scheduling_overhead

        # Warp divergence penalties
        num_divergent_ops = int(total_ops * self.warp_divergence_rate)
        divergence_energy = num_divergent_ops * self.warp_divergence_penalty

        # Memory coalescing overhead
        num_uncoalesced = int(num_memory_ops * self.uncoalesced_access_rate)
        coalescing_energy = num_uncoalesced * self.memory_coalescing_overhead

        # Synchronization barriers
        num_barriers = (total_ops // 1000) * self.barriers_per_1000_ops
        barrier_energy = num_barriers * self.barrier_sync_energy

        # ============================================================
        # 5. AGGREGATE OVERHEAD CATEGORIES
        # ============================================================

        # Compute overhead: instruction pipeline + register file
        compute_overhead_total = (instruction_fetch_energy_total +
                                 instruction_decode_energy_total +
                                 instruction_execute_energy_total +
                                 register_file_energy)

        # Data movement overhead: hierarchy + coalescing
        data_movement_overhead_total = (shared_mem_l1_energy + l2_energy + dram_energy +
                                coalescing_energy)

        # Control overhead: coherence + scheduling + divergence + barriers
        control_overhead = (coherence_energy + scheduling_energy +
                           divergence_energy + barrier_energy)

        explanation = (
            f"Data Parallel (GPU SIMT) Architecture Energy Events:\n"
            f"\n"
            f"1. COMPUTE UNITS (MAC/FLOP/IntOp Separation):\n"
            f"   Tensor Cores:       {tensor_core_mac_energy*1e12:.2f} pJ ({tensor_core_ops:,} TC ops, {tensor_core_macs:,} MACs @ {self.tensor_core_mac_energy*1e12:.2f} pJ/MAC)\n"
            f"   CUDA Cores (MACs):  {cuda_core_mac_energy*1e12:.2f} pJ ({cuda_core_macs:,} MACs @ {self.cuda_core_mac_energy*1e12:.2f} pJ/MAC)\n"
            f"   CUDA Cores (FLOPs): {cuda_core_flop_energy*1e12:.2f} pJ ({flops:,} FLOPs @ {self.cuda_core_flop_energy*1e12:.2f} pJ/FLOP)\n"
            f"   Integer ALUs:       {int_alu_energy*1e12:.2f} pJ ({intops:,} IntOps @ {self.int_alu_energy*1e12:.2f} pJ/IntOp)\n"
            f"   Register File:      {register_file_energy*1e12:.2f} pJ ({num_register_accesses:,} accesses)\n"
            f"\n"
            f"2. INSTRUCTION PIPELINE:\n"
            f"   Fetch:  {instruction_fetch_energy_total*1e12:.2f} pJ ({num_instructions:,} instructions)\n"
            f"   Decode: {instruction_decode_energy_total*1e12:.2f} pJ\n"
            f"   Execute: {instruction_execute_energy_total*1e12:.2f} pJ\n"
            f"\n"
            f"3. MEMORY HIERARCHY (NVIDIA Ampere Nomenclature):\n"
            f"   Shared Memory/L1 (unified): {shared_mem_l1_energy*1e12:.2f} pJ ({shared_mem_l1_accesses:,} accesses, {self.shared_mem_l1_hit_rate*100:.0f}% hit rate)\n"
            f"   L2 Cache:                   {l2_energy*1e12:.2f} pJ ({l2_accesses:,} hits, {self.l2_hit_rate*100:.0f}% of Shared/L1 misses)\n"
            f"   DRAM:                       {dram_energy*1e12:.2f} pJ ({dram_accesses:,} accesses)\n"
            f"\n"
            f"4. SIMT CONTROL OVERHEADS:\n"
            f"   Coherence Machinery:   {coherence_energy*1e12:.2f} pJ ← DOMINANT!\n"
            f"                          ({num_concurrent_warps:,} warps × {num_memory_ops:,} mem ops)\n"
            f"   Thread Scheduling:     {scheduling_energy*1e12:.2f} pJ ({concurrent_threads:,} threads)\n"
            f"   Warp Divergence:       {divergence_energy*1e12:.2f} pJ ({num_divergent_ops:,} divergent ops)\n"
            f"   Memory Coalescing:     {coalescing_energy*1e12:.2f} pJ ({num_uncoalesced:,} uncoalesced)\n"
            f"   Synchronization Barriers: {barrier_energy*1e12:.2f} pJ ({num_barriers:,} barriers)\n"
            f"\n"
            f"TOTAL OVERHEAD: {(compute_overhead_total + data_movement_overhead_total + control_overhead)*1e12:.2f} pJ\n"
            f"\n"
            f"KEY INSIGHT: Coherence machinery dominates at small batch sizes!\n"
            f"             GPU burns massive energy managing thousands of concurrent memory requests.\n"
            f"             Tensor Cores are 2.7× more efficient than CUDA cores (0.3 vs 0.8 pJ/MAC)."
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead_total,
            data_movement_overhead=data_movement_overhead_total,
            control_overhead=control_overhead,

            # NEW: MAC/FLOP/IntOp energy breakdown (Phase 3)
            mac_energy=total_mac_energy,
            flop_energy=total_flop_energy,
            intop_energy=total_intop_energy,

            # NEW: Operation counts (Phase 3)
            mac_ops_executed=macs,
            flop_ops_executed=flops,
            intop_ops_executed=intops,

            extra_details={
                # Compute units (detailed breakdown)
                'tensor_core_mac_energy': tensor_core_mac_energy,
                'tensor_core_macs': tensor_core_macs,
                'tensor_core_ops': tensor_core_ops,
                'cuda_core_mac_energy': cuda_core_mac_energy,
                'cuda_core_macs': cuda_core_macs,
                'cuda_core_flop_energy': cuda_core_flop_energy,
                'cuda_core_flops': flops,
                'int_alu_energy': int_alu_energy,
                'int_alu_intops': intops,
                'register_file_energy': register_file_energy,
                'num_register_accesses': num_register_accesses,

                # Energy model parameters (for hardware config display)
                'cuda_core_mac_energy_per_op': self.cuda_core_mac_energy,
                'cuda_core_flop_energy_per_op': self.cuda_core_flop_energy,
                'int_alu_energy_per_op': self.int_alu_energy,
                'tensor_core_mac_energy_per_op': self.tensor_core_mac_energy,
                'register_file_energy_per_access': self.register_file_energy_per_access,

                # Instruction pipeline
                'instruction_fetch_energy': instruction_fetch_energy_total,
                'instruction_decode_energy': instruction_decode_energy_total,
                'instruction_execute_energy': instruction_execute_energy_total,
                'num_instructions': num_instructions,

                # Memory hierarchy (NVIDIA Ampere nomenclature)
                'shared_mem_l1_unified_energy': shared_mem_l1_energy,
                'shared_mem_l1_accesses': shared_mem_l1_accesses,
                'shared_mem_bytes': shared_mem_l1_accesses * 4,  # For arithmetic intensity
                'l1_bytes': shared_mem_l1_accesses * 4,          # Alias for shared_mem_bytes
                'l2_cache_energy': l2_energy,
                'l2_accesses': l2_accesses,
                'l2_bytes': l2_accesses * 4,                     # For arithmetic intensity
                'dram_energy': dram_energy,
                'dram_accesses': dram_accesses,
                'dram_bytes': dram_accesses * 4,                 # For arithmetic intensity

                # Workload data movement (for consistent AI calculation across architectures)
                'bytes_transferred': bytes_transferred,

                # SIMT control overheads
                'coherence_energy': coherence_energy,
                'num_concurrent_warps': num_concurrent_warps,
                'num_memory_ops': num_memory_ops,
                'scheduling_energy': scheduling_energy,
                'concurrent_threads': concurrent_threads,
                'divergence_energy': divergence_energy,
                'num_divergent_ops': num_divergent_ops,
                'coalescing_energy': coalescing_energy,
                'num_uncoalesced': num_uncoalesced,
                'barrier_energy': barrier_energy,
                'num_barriers': num_barriers,
            },
            explanation=explanation
        )


@dataclass
class SystolicArrayEnergyModel(ArchitecturalEnergyModel):
    """
    Energy model for systolic array architectures (Google TPU).

    First-Principles Control Overhead Model:
    TPU control orchestrates complex dataflow through memory hierarchy and systolic array.
    Control unit is ~2% of die area (TPU v1 paper) running at peak core clock.

    Control Operations (per matrix operation):
    1. DMA Setup: Configure transfers from DRAM → Weight Memory (SRAM)
    2. Weight Loader: Shift weights into systolic array columns
    3. Unified Buffer Controller: Manage activation scratchpad
    4. Systolic Array Sequencer: Control weight-stationary dataflow
    5. Accumulator Controller: Manage partial sum staging
    6. Address Generators: Compute tiled matrix addresses
    7. Instruction Decoder: Decode matrix multiply instruction

    Why TPU control is efficient vs CPU/GPU:
    - CPU: 1 instruction per ~2 operations → high control:compute ratio
    - GPU: Cache coherence per memory access → high control per byte
    - TPU: 1 instruction per 16K MACs (systolic array) → low control:compute ratio

    But: Control sequencers run at peak clock and consume real energy!
    """

    # ============================================================
    # CONTROL ENERGY (First-Principles, per operation)
    # ============================================================

    # Instruction-level control (per matrix operation, not per MAC!)
    instruction_decode_energy: float = 5.0e-12        # ~5 pJ per matrix op (not per MAC!)

    # DMA Controller (setup per transfer, amortized over transfer size)
    dma_descriptor_setup: float = 10.0e-9             # ~10 nJ per DMA transfer
    dma_address_gen_per_cacheline: float = 0.1e-12    # ~0.1 pJ per cache line address

    # Weight Loading Sequencer (shift weights into systolic array)
    weight_shift_control_per_element: float = 0.05e-12  # ~0.05 pJ per weight element
    weight_column_select_per_cycle: float = 0.1e-12     # ~0.1 pJ per column per cycle

    # Unified Buffer Controller (activation scratchpad management)
    unified_buffer_address_gen: float = 0.1e-12      # ~0.1 pJ per address
    unified_buffer_arbitration: float = 0.2e-12      # ~0.2 pJ per request

    # Accumulator Controller (partial sum management)
    accumulator_read_control: float = 0.2e-12        # ~0.2 pJ per read
    accumulator_write_control: float = 0.2e-12       # ~0.2 pJ per write
    accumulator_address_gen: float = 0.1e-12         # ~0.1 pJ per address

    # Tile Iteration Control (for tiled matrix operations)
    tile_loop_control_per_tile: float = 1.0e-12      # ~1 pJ per tile iteration

    # Data injection/extraction (spatial array interface)
    data_injection_per_element: float = 0.5e-12      # ~0.5 pJ per element
    data_extraction_per_element: float = 0.5e-12     # ~0.5 pJ per element

    # Architectural efficiency multiplier (vs stored program baseline)
    # These represent the SAVINGS from eliminating instruction fetch, contention, etc.
    compute_efficiency: float = 0.15                 # 15% overhead (85% reduction!)
    memory_efficiency: float = 0.20                  # 20% overhead (80% reduction!)

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        data_movement_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # ========================================================================
        # FIRST-PRINCIPLES TPU CONTROL OVERHEAD MODEL
        # ========================================================================

        # Get systolic array dimension (from execution context or default)
        array_dimension = execution_context.get('array_dimension', 128)  # Default: 128×128
        macs_per_cycle = array_dimension * array_dimension
        num_systolic_cycles = max(1, int((ops + macs_per_cycle - 1) // macs_per_cycle))

        # Calculate number of matrix operations (one per MLP layer, not per MAC!)
        # For small workloads, we might have just 1 matrix operation
        num_matrix_ops = max(1, execution_context.get('num_matrix_ops', 1))

        # 1. INSTRUCTION DECODE (per matrix operation, not per MAC!)
        instruction_decode = num_matrix_ops * self.instruction_decode_energy

        # 2. DMA CONTROLLER
        # Calculate cache lines transferred (64-byte cache lines)
        cache_line_size = execution_context.get('cache_line_size', 64)
        num_cache_lines = max(1, (bytes_transferred + cache_line_size - 1) // cache_line_size)

        # DMA descriptor setup (one per transfer from DRAM → Weight Memory)
        num_dma_transfers = max(1, num_matrix_ops)  # One transfer per matrix op
        dma_setup = num_dma_transfers * self.dma_descriptor_setup

        # DMA address generation (per cache line)
        dma_address_gen = num_cache_lines * self.dma_address_gen_per_cacheline

        # 3. WEIGHT LOADING SEQUENCER
        # Shift weights into systolic array columns (weight-stationary dataflow)
        num_weight_elements = max(1, bytes_transferred // 4)  # Assume 4-byte elements
        weight_shift_control = num_weight_elements * self.weight_shift_control_per_element

        # Column select per cycle (control which columns receive weights)
        weight_column_select = num_systolic_cycles * array_dimension * self.weight_column_select_per_cycle

        # 4. UNIFIED BUFFER CONTROLLER
        # Activation scratchpad management (address generation + arbitration)
        num_ub_accesses = num_weight_elements  # Read activations from unified buffer
        ub_address_gen = num_ub_accesses * self.unified_buffer_address_gen
        ub_arbitration = num_ub_accesses * self.unified_buffer_arbitration

        # 5. ACCUMULATOR CONTROLLER
        # Partial sum staging (read + write + address generation)
        num_accumulator_ops = num_systolic_cycles * array_dimension  # One accumulator per row
        accumulator_read = num_accumulator_ops * self.accumulator_read_control
        accumulator_write = num_accumulator_ops * self.accumulator_write_control
        accumulator_address = num_accumulator_ops * self.accumulator_address_gen

        # 6. TILE ITERATION CONTROL
        # For tiled matrix operations (large matrices don't fit in array)
        # Estimate number of tiles: ops / (array_dimension^2) gives approximate tile count
        num_tiles = max(1, int((ops + macs_per_cycle - 1) // macs_per_cycle))
        tile_loop_control = num_tiles * self.tile_loop_control_per_tile

        # 7. DATA INJECTION/EXTRACTION (spatial array interface)
        num_elements = max(1, bytes_transferred // 4)  # Assume 4-byte elements
        injection_energy = num_elements * self.data_injection_per_element
        extraction_energy = num_elements * self.data_extraction_per_element

        # 8. REGISTER-TO-REGISTER FORWARDING THROUGH SYSTOLIC ARRAY
        # This is the energy to shift data through the array (PE-to-PE register writes)
        # TPU shifts 3 matrices: weights (loaded once), inputs (streamed), outputs (accumulated)
        #
        # For weight-stationary dataflow:
        # - Weights: loaded into PE registers (one-time setup)
        # - Inputs: shift horizontally through array
        # - Outputs: shift vertically through array (partial sums)
        #
        # Number of register writes ≈ elements × (array_dimension / elements_per_pass)
        # For simplicity: assume each element is written to ~array_dimension PEs
        #
        # Energy per register write: 0.3 pJ (typical register file write energy)
        register_write_energy = 0.3e-12  # 0.3 pJ per register write

        # Weight matrix forwarding (load weights into array registers)
        weight_bytes = int(bytes_transferred * 0.5)  # Assume 50% weights
        weight_elements = weight_bytes // 4
        weight_forwarding = weight_elements * register_write_energy

        # Input matrix forwarding (stream through array horizontally)
        input_bytes = int(bytes_transferred * 0.3)  # Assume 30% inputs
        input_elements = input_bytes // 4
        input_forwarding = input_elements * array_dimension * register_write_energy

        # Output matrix forwarding (accumulate and shift vertically)
        output_bytes = int(bytes_transferred * 0.2)  # Assume 20% outputs
        output_elements = output_bytes // 4
        output_forwarding = output_elements * array_dimension * register_write_energy

        systolic_forwarding_energy = weight_forwarding + input_forwarding + output_forwarding

        # ========================================================================
        # TOTAL CONTROL OVERHEAD
        # ========================================================================

        control_overhead_total = (
            instruction_decode +
            dma_setup +
            dma_address_gen +
            weight_shift_control +
            weight_column_select +
            ub_address_gen +
            ub_arbitration +
            accumulator_read +
            accumulator_write +
            accumulator_address +
            tile_loop_control +
            systolic_forwarding_energy  # PE-to-PE register writes through array
        )

        # ========================================================================
        # ARCHITECTURAL EFFICIENCY (vs Stored Program)
        # ========================================================================

        # Architectural benefit: Reduce compute overhead dramatically
        # Because no instruction fetch, decode, dispatch PER OPERATION
        compute_overhead_reduction = -compute_energy_baseline * (1.0 - self.compute_efficiency)

        # Memory benefit: Spatial data flows eliminate contention overhead
        data_movement_overhead_reduction = -data_movement_energy_baseline * (1.0 - self.memory_efficiency)

        total_data_movement = injection_energy + extraction_energy

        # ========================================================================
        # EXPLANATION
        # ========================================================================

        explanation = (
            f"Systolic Array (TPU) First-Principles Control Overhead:\n"
            f"  1. Instruction Decode: {instruction_decode*1e12:.2f} pJ "
            f"({num_matrix_ops} matrix ops × {self.instruction_decode_energy*1e12:.1f} pJ)\n"
            f"  2. DMA Controller:\n"
            f"     - Descriptor Setup: {dma_setup*1e12:.2f} pJ ({num_dma_transfers} transfers)\n"
            f"     - Address Generation: {dma_address_gen*1e12:.2f} pJ ({num_cache_lines} cache lines)\n"
            f"  3. Weight Loading Sequencer:\n"
            f"     - Weight Shift Control: {weight_shift_control*1e12:.2f} pJ ({num_weight_elements} elements)\n"
            f"     - Column Select: {weight_column_select*1e12:.2f} pJ ({num_systolic_cycles} cycles)\n"
            f"  4. Unified Buffer Controller:\n"
            f"     - Address Generation: {ub_address_gen*1e12:.2f} pJ\n"
            f"     - Arbitration: {ub_arbitration*1e12:.2f} pJ\n"
            f"  5. Accumulator Controller:\n"
            f"     - Read Control: {accumulator_read*1e12:.2f} pJ\n"
            f"     - Write Control: {accumulator_write*1e12:.2f} pJ\n"
            f"     - Address Generation: {accumulator_address*1e12:.2f} pJ\n"
            f"  6. Tile Loop Control: {tile_loop_control*1e12:.2f} pJ ({num_tiles} tiles)\n"
            f"  7. Data Injection/Extraction:\n"
            f"     - Injection: {injection_energy*1e12:.2f} pJ\n"
            f"     - Extraction: {extraction_energy*1e12:.2f} pJ\n"
            f"\n"
            f"  TOTAL CONTROL OVERHEAD: {control_overhead_total*1e12:.2f} pJ\n"
            f"  Control Overhead per MAC: {(control_overhead_total/ops*1e12) if ops > 0 else 0:.4f} pJ\n"
            f"\n"
            f"  Why TPU control is efficient:\n"
            f"  - CPU: ~1 instruction per 2 MACs → {5.0/2:.2f} pJ control per MAC\n"
            f"  - GPU: Coherence per memory op → ~0.5 pJ control per byte\n"
            f"  - TPU: 1 instruction per {macs_per_cycle:,} MACs → {(control_overhead_total/ops*1e12) if ops > 0 else 0:.4f} pJ control per MAC\n"
            f"\n"
            f"  Architectural Efficiency (vs Stored Program):\n"
            f"    - Compute overhead eliminated: {-compute_overhead_reduction*1e12:.2f} pJ saved\n"
            f"    - Memory contention eliminated: {-data_movement_overhead_reduction*1e12:.2f} pJ saved\n"
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead_reduction,
            data_movement_overhead=data_movement_overhead_reduction + total_data_movement,
            control_overhead=control_overhead_total,
            extra_details={
                'instruction_decode': instruction_decode,
                'dma_setup': dma_setup,
                'dma_address_gen': dma_address_gen,
                'weight_shift_control': weight_shift_control,
                'weight_column_select': weight_column_select,
                'ub_address_gen': ub_address_gen,
                'ub_arbitration': ub_arbitration,
                'accumulator_read': accumulator_read,
                'accumulator_write': accumulator_write,
                'accumulator_address': accumulator_address,
                'tile_loop_control': tile_loop_control,
                'injection_energy': injection_energy,
                'extraction_energy': extraction_energy,
                'num_systolic_cycles': num_systolic_cycles,
                'array_dimension': array_dimension,
                'num_tiles': num_tiles,
                'num_matrix_ops': num_matrix_ops,
                'control_overhead_per_mac_pj': (control_overhead_total/ops*1e12) if ops > 0 else 0,
                # For arithmetic intensity calculation
                'total_macs': ops // 2,  # MACs (each MAC = 2 ops: multiply + accumulate)
                'dma_bytes': bytes_transferred,  # All bytes go through DMA
                'on_chip_buffer_bytes': bytes_transferred,  # Unified buffer holds all activations

                # Workload data movement (for consistent AI calculation across architectures)
                'bytes_transferred': bytes_transferred,

                # Occurrence counts
                'num_dma_transfers': num_dma_transfers,
                'num_cache_lines': num_cache_lines,
                'num_weight_elements': num_weight_elements,
                'num_ub_accesses': num_ub_accesses,
                'num_accumulator_ops': num_accumulator_ops,
                'num_elements': num_elements,
            },
            explanation=explanation
        )


@dataclass
class DomainFlowEnergyModel(ArchitecturalEnergyModel):
    """
    Energy model for Domain Flow Architectures (Stillwater KPU).

    Programmable systolic processor with domain tracking.

    Key characteristics:
    - SURE/SARE network overlay models variable dependencies
    - Domain checks track computation through N-dimensional space
    - Execution wavefronts can change dynamically without energy consumption
    - Still eliminates instruction fetch per operation
    - More flexible than fixed systolic, more efficient than stored program
    - Data is streamed through spatial domains with dynamic control

    Historical context: Evolved from Data Flow Machine concepts.
    Modern innovation: x86 uses DFM concepts for register renaming.
    """

    # Domain flow management overhead: CAM-like tracking per operation
    domainflow_tracking_per_op: float = 1.0e-12        # ~1 pJ per operation
  
    # Dataflow flexibility cost (more than fixed systolic, less than stored program)
    dataflow_adaptation_energy: float = 50.0e-12   # ~50 pJ per schedule change
    # if we model that, we would allocate that to the data token routing energy cost

    # Data movement of domains into the fabric
    domain_data_injection: float = 0.5e-12         # ~0.5 pJ per element
    domain_data_extraction: float = 0.5e-12        # ~0.5 pJ per element
    # assume we tile which create some overfetching, typically on one domain
    domain_data_overfetch_factor: float = 1.1      # 10% overfetching overhead: only on input domains

    # Efficiency vs stored program (less efficient than fixed systolic, but programmable)
    compute_efficiency: float = 0.75               # 75% of peak
    memory_efficiency: float = 0.75                # 75% of peak

    # Kernel update frequency (adaptive based on computation structure)
    kernel_changes: int = 10                       # ~10 kernel changes per workload
    # kernel programs get streamed into the array with very little energy cost
    # so first order, we ignore the energy cost of loading new programs

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        data_movement_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # Domain tracking per operation
        domain_tracking_energy = ops * self.domainflow_tracking_per_op

        # Kernel changes (when computation pattern changes)
        num_kernel_changes = execution_context.get('kernel_changes', 1)
        kernel_load_energy = num_kernel_changes * self.dataflow_adaptation_energy

        # Data movement through domains
        num_elements = max(1, bytes_transferred // 4)
        injection_energy = num_elements * self.domain_data_injection
        extraction_energy = num_elements * self.domain_data_extraction

        # Architectural benefit: Still eliminates instruction fetch
        compute_overhead_reduction = -compute_energy_baseline * (1.0 - self.compute_efficiency)
        data_movement_overhead_reduction = -data_movement_energy_baseline * (1.0 - self.memory_efficiency)

        total_domain_overhead = domain_tracking_energy + kernel_load_energy
        total_data_movement = injection_energy + extraction_energy

        explanation = (
            f"Domain Flow Architecture Energy Events:\n"
            f"  Domain Tracking:\n"
            f"    - Operations: {ops:,}\n"
            f"    - Energy: {domain_tracking_energy*1e12:.2f} pJ "
            f"({ops:,} x {self.domainflow_tracking_per_op*1e12:.2f} pJ)\n"
            f"  Schedule Adaptation:\n"
            f"    - Schedule changes: {num_kernel_changes}\n"
            f"    - Energy: {kernel_load_energy*1e12:.2f} pJ\n"
            f"  Domain Data Movement:\n"
            f"    - Elements: {num_elements:,}\n"
            f"    - Injection: {injection_energy*1e12:.2f} pJ\n"
            f"    - Extraction: {extraction_energy*1e12:.2f} pJ\n"
            f"  Architectural Efficiency (vs Stored Program):\n"
            f"    - Compute overhead eliminated: {-compute_overhead_reduction*1e12:.2f} pJ saved\n"
            f"    - Memory overhead eliminated: {-data_movement_overhead_reduction*1e12:.2f} pJ saved\n"
            f"  Efficiency vs Stored Program:\n"
            f"    - Compute: {self.compute_efficiency*100:.0f}% overhead "
            f"({(1-self.compute_efficiency)*100:.0f}% reduction)\n"
            f"    - Memory: {self.memory_efficiency*100:.0f}% overhead "
            f"({(1-self.memory_efficiency)*100:.0f}% reduction)\n"
            f"\n"
            f"KEY: Programmable spatial scheduling with domain tracking.\n"
            f"     More flexible than fixed systolic, still eliminates instruction fetch.\n"
            f"     Systolic array is special case (all dynamic control → one function)."
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead_reduction,
            data_movement_overhead=data_movement_overhead_reduction + total_data_movement,
            control_overhead=total_domain_overhead,
            extra_details={
                'domain_tracking_energy': domain_tracking_energy,
                'kernel_load_energy': kernel_load_energy,
                'injection_energy': injection_energy,
                'extraction_energy': extraction_energy,
            },
            explanation=explanation
        )


@dataclass
class DataFlowMachineEnergyModel(ArchitecturalEnergyModel):
    """
    Energy model for Data Flow Machines (Jack Dennis, Petri nets).

    Token-based execution with CAM for instruction matching.

    Key characteristics:
    - CAM (Content Addressable Memory) holds instruction tokens
    - Token matching determines ready operations
    - Sea of processing elements execute ready tokens
    - No program counter, data availability triggers execution

    Why no commercial implementations: CAM energy overhead limits benefits.

    Modern influence: x86 register renaming uses DFM concepts internally.
    """

    # CAM is energy-intensive (associative search)
    cam_lookup_per_cycle: float = 5.0e-12          # ~5 pJ per CAM lookup
    token_matching_energy: float = 3.0e-12         # ~3 pJ per token match
    token_queue_management: float = 1.5e-12        # ~1.5 pJ per queue operation

    # Dataflow graph traversal
    graph_traversal_per_node: float = 2.0e-12      # ~2 pJ per graph node

    # Efficiency (CAM overhead limits gains)
    compute_efficiency: float = 0.40               # 40% overhead (60% reduction)
    memory_efficiency: float = 0.45                # 45% overhead (55% reduction)

    # Matching frequency
    cam_lookups_per_op: float = 0.5                # ~1 CAM lookup per 2 ops

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        data_movement_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # CAM lookups (expensive!)
        num_cam_lookups = int(ops * self.cam_lookups_per_op)
        cam_energy = num_cam_lookups * self.cam_lookup_per_cycle

        # Token matching
        num_token_matches = ops  # One match per operation
        matching_energy = num_token_matches * self.token_matching_energy

        # Token queue management
        queue_energy = num_token_matches * self.token_queue_management

        # Graph traversal
        num_nodes = execution_context.get('graph_nodes', ops // 10)
        traversal_energy = num_nodes * self.graph_traversal_per_node

        # Architectural benefit: No instruction fetch, but CAM limits savings
        compute_overhead_reduction = -compute_energy_baseline * (1.0 - self.compute_efficiency)
        data_movement_overhead_reduction = -data_movement_energy_baseline * (1.0 - self.memory_efficiency)

        total_dfm_overhead = cam_energy + matching_energy + queue_energy + traversal_energy

        explanation = (
            f"Data Flow Machine Energy Events:\n"
            f"  CAM (Content Addressable Memory):\n"
            f"    - Lookups: {num_cam_lookups:,}\n"
            f"    - Energy: {cam_energy*1e12:.2f} pJ "
            f"({num_cam_lookups:,} × {self.cam_lookup_per_cycle*1e12:.2f} pJ)\n"
            f"    WARNING: CAM is energy-intensive!\n"
            f"  Token Matching:\n"
            f"    - Matches: {num_token_matches:,}\n"
            f"    - Energy: {matching_energy*1e12:.2f} pJ\n"
            f"  Token Queue Management:\n"
            f"    - Queue ops: {num_token_matches:,}\n"
            f"    - Energy: {queue_energy*1e12:.2f} pJ\n"
            f"  Dataflow Graph Traversal:\n"
            f"    - Nodes: {num_nodes:,}\n"
            f"    - Energy: {traversal_energy*1e12:.2f} pJ\n"
            f"  Architectural Efficiency (vs Stored Program):\n"
            f"    - Compute overhead eliminated: {-compute_overhead_reduction*1e12:.2f} pJ saved\n"
            f"    - Memory overhead eliminated: {-data_movement_overhead_reduction*1e12:.2f} pJ saved\n"
            f"  Efficiency vs Stored Program:\n"
            f"    - Compute: {self.compute_efficiency*100:.0f}% overhead "
            f"({(1-self.compute_efficiency)*100:.0f}% reduction)\n"
            f"    - Memory: {self.memory_efficiency*100:.0f}% overhead "
            f"({(1-self.memory_efficiency)*100:.0f}% reduction)\n"
            f"\n"
            f"KEY: Token-based execution eliminates program counter,\n"
            f"     but CAM energy overhead limits benefits.\n"
            f"     Why no commercial implementations: CAM cost > savings."
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead_reduction,
            data_movement_overhead=data_movement_overhead_reduction,
            control_overhead=total_dfm_overhead,
            extra_details={
                'cam_energy': cam_energy,
                'matching_energy': matching_energy,
                'queue_energy': queue_energy,
                'traversal_energy': traversal_energy,
            },
            explanation=explanation
        )


@dataclass
class SpatialPartitionEnergyModel(ArchitecturalEnergyModel):
    """
    Energy model for Spatial Partition architectures (Cerebras WSE, Hailo).

    Mesh of processing elements with graph partitioning.

    Key characteristics:
    - Computational graph partitioned into sub-graphs
    - Sub-graphs mapped to physical mesh topology
    - Inter-partition communication via mesh links
    - Data locality optimization critical for efficiency

    Energy efficiency depends on partitioning quality:
    - Good partitioning → high locality → low inter-partition traffic
    - Poor partitioning → high communication → energy approaches stored program
    """

    # Graph partitioning (compile-time, amortized)
    graph_partitioning_energy: float = 1000.0e-12  # ~1 nJ per graph partition
    partition_quality_factor: float = 0.85          # 0.0-1.0 (0.85 = good compiler)

    # Inter-partition communication (THE critical energy component)
    mesh_link_energy_per_hop: float = 3.0e-12      # ~3 pJ per mesh hop
    average_partition_distance: int = 4             # Average hops between partitions
    ingress_buffer_energy: float = 1.0e-12         # ~1 pJ per element (receive)
    egress_buffer_energy: float = 1.0e-12          # ~1 pJ per element (send)

    # Partition-local computation (very efficient once data is local)
    pe_local_compute_per_op: float = 0.3e-12       # ~0.3 pJ per local operation
    pe_local_memory_access: float = 0.5e-12        # ~0.5 pJ per local SRAM access

    # Synchronization overhead
    barrier_sync_energy: float = 10.0e-12          # ~10 pJ per barrier
    barrier_frequency: float = 0.01                # Barriers per operation (1%)

    # Efficiency vs stored program (depends on partition quality)
    compute_efficiency: float = 0.20               # 20% overhead (80% reduction)
    memory_efficiency: float = 0.25                # 25% overhead (75% reduction)

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        data_movement_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # Graph partitioning (amortized)
        num_partitions = execution_context.get('num_partitions', 100)
        partitioning_energy = self.graph_partitioning_energy * num_partitions

        # Amortize over number of inferences
        inferences_per_partition = execution_context.get('inferences_per_partition', 1000)
        partitioning_energy_amortized = partitioning_energy / inferences_per_partition

        # Inter-partition communication (CRITICAL for efficiency)
        total_data_movement = max(1, bytes_transferred // 4)  # Elements

        # Boundary crossings depend on partition quality
        # partition_quality_factor: 1.0 = perfect (no crossings), 0.0 = terrible (all cross)
        boundary_crossings = int(total_data_movement * (1.0 - self.partition_quality_factor))

        # Energy per crossing: ingress + mesh hops + egress
        energy_per_crossing = (
            self.ingress_buffer_energy +
            (self.mesh_link_energy_per_hop * self.average_partition_distance) +
            self.egress_buffer_energy
        )

        inter_partition_energy = boundary_crossings * energy_per_crossing

        # Partition-local computation (very efficient)
        local_data = int(total_data_movement * self.partition_quality_factor)
        local_compute_energy = ops * self.pe_local_compute_per_op
        local_memory_energy = local_data * self.pe_local_memory_access

        # Synchronization overhead
        num_barriers = int(ops * self.barrier_frequency)
        sync_energy = num_barriers * self.barrier_sync_energy

        # Architectural benefit
        compute_overhead_reduction = -compute_energy_baseline * (1.0 - self.compute_efficiency)
        data_movement_overhead_reduction = -data_movement_energy_baseline * (1.0 - self.memory_efficiency)

        total_compute = local_compute_energy
        total_memory = inter_partition_energy + local_memory_energy
        total_control = partitioning_energy_amortized + sync_energy

        explanation = (
            f"Spatial Partition Architecture Energy Events:\n"
            f"  Graph Partitioning (compile-time, amortized):\n"
            f"    - Partitions: {num_partitions}, Quality: {self.partition_quality_factor*100:.0f}%\n"
            f"    - Energy: {partitioning_energy_amortized*1e12:.2f} pJ "
            f"(amortized over {inferences_per_partition} inferences)\n"
            f"  Inter-Partition Communication (CRITICAL):\n"
            f"    - Total data movement: {total_data_movement:,} elements\n"
            f"    - Boundary crossings: {boundary_crossings:,} elements "
            f"({(1.0-self.partition_quality_factor)*100:.0f}% crosses partitions)\n"
            f"    - Energy per crossing: {energy_per_crossing*1e12:.2f} pJ "
            f"(ingress + {self.average_partition_distance} hops + egress)\n"
            f"    - Total: {inter_partition_energy*1e12:.2f} pJ\n"
            f"  Partition-Local (Efficient):\n"
            f"    - Local data: {local_data:,} elements "
            f"({self.partition_quality_factor*100:.0f}% stays local)\n"
            f"    - Compute: {local_compute_energy*1e12:.2f} pJ\n"
            f"    - Memory: {local_memory_energy*1e12:.2f} pJ\n"
            f"  Synchronization:\n"
            f"    - Barriers: {num_barriers}\n"
            f"    - Energy: {sync_energy*1e12:.2f} pJ\n"
            f"  Architectural Efficiency (vs Stored Program):\n"
            f"    - Compute overhead eliminated: {-compute_overhead_reduction*1e12:.2f} pJ saved\n"
            f"    - Memory overhead eliminated: {-data_movement_overhead_reduction*1e12:.2f} pJ saved\n"
            f"  Efficiency vs Stored Program:\n"
            f"    - Compute: {self.compute_efficiency*100:.0f}% overhead "
            f"({(1-self.compute_efficiency)*100:.0f}% reduction)\n"
            f"    - Memory: {self.memory_efficiency*100:.0f}% overhead "
            f"({(1-self.memory_efficiency)*100:.0f}% reduction)\n"
            f"\n"
            f"KEY INSIGHT: Partition quality ({self.partition_quality_factor*100:.0f}%) determines "
            f"{boundary_crossings/total_data_movement*100:.0f}% of data crosses partitions.\n"
            f"Better compiler → better partitioning → less communication → lower energy!"
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead_reduction + total_compute,
            data_movement_overhead=data_movement_overhead_reduction + total_memory,
            control_overhead=total_control,
            extra_details={
                'inter_partition_energy': inter_partition_energy,
                'partition_quality': self.partition_quality_factor,
                'boundary_crossings': boundary_crossings,
                'local_data': local_data,
            },
            explanation=explanation
        )


@dataclass
class AdaptiveDatapathEnergyModel(ArchitecturalEnergyModel):
    """
    Energy model for Adaptive Datapath architectures (FPGA, CGRA, DPU).

    Reconfigurable fabric with spatial routing.

    Key characteristics:
    - Algorithm mapped to arrangement of fixed datapath elements
    - Reconfiguration infrastructure (LUTs, routing, hard macros)
    - High reconfiguration cost (amortized per workload)
    - Runtime efficiency is good, but reconfiguration limits general purpose use

    Subcategories:
    - FPGA: Fine-grained LUT-based logic
    - CGRA: Coarse-grained word-level datapath
    - DPU: FPGA with pre-defined hard macros (Conv, Linear only)
    """

    # Reconfiguration costs (amortized per workload)
    reconfiguration_energy_per_lut: float = 100.0e-12      # ~100 pJ per LUT
    routing_config_energy_per_element: float = 50.0e-12   # ~50 pJ per routing element
    total_luts: int = 500_000                              # Typical FPGA size
    total_routing_elements: int = 500_000

    # Runtime overhead (after configuration)
    routing_overhead_per_op: float = 2.5e-12               # ~2.5 pJ (multiplexers)
    hard_macro_activation: float = 1.0e-12                 # ~1 pJ (DSP block switching)

    # Efficiency (runtime only, excludes reconfiguration)
    compute_efficiency: float = 0.25                       # 25% overhead (75% reduction)
    memory_efficiency: float = 0.30                        # 30% overhead (70% reduction)

    def compute_reconfiguration_overhead(
        self,
        execution_context: Dict
    ) -> float:
        """
        Compute per-inference reconfiguration overhead.

        Context must include:
        - kernel_switches_per_inference: How many times workload changes
        - inferences_per_config: How many inferences before reconfiguration
        """
        kernel_switches = execution_context.get('kernel_switches_per_inference', 1)
        inferences_per_config = execution_context.get('inferences_per_config', 1000)

        # Total reconfiguration energy
        total_reconfig = (
            self.total_luts * self.reconfiguration_energy_per_lut +
            self.total_routing_elements * self.routing_config_energy_per_element
        )

        # Amortized per inference
        reconfig_per_inference = (total_reconfig * kernel_switches) / inferences_per_config

        return reconfig_per_inference

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        data_movement_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # Reconfiguration overhead (amortized)
        reconfig_energy = self.compute_reconfiguration_overhead(execution_context)

        # Runtime routing overhead
        routing_energy = ops * self.routing_overhead_per_op

        # Hard macro activation (DSP blocks, BRAMs)
        num_hard_macros = execution_context.get('hard_macros_used', ops // 1000)
        macro_energy = num_hard_macros * self.hard_macro_activation

        # Architectural benefit (runtime efficiency)
        compute_overhead_reduction = -compute_energy_baseline * (1.0 - self.compute_efficiency)
        data_movement_overhead_reduction = -data_movement_energy_baseline * (1.0 - self.memory_efficiency)

        total_runtime_overhead = routing_energy + macro_energy

        kernel_switches = execution_context.get('kernel_switches_per_inference', 1)
        inferences_per_config = execution_context.get('inferences_per_config', 1000)

        explanation = (
            f"Adaptive Datapath Architecture Energy Events:\n"
            f"  Reconfiguration (amortized per workload):\n"
            f"    - LUTs: {self.total_luts:,} × {self.reconfiguration_energy_per_lut*1e12:.2f} pJ\n"
            f"    - Routing: {self.total_routing_elements:,} × {self.routing_config_energy_per_element*1e12:.2f} pJ\n"
            f"    - Kernel switches per inference: {kernel_switches}\n"
            f"    - Amortized over: {inferences_per_config} inferences\n"
            f"    - Energy per inference: {reconfig_energy*1e12:.2f} pJ\n"
            f"  Runtime Overhead:\n"
            f"    - Routing (multiplexers): {routing_energy*1e12:.2f} pJ "
            f"({ops:,} x {self.routing_overhead_per_op*1e12:.2f} pJ)\n"
            f"    - Hard macros: {macro_energy*1e12:.2f} pJ "
            f"({num_hard_macros:,} activations)\n"
            f"  Architectural Efficiency (runtime, vs Stored Program):\n"
            f"    - Compute overhead eliminated: {-compute_overhead_reduction*1e12:.2f} pJ saved\n"
            f"    - Memory overhead eliminated: {-data_movement_overhead_reduction*1e12:.2f} pJ saved\n"
            f"  Efficiency vs Stored Program:\n"
            f"    - Compute: {self.compute_efficiency*100:.0f}% overhead "
            f"({(1-self.compute_efficiency)*100:.0f}% reduction)\n"
            f"    - Memory: {self.memory_efficiency*100:.0f}% overhead "
            f"({(1-self.memory_efficiency)*100:.0f}% reduction)\n"
            f"\n"
            f"KEY: Runtime efficiency is good, but reconfiguration cost limits\n"
            f"     general purpose use. For dynamic workloads (frequent kernel\n"
            f"     switches), reconfiguration dominates! DPU limited to simple ops."
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead_reduction,
            data_movement_overhead=data_movement_overhead_reduction + total_runtime_overhead,
            control_overhead=reconfig_energy,
            extra_details={
                'reconfiguration_energy': reconfig_energy,
                'routing_energy': routing_energy,
                'macro_energy': macro_energy,
            },
            explanation=explanation
        )


@dataclass
class TPUTileEnergyModel:
    """
    Tile-based energy model for TPU systolic array architectures.

    Captures the tile-based data movement through TPU memory hierarchy:
    Weight Memory → Weight FIFO → Matrix Unit → Accumulators → Unified Buffer

    This model is based on the TPU v1 architecture (ISCA 2017 paper) and
    generalizes to v3/v4/v5 with generation-specific parameters.

    Key energy events per tile:
    1. Weight tile loading (DRAM/HBM → Weight FIFO → Matrix Unit)
    2. Input activation streaming (Unified Buffer → Matrix Unit)
    3. Systolic array computation (MACs)
    4. Accumulator management (partial sum staging)
    5. Output write (Accumulators → Unified Buffer)
    6. Pipeline fill/drain overhead

    Energy is amortized by batch size (weights loaded once, reused across batch).
    """

    # ============================================================
    # Architectural Parameters (generation-specific)
    # ============================================================

    # Systolic array configuration
    array_width: int  # 256 (v1) or 128 (v3/v4/v5)
    array_height: int  # 256 (v1) or 128 (v3/v4/v5)
    num_arrays: int  # 1 (v1, Coral) or 2 (v3/v4/v5)

    # Weight tile configuration
    weight_tile_size: int  # 64 KiB (v1) or 32 KiB (v3+)
    weight_fifo_depth: int  # 4 tiles (v1) or 2 tiles (v3+, estimated)

    # Pipeline depth
    pipeline_fill_cycles: int  # 256 (v1) or 128 (v3+)
    clock_frequency_hz: float  # 700 MHz (v1), 1050 MHz (v4), etc.

    # Accumulator configuration
    accumulator_size: int  # 4 MiB (v1), 2 MiB per MXU (v3+)
    accumulator_width: int  # 256 (v1) or 128 (v3+)

    # Unified Buffer size
    unified_buffer_size: int  # 24 MiB (v1), 32 MiB (v4, estimated)

    # ============================================================
    # Energy Coefficients (technology-dependent)
    # ============================================================

    # Memory energy (varies by generation)
    weight_memory_energy_per_byte: float  # 10 pJ (DDR3), 5-10 pJ (HBM), 20 pJ (USB)
    weight_fifo_energy_per_byte: float  # 0.5 pJ (on-chip SRAM buffering)
    unified_buffer_read_energy_per_byte: float  # 0.5 pJ (on-chip SRAM)
    unified_buffer_write_energy_per_byte: float  # 0.5 pJ (on-chip SRAM)

    # Accumulator energy (on-chip SRAM, 32-bit wide)
    accumulator_write_energy_per_element: float  # 0.4 pJ (32-bit write)
    accumulator_read_energy_per_element: float  # 0.3 pJ (32-bit read)

    # Matrix unit data movement
    weight_shift_in_energy_per_element: float  # 0.3 pJ (shift register energy)
    activation_stream_energy_per_element: float  # 0.2 pJ (stream into array)

    # Computation energy
    mac_energy: float  # 0.2 pJ (8-bit MAC), 0.25 pJ (BF16 MAC), 0.6 pJ (FP32)

    def compute_tile_energy(
        self,
        num_weight_tiles: int,
        ops_per_tile: int,
        input_elements_per_tile: int,
        output_elements_per_tile: int,
        batch_size: int = 1,
        precision: str = "INT8"
    ) -> Dict[str, float]:
        """
        Compute energy for tile-based matrix operation.

        This models the complete data flow through the TPU memory hierarchy
        for a tiled matrix operation (Conv2D, Linear, MatMul).

        Args:
            num_weight_tiles: Number of weight tiles to load
            ops_per_tile: MACs per weight tile
            input_elements_per_tile: Input activation elements per tile
            output_elements_per_tile: Output elements per tile (accumulator outputs)
            batch_size: Batch size (weight reuse factor)
            precision: Operation precision (INT8, BF16, FP32, FP8)

        Returns:
            Dictionary with detailed energy breakdown (all in Joules)
        """

        # ============================================================
        # 1. Weight Tile Loading (amortized by batch size)
        # ============================================================

        # Weight Memory → Weight FIFO (off-chip DDR3/HBM or USB for Coral)
        # This is the MOST EXPENSIVE operation for small batch sizes
        weight_dram_energy = (
            num_weight_tiles * self.weight_tile_size *
            self.weight_memory_energy_per_byte
        ) / max(1, batch_size)  # Amortized: weights loaded once, reused for batch

        # Weight FIFO buffering (on-chip staging, 256 KiB buffer)
        weight_fifo_energy = (
            num_weight_tiles * self.weight_tile_size *
            self.weight_fifo_energy_per_byte
        )

        # Weight shift-in to Matrix Unit (shift register energy, 256 or 128 cycles)
        bytes_per_element = self._get_bytes_per_element(precision)
        elements_per_tile = self.weight_tile_size // bytes_per_element
        weight_shift_energy = (
            num_weight_tiles * elements_per_tile *
            self.weight_shift_in_energy_per_element
        )

        total_weight_energy = weight_dram_energy + weight_fifo_energy + weight_shift_energy

        # ============================================================
        # 2. Input Activation Loading (DRAM → Unified Buffer → Matrix Unit)
        # ============================================================

        input_bytes = (
            input_elements_per_tile * num_weight_tiles * batch_size * bytes_per_element
        )

        # IMPORTANT: Activations must come from DRAM → Unified Buffer first!
        # The Unified Buffer is only 2-32 MiB, so for large activations we need
        # to stage them from DRAM (just like weights).
        #
        # Energy breakdown:
        # 1. DRAM → Unified Buffer (12 pJ/byte for LPDDR5, same as weights)
        # 2. Unified Buffer → Matrix Unit (0.5 pJ/byte read + 0.2 pJ/byte stream)
        #
        # This was MISSING and caused TPU memory energy to be 38% too low!

        # DRAM → Unified Buffer (activations staged from main memory)
        input_dram_energy = input_bytes * self.weight_memory_energy_per_byte

        # Unified Buffer read (activations staged in 2-32 MiB UB)
        input_read_energy = input_bytes * self.unified_buffer_read_energy_per_byte

        # Stream into Matrix Unit (spatial data flow through systolic array)
        total_input_elements = input_elements_per_tile * num_weight_tiles * batch_size
        activation_stream_energy = (
            total_input_elements * self.activation_stream_energy_per_element
        )

        total_input_energy = input_dram_energy + input_read_energy + activation_stream_energy

        # ============================================================
        # 3. Computation (Systolic Array MACs)
        # ============================================================

        # Adjust MAC energy for precision (INT8 most efficient, FP32 least)
        mac_energy = self._get_mac_energy(precision)
        total_ops = ops_per_tile * num_weight_tiles * batch_size
        compute_energy = total_ops * mac_energy

        # ============================================================
        # 4. Accumulator Management
        # ============================================================

        # Partial sums written to accumulators during computation
        # Accumulators sized to reach roofline knee (~1350 ops/byte for v1)
        total_output_elements = output_elements_per_tile * num_weight_tiles * batch_size

        # Write partial sums (during computation, 256 elements/cycle)
        accumulator_write_energy = (
            total_output_elements * self.accumulator_write_energy_per_element
        )

        # Read completed results (after tile finishes, DMA to Unified Buffer)
        accumulator_read_energy = (
            total_output_elements * self.accumulator_read_energy_per_element
        )

        total_accumulator_energy = accumulator_write_energy + accumulator_read_energy

        # ============================================================
        # 5. Output Write (Accumulators → Unified Buffer → DRAM)
        # ============================================================

        output_bytes = total_output_elements * bytes_per_element

        # Accumulator → Unified Buffer (0.5 pJ/byte write)
        output_write_energy = output_bytes * self.unified_buffer_write_energy_per_byte

        # Unified Buffer → DRAM (12 pJ/byte for LPDDR5)
        # Outputs must be written back to main memory for next layer!
        output_dram_energy = output_bytes * self.weight_memory_energy_per_byte

        total_output_energy = output_write_energy + output_dram_energy

        # ============================================================
        # Energy Breakdown
        # ============================================================
        # Note: Pipeline fill/drain overhead (128-256 cycles) is a LATENCY concern,
        # not an energy concern. The energy cost is already captured in weight_shift_in_energy.
        # The latency impact should be modeled in the TPUMapper's latency calculations.

        total_energy = (
            total_weight_energy + total_input_energy + compute_energy +
            total_accumulator_energy + total_output_energy
        )

        # Calculate arithmetic intensity (ops per byte transferred)
        total_bytes_transferred = input_bytes + output_bytes + (num_weight_tiles * self.weight_tile_size)
        arithmetic_intensity = total_ops / total_bytes_transferred if total_bytes_transferred > 0 else 0

        return {
            # Weight loading (off-chip → on-chip, THE MOST EXPENSIVE)
            'weight_dram_energy_j': weight_dram_energy,
            'weight_fifo_energy_j': weight_fifo_energy,
            'weight_shift_energy_j': weight_shift_energy,
            'total_weight_energy_j': total_weight_energy,

            # Input activation loading (DRAM → Unified Buffer → Matrix Unit)
            'input_dram_energy_j': input_dram_energy,
            'input_read_energy_j': input_read_energy,
            'activation_stream_energy_j': activation_stream_energy,
            'total_input_energy_j': total_input_energy,

            # Computation
            'compute_energy_j': compute_energy,

            # Accumulator management
            'accumulator_write_energy_j': accumulator_write_energy,
            'accumulator_read_energy_j': accumulator_read_energy,
            'total_accumulator_energy_j': total_accumulator_energy,

            # Output staging (Accumulator → Unified Buffer → DRAM)
            'output_write_energy_j': output_write_energy,
            'output_dram_energy_j': output_dram_energy,
            'total_output_energy_j': total_output_energy,

            # Total
            'total_energy_j': total_energy,

            # Metrics
            'num_tiles': num_weight_tiles,
            'batch_size': batch_size,
            'weight_reuse_factor': batch_size,
            'arithmetic_intensity': arithmetic_intensity,
            'total_ops': total_ops,
            'total_bytes': total_bytes_transferred,
        }

    def _get_bytes_per_element(self, precision: str) -> int:
        """Get bytes per element for precision"""
        return {
            'FP32': 4,
            'BF16': 2,
            'FP16': 2,
            'INT8': 1,
            'FP8': 1,
            'FP8_E4M3': 1,
            'FP8_E5M2': 1,
        }.get(precision, 4)

    def _get_mac_energy(self, precision: str) -> float:
        """
        Get MAC energy for precision.

        INT8 is most efficient (systolic array optimized for it).
        BF16 is ~1.5× INT8 (wider datapath, more switching).
        FP32 is ~3× INT8 (much wider, not native on most TPUs).
        """
        base_int8_energy = self.mac_energy
        return {
            'INT8': base_int8_energy,
            'FP8': base_int8_energy,
            'FP8_E4M3': base_int8_energy,
            'FP8_E5M2': base_int8_energy,
            'BF16': base_int8_energy * 1.5,  # BF16 ~1.5× INT8 energy
            'FP16': base_int8_energy * 1.5,
            'FP32': base_int8_energy * 3.0,  # FP32 ~3× INT8 energy (often emulated)
        }.get(precision, base_int8_energy)


@dataclass
class KPUTileEnergyModel:
    """
    Tile-based energy model for KPU Domain Flow Architecture (DFA).

    The KPU differs from TPU in fundamental ways:
    1. 4-stage memory hierarchy (vs TPU's 2-stage): DRAM → L3 → L2 → L1 → Fabric
    2. Token-based spatial dataflow with signature matching
    3. Programmable SURE execution (supports all BLAS operators)
    4. 3 data movement engines: DMA, BlockMover, Streamer
    5. Distributed L3 scratchpad with variable routing distance
    6. Automatic operator fusion in hardware
    7. Multi-engine coordination overhead

    Energy Model Comparison:
    - TPU: Fixed systolic schedule, minimal control (~0.8 pJ/MAC)
    - KPU: Programmable spatial dataflow, token routing (~1.1 pJ/MAC)
    - GPU: SIMT with massive coherence (~1.5 pJ/MAC)
    - CPU: Sequential with instruction fetch (~5 pJ/MAC)

    Key Energy Events:
    1. 4-stage memory hierarchy (DRAM → L3 → L2 → L1)
    2. Token signature matching (distributed CAM-like operation)
    3. SURE program loading (per-operator broadcast)
    4. Data movement engines (DMA, BlockMover, Streamer)
    5. PE computation (BLAS operators in spatial fabric)
    6. Token routing through fabric (signature matching + handshake)
    7. L3 distributed scratchpad (variable distance routing)
    8. Operator fusion coordination
    """

    # ============================================================
    # Architectural Parameters (product-specific)
    # ============================================================

    # Processing element configuration (NO DEFAULTS - must come first!)
    num_tiles: int  # 64 (T64), 256 (T256), 768 (T768)
    pes_per_tile: int  # an array of 16x16 PEs = 256 typical
    tile_mesh_dimensions: tuple  # (8, 8) for T64, (16, 16) for T256, (24, 32) for T768

    # Memory hierarchy configuration (4-stage)
    dram_bandwidth_gb_s: float  # 25.6 (T64/DDR4), 204.8 (T256/LPDDR5), 1638.4 (T768/HBM2)
    l3_size_per_tile: int  # 256 KiB per tile (distributed scratchpad)
    l2_size_per_tile: int  # 32 KiB per tile
    l1_size_per_pe: int  # 4 KiB per PE

    # Clock frequency
    clock_frequency_hz: float  # 800 MHz (T64), 1200 MHz (T256), 1500 MHz (T768)

    # ============================================================
    # Energy Coefficients (technology-dependent)
    # ============================================================

    # Memory hierarchy energy (Joules per byte)
    # CRITICAL: KPU has 4 stages vs TPU's 2 stages
    dram_read_energy_per_byte: float  # 10 pJ (DDR4) or 5 pJ (HBM2)
    dram_write_energy_per_byte: float  # 12 pJ (DDR4) or 6 pJ (HBM2)
    l3_read_energy_per_byte: float  # 2.0 pJ (distributed SRAM)
    l3_write_energy_per_byte: float  # 2.5 pJ (distributed SRAM)
    l2_read_energy_per_byte: float  # 0.8 pJ (tile-local SRAM)
    l2_write_energy_per_byte: float  # 1.0 pJ (tile-local SRAM)
    l1_read_energy_per_byte: float  # 0.3 pJ (PE-local SRAM)
    l1_write_energy_per_byte: float  # 0.4 pJ (PE-local SRAM)

    # Computation energy (BLAS operators on PE)
    mac_energy_int8: float  # ~0.3 pJ (slightly higher than TPU due to programmability)
    mac_energy_bf16: float  # ~0.45 pJ (1.5× INT8)
    mac_energy_fp32: float  # ~0.9 pJ (3× INT8)

    # Token-based execution (WITH DEFAULTS - must come after!)
    token_payload_bytes: int = 4  # Data payload size
    token_signature_bytes: int = 4  # Signature for matching
    max_tokens_in_flight: int = 8192  # 32 entry CAM * 256 PEs =  Per tile

    # SURE program configuration
    sure_program_size_bytes: int = 256 # Typical SURE program
    sure_program_cache_size: int = 4  # Programs cached per tile

    # Data movement engines
    dma_engines_per_tile: int = 4  # DRAM ↔ L3
    blockmover_engines_per_tile: int = 2  # L3 ↔ L2 (inter-tile)
    streamer_engines_per_tile: int = 4  # L2 ↔ L1 (intra-tile)

    # Data movement engine energy (UNIQUE TO KPU)
    dma_transfer_energy_per_byte: float = 1.5e-12  # ~1.5 pJ (DRAM ↔ L3)
    blockmover_energy_per_byte: float = 0.8e-12  # ~0.8 pJ (L3 ↔ L2 inter-tile)
    streamer_energy_per_byte: float = 0.3e-12  # ~0.3 pJ (L2 ↔ L1 intra-tile)

    # Token routing energy (UNIQUE TO KPU - THE KEY DIFFERENTIATOR)
    token_signature_matching_energy: float = 0.6e-12  # ~0.6 pJ per match (distributed CAM-like)
    token_dispatch_energy: float = 0.2e-12  # ~0.2 pJ per instruction token dispatch (firing ready tokens)
    token_routing_per_hop: float = 0.15e-12  # ~0.15 pJ per mesh hop

    # SURE program management (UNIQUE TO KPU)
    sure_program_load_energy: float = 50e-12  # ~50 pJ per program broadcast
    sure_program_cache_hit_energy: float = 1e-12  # ~1 pJ (cache hit, no broadcast)

    # L3 distributed scratchpad routing (UNIQUE TO KPU)
    l3_routing_distance_factor: float = 1.2  # Average routing distance multiplier
    l3_noc_energy_per_hop: float = 0.5e-12  # ~0.5 pJ per NoC hop

    # Operator fusion benefits (UNIQUE TO KPU - HARDWARE FUSION)
    fusion_l2_traffic_reduction: float = 0.7  # 70% of intermediate data eliminated
    fusion_coordination_overhead: float = 5e-12  # ~5 pJ per fusion boundary

    def compute_gemm_energy(
        self,
        M: int,  # Output rows
        N: int,  # Output cols
        K: int,  # Inner dimension
        batch_size: int = 1,
        precision: str = "BF16",
        enable_fusion: bool = False,
        num_fused_ops: int = 1
    ) -> Dict[str, float]:
        """
        Compute energy for GEMM operation on KPU.

        This models the complete 8-component energy breakdown:
        1. 4-stage memory hierarchy
        2. 3 data movement engines
        3. Token signature matching
        4. SURE program loading
        5. Distributed L3 scratchpad
        6. Automatic operator fusion
        7. Token routing overhead
        8. Multi-engine coordination

        Args:
            M, N, K: GEMM dimensions (Y = X @ W, X=[M,K], W=[K,N], Y=[M,N])
            batch_size: Batch size (weight reuse)
            precision: Operation precision (INT8, BF16, FP32)
            enable_fusion: Enable hardware operator fusion
            num_fused_ops: Number of operators fused together

        Returns:
            Dictionary with 8-component energy breakdown (all in Joules)
        """

        # ============================================================
        # Component 1: 4-Stage Memory Hierarchy
        # ============================================================

        bytes_per_element = self._get_bytes_per_element(precision)

        # Input data (activations): [batch, M, K]
        input_bytes = batch_size * M * K * bytes_per_element

        # Weights: [K, N] (loaded once, reused across batch)
        weight_bytes = K * N * bytes_per_element

        # Output: [batch, M, N]
        output_bytes = batch_size * M * N * bytes_per_element

        # DRAM → L3 (via DMA engines)
        dram_l3_bytes = weight_bytes / batch_size + input_bytes  # Weights amortized
        dram_read_energy = dram_l3_bytes * self.dram_read_energy_per_byte
        dram_write_energy = output_bytes * self.dram_write_energy_per_byte

        # L3 → L2 (via BlockMover engines, distributed routing)
        # L3 is distributed, so routing distance matters
        average_l3_hops = self._estimate_l3_routing_distance()
        l3_noc_energy = dram_l3_bytes * average_l3_hops * self.l3_noc_energy_per_hop
        l3_read_energy = dram_l3_bytes * self.l3_read_energy_per_byte + l3_noc_energy
        l3_write_energy = output_bytes * self.l3_write_energy_per_byte

        # L2 → L1 (via Streamer engines, tile-local)
        # Fusion reduces intermediate L2 traffic
        l2_traffic_factor = (1.0 - self.fusion_l2_traffic_reduction) if enable_fusion else 1.0
        l2_l1_bytes = (input_bytes + weight_bytes) * l2_traffic_factor
        l2_read_energy = l2_l1_bytes * self.l2_read_energy_per_byte
        l2_write_energy = output_bytes * self.l2_write_energy_per_byte * l2_traffic_factor

        # L1 PE-local access
        l1_read_energy = l2_l1_bytes * self.l1_read_energy_per_byte
        l1_write_energy = output_bytes * self.l1_write_energy_per_byte

        total_memory_hierarchy_energy = (
            dram_read_energy + dram_write_energy +
            l3_read_energy + l3_write_energy +
            l2_read_energy + l2_write_energy +
            l1_read_energy + l1_write_energy
        )

        # ============================================================
        # Component 2: 3 Data Movement Engines
        # ============================================================

        # DMA: DRAM ↔ L3
        dma_energy = dram_l3_bytes * self.dma_transfer_energy_per_byte

        # BlockMover: L3 ↔ L2 (inter-tile)
        blockmover_energy = dram_l3_bytes * self.blockmover_energy_per_byte

        # Streamer: L2 ↔ L1 (intra-tile)
        streamer_energy = l2_l1_bytes * self.streamer_energy_per_byte

        total_dme_energy = dma_energy + blockmover_energy + streamer_energy

        # ============================================================
        # Component 3: Token Signature Matching (UNIQUE TO KPU!)
        # ============================================================

        # Each data transfer is a token with (payload, signature)
        # Signature matching happens at EVERY handshake point
        num_tokens = (input_bytes + weight_bytes + output_bytes) // self.token_payload_bytes

        # Distributed CAM-like matching at each routing point
        # GEMM typically routes through 2-4 matching points
        average_matching_points = 3
        signature_matching_energy = (
            num_tokens * average_matching_points * self.token_signature_matching_energy
        )

        # Instruction token dispatch (firing ready tokens in dataflow execution)
        dispatch_energy = num_tokens * self.token_dispatch_energy

        total_token_matching_energy = signature_matching_energy + dispatch_energy

        # ============================================================
        # Component 4: SURE Program Loading (UNIQUE TO KPU!)
        # ============================================================

        # GEMM requires SURE program broadcast to all participating PEs
        # Check if program is cached
        cache_hit_rate = 0.8 if num_fused_ops == 1 else 0.5  # Fusion reduces cache hits

        num_program_loads = 1  # GEMM is one operator
        if enable_fusion:
            num_program_loads = num_fused_ops  # Each fused op needs program

        program_load_energy = (
            num_program_loads * (1 - cache_hit_rate) * self.sure_program_load_energy +
            num_program_loads * cache_hit_rate * self.sure_program_cache_hit_energy
        )

        # ============================================================
        # Component 5: Distributed L3 Scratchpad Routing
        # ============================================================

        # L3 is distributed across tiles, routing distance varies
        # Already captured in l3_noc_energy above, but track separately for reporting
        distributed_l3_routing_energy = l3_noc_energy

        # ============================================================
        # Component 6: Automatic Operator Fusion
        # ============================================================

        fusion_energy = 0.0
        fusion_savings = 0.0

        if enable_fusion and num_fused_ops > 1:
            # Fusion coordination overhead (synchronization between fused ops)
            fusion_energy = (num_fused_ops - 1) * self.fusion_coordination_overhead

            # Fusion savings (reduced L2 traffic)
            # Intermediate results stay in L1/L2, not written back to L3
            intermediate_bytes = output_bytes * (num_fused_ops - 1)
            fusion_savings = (
                intermediate_bytes * self.fusion_l2_traffic_reduction *
                (self.l2_write_energy_per_byte + self.l2_read_energy_per_byte)
            )

        total_fusion_net_energy = fusion_energy - fusion_savings

        # ============================================================
        # Component 7: Token Routing Overhead
        # ============================================================

        # Tokens route through 2D mesh, distance depends on tile allocation
        average_routing_distance = self._estimate_token_routing_distance()

        token_routing_energy = (
            num_tokens * average_routing_distance * self.token_routing_per_hop
        )

        # ============================================================
        # Component 8: PE-to-PE Streaming Forwarding (MISSING!)
        # ============================================================

        # KPU streams weights and inputs through the PE array (16×8 PEs per tile)
        # Each PE forwards data to the next PE with token control overhead
        # Token control adds ~1.5× energy vs simple register write (0.3 pJ → 0.45 pJ)
        #
        # KPU streams 2 matrices (weights + inputs), outputs stay local
        # Number of PE hops ≈ tile dimension (16 PEs wide)

        pe_forwarding_energy_per_byte = 0.45e-12  # 0.45 pJ (0.3 pJ + 50% token overhead)
        tile_dimension = 16  # 16×8 PEs per tile

        # Weight + Input bytes need to be streamed through PEs
        weight_input_bytes = weight_bytes + input_bytes
        pe_streaming_energy = weight_input_bytes * tile_dimension * pe_forwarding_energy_per_byte

        # ============================================================
        # Component 9: Computation (PE BLAS Operators)
        # ============================================================

        total_ops = 2 * batch_size * M * N * K  # 2× for MAC (multiply + add)
        mac_energy = self._get_mac_energy(precision)
        compute_energy = total_ops * mac_energy

        # ============================================================
        # Total Energy Breakdown
        # ============================================================

        total_energy = (
            total_memory_hierarchy_energy +
            total_dme_energy +
            total_token_matching_energy +
            program_load_energy +
            distributed_l3_routing_energy +  # Already in memory hierarchy
            total_fusion_net_energy +
            token_routing_energy +
            pe_streaming_energy +  # PE-to-PE forwarding through tile
            compute_energy
        )

        # Adjust for double-counting L3 routing (included in both hierarchy and component 5)
        total_energy -= distributed_l3_routing_energy

        # Calculate metrics
        total_bytes_transferred = input_bytes + weight_bytes + output_bytes
        arithmetic_intensity = total_ops / total_bytes_transferred if total_bytes_transferred > 0 else 0
        energy_per_mac = total_energy / (total_ops / 2) if total_ops > 0 else 0

        # Compute vs Memory bound analysis
        compute_percentage = (compute_energy / total_energy * 100) if total_energy > 0 else 0

        # Calculate memory access counts (different cache line sizes per level)
        l1_line_size = 32    # bytes (PE-local, smaller lines)
        l2_line_size = 64    # bytes (tile-local)
        l3_line_size = 128   # bytes (distributed scratchpad, larger lines)
        dram_line_size = 256 # bytes (burst transfers for efficiency)

        # Calculate accesses based on bytes transferred at each level
        dram_accesses = int((dram_l3_bytes + dram_line_size - 1) // dram_line_size) if dram_l3_bytes > 0 else 0
        l3_accesses = int((dram_l3_bytes + l3_line_size - 1) // l3_line_size) if dram_l3_bytes > 0 else 0
        l2_accesses = int((l2_l1_bytes + l2_line_size - 1) // l2_line_size) if l2_l1_bytes > 0 else 0
        l1_accesses = int((l2_l1_bytes + l1_line_size - 1) // l1_line_size) if l2_l1_bytes > 0 else 0

        return {
            # Component 1: 4-Stage Memory Hierarchy
            'dram_read_energy_j': dram_read_energy,
            'dram_write_energy_j': dram_write_energy,
            'l3_read_energy_j': l3_read_energy,
            'l3_write_energy_j': l3_write_energy,
            'l2_read_energy_j': l2_read_energy,
            'l2_write_energy_j': l2_write_energy,
            'l1_read_energy_j': l1_read_energy,
            'l1_write_energy_j': l1_write_energy,
            'total_memory_hierarchy_energy_j': total_memory_hierarchy_energy,

            # Component 2: Data Movement Engines
            'dma_energy_j': dma_energy,
            'blockmover_energy_j': blockmover_energy,
            'streamer_energy_j': streamer_energy,
            'total_dme_energy_j': total_dme_energy,
            'dma_bytes': dram_l3_bytes,
            'blockmover_bytes': dram_l3_bytes,
            'streamer_bytes': l2_l1_bytes,

            # Component 3: Token Signature Matching (UNIQUE!)
            'signature_matching_energy_j': signature_matching_energy,
            'dispatch_energy_j': dispatch_energy,
            'total_token_matching_energy_j': total_token_matching_energy,
            'num_signature_matches': num_tokens * average_matching_points,
            'num_tokens': num_tokens,

            # Component 4: SURE Program Loading (UNIQUE!)
            'program_load_energy_j': program_load_energy,
            'cache_hit_rate': cache_hit_rate,

            # Component 5: Distributed L3 Scratchpad
            'distributed_l3_routing_energy_j': distributed_l3_routing_energy,
            'average_l3_hops': average_l3_hops,
            'l3_routing_accesses': l3_accesses,  # Same as L3 cache accesses

            # Component 6: Operator Fusion (UNIQUE!)
            'fusion_overhead_energy_j': fusion_energy,
            'fusion_savings_energy_j': fusion_savings,
            'fusion_net_energy_j': total_fusion_net_energy,

            # Component 7: Token Routing
            'token_routing_energy_j': token_routing_energy,
            'average_routing_distance': average_routing_distance,
            'num_tokens': num_tokens,

            # Component 8: PE-to-PE Streaming Forwarding
            'pe_streaming_energy_j': pe_streaming_energy,
            'tile_dimension': tile_dimension,

            # Component 9: Computation
            'compute_energy_j': compute_energy,

            # Total
            'total_energy_j': total_energy,

            # Metrics
            'total_ops': total_ops,
            'total_bytes': total_bytes_transferred,
            'arithmetic_intensity': arithmetic_intensity,
            'energy_per_mac_j': energy_per_mac,
            'energy_per_mac_pj': energy_per_mac * 1e12,
            'compute_percentage': compute_percentage,
            'batch_size': batch_size,
            'fusion_enabled': enable_fusion,
            'num_fused_ops': num_fused_ops,

            # Memory access counts and bytes
            'dram_bytes': dram_l3_bytes,
            'l3_bytes': dram_l3_bytes,
            'l2_bytes': l2_l1_bytes,
            'l1_bytes': l2_l1_bytes,
            'dram_accesses': dram_accesses,
            'l3_accesses': l3_accesses,
            'l2_accesses': l2_accesses,
            'l1_accesses': l1_accesses,

            # Hardware configuration (for TOPS/W calculation)
            'clock_frequency_hz': self.clock_frequency_hz,
            'num_tiles': self.num_tiles,
            'ops_per_cycle_per_tile': self._get_ops_per_cycle(precision),
        }

    def _get_bytes_per_element(self, precision: str) -> int:
        """Get bytes per element for precision"""
        return {
            'FP32': 4,
            'BF16': 2,
            'FP16': 2,
            'INT8': 1,
            'FP8': 1,
            'FP8_E4M3': 1,
            'FP8_E5M2': 1,
        }.get(precision.upper(), 4)

    def _get_mac_energy(self, precision: str) -> float:
        """Get MAC energy for precision"""
        return {
            'INT8': self.mac_energy_int8,
            'FP8': self.mac_energy_int8,
            'FP8_E4M3': self.mac_energy_int8,
            'FP8_E5M2': self.mac_energy_int8,
            'BF16': self.mac_energy_bf16,
            'FP16': self.mac_energy_bf16,
            'FP32': self.mac_energy_fp32,
        }.get(precision.upper(), self.mac_energy_bf16)

    def _get_ops_per_cycle(self, precision: str) -> float:
        """
        Get ops per cycle per tile for a given precision.

        KPU tiles have an array of 16x16 PEs, each can do 1 MAC per cycle.
        INT8 can be packed 2x (dual MAC units).
        FP32 runs at half rate.

        Returns:
            Ops per cycle per tile
        """
        # Base: pes_per_tile MACs per cycle (16x16 = 256 typical)
        base_ops = self.pes_per_tile * 2  # 2 ops per MAC (multiply + accumulate)

        return {
            'INT8': base_ops * 2,    # 2× throughput (dual INT8 units)
            'FP8': base_ops * 2,
            'BF16': base_ops,        # 1× throughput (base)
            'FP16': base_ops,
            'FP32': base_ops * 0.5,  # 0.5× throughput (half rate)
        }.get(precision.upper(), base_ops)

    def _estimate_l3_routing_distance(self) -> float:
        """
        Estimate average L3 routing distance (mesh hops).

        L3 is distributed across tiles, so data must route through NoC.
        Average distance depends on mesh dimensions.

        For 2D mesh: average distance ≈ (width + height) / 3
        """
        width, height = self.tile_mesh_dimensions
        return (width + height) / 3.0 * self.l3_routing_distance_factor

    def _estimate_token_routing_distance(self) -> float:
        """
        Estimate average token routing distance (PE-to-PE hops).

        Token routing implements SURE recurrence dependencies between PEs.
        For GEMM (matmul), SURE dataflow has nearest-neighbor dependencies:
        - Each PE receives data from adjacent PE (1 hop)
        - Each PE forwards results to adjacent PE (1 hop)
        - Systolic-like wave propagation through PE array

        For matmul, all dependencies are nearest-neighbor (1 hop).
        For complex multi-op graphs with non-local dependencies,
        routing distance would be higher.
        """
        # For matmul: nearest-neighbor SURE dependencies (1 hop between PEs)
        return 1.0


# ============================================================================
# KPU Tile Energy Model Adapter
# ============================================================================


class KPUTileEnergyAdapter(ArchitecturalEnergyModel):
    """
    Adapter to use KPUTileEnergyModel within the ArchitecturalEnergyModel interface.

    This wraps the detailed KPUTileEnergyModel and exposes it via the standard
    compute_architectural_energy() interface expected by the comparison tool.
    """

    def __init__(self, tile_model: KPUTileEnergyModel):
        """
        Initialize adapter with a configured KPUTileEnergyModel instance.

        Args:
            tile_model: Configured KPUTileEnergyModel with product-specific parameters
        """
        self.tile_model = tile_model

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        data_movement_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:
        """
        Compute architectural energy using the detailed KPU tile model.

        This adapter extracts GEMM dimensions from execution_context and delegates
        to KPUTileEnergyModel.compute_gemm_energy(), then maps the 8-component
        breakdown to the ArchEnergyBreakdown format.
        """
        if execution_context is None:
            execution_context = {}

        # Extract GEMM dimensions from context (comparison tool should provide these)
        # For MLP: M=batch_size, N=out_features, K=in_features
        batch_size = execution_context.get('batch_size', 1)
        mlp_input_dim = execution_context.get('mlp_input_dim', 256)
        mlp_output_dim = execution_context.get('mlp_output_dim', 256)

        # Call the detailed tile model
        breakdown = self.tile_model.compute_gemm_energy(
            M=batch_size,
            N=mlp_output_dim,
            K=mlp_input_dim,
            batch_size=1,  # Already included in M
            precision='FP32',
            enable_fusion=False,
            num_fused_ops=1
        )

        # For micro-benchmarks, disable program load overhead to get pure compute
        disable_overhead = execution_context.get('disable_launch_overhead', False)
        if disable_overhead:
            breakdown['program_load_energy_j'] = 0.0

        # Map 8-component breakdown to ArchEnergyBreakdown
        # Component 1: 4-Stage Memory Hierarchy
        dram_energy = breakdown['dram_read_energy_j'] + breakdown['dram_write_energy_j']
        l3_energy = breakdown['l3_read_energy_j'] + breakdown['l3_write_energy_j']
        l2_energy = breakdown['l2_read_energy_j'] + breakdown['l2_write_energy_j']
        l1_energy = breakdown['l1_read_energy_j'] + breakdown['l1_write_energy_j']

        # Component 2: Data Movement Engines
        dma_energy = breakdown['dma_energy_j']
        blockmover_energy = breakdown['blockmover_energy_j']
        streamer_energy = breakdown['streamer_energy_j']

        # Component 3: Token Signature Matching
        token_matching_energy = breakdown['total_token_matching_energy_j']

        # Component 4: SURE Program Loading
        program_load_energy = breakdown['program_load_energy_j']

        # Component 5: Distributed L3 Scratchpad
        l3_routing_energy = breakdown['distributed_l3_routing_energy_j']

        # Component 6: Operator Fusion
        fusion_energy = breakdown['fusion_net_energy_j']

        # Component 7: Token Routing
        token_routing_energy = breakdown['token_routing_energy_j']

        # Component 8: PE-to-PE Streaming Forwarding
        pe_streaming_energy = breakdown['pe_streaming_energy_j']

        # Component 9: Computation
        compute_energy = breakdown['compute_energy_j']

        # Calculate total overhead (everything except base compute)
        total_overhead = (
            dram_energy + l3_energy + l2_energy + l1_energy +
            dma_energy + blockmover_energy + streamer_energy +
            token_matching_energy + program_load_energy +
            l3_routing_energy + fusion_energy + token_routing_energy +
            pe_streaming_energy
        )

        # Categorize into compute/data_movement/control overhead
        # Data movement overhead: memory hierarchy + data movement engines + L3 NoC routing
        # L3 routing is energy to route data through distributed L3 scratchpad (NoC hops)
        data_movement_overhead = (
            dram_energy + l3_energy + l2_energy + l1_energy +
            dma_energy + blockmover_energy + streamer_energy +
            l3_routing_energy  # Moved from compute_overhead (data movement, not computation)
        )

        # Control overhead: token matching + program loading + token routing + PE streaming
        # PE streaming is the energy to forward data between PEs within a tile
        control_overhead = (
            token_matching_energy + program_load_energy + token_routing_energy +
            pe_streaming_energy  # PE-to-PE forwarding through array
        )

        # Compute overhead: fusion coordination only
        # (L3 routing moved to data_movement_overhead since it's data movement)
        compute_overhead = fusion_energy

        # Create extra_details dictionary with all 8 components
        extra_details = {
            # Component 1: 4-Stage Memory Hierarchy
            'dram_energy': dram_energy,
            'l3_energy': l3_energy,
            'l2_energy': l2_energy,
            'l1_energy': l1_energy,
            'dram_bytes': breakdown['dram_bytes'],
            'l3_bytes': breakdown['l3_bytes'],
            'l2_bytes': breakdown['l2_bytes'],
            'l1_bytes': breakdown['l1_bytes'],
            'dram_accesses': breakdown['dram_accesses'],
            'l3_accesses': breakdown['l3_accesses'],
            'l2_accesses': breakdown['l2_accesses'],
            'l1_accesses': breakdown['l1_accesses'],

            # Component 2: Data Movement Engines
            'dma_energy': dma_energy,
            'blockmover_energy': blockmover_energy,
            'streamer_energy': streamer_energy,
            'dma_bytes': breakdown['dma_bytes'],
            'blockmover_bytes': breakdown['blockmover_bytes'],
            'streamer_bytes': breakdown['streamer_bytes'],

            # Component 3: Token Signature Matching
            'token_matching_energy': token_matching_energy,
            'signature_matching_energy': breakdown['signature_matching_energy_j'],
            'dispatch_energy': breakdown['dispatch_energy_j'],
            'num_signature_matches': breakdown['num_signature_matches'],
            'num_tokens': breakdown['num_tokens'],

            # Component 4: SURE Program Loading
            'program_load_energy': program_load_energy,
            'cache_hit_rate': breakdown['cache_hit_rate'],

            # Component 5: Distributed L3 Scratchpad
            'l3_routing_energy': l3_routing_energy,
            'average_l3_hops': breakdown['average_l3_hops'],
            'l3_routing_accesses': breakdown['l3_routing_accesses'],

            # Component 6: Operator Fusion
            'fusion_overhead_energy': breakdown['fusion_overhead_energy_j'],
            'fusion_savings_energy': breakdown['fusion_savings_energy_j'],
            'fusion_net_energy': fusion_energy,

            # Component 7: Token Routing
            'token_routing_energy': token_routing_energy,
            'average_routing_distance': breakdown['average_routing_distance'],
            'num_tokens': breakdown['num_tokens'],

            # Component 8: Computation
            'compute_energy': compute_energy,

            # Metrics
            'total_ops': breakdown['total_ops'],
            'total_bytes': breakdown['total_bytes'],
            'arithmetic_intensity': breakdown['arithmetic_intensity'],
            'energy_per_mac_pj': breakdown['energy_per_mac_pj'],
            'compute_percentage': breakdown['compute_percentage'],
        }

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead,
            data_movement_overhead=data_movement_overhead,
            control_overhead=control_overhead,
            extra_details=extra_details,
            explanation=(
                f"KPU Domain-Flow 8-component breakdown:\n"
                f"  1. Memory Hierarchy: DRAM={dram_energy*1e6:.3f} μJ, L3={l3_energy*1e6:.3f} μJ, "
                f"L2={l2_energy*1e6:.3f} μJ, L1={l1_energy*1e6:.3f} μJ\n"
                f"  2. Data Movement: DMA={dma_energy*1e6:.3f} μJ, BlockMover={blockmover_energy*1e6:.3f} μJ, "
                f"Streamer={streamer_energy*1e6:.3f} μJ\n"
                f"  3. Token Matching: {token_matching_energy*1e6:.3f} μJ\n"
                f"  4. Program Loading: {program_load_energy*1e6:.3f} μJ\n"
                f"  5. L3 Routing: {l3_routing_energy*1e6:.3f} μJ\n"
                f"  6. Fusion: {fusion_energy*1e6:.3f} μJ\n"
                f"  7. Token Routing: {token_routing_energy*1e6:.3f} μJ\n"
                f"  8. Compute: {compute_energy*1e6:.3f} μJ"
            )
        )
