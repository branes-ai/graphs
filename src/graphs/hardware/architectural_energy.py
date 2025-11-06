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
    Result of architectural energy calculation.

    Contains energy overheads (positive = cost, negative = savings)
    and human-readable explanation.
    """
    compute_overhead: float  # Additional compute energy (Joules)
    memory_overhead: float   # Additional memory energy (Joules)
    control_overhead: float  # Control/coordination energy (Joules)

    # Additional details for specific architectures
    extra_details: Dict[str, float] = field(default_factory=dict)

    # Human-readable explanation
    explanation: str = ""

    @property
    def total_overhead(self) -> float:
        """Total architectural overhead"""
        return self.compute_overhead + self.memory_overhead + self.control_overhead


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
        memory_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:
        """
        Compute architectural energy overhead.

        Args:
            ops: Number of operations (FLOPs, MACs, etc.)
            bytes_transferred: Total bytes read/written
            compute_energy_baseline: Baseline compute energy (ops x energy_per_op)
            memory_energy_baseline: Baseline memory energy (bytes x energy_per_byte)
            execution_context: Additional context (threads, batch size, etc.)

        Returns:
            ArchitecturalEnergyBreakdown with overheads and explanation
        """
        pass


@dataclass
class StoredProgramEnergyModel(ArchitecturalEnergyModel):
    """
    Energy model for stored program architectures (CPU, DSP) - Sequential/Modest Parallelism.

    Key energy events:
    - Instruction fetch: Energy to request/receive each instruction from memory
    - Operand fetch: Explicit load/store overhead (request/reply cycles)
    - Pipeline control: Decode, dispatch, ordering, writeback machinery
    - Out-of-order execution overhead
    - Branch prediction

    The memory wall: Request/reply latency is significant AND energy intensive.
    """

    # Energy coefficients (Joules per event)
    instruction_fetch_energy: float = 2.0e-12      # ~2 pJ per instruction
    operand_fetch_overhead: float = 10.0e-12       # ~10 pJ per memory operation
    pipeline_control_per_cycle: float = 0.5e-12    # ~0.5 pJ per cycle
    branch_prediction_overhead: float = 0.3e-12    # ~0.3 pJ per branch

    # Instructions per operation (typical for AI workloads)
    instructions_per_op: float = 0.1               # ~1 instruction per 10 ops
    branches_per_1000_ops: int = 50                # ~50 branches per 1000 ops

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        memory_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # Estimate instruction count
        num_instructions = int(ops * self.instructions_per_op)

        # Estimate memory operations (cache line granularity)
        cache_line_size = execution_context.get('cache_line_size', 64)
        num_memory_ops = max(1, int(bytes_transferred / cache_line_size))

        # Instruction fetch overhead
        instruction_energy = num_instructions * self.instruction_fetch_energy

        # Memory request overhead (beyond data transfer)
        memory_overhead = num_memory_ops * self.operand_fetch_overhead

        # Branch prediction
        num_branches = (ops // 1000) * self.branches_per_1000_ops
        branch_energy = num_branches * self.branch_prediction_overhead

        # Total control overhead
        control_overhead = instruction_energy + branch_energy

        explanation = (
            f"Stored Program Architecture (CPU/DSP) Energy Events:\n"
            f"  Instruction Fetch:\n"
            f"    - Instructions: {num_instructions:,} ({self.instructions_per_op} per op)\n"
            f"    - Energy: {instruction_energy*1e12:.2f} pJ "
            f"({num_instructions:,} × {self.instruction_fetch_energy*1e12:.2f} pJ)\n"
            f"  Memory Request Overhead:\n"
            f"    - Memory ops: {num_memory_ops:,} ({cache_line_size}-byte cache lines)\n"
            f"    - Energy: {memory_overhead*1e12:.2f} pJ "
            f"({num_memory_ops:,} × {self.operand_fetch_overhead*1e12:.2f} pJ)\n"
            f"  Branch Prediction:\n"
            f"    - Branches: {num_branches:,}\n"
            f"    - Energy: {branch_energy*1e12:.2f} pJ\n"
            f"  Total Architectural Overhead: {(control_overhead + memory_overhead)*1e12:.2f} pJ\n"
            f"\n"
            f"KEY: Sequential execution with modest parallelism (8-16 cores).\n"
            f"     Memory wall creates request/reply energy overhead."
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=0.0,
            memory_overhead=memory_overhead,
            control_overhead=control_overhead,
            extra_details={
                'instruction_energy': instruction_energy,
                'branch_energy': branch_energy,
            },
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

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        memory_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # CPU-like overheads (instruction fetch, operand fetch)
        num_instructions = int(ops * self.instructions_per_op)
        instruction_energy = num_instructions * self.instruction_fetch_energy

        cache_line_size = execution_context.get('cache_line_size', 128)  # H100 uses 128B
        num_memory_ops = max(1, int(bytes_transferred / cache_line_size))
        memory_overhead = num_memory_ops * self.operand_fetch_overhead

        # GPU-specific: Coherence machinery (THE CRITICAL ENERGY COMPONENT)
        concurrent_threads = execution_context.get('concurrent_threads', 200_000)
        warp_size = execution_context.get('warp_size', 32)
        num_concurrent_warps = max(1, concurrent_threads // warp_size)

        # Coherence energy scales with concurrent warps × memory operations
        coherence_energy = num_concurrent_warps * self.coherence_energy_per_request * num_memory_ops

        # Thread scheduling overhead
        scheduling_energy = concurrent_threads * self.thread_scheduling_overhead

        # Warp divergence penalties (control flow causes some warps to diverge)
        num_divergent_ops = int(ops * self.warp_divergence_rate)
        divergence_energy = num_divergent_ops * self.warp_divergence_penalty

        # Memory coalescing overhead (uncoalesced accesses are expensive)
        num_uncoalesced = int(num_memory_ops * self.uncoalesced_access_rate)
        coalescing_energy = num_uncoalesced * self.memory_coalescing_overhead

        # Synchronization barriers
        num_barriers = (ops // 1000) * self.barriers_per_1000_ops
        barrier_energy = num_barriers * self.barrier_sync_energy

        # Total control overhead
        control_overhead = (instruction_energy + coherence_energy + scheduling_energy +
                           divergence_energy + barrier_energy)

        # Memory overhead includes coalescing
        total_memory_overhead = memory_overhead + coalescing_energy

        explanation = (
            f"Data Parallel (GPU SIMT) Architecture Energy Events:\n"
            f"  CPU-Like Overheads:\n"
            f"    - Instruction fetch: {instruction_energy*1e12:.2f} pJ "
            f"({num_instructions:,} instructions)\n"
            f"    - Memory requests: {memory_overhead*1e12:.2f} pJ "
            f"({num_memory_ops:,} ops)\n"
            f"  GPU-Specific SIMT Overheads:\n"
            f"    - Coherence machinery: {coherence_energy*1e12:.2f} pJ "
            f"({num_concurrent_warps:,} warps × {num_memory_ops:,} mem ops)\n"
            f"      THIS IS THE DOMINANT ENERGY COMPONENT!\n"
            f"    - Thread scheduling: {scheduling_energy*1e12:.2f} pJ "
            f"({concurrent_threads:,} threads)\n"
            f"    - Warp divergence: {divergence_energy*1e12:.2f} pJ "
            f"({num_divergent_ops:,} divergent ops)\n"
            f"    - Memory coalescing: {coalescing_energy*1e12:.2f} pJ "
            f"({num_uncoalesced:,} uncoalesced)\n"
            f"    - Synchronization barriers: {barrier_energy*1e12:.2f} pJ "
            f"({num_barriers:,} barriers)\n"
            f"  Total Architectural Overhead: {(control_overhead + total_memory_overhead)*1e12:.2f} pJ\n"
            f"\n"
            f"KEY INSIGHT: Coherence machinery dominates! GPU burns energy to manage\n"
            f"             thousands of concurrent memory requests. This is only worth\n"
            f"             it at large batch sizes where latency hiding pays off.\n"
            f"             At small batches, coherence overhead >> computation benefit."
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=0.0,
            memory_overhead=total_memory_overhead,
            control_overhead=control_overhead,
            extra_details={
                'instruction_energy': instruction_energy,
                'coherence_energy': coherence_energy,
                'scheduling_energy': scheduling_energy,
                'divergence_energy': divergence_energy,
                'coalescing_energy': coalescing_energy,
                'barrier_energy': barrier_energy,
            },
            explanation=explanation
        )


@dataclass
class SystolicArrayEnergyModel(ArchitecturalEnergyModel):
    """
    Energy model for systolic array architectures (Google TPU).

    Key advantages over stored program:
    - No instruction fetch during execution (schedule is predetermined)
    - No contention (spatial separation of data flows)
    - No ordering machinery (schedule known a priori)
    - Direct data injection into 2D array

    Result: 5-10x lower energy per operation (Google TCO data validates this)
    """

    # Energy coefficients (much lower than stored program)
    schedule_setup_energy: float = 100.0e-12       # ~100 pJ per kernel (one-time)
    data_injection_per_element: float = 0.5e-12    # ~0.5 pJ per element
    data_extraction_per_element: float = 0.5e-12   # ~0.5 pJ per element

    # Architectural efficiency multiplier (vs stored program baseline)
    # These represent the SAVINGS from eliminating instruction fetch, contention, etc.
    compute_efficiency: float = 0.15               # 15% overhead (85% reduction!)
    memory_efficiency: float = 0.20                # 20% overhead (80% reduction!)

    def compute_architectural_energy(
        self,
        ops: int,
        bytes_transferred: int,
        compute_energy_baseline: float,
        memory_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # One-time schedule setup (amortized over all operations)
        schedule_energy = self.schedule_setup_energy

        # Data injection/extraction (much more efficient than load/store)
        num_elements = max(1, bytes_transferred // 4)  # Assume 4-byte elements
        injection_energy = num_elements * self.data_injection_per_element
        extraction_energy = num_elements * self.data_extraction_per_element

        # Architectural benefit: Reduce compute overhead dramatically
        # Because no instruction fetch, decode, dispatch per operation
        compute_overhead_reduction = -compute_energy_baseline * (1.0 - self.compute_efficiency)

        # Memory benefit: Spatial data flows eliminate contention overhead
        memory_overhead_reduction = -memory_energy_baseline * (1.0 - self.memory_efficiency)

        total_injection = injection_energy + extraction_energy + schedule_energy

        explanation = (
            f"Systolic Array Architecture Energy Events:\n"
            f"  Schedule Setup (one-time, amortized):\n"
            f"    - Energy: {schedule_energy*1e12:.2f} pJ\n"
            f"  Data Injection/Extraction:\n"
            f"    - Elements: {num_elements:,}\n"
            f"    - Injection: {injection_energy*1e12:.2f} pJ "
            f"({num_elements:,} × {self.data_injection_per_element*1e12:.2f} pJ)\n"
            f"    - Extraction: {extraction_energy*1e12:.2f} pJ\n"
            f"  Architectural Efficiency (vs Stored Program):\n"
            f"    - Compute overhead eliminated: {-compute_overhead_reduction*1e12:.2f} pJ saved\n"
            f"      (No instruction fetch, decode, dispatch per op)\n"
            f"    - Memory contention eliminated: {-memory_overhead_reduction*1e12:.2f} pJ saved\n"
            f"      (Spatial data flows, no request/reply overhead)\n"
            f"  Efficiency vs Stored Program:\n"
            f"    - Compute: {self.compute_efficiency*100:.0f}% overhead "
            f"({(1-self.compute_efficiency)*100:.0f}% reduction)\n"
            f"    - Memory: {self.memory_efficiency*100:.0f}% overhead "
            f"({(1-self.memory_efficiency)*100:.0f}% reduction)\n"
            f"\n"
            f"KEY: Pre-designed spatial schedule eliminates instruction fetch\n"
            f"     and contention overhead. 5-10x more energy efficient!"
        )

        return ArchitecturalEnergyBreakdown(
            compute_overhead=compute_overhead_reduction,
            memory_overhead=memory_overhead_reduction + total_injection,
            control_overhead=schedule_energy,
            extra_details={
                'injection_energy': injection_energy,
                'extraction_energy': extraction_energy,
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
        memory_energy_baseline: float,
        execution_context: Optional[Dict] = None
    ) -> ArchitecturalEnergyBreakdown:

        if execution_context is None:
            execution_context = {}

        # Domain tracking per operation
        domain_tracking_energy = ops * self.domain_tracking_per_op

        # Kernel changes (when computation pattern changes)
        num_kernel_changes = execution_context.get('kernel_changes', 1)
        kernel_load_energy = num_kernel_changes * self.schedule_adaptation_energy

        # Data movement through domains
        num_elements = max(1, bytes_transferred // 4)
        injection_energy = num_elements * self.domain_data_injection
        extraction_energy = num_elements * self.domain_data_extraction

        # Architectural benefit: Still eliminates instruction fetch
        compute_overhead_reduction = -compute_energy_baseline * (1.0 - self.compute_efficiency)
        memory_overhead_reduction = -memory_energy_baseline * (1.0 - self.memory_efficiency)

        total_domain_overhead = domain_tracking_energy + wavefront_energy + schedule_energy
        total_data_movement = injection_energy + extraction_energy

        explanation = (
            f"Domain Flow Architecture Energy Events:\n"
            f"  Domain Tracking:\n"
            f"    - Operations: {ops:,}\n"
            f"    - Energy: {domain_tracking_energy*1e12:.2f} pJ "
            f"({ops:,} x {self.domain_tracking_per_op*1e12:.2f} pJ)\n"
            f"  Wavefront Management:\n"
            f"    - Wavefronts: {num_wavefronts:,}\n"
            f"    - Energy: {wavefront_energy*1e12:.2f} pJ\n"
            f"  Schedule Adaptation:\n"
            f"    - Schedule changes: {num_schedule_changes}\n"
            f"    - Energy: {schedule_energy*1e12:.2f} pJ\n"
            f"  Domain Data Movement:\n"
            f"    - Elements: {num_elements:,}\n"
            f"    - Injection: {injection_energy*1e12:.2f} pJ\n"
            f"    - Extraction: {extraction_energy*1e12:.2f} pJ\n"
            f"  Architectural Efficiency (vs Stored Program):\n"
            f"    - Compute overhead eliminated: {-compute_overhead_reduction*1e12:.2f} pJ saved\n"
            f"    - Memory overhead eliminated: {-memory_overhead_reduction*1e12:.2f} pJ saved\n"
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
            memory_overhead=memory_overhead_reduction + total_data_movement,
            control_overhead=total_domain_overhead,
            extra_details={
                'domain_tracking_energy': domain_tracking_energy,
                'wavefront_energy': wavefront_energy,
                'schedule_energy': schedule_energy,
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
        memory_energy_baseline: float,
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
        memory_overhead_reduction = -memory_energy_baseline * (1.0 - self.memory_efficiency)

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
            f"    - Memory overhead eliminated: {-memory_overhead_reduction*1e12:.2f} pJ saved\n"
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
            memory_overhead=memory_overhead_reduction,
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
        memory_energy_baseline: float,
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
        memory_overhead_reduction = -memory_energy_baseline * (1.0 - self.memory_efficiency)

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
            f"    - Memory overhead eliminated: {-memory_overhead_reduction*1e12:.2f} pJ saved\n"
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
            memory_overhead=memory_overhead_reduction + total_memory,
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
        memory_energy_baseline: float,
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
        memory_overhead_reduction = -memory_energy_baseline * (1.0 - self.memory_efficiency)

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
            f"    - Memory overhead eliminated: {-memory_overhead_reduction*1e12:.2f} pJ saved\n"
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
            memory_overhead=memory_overhead_reduction + total_runtime_overhead,
            control_overhead=reconfig_energy,
            extra_details={
                'reconfiguration_energy': reconfig_energy,
                'routing_energy': routing_energy,
                'macro_energy': macro_energy,
            },
            explanation=explanation
        )
