"""
Operand Fetch Energy Models - Register-to-ALU Delivery Infrastructure

This module models the energy required to deliver operands from local storage
(register files, PE registers) to ALU inputs. This is the "last mile" of
operand delivery and is SEPARATE from memory hierarchy (load/store operations).

Key Insight:
The actual ALU energy (ADD/MUL/FMA circuit) is nearly identical across
architectures at the same process node (~0.7 pJ for FP32 FMA at 7nm).
What differs dramatically is the operand fetch infrastructure:

  CPU:  ~6 pJ per operand pair (register file with 2R+1W ports)
  GPU:  ~10 pJ per operand pair (operand collectors + crossbar + bank conflicts)
  TPU:  ~0.2 pJ per operand pair (PE-to-PE forwarding, spatial reuse)
  KPU:  ~0.3 pJ per operand pair (domain flow forwarding)

For stored-program architectures (CPU/GPU), operand fetch dominates (90%+ of
operation energy). For spatial architectures (TPU/KPU), the ALU dominates
because operands are already at the PE through spatial reuse.

This is NOT memory hierarchy:
- Memory hierarchy = LOAD/STORE instructions moving data between memory levels
- Operand fetch = delivering operands from registers to ALU every cycle
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graphs.hardware.technology_profile import TechnologyProfile


# =============================================================================
# Operand Fetch Breakdown
# =============================================================================

@dataclass
class OperandFetchBreakdown:
    """
    Detailed breakdown of operand fetch energy.

    Operand fetch = register-to-ALU delivery.
    Does NOT include memory hierarchy (load/store are separate instructions).

    Components:
    - Register file access (CPU/GPU): Reading source operands, writing results
    - Operand routing (GPU): Collectors, crossbars, bank conflict resolution
    - Spatial forwarding (TPU/KPU): PE-to-PE data movement within array
    """

    # =========================================================================
    # Register File Access (CPU/GPU style architectures)
    # =========================================================================
    register_read_energy: float = 0.0     # Energy to read source operands (J)
    register_write_energy: float = 0.0    # Energy to write result back (J)

    # =========================================================================
    # Operand Routing Infrastructure (GPU)
    # =========================================================================
    operand_collector_energy: float = 0.0  # GPU: gather operands for warp (J)
    crossbar_routing_energy: float = 0.0   # Energy to route operands to ALUs (J)
    bank_conflict_penalty: float = 0.0     # GPU: extra energy from bank conflicts (J)

    # =========================================================================
    # Spatial Array Forwarding (TPU/KPU style architectures)
    # =========================================================================
    pe_forwarding_energy: float = 0.0      # PE-to-PE register forwarding (J)
    array_injection_energy: float = 0.0    # Energy to inject at array boundary (J)
    array_extraction_energy: float = 0.0   # Energy to extract from array boundary (J)

    # =========================================================================
    # Domain Flow Control (KPU)
    # =========================================================================
    domain_tracking_energy: float = 0.0    # Domain flow routing control (J)

    # =========================================================================
    # Reuse Accounting
    # =========================================================================
    operands_from_registers: int = 0       # Operands read from register file
    operands_from_forwarding: int = 0      # Operands received via PE forwarding
    operand_reuse_factor: float = 1.0      # Average times each operand is reused

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def total_fetch_energy(self) -> float:
        """Total operand fetch energy in Joules (excludes memory hierarchy)."""
        return (self.register_read_energy +
                self.register_write_energy +
                self.operand_collector_energy +
                self.crossbar_routing_energy +
                self.bank_conflict_penalty +
                self.pe_forwarding_energy +
                self.array_injection_energy +
                self.array_extraction_energy +
                self.domain_tracking_energy)

    @property
    def total_operands_delivered(self) -> int:
        """Total operands delivered to ALUs."""
        return self.operands_from_registers + self.operands_from_forwarding

    @property
    def energy_per_operand(self) -> float:
        """Average energy per operand delivered (Joules)."""
        if self.total_operands_delivered > 0:
            return self.total_fetch_energy / self.total_operands_delivered
        return 0.0

    @property
    def energy_per_operation(self) -> float:
        """Average operand fetch energy per arithmetic operation (Joules).

        Each operation needs 2 input operands, so divide by 2.
        """
        if self.total_operands_delivered > 0:
            return self.total_fetch_energy / (self.total_operands_delivered / 2)
        return 0.0

    @property
    def forwarding_ratio(self) -> float:
        """Fraction of operands served via forwarding (0.0-1.0).

        Higher is better - means more spatial reuse.
        CPU/GPU: 0.0 (no spatial forwarding)
        TPU/KPU: 0.99+ (almost all operands forwarded)
        """
        total = self.total_operands_delivered
        if total > 0:
            return self.operands_from_forwarding / total
        return 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Operand Fetch Energy Breakdown:",
            f"  Total Fetch Energy: {self.total_fetch_energy * 1e6:.3f} uJ",
            f"  Energy per Operation: {self.energy_per_operation * 1e12:.2f} pJ",
            f"  Operand Reuse Factor: {self.operand_reuse_factor:.1f}x",
            f"  Forwarding Ratio: {self.forwarding_ratio:.1%}",
            "",
            "  Components:",
        ]

        if self.register_read_energy > 0:
            lines.append(f"    Register Reads:    {self.register_read_energy * 1e6:.3f} uJ")
        if self.register_write_energy > 0:
            lines.append(f"    Register Writes:   {self.register_write_energy * 1e6:.3f} uJ")
        if self.operand_collector_energy > 0:
            lines.append(f"    Operand Collector: {self.operand_collector_energy * 1e6:.3f} uJ")
        if self.crossbar_routing_energy > 0:
            lines.append(f"    Crossbar Routing:  {self.crossbar_routing_energy * 1e6:.3f} uJ")
        if self.bank_conflict_penalty > 0:
            lines.append(f"    Bank Conflicts:    {self.bank_conflict_penalty * 1e6:.3f} uJ")
        if self.pe_forwarding_energy > 0:
            lines.append(f"    PE Forwarding:     {self.pe_forwarding_energy * 1e6:.3f} uJ")
        if self.array_injection_energy > 0:
            lines.append(f"    Array Injection:   {self.array_injection_energy * 1e6:.3f} uJ")
        if self.array_extraction_energy > 0:
            lines.append(f"    Array Extraction:  {self.array_extraction_energy * 1e6:.3f} uJ")
        if self.domain_tracking_energy > 0:
            lines.append(f"    Domain Tracking:   {self.domain_tracking_energy * 1e6:.3f} uJ")

        return "\n".join(lines)


# =============================================================================
# Base Class for Operand Fetch Models
# =============================================================================

@dataclass
class OperandFetchEnergyModel(ABC):
    """
    Base class for architecture-specific operand fetch energy modeling.

    Models the "last mile" delivery of operands from local storage to ALU inputs.
    This is SEPARATE from memory hierarchy (load/store operations).

    The energy to perform an arithmetic operation has two components:
    1. ALU Energy: The circuit energy of the actual ADD/MUL/FMA (~0.7 pJ @ 7nm)
    2. Operand Fetch Energy: The energy to deliver operands to the ALU

    For stored-program architectures, operand fetch energy >> ALU energy!
    For spatial architectures (systolic, domain flow), operand fetch is minimized
    through spatial reuse.
    """
    tech_profile: 'TechnologyProfile'

    @abstractmethod
    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 1.0,
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """
        Compute the energy required to deliver operands for arithmetic operations.

        Args:
            num_operations: Total arithmetic operations (ADD, MUL, FMA, etc.)
            operand_width_bytes: Size of each operand (4 for FP32, 2 for FP16/BF16)
            spatial_reuse_factor: How many times each operand is reused spatially
                                  (1.0 for CPU/GPU, 128+ for systolic arrays)
            execution_context: Architecture-specific context (array dimensions, etc.)

        Returns:
            OperandFetchBreakdown with detailed energy components
        """
        pass

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        """Name of the architecture (for reporting)."""
        pass


# =============================================================================
# CPU Operand Fetch Model
# =============================================================================

@dataclass
class CPUOperandFetchModel(OperandFetchEnergyModel):
    """
    CPU operand fetch: Small register file with multi-port access.

    Architecture:
    - 16-32 architectural registers (x86-64: 16 GPRs, ARM64: 31)
    - 2 read ports + 1 write port per core (for 2-operand instructions)
    - Register renaming adds ~50-100 physical registers
    - Operands read during decode/issue stage
    - Results written during writeback stage

    Energy Profile:
    - Register read: ~3.0 pJ (multi-ported SRAM with CAM for renaming)
    - Register write: ~3.0 pJ
    - No spatial reuse - each instruction fetches operands independently

    Key Characteristic:
    - Every operation requires 2 register reads + 1 register write
    - No operand reuse between instructions (except through bypass/forwarding paths)
    - Register file energy dominates for small operations
    """

    # Register file characteristics (configurable)
    num_architectural_registers: int = 16    # x86-64 GPRs
    num_physical_registers: int = 64         # With renaming
    read_ports: int = 2                      # Simultaneous reads per cycle
    write_ports: int = 1                     # Simultaneous writes per cycle

    # Bypass/forwarding network (reduces some register writes)
    bypass_utilization: float = 0.20         # 20% of results bypass register file

    # Derived energy values (set in __post_init__)
    register_read_energy_pj: float = field(init=False)
    register_write_energy_pj: float = field(init=False)
    bypass_network_energy_pj: float = field(init=False)

    def __post_init__(self):
        tp = self.tech_profile
        # CPU registers are multi-ported SRAM with CAM for renaming
        self.register_read_energy_pj = tp.register_read_energy_pj
        self.register_write_energy_pj = tp.register_write_energy_pj
        # Bypass network is cheaper than full register write
        self.bypass_network_energy_pj = tp.register_read_energy_pj * 0.3

    @property
    def architecture_name(self) -> str:
        return "CPU (Stored Program)"

    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 1.0,  # Always 1.0 for CPU (no spatial reuse)
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """
        CPU operand fetch: 2 register reads + 1 register write per operation.

        CPU has no spatial reuse - every operation independently fetches its
        operands from the register file.
        """
        if execution_context is None:
            execution_context = {}

        # Override bypass utilization if provided
        bypass_util = execution_context.get('bypass_utilization', self.bypass_utilization)

        # Each operation needs 2 source operands read from register file
        num_register_reads = num_operations * 2

        # Results: some bypass register file (forwarding), rest are written
        num_register_writes = int(num_operations * (1.0 - bypass_util))
        num_bypass_forwards = int(num_operations * bypass_util)

        # Energy calculation (convert pJ to J)
        read_energy = num_register_reads * self.register_read_energy_pj * 1e-12
        write_energy = num_register_writes * self.register_write_energy_pj * 1e-12
        bypass_energy = num_bypass_forwards * self.bypass_network_energy_pj * 1e-12

        return OperandFetchBreakdown(
            register_read_energy=read_energy,
            register_write_energy=write_energy + bypass_energy,
            operands_from_registers=num_register_reads,
            operands_from_forwarding=0,  # No spatial forwarding in CPU
            operand_reuse_factor=1.0     # No spatial reuse
        )


# =============================================================================
# GPU Operand Fetch Model
# =============================================================================

@dataclass
class GPUOperandFetchModel(OperandFetchEnergyModel):
    """
    GPU operand fetch: Banked register files with operand collectors.

    Architecture:
    - 64K registers per SM (shared across 2048 threads max)
    - Each thread gets 32-255 registers (configurable at launch)
    - Register file is banked (32 banks for 32-thread warps)
    - Operand collectors gather operands for warp-wide execution
    - Must deliver 128 operands per cycle for CUDA cores (32 threads x 2 ops x 2 operands)

    Energy Profile:
    - Register access: ~0.75 pJ (simpler than CPU - no renaming, in-order per thread)
    - Operand collector: ~0.5 pJ per operand (arbitration, staging buffers)
    - Bank conflict penalty: +1.0 pJ when multiple threads hit same bank
    - Crossbar: ~0.3 pJ per operand (route from bank to collector to ALU)

    Key Characteristics:
    - Massive parallelism requires massive operand bandwidth
    - Operand collectors are energy-hungry (arbitrate 128+ operands/cycle)
    - Bank conflicts add significant energy overhead (~10% of accesses)
    - No spatial reuse - each thread fetches independently
    """

    # Warp and register file characteristics
    warp_size: int = 32
    registers_per_sm: int = 65536         # 64K registers per SM
    register_banks: int = 32              # One bank per warp lane

    # Bank conflict rate (depends on access pattern)
    bank_conflict_rate: float = 0.10      # 10% of accesses have conflicts

    # Derived energy values (set in __post_init__)
    register_access_energy_pj: float = field(init=False)
    operand_collector_energy_pj: float = field(init=False)
    crossbar_energy_pj: float = field(init=False)
    bank_conflict_penalty_pj: float = field(init=False)

    def __post_init__(self):
        tp = self.tech_profile
        # GPU registers are simpler than CPU (no renaming, in-order per thread)
        # but register file is much larger and banked
        self.register_access_energy_pj = tp.register_read_energy_pj * 0.25
        # Operand collector arbitration and staging
        self.operand_collector_energy_pj = 0.5
        # Crossbar routing from bank to collector to ALU
        self.crossbar_energy_pj = 0.3
        # Bank conflict penalty (extra cycles = extra energy)
        self.bank_conflict_penalty_pj = 1.0

    @property
    def architecture_name(self) -> str:
        return "GPU (SIMT Data Parallel)"

    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 1.0,  # Always 1.0 for GPU (no spatial reuse)
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """
        GPU operand fetch: Register access + operand collector + crossbar per operation.

        GPU has no spatial reuse - each thread independently fetches its operands
        through the operand collector infrastructure.
        """
        if execution_context is None:
            execution_context = {}

        # Override bank conflict rate if provided
        conflict_rate = execution_context.get('bank_conflict_rate', self.bank_conflict_rate)

        # Each operation needs 2 operands per thread
        num_operands = num_operations * 2

        # Register file access (convert pJ to J)
        register_energy = num_operands * self.register_access_energy_pj * 1e-12

        # Operand collector energy (gather operands for warp execution)
        collector_energy = num_operands * self.operand_collector_energy_pj * 1e-12

        # Crossbar routing
        crossbar_energy = num_operands * self.crossbar_energy_pj * 1e-12

        # Bank conflict penalty
        num_conflicts = int(num_operands * conflict_rate)
        conflict_energy = num_conflicts * self.bank_conflict_penalty_pj * 1e-12

        # Result writeback (register write)
        write_energy = num_operations * self.register_access_energy_pj * 1e-12

        return OperandFetchBreakdown(
            register_read_energy=register_energy,
            register_write_energy=write_energy,
            operand_collector_energy=collector_energy,
            crossbar_routing_energy=crossbar_energy,
            bank_conflict_penalty=conflict_energy,
            operands_from_registers=num_operands,
            operands_from_forwarding=0,  # No spatial forwarding in GPU
            operand_reuse_factor=1.0     # No spatial reuse
        )


# =============================================================================
# TPU Operand Fetch Model (Systolic Array)
# =============================================================================

@dataclass
class TPUOperandFetchModel(OperandFetchEnergyModel):
    """
    TPU operand fetch: Systolic array with PE-to-PE forwarding.

    Architecture:
    - 128x128 or 256x256 systolic array (MXU - Matrix Multiply Unit)
    - Weight-stationary dataflow: weights loaded into PE registers, stay there
    - Inputs stream horizontally through array
    - Partial sums accumulate vertically
    - Each PE has: weight register, input latch, accumulator

    Energy Profile:
    - Weight load (once per tile): ~0.3 pJ per element
    - PE-to-PE forwarding: ~0.1 pJ (very short wire to neighbor)
    - Array boundary injection: ~0.35 pJ per element
    - Array boundary extraction: ~0.35 pJ per element

    Key Efficiency:
    - Weights loaded ONCE, reused for entire input batch (128+ reuses)
    - Inputs forwarded through array, not refetched from register file
    - Operand fetch energy amortized over massive spatial reuse
    - PE-to-PE wire is ~10x shorter than register file access path

    Example (128x128 matmul):
    - CPU: 128*128*128 = 2M operations, each needs 2 reg reads = 4M fetches
    - TPU: Load 16K weights once + inject 16K inputs = 32K fetches
    - Reuse factor: 4M / 32K = 128x fewer operand fetches!
    """

    # Array dimensions (configurable)
    array_rows: int = 128
    array_cols: int = 128

    # PE local storage characteristics
    weight_register_bits: int = 16    # BF16 weight storage
    accumulator_bits: int = 32        # FP32 accumulator

    # Derived energy values (set in __post_init__)
    weight_load_energy_pj: float = field(init=False)
    pe_forwarding_energy_pj: float = field(init=False)
    array_injection_energy_pj: float = field(init=False)
    array_extraction_energy_pj: float = field(init=False)

    def __post_init__(self):
        tp = self.tech_profile
        # Systolic forwarding is much cheaper than register file access
        # because wires are very short (PE to neighbor PE)
        self.weight_load_energy_pj = 0.3            # Load weight into PE register
        self.pe_forwarding_energy_pj = 0.1          # Very short wire to neighbor
        # Array boundary injection/extraction uses systolic MAC energy as reference
        self.array_injection_energy_pj = tp.systolic_mac_energy_pj * 0.5
        self.array_extraction_energy_pj = tp.systolic_mac_energy_pj * 0.5

    @property
    def architecture_name(self) -> str:
        return "TPU (Systolic Array)"

    @property
    def macs_per_array_cycle(self) -> int:
        """MACs executed per systolic array cycle."""
        return self.array_rows * self.array_cols

    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 128.0,  # Default to array dimension
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """
        TPU operand fetch: Boundary injection + PE-to-PE forwarding with spatial reuse.

        Systolic arrays achieve efficiency through spatial reuse:
        - Weights stay in PE registers (weight-stationary dataflow)
        - Inputs stream horizontally through array, reused by each row
        - Partial sums accumulate vertically

        Reuse Definition (physical reality):
        - Each weight loaded into a PE is reused for `array_cols` input activations
          that stream through horizontally
        - Each input activation injected at the array edge is reused by `array_rows`
          PEs as it flows through the column
        - The reuse factor is the ARRAY DIMENSION (e.g., 128 for 128x128 array),
          NOT the total array size

        Example (128x128 systolic array processing a tile):
        - Weight reuse: each weight sees 128 different inputs streaming past
        - Input reuse: each input contributes to 128 different outputs
        - Reuse factor = 128 (the streaming dimension)
        """
        if execution_context is None:
            execution_context = {}

        # Reuse is determined by the array dimension (streaming length)
        # For weight-stationary: reuse = number of inputs streaming through = array_cols
        # This can be overridden via execution_context for different dataflows
        array_reuse = execution_context.get('array_reuse', float(self.array_cols))

        # Number of operands that must be fetched from outside the array
        # Each operand fetched is reused `array_reuse` times
        # Total operands needed = 2 per operation (weight + activation)
        # Operands fetched = total_needed / reuse
        operands_needed = num_operations * 2
        operands_fetched = max(1, int(operands_needed / array_reuse))

        # Split fetched operands into weights and inputs
        # In weight-stationary: weights loaded once per tile, inputs stream continuously
        num_weight_elements = max(1, operands_fetched // 2)
        num_input_elements = operands_fetched - num_weight_elements

        # Weight loading (once per tile, amortized over streaming inputs)
        weight_load_energy = num_weight_elements * self.weight_load_energy_pj * 1e-12

        # Array boundary injection (inputs enter at array edge)
        injection_energy = num_input_elements * self.array_injection_energy_pj * 1e-12

        # PE-to-PE forwarding (data flows through array)
        # Each input travels through array_cols PEs horizontally
        # Each partial sum travels through array_rows PEs vertically
        # But this is the LOCAL forwarding cost, separate from reuse accounting
        avg_hops = (self.array_cols + self.array_rows) / 2
        forwarding_energy = num_input_elements * avg_hops * self.pe_forwarding_energy_pj * 1e-12

        # Array boundary extraction (outputs leave array)
        num_outputs = max(1, num_operations // self.macs_per_array_cycle)
        extraction_energy = num_outputs * self.array_extraction_energy_pj * 1e-12

        # Operands served via forwarding = total needed - fetched from outside
        operands_forwarded = operands_needed - operands_fetched

        return OperandFetchBreakdown(
            pe_forwarding_energy=forwarding_energy + weight_load_energy,
            array_injection_energy=injection_energy,
            array_extraction_energy=extraction_energy,
            operands_from_registers=operands_fetched,
            operands_from_forwarding=max(0, operands_forwarded),
            operand_reuse_factor=array_reuse  # Physical reuse = array dimension
        )


# =============================================================================
# KPU Operand Fetch Model (Domain Flow)
# =============================================================================

@dataclass
class KPUOperandFetchModel(OperandFetchEnergyModel):
    """
    KPU operand fetch: Domain flow with programmable spatial routing.

    Architecture:
    - Tiles contain array processors executing SURE/SARE recurrence equations
    - Each PE has local registers + programmable routing to neighbors
    - Domain flow enables dynamic spatial reuse patterns
    - Recirculation extends reuse beyond single tile dimension

    Energy Profile:
    - PE register access: ~0.25 pJ (simpler than GPU, no banking overhead)
    - PE-to-PE forwarding: ~0.1 pJ (same as TPU - short wire)
    - Array injection: ~0.35 pJ per element
    - Domain tracking: ~0.07 pJ per operation (routing control via domain tags)

    Reuse Definition (physical reality):
    - Base reuse = tile dimension (e.g., 16 for 16x16 tile)
    - Recirculation multiplies reuse (e.g., 4x recirculation -> 64x total)
    - Total reuse = tile_dimension * recirculation_factor
    - The computational domain size determines the effective reuse

    Example (16x16 tile with 4x recirculation):
    - Base streaming reuse: 16 (tile dimension)
    - Recirculation factor: 4 (data recirculates through tile 4 times)
    - Total reuse: 16 * 4 = 64x
    """

    # Tile configuration (configurable)
    tiles_per_chip: int = 64              # KPU-T64 configuration
    pes_per_tile: int = 256               # 16x16 array per tile
    tile_dimension: int = 16              # sqrt(pes_per_tile) - streaming dimension
    recirculation_factor: int = 4         # How many times data recirculates

    # Derived energy values (set in __post_init__)
    pe_register_energy_pj: float = field(init=False)
    pe_forwarding_energy_pj: float = field(init=False)
    array_injection_energy_pj: float = field(init=False)
    array_extraction_energy_pj: float = field(init=False)
    domain_tracking_energy_pj: float = field(init=False)

    def __post_init__(self):
        tp = self.tech_profile
        # KPU PE registers are simple (no banking, no renaming)
        self.pe_register_energy_pj = 0.25
        # PE-to-PE forwarding same as TPU (short wire)
        self.pe_forwarding_energy_pj = 0.1
        # Array boundary uses domain flow MAC energy as reference
        self.array_injection_energy_pj = tp.domain_flow_mac_energy_pj * 0.5
        self.array_extraction_energy_pj = tp.domain_flow_mac_energy_pj * 0.5
        # Domain tracking overhead (distributed routing control)
        self.domain_tracking_energy_pj = tp.domain_flow_mac_energy_pj * 0.1

    @property
    def architecture_name(self) -> str:
        return "KPU (Domain Flow)"

    @property
    def total_pes(self) -> int:
        """Total PEs across all tiles."""
        return self.tiles_per_chip * self.pes_per_tile

    @property
    def effective_reuse(self) -> int:
        """Effective reuse = tile_dimension * recirculation_factor."""
        return self.tile_dimension * self.recirculation_factor

    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 64.0,  # Default: 16 * 4 = 64
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """
        KPU operand fetch: Domain-controlled forwarding with programmable routing.

        Domain flow achieves efficiency through programmable spatial reuse:
        - Base reuse from tile dimension (streaming through 16x16 array)
        - Recirculation extends reuse by reusing data multiple passes
        - Total reuse = tile_dimension * recirculation_factor

        Reuse is physically determined by:
        1. Tile dimension: how far data streams through the array (16)
        2. Recirculation: how many times data loops back (4x default)
        3. Computational domain: the product defines the reuse (64x)
        """
        if execution_context is None:
            execution_context = {}

        # Get reuse components from context or use defaults
        tile_dim = execution_context.get('tile_dimension', self.tile_dimension)
        recirc = execution_context.get('recirculation_factor', self.recirculation_factor)

        # Physical reuse = tile streaming dimension * recirculation passes
        physical_reuse = float(tile_dim * recirc)

        # Allow override if explicitly specified
        reuse = execution_context.get('reuse_factor', physical_reuse)

        # Total operands needed without reuse
        operands_needed = num_operations * 2

        # Operands that must be fetched from outside the array
        operands_fetched = max(1, int(operands_needed / reuse))

        # Operands served via PE-to-PE forwarding within the domain
        operands_forwarded = operands_needed - operands_fetched

        # Array boundary injection (operands entering the computational domain)
        injection_energy = operands_fetched * self.array_injection_energy_pj * 1e-12

        # PE-to-PE forwarding energy (within tile, including recirculation)
        # Average hops includes recirculation paths
        avg_hops = tile_dim * recirc / 2  # Average distance through domain
        forwarding_energy = operands_fetched * avg_hops * self.pe_forwarding_energy_pj * 1e-12

        # Domain tracking overhead (per operation, not per operand)
        # This is the cost of the domain flow control plane
        domain_energy = num_operations * self.domain_tracking_energy_pj * 1e-12

        # Array boundary extraction (outputs leaving the computational domain)
        num_outputs = execution_context.get(
            'output_elements',
            max(1, num_operations // self.pes_per_tile)
        )
        extraction_energy = num_outputs * self.array_extraction_energy_pj * 1e-12

        return OperandFetchBreakdown(
            pe_forwarding_energy=forwarding_energy,
            array_injection_energy=injection_energy,
            array_extraction_energy=extraction_energy,
            domain_tracking_energy=domain_energy,
            operands_from_registers=operands_fetched,
            operands_from_forwarding=operands_forwarded,
            operand_reuse_factor=reuse  # Physical reuse from domain size
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_operand_fetch_model(
    architecture: str,
    tech_profile: 'TechnologyProfile',
    **kwargs
) -> OperandFetchEnergyModel:
    """
    Factory function to create architecture-specific operand fetch models.

    Args:
        architecture: One of 'cpu', 'gpu', 'tpu', 'kpu'
        tech_profile: Technology profile for energy parameters
        **kwargs: Architecture-specific parameters

    Returns:
        Appropriate OperandFetchEnergyModel subclass

    Example:
        model = create_operand_fetch_model('tpu', tech_profile, array_rows=256, array_cols=256)
    """
    arch_lower = architecture.lower()

    if arch_lower == 'cpu':
        return CPUOperandFetchModel(tech_profile=tech_profile, **kwargs)
    elif arch_lower == 'gpu':
        return GPUOperandFetchModel(tech_profile=tech_profile, **kwargs)
    elif arch_lower == 'tpu':
        return TPUOperandFetchModel(tech_profile=tech_profile, **kwargs)
    elif arch_lower == 'kpu':
        return KPUOperandFetchModel(tech_profile=tech_profile, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                         f"Supported: cpu, gpu, tpu, kpu")


# =============================================================================
# Comparison Utility
# =============================================================================

def compare_operand_fetch_energy(
    num_operations: int,
    tech_profile: 'TechnologyProfile',
    operand_width_bytes: int = 4
) -> Dict[str, OperandFetchBreakdown]:
    """
    Compare operand fetch energy across all architectures.

    Args:
        num_operations: Number of arithmetic operations
        tech_profile: Technology profile for energy parameters
        operand_width_bytes: Size of each operand (4 for FP32)

    Returns:
        Dict mapping architecture name to OperandFetchBreakdown
    """
    results = {}

    # CPU
    cpu_model = CPUOperandFetchModel(tech_profile=tech_profile)
    results['CPU'] = cpu_model.compute_operand_fetch_energy(
        num_operations, operand_width_bytes, spatial_reuse_factor=1.0
    )

    # GPU
    gpu_model = GPUOperandFetchModel(tech_profile=tech_profile)
    results['GPU'] = gpu_model.compute_operand_fetch_energy(
        num_operations, operand_width_bytes, spatial_reuse_factor=1.0
    )

    # TPU (128x128 systolic array)
    tpu_model = TPUOperandFetchModel(tech_profile=tech_profile)
    results['TPU'] = tpu_model.compute_operand_fetch_energy(
        num_operations, operand_width_bytes, spatial_reuse_factor=128.0
    )

    # KPU (64 tiles)
    kpu_model = KPUOperandFetchModel(tech_profile=tech_profile)
    results['KPU'] = kpu_model.compute_operand_fetch_energy(
        num_operations, operand_width_bytes, spatial_reuse_factor=64.0
    )

    return results


def format_comparison_table(
    results: Dict[str, OperandFetchBreakdown],
    num_operations: int,
    alu_energy_per_op_pj: float = 0.7
) -> str:
    """
    Format operand fetch comparison as a table.

    Args:
        results: Dict from compare_operand_fetch_energy()
        num_operations: Number of operations (for context)
        alu_energy_per_op_pj: Pure ALU energy per operation in pJ

    Returns:
        Formatted table string
    """
    lines = [
        f"Operand Fetch Energy Comparison ({num_operations:,} operations)",
        "=" * 80,
        "",
        f"{'Architecture':<15} {'Fetch Energy':>12} {'Energy/Op':>12} {'Reuse':>8} {'ALU/Fetch':>10}",
        "-" * 80,
    ]

    total_alu_energy = num_operations * alu_energy_per_op_pj * 1e-12  # Convert to J

    for arch, breakdown in results.items():
        fetch_energy = breakdown.total_fetch_energy
        energy_per_op = breakdown.energy_per_operation * 1e12  # Convert to pJ
        reuse = breakdown.operand_reuse_factor

        # ALU/Fetch ratio (higher = ALU dominated = more efficient architecture)
        if fetch_energy > 0:
            alu_fetch_ratio = total_alu_energy / fetch_energy
        else:
            alu_fetch_ratio = float('inf')

        # Format energy with appropriate units
        if fetch_energy >= 1e-3:
            fetch_str = f"{fetch_energy * 1e3:.2f} mJ"
        elif fetch_energy >= 1e-6:
            fetch_str = f"{fetch_energy * 1e6:.2f} uJ"
        else:
            fetch_str = f"{fetch_energy * 1e9:.2f} nJ"

        lines.append(
            f"{arch:<15} {fetch_str:>12} {energy_per_op:>9.2f} pJ {reuse:>7.1f}x {alu_fetch_ratio:>9.2f}"
        )

    lines.append("-" * 80)
    lines.append("")
    lines.append("Key Insight:")
    lines.append(f"  Pure ALU energy: {alu_energy_per_op_pj:.2f} pJ/op (same for all architectures)")
    lines.append("  ALU/Fetch > 1.0: ALU-dominated (efficient spatial architecture)")
    lines.append("  ALU/Fetch < 1.0: Fetch-dominated (stored-program architecture)")

    return "\n".join(lines)
