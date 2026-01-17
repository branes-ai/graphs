# Plan: Operand Fetch Energy Modeling for Architecture Differentiation

## Executive Summary

The current energy model treats ALU energy (`energy_per_flop_fp32`) as the primary differentiator between architectures. However, the **operand fetch energy** - the energy required to deliver operands from local registers to ALU inputs - is the **true differentiator** between architecture classes.

### Critical Distinction: Operand Fetch vs Memory Hierarchy

**Operand Fetch** (THIS PLAN):
- The "last mile" delivery of operands from register file (or equivalent) to ALU inputs
- Happens every cycle for every arithmetic operation
- Includes: register file reads, operand collectors, crossbars, PE-to-PE forwarding
- Does NOT include: cache accesses, DRAM loads - these are separate load/store instructions

**Memory Hierarchy** (SEPARATE - already modeled elsewhere):
- Load/store operations that move data between memory levels
- Governed by explicit LOAD/STORE instructions in load/store ISAs
- Already modeled in `data_movement_overhead` component
- Completely separate from operand fetch energy

Key insight: The actual ADD/MUL/FMA circuit energy is nearly identical across architectures at the same process node (~0.7 pJ for FP32 FMA at 7nm). What differs dramatically is the **register-to-ALU delivery infrastructure**:

| Architecture | Operand Fetch Mechanism | Energy per Operand Pair |
|--------------|------------------------|------------------------|
| CPU          | Small register file (16-32 regs), 2R+1W ports, read during decode | ~6 pJ (2 reads @ 3 pJ) |
| GPU          | Operand collectors gather 128 operands from banked RF for warp | ~10 pJ (arbitration + routing) |
| TPU          | Systolic PE-to-PE forwarding, operands in neighbor registers | ~0.2 pJ (short wire) |
| KPU          | Domain flow array forwarding, spatial operand reuse | ~0.3 pJ (programmable routing) |

## Why This Matters

Consider a simple FP32 FMA operation:
- **Pure ALU energy**: ~0.7 pJ (the multiply-add circuit itself)
- **CPU operand fetch**: ~9 pJ (2 reads + 1 write @ 3 pJ each)
- **GPU operand fetch**: ~10 pJ (collector arbitration + bank conflicts + crossbar)
- **TPU operand fetch**: ~0.2 pJ (data already in neighbor PE register)
- **KPU operand fetch**: ~0.3 pJ (spatial forwarding through array)

The ALU is only 7-10% of CPU/GPU operation energy, but 70-80% of TPU/KPU operation energy!

## Current State Analysis

### What We Have

The `architectural_energy.py` module already models some operand-related energy:

1. **StoredProgramEnergyModel (CPU)**:
   - `register_file_read_energy`: 3.0 pJ per read
   - `register_file_write_energy`: 3.0 pJ per write
   - `register_ops_per_instruction`: 3 (2 reads + 1 write)
   - Memory hierarchy modeled SEPARATELY (correct!)

2. **DataParallelEnergyModel (GPU)**:
   - `register_file_energy_per_access`: 0.75 pJ (simpler than CPU due to no renaming)
   - But MISSING: operand collector energy, crossbar routing, bank conflict penalties

3. **SystolicArrayEnergyModel (TPU)**:
   - `data_injection_per_element`: 0.35 pJ (array boundary injection)
   - `data_extraction_per_element`: 0.35 pJ (array boundary extraction)
   - But MISSING: PE-to-PE forwarding energy (the key efficiency!)

4. **DomainFlowEnergyModel (KPU)**:
   - `pe_injection_energy_per_element`: 0.35 pJ
   - `pe_extraction_energy_per_element`: 0.35 pJ
   - But MISSING: intra-array forwarding, domain-based routing

### What's Missing

The current model does NOT explicitly:
1. **Separate ALU energy from operand fetch energy** in the base energy calculation
2. **Model the register-to-ALU delivery infrastructure** for each architecture
3. **Quantify spatial reuse** (how systolic/domain-flow avoids refetching operands)
4. **Show operand fetch as the key differentiator** in energy reports

## Proposed Architecture

### Phase 1: Core Data Structures

#### 1.1 Define `OperandFetchBreakdown` Data Class

```python
@dataclass
class OperandFetchBreakdown:
    """
    Detailed breakdown of operand fetch energy.

    Operand fetch = register-to-ALU delivery.
    Does NOT include memory hierarchy (load/store are separate).
    """
    # Register file access (CPU/GPU style)
    register_read_energy: float = 0.0     # Energy to read source operands
    register_write_energy: float = 0.0    # Energy to write result back

    # Operand routing infrastructure
    operand_collector_energy: float = 0.0  # GPU: gather operands for warp
    crossbar_routing_energy: float = 0.0   # Energy to route operands to ALUs
    bank_conflict_penalty: float = 0.0     # GPU: extra energy from bank conflicts

    # Spatial array forwarding (TPU/KPU style)
    pe_forwarding_energy: float = 0.0      # PE-to-PE register forwarding
    array_injection_energy: float = 0.0    # Energy to inject at array boundary
    array_extraction_energy: float = 0.0   # Energy to extract from array boundary

    # Reuse accounting
    operands_from_registers: int = 0       # Operands read from register file
    operands_from_forwarding: int = 0      # Operands received via PE forwarding
    operand_reuse_factor: float = 1.0      # Average times each operand is reused

    # Totals
    @property
    def total_fetch_energy(self) -> float:
        """Total operand fetch energy (excludes memory hierarchy)."""
        return (self.register_read_energy +
                self.register_write_energy +
                self.operand_collector_energy +
                self.crossbar_routing_energy +
                self.bank_conflict_penalty +
                self.pe_forwarding_energy +
                self.array_injection_energy +
                self.array_extraction_energy)

    @property
    def energy_per_operation(self) -> float:
        """Average operand fetch energy per arithmetic operation."""
        total_ops = self.operands_from_registers + self.operands_from_forwarding
        if total_ops > 0:
            return self.total_fetch_energy / (total_ops / 2)  # 2 operands per op
        return 0.0
```

#### 1.2 Introduce `OperandFetchEnergyModel` Base Class

```python
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
    For spatial architectures (systolic, domain flow), operand fetch is minimized.
    """
    tech_profile: 'TechnologyProfile'

    @abstractmethod
    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,      # FP32 = 4 bytes
        spatial_reuse_factor: float = 1.0, # 1.0 = no reuse, higher = better
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """
        Compute the energy required to deliver operands for arithmetic operations.

        Args:
            num_operations: Total arithmetic operations (ADD, MUL, FMA, etc.)
            operand_width_bytes: Size of each operand (4 for FP32, 2 for FP16)
            spatial_reuse_factor: How many times each operand is reused spatially
                                  (1.0 for CPU/GPU, 128+ for systolic arrays)
            execution_context: Architecture-specific context

        Returns:
            OperandFetchBreakdown with detailed energy components
        """
        pass
```

### Phase 2: Architecture-Specific Implementations

#### 2.1 CPU Operand Fetch Model

```python
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
    - No operand reuse between instructions (except through forwarding paths)
    - Register file energy dominates for small operations
    """

    # Register file characteristics
    num_architectural_registers: int = 16    # x86-64 GPRs
    num_physical_registers: int = 64         # With renaming
    read_ports: int = 2                      # Simultaneous reads
    write_ports: int = 1                     # Simultaneous writes

    # Energy per access (derived from tech_profile in __post_init__)
    register_read_energy_pj: float = field(init=False)
    register_write_energy_pj: float = field(init=False)

    # Bypass/forwarding (reduces some register writes)
    bypass_network_energy_pj: float = field(init=False)
    bypass_utilization: float = 0.20         # 20% of results bypass register file

    def __post_init__(self):
        tp = self.tech_profile
        # CPU registers are multi-ported SRAM with CAM for renaming
        self.register_read_energy_pj = tp.register_read_energy_pj
        self.register_write_energy_pj = tp.register_write_energy_pj
        self.bypass_network_energy_pj = tp.register_read_energy_pj * 0.3  # Bypass is cheaper

    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 1.0,  # Always 1.0 for CPU
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """CPU: 2 register reads + 1 register write per operation."""

        # Each operation needs 2 source operands read
        num_register_reads = num_operations * 2

        # Results: some bypass register file, rest are written
        num_register_writes = int(num_operations * (1.0 - self.bypass_utilization))
        num_bypass_forwards = int(num_operations * self.bypass_utilization)

        # Energy calculation
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
```

#### 2.2 GPU Operand Fetch Model

```python
@dataclass
class GPUOperandFetchModel(OperandFetchEnergyModel):
    """
    GPU operand fetch: Banked register files with operand collectors.

    Architecture:
    - 64K registers per SM (shared across 2048 threads)
    - Each thread gets 32-255 registers
    - Register file is banked (32 banks for 32-thread warps)
    - Operand collectors gather operands for warp-wide execution
    - Must deliver 128 operands per cycle (32 threads x 2 operands x 2 ops)

    Energy Profile:
    - Register access: ~0.75 pJ (simpler than CPU - no renaming, in-order)
    - Operand collector: ~0.5 pJ per operand (arbitration, staging)
    - Bank conflict penalty: +1.0 pJ when multiple threads hit same bank
    - Crossbar: ~0.3 pJ per operand (route from bank to collector)

    Key Characteristics:
    - Massive parallelism requires massive operand bandwidth
    - Operand collectors are energy-hungry (arbitrate 128 operands/cycle)
    - Bank conflicts add significant energy overhead
    - No spatial reuse - each thread fetches independently
    """

    # Warp and register file characteristics
    warp_size: int = 32
    registers_per_sm: int = 65536
    register_banks: int = 32              # One bank per warp lane

    # Energy per access (derived from tech_profile)
    register_access_energy_pj: float = field(init=False)
    operand_collector_energy_pj: float = field(init=False)
    crossbar_energy_pj: float = field(init=False)
    bank_conflict_penalty_pj: float = field(init=False)

    # Bank conflict rate (depends on access pattern)
    bank_conflict_rate: float = 0.10      # 10% of accesses have conflicts

    def __post_init__(self):
        tp = self.tech_profile
        # GPU registers are simpler (no renaming) but need more ports
        self.register_access_energy_pj = tp.register_read_energy_pj * 0.25
        self.operand_collector_energy_pj = 0.5  # Collector arbitration
        self.crossbar_energy_pj = 0.3           # Operand routing
        self.bank_conflict_penalty_pj = 1.0     # Extra cycles for conflicts

    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 1.0,  # Always 1.0 for GPU
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """GPU: Register access + operand collector + crossbar per operation."""

        # Each operation needs 2 operands per thread, across warp
        num_operands = num_operations * 2

        # Register file access
        register_energy = num_operands * self.register_access_energy_pj * 1e-12

        # Operand collector energy (gather operands for warp execution)
        collector_energy = num_operands * self.operand_collector_energy_pj * 1e-12

        # Crossbar routing
        crossbar_energy = num_operands * self.crossbar_energy_pj * 1e-12

        # Bank conflict penalty
        num_conflicts = int(num_operands * self.bank_conflict_rate)
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
            operands_from_forwarding=0,
            operand_reuse_factor=1.0
        )
```

#### 2.3 TPU Operand Fetch Model

```python
@dataclass
class TPUOperandFetchModel(OperandFetchEnergyModel):
    """
    TPU operand fetch: Systolic array with PE-to-PE forwarding.

    Architecture:
    - 128x128 or 256x256 systolic array (MXU)
    - Weight-stationary dataflow: weights loaded into PE registers, stay there
    - Inputs stream horizontally through array
    - Partial sums accumulate vertically
    - Each PE has: weight register, input latch, accumulator

    Energy Profile:
    - Weight load (once per tile): 0.3 pJ per element
    - PE-to-PE forwarding: ~0.1 pJ (very short wire to neighbor)
    - Array boundary injection: 0.35 pJ per element
    - Array boundary extraction: 0.35 pJ per element

    Key Efficiency:
    - Weights loaded ONCE, reused for entire input batch (128+ reuses)
    - Inputs forwarded through array, not refetched
    - Operand fetch energy amortized over massive reuse
    - PE-to-PE wire is ~10x shorter than register file access

    Example (128x128 matmul):
    - CPU: 128x128x128 = 2M operations, each needs 2 reg reads = 4M fetches
    - TPU: Load 16K weights once + inject 16K inputs = 32K fetches
    - Reuse factor: 4M / 32K = 128x fewer operand fetches!
    """

    # Array dimensions
    array_rows: int = 128
    array_cols: int = 128

    # PE local storage
    weight_register_bits: int = 16    # BF16 weight
    accumulator_bits: int = 32        # FP32 accumulator

    # Energy per element (derived from tech_profile)
    weight_load_energy_pj: float = field(init=False)
    pe_forwarding_energy_pj: float = field(init=False)
    array_injection_energy_pj: float = field(init=False)
    array_extraction_energy_pj: float = field(init=False)

    def __post_init__(self):
        tp = self.tech_profile
        # Systolic forwarding is much cheaper than register file
        self.weight_load_energy_pj = 0.3           # Load weight into PE
        self.pe_forwarding_energy_pj = 0.1         # Very short wire
        self.array_injection_energy_pj = tp.systolic_mac_energy_pj * 0.5
        self.array_extraction_energy_pj = tp.systolic_mac_energy_pj * 0.5

    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 128.0,  # Depends on array size
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """TPU: Boundary injection + PE forwarding, massive reuse."""

        if execution_context is None:
            execution_context = {}

        # Get actual reuse from context or use spatial_reuse_factor
        weight_reuse = execution_context.get('weight_reuse', spatial_reuse_factor)
        input_reuse = execution_context.get('input_reuse', self.array_cols)

        # Number of unique operands needed (before reuse)
        # For matmul MxNxK: weights = MxK, inputs = KxN
        num_weight_elements = execution_context.get('weight_elements',
                                                     num_operations // (self.array_rows * weight_reuse))
        num_input_elements = execution_context.get('input_elements',
                                                    num_operations // (self.array_cols * input_reuse))

        # Weight loading (once per tile, amortized)
        weight_load_energy = num_weight_elements * self.weight_load_energy_pj * 1e-12

        # Array boundary injection (inputs enter array edge)
        injection_energy = num_input_elements * self.array_injection_energy_pj * 1e-12

        # PE-to-PE forwarding (data flows through array)
        # Each input travels through array_cols PEs
        num_forwards = num_input_elements * self.array_cols
        forwarding_energy = num_forwards * self.pe_forwarding_energy_pj * 1e-12

        # Array boundary extraction (outputs leave array)
        num_outputs = num_operations // (self.array_rows * self.array_cols)
        extraction_energy = num_outputs * self.array_extraction_energy_pj * 1e-12

        # Calculate effective reuse
        total_operands_needed = num_operations * 2  # Without reuse
        operands_actually_fetched = num_weight_elements + num_input_elements
        effective_reuse = total_operands_needed / max(1, operands_actually_fetched)

        return OperandFetchBreakdown(
            pe_forwarding_energy=forwarding_energy + weight_load_energy,
            array_injection_energy=injection_energy,
            array_extraction_energy=extraction_energy,
            operands_from_registers=0,  # No traditional register file
            operands_from_forwarding=num_forwards,
            operand_reuse_factor=effective_reuse
        )
```

#### 2.4 KPU Operand Fetch Model

```python
@dataclass
class KPUOperandFetchModel(OperandFetchEnergyModel):
    """
    KPU operand fetch: Domain flow with programmable spatial routing.

    Architecture:
    - Tiles contain array processors executing SURE/SARE
    - Each PE has local registers + programmable routing
    - Domain flow enables dynamic spatial reuse patterns
    - More flexible than TPU but slightly higher control overhead

    Energy Profile:
    - PE register access: ~0.25 pJ (simpler than GPU)
    - PE-to-PE forwarding: ~0.1 pJ (same as TPU)
    - Array injection: ~0.35 pJ per element
    - Domain tracking: ~0.07 pJ per operation (routing control)

    Key Characteristics:
    - Programmable dataflow (not fixed weight-stationary)
    - Can adapt reuse pattern to workload
    - Handles irregular operations (attention, softmax) efficiently
    - Slightly higher overhead than TPU for simple matmul
    - Significantly better than TPU for complex dataflows
    """

    # Tile configuration
    tiles_per_chip: int = 64
    pes_per_tile: int = 256           # 16x16 array

    # Energy per access (derived from tech_profile)
    pe_register_energy_pj: float = field(init=False)
    pe_forwarding_energy_pj: float = field(init=False)
    array_injection_energy_pj: float = field(init=False)
    array_extraction_energy_pj: float = field(init=False)
    domain_tracking_energy_pj: float = field(init=False)

    def __post_init__(self):
        tp = self.tech_profile
        self.pe_register_energy_pj = 0.25
        self.pe_forwarding_energy_pj = 0.1
        self.array_injection_energy_pj = tp.domain_flow_mac_energy_pj * 0.5
        self.array_extraction_energy_pj = tp.domain_flow_mac_energy_pj * 0.5
        self.domain_tracking_energy_pj = tp.domain_flow_mac_energy_pj * 0.1

    def compute_operand_fetch_energy(
        self,
        num_operations: int,
        operand_width_bytes: int = 4,
        spatial_reuse_factor: float = 64.0,
        execution_context: Optional[Dict] = None
    ) -> OperandFetchBreakdown:
        """KPU: Domain-controlled forwarding with programmable routing."""

        if execution_context is None:
            execution_context = {}

        # Get reuse factor from context
        reuse = execution_context.get('reuse_factor', spatial_reuse_factor)

        # Operands fetched vs forwarded
        operands_needed = num_operations * 2
        operands_fetched = int(operands_needed / reuse)
        operands_forwarded = operands_needed - operands_fetched

        # Array boundary injection
        injection_energy = operands_fetched * self.array_injection_energy_pj * 1e-12

        # PE-to-PE forwarding
        forwarding_energy = operands_forwarded * self.pe_forwarding_energy_pj * 1e-12

        # Domain tracking overhead (per operation)
        domain_energy = num_operations * self.domain_tracking_energy_pj * 1e-12

        # Array boundary extraction
        num_outputs = execution_context.get('output_elements', num_operations // 256)
        extraction_energy = num_outputs * self.array_extraction_energy_pj * 1e-12

        return OperandFetchBreakdown(
            pe_forwarding_energy=forwarding_energy + domain_energy,
            array_injection_energy=injection_energy,
            array_extraction_energy=extraction_energy,
            operands_from_registers=operands_fetched,
            operands_from_forwarding=operands_forwarded,
            operand_reuse_factor=reuse
        )
```

### Phase 3: Integration with Existing Energy Model

#### 3.1 Extend `ArchitecturalEnergyBreakdown`

Add new fields to capture operand fetch energy separately:

```python
@dataclass
class ArchitecturalEnergyBreakdown:
    # ... existing fields ...

    # NEW: Explicit ALU vs Operand Fetch separation
    pure_alu_energy: float = 0.0           # Just the arithmetic circuit
    operand_fetch_energy: float = 0.0      # Register-to-ALU delivery
    operand_fetch_breakdown: Optional[OperandFetchBreakdown] = None

    @property
    def alu_to_fetch_ratio(self) -> float:
        """Ratio of pure ALU energy to operand fetch energy."""
        if self.operand_fetch_energy > 0:
            return self.pure_alu_energy / self.operand_fetch_energy
        return float('inf')

    @property
    def fetch_dominance(self) -> str:
        """Describe whether operand fetch or ALU dominates."""
        ratio = self.alu_to_fetch_ratio
        if ratio > 2.0:
            return "ALU-dominated (spatial architecture)"
        elif ratio > 0.5:
            return "Balanced"
        else:
            return "Fetch-dominated (stored-program architecture)"
```

#### 3.2 Modify Existing ArchitecturalEnergyModels

Update each model to use the new operand fetch classes:

```python
@dataclass
class StoredProgramEnergyModel(ArchitecturalEnergyModel):
    # ... existing fields ...
    operand_fetch_model: CPUOperandFetchModel = field(init=False)

    def __post_init__(self):
        # ... existing init ...
        self.operand_fetch_model = CPUOperandFetchModel(tech_profile=self.tech_profile)

    def compute_architectural_energy(...) -> ArchitecturalEnergyBreakdown:
        # ... existing computation ...

        # NEW: Use operand fetch model
        fetch_breakdown = self.operand_fetch_model.compute_operand_fetch_energy(
            num_operations=ops,
            operand_width_bytes=4,  # FP32
            spatial_reuse_factor=1.0
        )

        # Calculate pure ALU energy (separate from operand fetch)
        pure_alu = ops * self.alu_energy_per_op  # Just the FMA circuit

        return ArchitecturalEnergyBreakdown(
            # ... existing fields ...
            pure_alu_energy=pure_alu,
            operand_fetch_energy=fetch_breakdown.total_fetch_energy,
            operand_fetch_breakdown=fetch_breakdown
        )
```

### Phase 4: Reporting

#### 4.1 Operand Fetch Comparison Table

```
Operand Fetch Energy Analysis:
==============================

Operation: 128x128 MatMul (2,097,152 MACs)

| Architecture | Pure ALU   | Operand Fetch | Total/Op | ALU/Fetch Ratio |
|--------------|------------|---------------|----------|-----------------|
| CPU (x86)    | 1.47 mJ    | 18.9 mJ       | 9.7 pJ   | 0.08 (fetch!)   |
| GPU (H100)   | 1.47 mJ    | 20.9 mJ       | 10.7 pJ  | 0.07 (fetch!)   |
| TPU v4       | 1.47 mJ    | 0.15 mJ       | 0.77 pJ  | 9.8 (ALU!)      |
| KPU T64      | 1.47 mJ    | 0.21 mJ       | 0.80 pJ  | 7.0 (ALU!)      |

Key Insight:
- CPU/GPU: 92-93% of energy goes to operand fetch infrastructure
- TPU/KPU: 80-90% of energy goes to actual computation (ALU)
- Spatial architectures achieve 100x lower operand fetch energy through reuse!
```

#### 4.2 Operand Reuse Visualization

```
Operand Reuse Analysis (128x128 MatMul):
========================================

Total operands needed (without reuse): 4,194,304 (2M ops x 2 operands)

| Architecture | Fetched    | Reused      | Reuse Factor | Fetch Energy |
|--------------|------------|-------------|--------------|--------------|
| CPU          | 4,194,304  | 0           | 1.0x         | 18.9 mJ      |
| GPU          | 4,194,304  | 0           | 1.0x         | 20.9 mJ      |
| TPU          | 32,768     | 4,161,536   | 128x         | 0.15 mJ      |
| KPU          | 65,536     | 4,128,768   | 64x          | 0.21 mJ      |

TPU reuse: Weight-stationary (weights stay in PE, reused 128x)
KPU reuse: Domain flow (programmable routing enables 64x spatial reuse)
```

## Implementation Plan

### Step 1: Add Core Data Structures
- [ ] Add `OperandFetchBreakdown` dataclass
- [ ] Add `OperandFetchEnergyModel` abstract base class
- [ ] Update `ArchitecturalEnergyBreakdown` with operand fetch fields

### Step 2: Implement Architecture-Specific Models
- [ ] Implement `CPUOperandFetchModel`
- [ ] Implement `GPUOperandFetchModel`
- [ ] Implement `TPUOperandFetchModel`
- [ ] Implement `KPUOperandFetchModel`

### Step 3: Integrate with Existing Analyzers
- [ ] Modify `StoredProgramEnergyModel` to use `CPUOperandFetchModel`
- [ ] Modify `DataParallelEnergyModel` to use `GPUOperandFetchModel`
- [ ] Modify `SystolicArrayEnergyModel` to use `TPUOperandFetchModel`
- [ ] Modify `DomainFlowEnergyModel` to use `KPUOperandFetchModel`

### Step 4: Update TechnologyProfile
- [ ] Ensure all operand fetch energy parameters are in `TechnologyProfile`
- [ ] Add PE forwarding energy parameters
- [ ] Add operand collector parameters for GPU

### Step 5: Reporting and Validation
- [ ] Add operand fetch breakdown to `EnergyReport`
- [ ] Add ALU/Fetch ratio to comparison tables
- [ ] Create validation tests comparing architectures
- [ ] Verify energy numbers against published data

## Energy Parameter Reference

### Pure ALU Energy (Process Node Dependent)

The actual arithmetic circuit energy:

| Node | FP32 FMA | BF16 FMA | INT8 MAC |
|------|----------|----------|----------|
| 3nm  | 0.5 pJ   | 0.25 pJ  | 0.06 pJ  |
| 5nm  | 0.7 pJ   | 0.35 pJ  | 0.09 pJ  |
| 7nm  | 0.8 pJ   | 0.40 pJ  | 0.10 pJ  |
| 16nm | 1.5 pJ   | 0.75 pJ  | 0.19 pJ  |

### Operand Fetch Energy (Architecture Dependent)

Energy to deliver operands from local storage to ALU:

| Component               | CPU     | GPU     | TPU     | KPU     |
|-------------------------|---------|---------|---------|---------|
| Register read           | 3.0 pJ  | 0.75 pJ | N/A     | 0.25 pJ |
| Register write          | 3.0 pJ  | 0.75 pJ | N/A     | 0.25 pJ |
| Operand collector       | N/A     | 0.5 pJ  | N/A     | N/A     |
| Crossbar/routing        | N/A     | 0.3 pJ  | N/A     | N/A     |
| PE-to-PE forwarding     | N/A     | N/A     | 0.1 pJ  | 0.1 pJ  |
| Array injection         | N/A     | N/A     | 0.35 pJ | 0.35 pJ |
| Domain tracking         | N/A     | N/A     | N/A     | 0.07 pJ |

### Total Energy per FP32 Operation

| Architecture | ALU    | Operand Fetch | Total  | Fetch % |
|--------------|--------|---------------|--------|---------|
| CPU (7nm)    | 0.8 pJ | 9.0 pJ        | 9.8 pJ | 92%     |
| GPU (5nm)    | 0.7 pJ | 10.3 pJ       | 11.0 pJ| 94%     |
| TPU (7nm)    | 0.8 pJ | 0.07 pJ*      | 0.87 pJ| 8%      |
| KPU (16nm)   | 1.5 pJ | 0.12 pJ*      | 1.62 pJ| 7%      |

*Amortized over spatial reuse

## References

1. Horowitz, M. (2014). "Computing's Energy Problem" - ISSCC Keynote
2. Jouppi et al. (2017). "TPU v1" - ISCA (Table 2: Energy breakdown)
3. NVIDIA H100 Architecture Whitepaper (2022)
4. Sze et al. (2017). "Efficient Processing of DNNs" - IEEE Proceedings

## Document History

- 2025-12-05: Revised to separate operand fetch from memory hierarchy
- 2025-12-04: Initial plan created
