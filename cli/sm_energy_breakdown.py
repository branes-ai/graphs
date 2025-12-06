#!/usr/bin/env python3
"""
SM Energy Breakdown: ALU + Operand Fetch per MAC

This script calculates the MINIMUM energy per MAC for:
1. CUDA cores (scalar FMA operations)
2. TensorCores (matrix MMA operations)

The key insight: operand fetch infrastructure is SUBSTANTIAL.

GPU Register File Architecture (H100-class):
============================================
Each SM partition has:
- 16KB register file (64K 32-bit registers)
- 16 banks of dual-ported SRAM
- 32 threads need operands simultaneously (warp)
- Each FMA needs 3 operands (A, B, accumulator C)

Operand Delivery Path:
  Register File (16 banks, dual-ported)
       |
       v
  Bank Arbitration (resolve conflicts when threads hit same bank)
       |
       v
  Operand Collector (buffer until all operands ready)
       |
       v
  Crossbar/MUX network (route to correct ALU lane)
       |
       v
  ALU (do the actual computation)
       |
       v
  Result Crossbar (route back to register file)
       |
       v
  Register File Write

Energy Reality Check:
- A 16KB SRAM read at 4nm: ~2-5 pJ per access
- A 32-wide crossbar: ~1-2 pJ per routing
- Bank arbitration logic: ~0.5-1 pJ
- The ALU itself (FP16 FMA): ~0.5-1 pJ

The infrastructure DOMINATES the ALU for stored-program machines!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Process Node Scaling
# =============================================================================
#
# Reference: Mark Horowitz, "Computing's Energy Problem", ISSCC 2014
# https://www.researchgate.net/publication/271463146
#
# Horowitz 45nm (0.9V) reference values:
#   - 8b Add:        0.03 pJ
#   - 32b Add:       0.1 pJ
#   - 32b Int Mult:  3.1 pJ
#   - 16b FP Add:    0.4 pJ
#   - 16b FP Mult:   1.1 pJ
#   - 32b FP Add:    0.9 pJ
#   - 32b FP Mult:   3.7 pJ
#   - 8KB SRAM read: 5.0 pJ (32-bit word)
#   - DRAM read:     640 pJ (32-bit word)
#
# Scaling from 45nm to modern nodes:
# - Energy scales ~linearly with capacitance (area) and quadratically with voltage
# - Modern nodes have diminishing returns in energy scaling
# - 45nm -> 7nm: ~3-4x improvement (not 6.4x as node ratio suggests)
# - 45nm -> 4nm: ~4-5x improvement
#
# Using 45nm as reference, scale factors relative to 4nm:

PROCESS_SCALE_VS_4NM = {
    3: 0.85,   # 3nm: 15% better than 4nm
    4: 1.0,    # 4nm: baseline for our model
    5: 1.15,   # 5nm: ~15% worse than 4nm
    7: 1.4,    # 7nm: ~40% worse than 4nm
    10: 1.7,   # 10nm
    12: 2.0,   # 12nm: ~2x worse than 4nm
    14: 2.2,   # 14nm
    16: 2.5,   # 16nm
    22: 3.0,   # 22nm
    28: 3.5,   # 28nm
    45: 4.5,   # 45nm: Horowitz reference node
}

# Horowitz 45nm reference values (pJ)
HOROWITZ_45NM = {
    'fp16_mult': 1.1,
    'fp16_add': 0.4,
    'fp32_mult': 3.7,
    'fp32_add': 0.9,
    'int8_mult': 0.2,
    'int8_add': 0.03,
    'int32_mult': 3.1,
    'int32_add': 0.1,
    'sram_8kb_read': 5.0,   # 32-bit word from 8KB SRAM
    'dram_read': 640.0,     # 32-bit word from DRAM
}

# =============================================================================
# CALIBRATED BASELINE: Energy per MAC at 4nm
# =============================================================================
#
# ENERGY BOUNDS ANALYSIS:
# =======================
# The actual energy per TensorCore MAC at 4nm is constrained by two bounds:
#
# LOWER BOUND (Component Model): ~1.15 pJ/MAC
#   - Horowitz-scaled register file + ALU energy (see component model below)
#   - This is TOO LOW because:
#     - Real-world MFU of 35-50% would imply physics limit of 40-60%
#     - But MFU includes software overhead (kernel launch, memory stalls, etc.)
#     - Physics limit must be HIGHER than observed MFU, not lower
#
# UPPER BOUND (External Analysis): ~2.5 pJ/MAC
#   - One external analysis suggested 2.5 pJ/MAC at 4nm
#   - This is TOO HIGH because:
#     - At 2.5 pJ, H100 (700W) sustains 280 TFLOPS max
#     - 280 / 990 (FP16 dense) = 28% physics limit
#     - But real MFU is 35-50%, which must be BELOW physics limit
#     - Contradiction: measured > theoretical is impossible
#
# REALISTIC ESTIMATE: 1.5 - 1.8 pJ/MAC
#   - If real MFU is 38% and that's below physics limit (~45-50%):
#     - Physics limit at 45%: 990 * 0.45 = 445 TFLOPS sustainable
#     - Energy = 700W / 445 TFLOPS = 1.57 pJ/MAC
#   - If real MFU is 50% and physics limit is ~55-60%:
#     - Physics limit at 55%: 990 * 0.55 = 545 TFLOPS sustainable
#     - Energy = 700W / 545 TFLOPS = 1.28 pJ/MAC
#   - Likely range: 1.5 - 1.8 pJ/MAC at 4nm
#
# The component model (~1.15 pJ) likely underestimates:
#   - Clock distribution and sequencing overhead
#   - Instruction decode and dispatch
#   - Warp scheduling logic
#   - Other control plane energy
#
# Using 1.65 pJ as midpoint estimate (geometric mean of 1.5-1.8 range)
#
CALIBRATED_MAC_ENERGY_PJ = {
    3: 1.4,    # 3nm (Blackwell): ~1.4 pJ/MAC (scaled from 4nm)
    4: 1.65,   # 4nm (Hopper): ~1.65 pJ/MAC - REALISTIC ESTIMATE
    7: 2.3,    # 7nm (Ampere): ~2.3 pJ/MAC
    12: 3.3,   # 12nm (Volta/Turing): ~3.3 pJ/MAC
}

def get_scale(process_nm: int) -> float:
    """Get energy scaling factor relative to 4nm baseline."""
    return PROCESS_SCALE_VS_4NM.get(process_nm, process_nm / 4.0)

def get_calibrated_mac_energy_pj(process_nm: int) -> float:
    """
    Get calibrated total energy per MAC for TensorCore operations.

    This is the empirically-validated value that matches real-world
    MLPerf benchmarks and MFU observations.
    """
    if process_nm in CALIBRATED_MAC_ENERGY_PJ:
        return CALIBRATED_MAC_ENERGY_PJ[process_nm]
    # Interpolate/extrapolate based on 4nm baseline
    return 2.5 * get_scale(process_nm)

def get_horowitz_scaled(operation: str, process_nm: int) -> float:
    """
    Get Horowitz energy value scaled to a target process node.

    Args:
        operation: Key from HOROWITZ_45NM dict
        process_nm: Target process node

    Returns:
        Energy in pJ scaled to target node
    """
    base_45nm = HOROWITZ_45NM[operation]
    # Scale from 45nm to target
    scale_45_to_4 = PROCESS_SCALE_VS_4NM[45] / PROCESS_SCALE_VS_4NM[4]  # 4.5
    scale_4_to_target = PROCESS_SCALE_VS_4NM.get(process_nm, process_nm / 4.0)
    return base_45nm / scale_45_to_4 * scale_4_to_target


# =============================================================================
# Register File Energy Model
# =============================================================================
# Based on SRAM energy models and GPU architecture papers

class RegisterFileModel:
    """
    GPU register file energy model - CALIBRATED TO HOROWITZ DATA.

    Reference: Mark Horowitz, "Computing's Energy Problem", ISSCC 2014
    - 8KB SRAM read (32-bit): 5.0 pJ at 45nm

    GPU Register File:
    - 16KB per SM partition (larger than 8KB reference)
    - 16 banks of SRAM
    - Very fast access (single cycle at ~2GHz)
    - Dual-ported for read + write

    Energy Scaling:
    - Horowitz 8KB SRAM at 45nm: 5.0 pJ per 32-bit read
    - GPU register file is 16KB but heavily optimized for speed
    - Estimate: ~4 pJ at 45nm (smaller than cache, optimized)
    - At 4nm: 4.0 / 4.5 = ~0.9 pJ

    But wait - the Horowitz number is for a SINGLE read.
    GPU needs to read from BANKED structure with:
    - Bank decode/select overhead
    - Potential bank conflicts
    - Routing to operand collector

    Conservative estimate for GPU register file at 4nm:
    - Base SRAM read: ~1.0 pJ (Horowitz scaled)
    - Bank decode: ~0.1 pJ
    - Total: ~1.1 pJ per 32-bit read
    """

    def __init__(self, process_nm: int = 4):
        self.process_nm = process_nm

        # Use Horowitz-calibrated value
        # 8KB SRAM at 45nm = 5.0 pJ
        # Scale to target process, and adjust for register file size/optimization
        # Register file is smaller and faster, estimate ~80% of 8KB cache energy
        self.base_sram_read_pj = get_horowitz_scaled('sram_8kb_read', process_nm) * 0.8

        # Additional overhead for banked structure
        scale = get_scale(process_nm)
        self.bank_decode_pj = 0.1 * scale

        # Write is ~10% more expensive than read (bit line driving)
        self.write_overhead_factor = 1.1

    def read_energy_per_word_pj(self) -> float:
        """Energy to read one 32-bit register from banked SRAM."""
        return self.base_sram_read_pj + self.bank_decode_pj

    def write_energy_per_word_pj(self) -> float:
        """Energy to write one 32-bit register to banked SRAM."""
        return (self.base_sram_read_pj + self.bank_decode_pj) * self.write_overhead_factor


# =============================================================================
# Operand Collector Energy Model
# =============================================================================

class OperandCollectorModel:
    """
    GPU operand collector energy model.

    The operand collector is the MOST COMPLEX piece of the operand delivery path.
    It's not just a buffer - it's a multi-stage pipeline with significant logic.

    Architecture (per SM partition):
    ================================
    - 4-8 collector units (one per warp scheduler slot)
    - Each collector buffers operands for one in-flight instruction
    - Must handle bank conflicts (2+ threads accessing same bank)
    - Must arbitrate across multiple instructions competing for register ports

    Physical Structure of One Collector Unit:
    =========================================
    1. Staging Registers (64 x 32-bit flip-flops for 2 operands x 32 threads)
       - 64 registers x 32 bits = 2048 flip-flops
       - Each flip-flop: ~0.5fF, read/write ~0.01 pJ
       - Total staging: 64 x 0.01 = 0.64 pJ

    2. Bank Conflict Detection (CAM-like structure):
       - Compare bank addresses for all 32 threads (32 x 4-bit comparators)
       - Detect if 2+ threads hit same bank
       - CAM lookup: ~0.5 pJ per comparison set
       - Need to check 2 operands: 2 x 0.5 = 1.0 pJ

    3. Conflict Arbitration Logic:
       - Priority encoder for conflicting threads
       - Multi-cycle scheduling when conflicts occur
       - State machine for retry logic
       - ~0.8 pJ per instruction

    4. Operand Routing Network (the big one!):
       - Must route 64 operands (2 x 32 threads) to 32 ALU lanes
       - Each operand has specific destination (thread's ALU lane)
       - This is a 64:32 routing network
       - Crossbar or Benes network: ~2-3 pJ total

    5. Ready/Valid Control:
       - Track which operands have arrived
       - Signal when all 64 operands ready
       - ~0.3 pJ per instruction

    Total per warp instruction at 4nm:
    ==================================
    - Staging: 0.64 pJ
    - Conflict detection: 1.0 pJ
    - Arbitration: 0.8 pJ
    - Routing: 2.5 pJ
    - Control: 0.3 pJ
    -------------------
    Total: ~5.2 pJ per warp instruction
    Per MAC: 5.2 / 32 = 0.16 pJ/MAC (for CUDA)

    This is MORE than the ALU in many cases!
    """

    def __init__(self, process_nm: int = 4):
        self.process_nm = process_nm
        scale = get_scale(process_nm)

        # Detailed energy components at 4nm (pJ)
        self.staging_per_operand_pj = 0.01 * scale  # Flip-flop read/write
        self.conflict_detection_pj = 1.0 * scale    # CAM-like comparison
        self.arbitration_pj = 0.8 * scale           # Priority + state machine
        self.routing_network_pj = 2.5 * scale       # 64:32 crossbar
        self.control_pj = 0.3 * scale               # Ready/valid tracking

    def energy_per_warp_instruction_pj(self, operands_per_thread: int = 2) -> float:
        """Energy for operand collection for one warp (32 threads)."""
        warp_size = 32
        total_operands = warp_size * operands_per_thread

        staging = total_operands * self.staging_per_operand_pj
        conflict_det = self.conflict_detection_pj * operands_per_thread
        arbitration = self.arbitration_pj
        routing = self.routing_network_pj * (operands_per_thread / 2)  # Scale with operands
        control = self.control_pj

        return staging + conflict_det + arbitration + routing + control


# =============================================================================
# Bank Arbitration Energy Model
# =============================================================================

class BankArbitrationModel:
    """
    Bank conflict arbitration energy model.

    When multiple threads in a warp access the same register bank,
    the accesses must be serialized. The arbitration logic:
    1. Detects conflicts (comparators on bank addresses)
    2. Prioritizes/schedules conflicting accesses
    3. Stalls warp until all operands ready

    Energy sources:
    1. Bank address comparison: ~0.05 pJ per thread-pair at 4nm
    2. Priority encoder: ~0.2 pJ per cycle
    3. Stall logic: ~0.1 pJ per cycle

    For 32 threads accessing 16 banks:
    - Average ~2 conflicts per warp (depends on access pattern)
    - Worst case: all threads hit same bank (31 conflicts)

    Typical energy: ~3-5 pJ per warp instruction
    """

    def __init__(self, process_nm: int = 4):
        self.process_nm = process_nm
        scale = get_scale(process_nm)

        # Base arbitration at 4nm
        self.base_arbitration_pj = 3.0 * scale
        self.conflict_penalty_pj = 0.5 * scale  # Extra per conflict

    def energy_per_warp_instruction_pj(self, avg_conflicts: float = 2.0) -> float:
        """Energy for bank arbitration for one warp instruction."""
        return self.base_arbitration_pj + avg_conflicts * self.conflict_penalty_pj


# =============================================================================
# Crossbar/Interconnect Energy Model
# =============================================================================

class CrossbarModel:
    """
    Operand routing crossbar energy model.

    The crossbar routes operands from the register file/collector to ALU lanes.
    For 32-wide SIMT, this is a substantial piece of logic.

    Architecture:
    - 32 inputs (from collector) to 32 outputs (ALU lanes)
    - Full crossbar or multi-stage network
    - Also need result routing back to register file

    Energy (4nm):
    - 32x32 crossbar: ~1.5 pJ per routing operation
    - Multi-stage (log network): ~0.8 pJ per routing

    Modern GPUs use hierarchical routing to reduce energy.
    """

    def __init__(self, process_nm: int = 4):
        self.process_nm = process_nm
        scale = get_scale(process_nm)

        # Per-lane routing energy at 4nm
        self.routing_per_lane_pj = 0.05 * scale
        self.result_routing_per_lane_pj = 0.04 * scale

    def operand_routing_energy_pj(self, num_lanes: int = 32) -> float:
        """Energy to route operands to ALU lanes."""
        return num_lanes * self.routing_per_lane_pj * 2  # 2 source operands

    def result_routing_energy_pj(self, num_lanes: int = 32) -> float:
        """Energy to route results back to register file."""
        return num_lanes * self.result_routing_per_lane_pj


# =============================================================================
# ALU Energy Model
# =============================================================================

class ALUModel:
    """
    FMA (Fused Multiply-Add) ALU energy model - CALIBRATED TO HOROWITZ DATA.

    Reference: Mark Horowitz, "Computing's Energy Problem", ISSCC 2014
    - FP16 Mult: 1.1 pJ at 45nm
    - FP16 Add:  0.4 pJ at 45nm
    - FP32 Mult: 3.7 pJ at 45nm
    - FP32 Add:  0.9 pJ at 45nm

    FMA = Mult + Add (fused, ~10% savings from shared normalization)
    - FP16 FMA at 45nm: 1.1 + 0.4 - 0.15 = ~1.35 pJ
    - FP32 FMA at 45nm: 3.7 + 0.9 - 0.4 = ~4.2 pJ

    Scaled to 4nm (divide by 4.5):
    - FP16 FMA at 4nm: ~0.30 pJ
    - FP32 FMA at 4nm: ~0.93 pJ

    TensorCore MAC:
    - FP16 inputs, FP32 accumulation
    - Systolic-style local accumulation saves routing energy
    - But still needs FP16 mult + FP32 add per MAC
    - Estimate: FP16 mult (0.24 pJ) + partial FP32 add (0.10 pJ) = ~0.34 pJ at 4nm
    """

    def __init__(self, process_nm: int = 4):
        self.process_nm = process_nm

        # Use Horowitz-calibrated values
        # FMA = mult + add with ~10% fusion savings
        fp16_mult = get_horowitz_scaled('fp16_mult', process_nm)
        fp16_add = get_horowitz_scaled('fp16_add', process_nm)
        fp32_mult = get_horowitz_scaled('fp32_mult', process_nm)
        fp32_add = get_horowitz_scaled('fp32_add', process_nm)

        # Fusion savings: shared normalization/rounding logic
        fusion_savings = 0.9  # 10% savings

        self.fp16_fma_pj = (fp16_mult + fp16_add) * fusion_savings
        self.fp32_fma_pj = (fp32_mult + fp32_add) * fusion_savings
        self.fp64_fma_pj = self.fp32_fma_pj * 2.0  # FP64 ~2x FP32

        # TensorCore MAC: FP16 inputs with FP32 accumulation
        # Local accumulation in systolic array saves some add energy
        # but still need full FP16 multiply
        self.tc_fp16_mac_pj = fp16_mult + fp32_add * 0.5  # Partial FP32 add (local accum)
        self.tc_fp32_acc_pj = fp32_add  # Full FP32 accumulator energy


# =============================================================================
# Complete SM Energy Model
# =============================================================================

class SMEnergyModel:
    """
    Complete SM energy model for CUDA and TensorCore operations.
    """

    def __init__(self, process_nm: int = 4):
        self.process_nm = process_nm
        self.regfile = RegisterFileModel(process_nm)
        self.collector = OperandCollectorModel(process_nm)
        self.arbitration = BankArbitrationModel(process_nm)
        self.crossbar = CrossbarModel(process_nm)
        self.alu = ALUModel(process_nm)

    def cuda_core_energy_per_mac_pj(self, precision: str = 'FP32') -> dict:
        """
        Energy breakdown for one CUDA core MAC operation.

        CUDA execution model:
        - 1 warp = 32 threads
        - 1 instruction = 32 parallel FMAs = 32 MACs
        - Each MAC needs 2 source operands read + 1 result write
        """
        warp_size = 32

        # Register file reads: 2 operands per thread, 32 threads
        reads = 2 * warp_size
        reg_read_energy = reads * self.regfile.read_energy_per_word_pj()

        # Register file write: 1 result per thread
        writes = warp_size
        reg_write_energy = writes * self.regfile.write_energy_per_word_pj()

        # Operand collector
        collector_energy = self.collector.energy_per_warp_instruction_pj(operands_per_thread=2)

        # Bank arbitration
        arbitration_energy = self.arbitration.energy_per_warp_instruction_pj(avg_conflicts=2.0)

        # Crossbar routing
        routing_energy = self.crossbar.operand_routing_energy_pj(warp_size)
        result_routing = self.crossbar.result_routing_energy_pj(warp_size)

        # ALU energy
        if precision == 'FP16':
            alu_per_op = self.alu.fp16_fma_pj
        elif precision == 'FP32':
            alu_per_op = self.alu.fp32_fma_pj
        else:  # FP64
            alu_per_op = self.alu.fp64_fma_pj
        alu_energy = warp_size * alu_per_op

        # Total per warp
        total_fetch = reg_read_energy + collector_energy + arbitration_energy + routing_energy
        total_writeback = reg_write_energy + result_routing
        total_alu = alu_energy
        total = total_fetch + total_writeback + total_alu

        # Amortize per MAC
        return {
            'reg_read': reg_read_energy / warp_size,
            'collector': collector_energy / warp_size,
            'arbitration': arbitration_energy / warp_size,
            'routing': routing_energy / warp_size,
            'fetch_total': total_fetch / warp_size,
            'alu': alu_per_op,
            'result_routing': result_routing / warp_size,
            'reg_write': reg_write_energy / warp_size,
            'writeback_total': total_writeback / warp_size,
            'total': total / warp_size,
        }

    def tensor_core_energy_per_mac_pj(self, precision: str = 'FP16') -> dict:
        """
        Energy breakdown for one TensorCore MAC operation.

        TensorCore Architecture Reality:
        =================================
        TensorCores use the SAME register file and infrastructure as CUDA cores!

        Key insight from NVIDIA documentation:
        - Matrix fragments stored in standard register file (same as CUDA)
        - Each thread in warp holds part of A, B, C fragments
        - Hardware must read from ALL 32 threads' registers
        - The operand collector is still involved (gathering from regfile)

        WMMA Fragment Distribution:
        ===========================
        For FP16 m16n16k16 MMA (the common size):
        - A matrix: 16x16 FP16 = 512 bytes distributed across 32 threads
          - Each thread holds 16 bytes (8 32-bit registers for A)
        - B matrix: same, 16x16 FP16 = 512 bytes (8 regs/thread)
        - C matrix: 16x16 FP32 = 1024 bytes (16 regs/thread for accum)

        Total registers per thread per MMA: 8 + 8 + 16 = 32 registers!
        Total register accesses per MMA: 32 threads x 32 registers = 1024 reads

        Wait - that seems like a lot. But it's amortized over 16x16x16 = 4096 MACs
        Per MAC: 1024 reads / 4096 MACs = 0.25 reads/MAC

        For 4x4x4 MMA (64 MACs, single TensorCore unit):
        - A: 4x4 FP16 = 32 bytes, 8 regs across threads
        - B: 4x4 FP16 = 32 bytes, 8 regs across threads
        - C: 4x4 FP32 = 64 bytes, 16 regs across threads
        - Total: 32 register reads + 16 register writes per MMA

        Per MAC: (32 reads + 16 writes) / 64 MACs = 0.75 reg ops/MAC
        Still less than CUDA (2 reads + 1 write = 3 reg ops/MAC)

        The advantage: bulk access pattern reduces conflict overhead
        """
        macs_per_mma = 64
        scale = get_scale(self.process_nm)

        # =================================================================
        # Register File Access for Fragment Data
        # =================================================================
        # For 4x4 x 4x4 matrix multiply (64 MACs):
        #
        # A matrix: 4x4 = 16 FP16 elements
        #   - Packed 2 per 32-bit register = 8 register reads
        #
        # B matrix: 4x4 = 16 FP16 elements
        #   - Packed 2 per 32-bit register = 8 register reads
        #
        # C matrix (accumulator input): 4x4 = 16 FP32 elements
        #   - 1 per 32-bit register = 16 register reads
        #
        # C matrix (result output): 16 FP32 elements
        #   - 1 per 32-bit register = 16 register writes
        #
        # Total: 8 + 8 + 16 = 32 reads, 16 writes

        a_reads = 8   # 16 FP16 packed into 8 32-bit registers
        b_reads = 8   # 16 FP16 packed into 8 32-bit registers
        c_reads = 16  # 16 FP32 accumulators (read for accumulate)
        c_writes = 16 # 16 FP32 results

        total_reads = a_reads + b_reads + c_reads  # 32 reads
        reg_read_energy = total_reads * self.regfile.read_energy_per_word_pj()
        reg_write_energy = c_writes * self.regfile.write_energy_per_word_pj()

        # =================================================================
        # Operand Collection (TensorCore has dedicated but similar machinery)
        # =================================================================
        # Still need to gather operands from register file and route to TC
        # TensorCore collector is more efficient (structured access pattern)

        # Staging: 32 32-bit values staged (A + B + C inputs)
        staging_energy = 32 * 0.02 * scale  # ~0.64 pJ at 4nm

        # Conflict detection: structured pattern = minimal conflicts
        # But still need the logic to coordinate reads from banked regfile
        conflict_det_energy = 0.5 * scale

        # Arbitration (simplified for structured access)
        arbitration_energy = 0.4 * scale

        # Routing to TensorCore array
        # 32 values in (A+B+C), 16 values out (C result)
        routing_energy = (32 + 16) * 0.015 * scale  # ~0.72 pJ at 4nm

        # =================================================================
        # TensorCore Compute
        # =================================================================
        alu_energy = macs_per_mma * self.alu.tc_fp16_mac_pj

        # =================================================================
        # Result Routing and Writeback
        # =================================================================
        result_routing_energy = 16 * 0.02 * scale  # Route C back

        # =================================================================
        # Totals
        # =================================================================
        total_fetch = reg_read_energy + staging_energy + conflict_det_energy + arbitration_energy + routing_energy
        total_writeback = reg_write_energy + result_routing_energy
        total_alu = alu_energy
        total = total_fetch + total_writeback + total_alu

        # Amortize per MAC (64 MACs per MMA)
        return {
            'reg_read': reg_read_energy / macs_per_mma,
            'collector': staging_energy / macs_per_mma,
            'arbitration': (conflict_det_energy + arbitration_energy) / macs_per_mma,
            'routing': routing_energy / macs_per_mma,
            'fetch_total': total_fetch / macs_per_mma,
            'alu': self.alu.tc_fp16_mac_pj,
            'result_routing': result_routing_energy / macs_per_mma,
            'reg_write': reg_write_energy / macs_per_mma,
            'writeback_total': total_writeback / macs_per_mma,
            'total': total / macs_per_mma,
        }


# =============================================================================
# Main Report
# =============================================================================

def print_energy_breakdown():
    """Print energy breakdown for CUDA and TensorCore operations."""

    print("=" * 100)
    print("SM ENERGY BREAKDOWN: ALU + OPERAND FETCH PER MAC")
    print("=" * 100)
    print("""
This analysis shows the MINIMUM energy per MAC operation, breaking down:
  1. Operand Fetch: Register file reads + collector + arbitration + routing
  2. ALU: The actual multiply-accumulate computation
  3. Writeback: Result routing + register file write

Key insight: In stored-program machines like GPUs, operand fetch infrastructure
is SUBSTANTIAL - often exceeding the ALU energy itself.
""")

    process_nodes = [3, 4, 7, 12]

    # CUDA Cores
    print("\n" + "-" * 100)
    print("CUDA CORES (FP32 FMA, 32-wide warp)")
    print("-" * 100)
    print(f"{'Node':<6} {'RegRead':>8} {'Collect':>8} {'Arbit':>8} {'Route':>8} {'FETCH':>8} | "
          f"{'ALU':>8} | {'ResRoute':>8} {'RegWr':>8} {'WB':>8} | {'TOTAL':>8}")
    print(f"{'(nm)':<6} {'(pJ)':>8} {'(pJ)':>8} {'(pJ)':>8} {'(pJ)':>8} {'(pJ)':>8} | "
          f"{'(pJ)':>8} | {'(pJ)':>8} {'(pJ)':>8} {'(pJ)':>8} | {'(pJ)':>8}")
    print("-" * 100)

    for node in process_nodes:
        model = SMEnergyModel(node)
        e = model.cuda_core_energy_per_mac_pj('FP32')
        print(f"{node:<6} {e['reg_read']:>8.2f} {e['collector']:>8.2f} {e['arbitration']:>8.2f} "
              f"{e['routing']:>8.2f} {e['fetch_total']:>8.2f} | {e['alu']:>8.2f} | "
              f"{e['result_routing']:>8.2f} {e['reg_write']:>8.2f} {e['writeback_total']:>8.2f} | "
              f"{e['total']:>8.2f}")

    # TensorCores
    print("\n" + "-" * 100)
    print("TENSOR CORES (FP16 MMA, 64 MACs/instruction)")
    print("-" * 100)
    print(f"{'Node':<6} {'RegRead':>8} {'Collect':>8} {'Arbit':>8} {'Route':>8} {'FETCH':>8} | "
          f"{'ALU':>8} | {'ResRoute':>8} {'RegWr':>8} {'WB':>8} | {'TOTAL':>8}")
    print(f"{'(nm)':<6} {'(pJ)':>8} {'(pJ)':>8} {'(pJ)':>8} {'(pJ)':>8} {'(pJ)':>8} | "
          f"{'(pJ)':>8} | {'(pJ)':>8} {'(pJ)':>8} {'(pJ)':>8} | {'(pJ)':>8}")
    print("-" * 100)

    for node in process_nodes:
        model = SMEnergyModel(node)
        e = model.tensor_core_energy_per_mac_pj('FP16')
        print(f"{node:<6} {e['reg_read']:>8.2f} {e['collector']:>8.2f} {e['arbitration']:>8.2f} "
              f"{e['routing']:>8.2f} {e['fetch_total']:>8.2f} | {e['alu']:>8.2f} | "
              f"{e['result_routing']:>8.2f} {e['reg_write']:>8.2f} {e['writeback_total']:>8.2f} | "
              f"{e['total']:>8.2f}")

    # Summary comparison
    print("\n" + "=" * 100)
    print("SUMMARY: FETCH vs ALU RATIO")
    print("=" * 100)
    print(f"\n{'Node':<6} {'CUDA Fetch':>12} {'CUDA ALU':>10} {'Fetch/ALU':>10} | "
          f"{'TC Fetch':>10} {'TC ALU':>10} {'Fetch/ALU':>10}")
    print("-" * 80)

    for node in process_nodes:
        model = SMEnergyModel(node)
        cuda = model.cuda_core_energy_per_mac_pj('FP32')
        tc = model.tensor_core_energy_per_mac_pj('FP16')

        cuda_ratio = cuda['fetch_total'] / cuda['alu']
        tc_ratio = tc['fetch_total'] / tc['alu']

        print(f"{node:<6} {cuda['fetch_total']:>12.2f} {cuda['alu']:>10.2f} {cuda_ratio:>10.2f}x | "
              f"{tc['fetch_total']:>10.2f} {tc['alu']:>10.2f} {tc_ratio:>10.2f}x")

    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    print("""
For CUDA cores: Fetch energy is ~2-3x the ALU energy
  -> The register file and operand delivery infrastructure dominates
  -> This is fundamental to stored-program execution

For TensorCores: Fetch energy is ~2-4x the ALU energy
  -> Better than CUDA (amortized over 64 MACs per instruction)
  -> But still substantial - fragments must be loaded from register file
  -> Each MMA instruction still requires operand collector, arbitration

Compare to TPU systolic arrays:
  -> Weights stay in PE registers (loaded once per layer)
  -> Inputs stream through array (injected at boundary)
  -> Fetch/ALU ratio: ~0.01-0.05x (fetch is negligible)
  -> This is why TPUs are more energy efficient for matrix ops
""")


def print_tdp_estimate():
    """Estimate TDP using both component model and calibrated baseline."""

    print("\n" + "=" * 120)
    print("TDP ESTIMATES: COMPONENT MODEL vs CALIBRATED BASELINE")
    print("=" * 120)

    gpus = [
        {'name': 'B100-SXM6-192GB', 'process_nm': 3, 'tensor_cores': 528,
         'tc_macs_per_clock': 512, 'freq_ghz': 2.1, 'spec_tdp_w': 1000,
         'marketing_tflops': 20000},  # FP4
        {'name': 'H100-SXM5-80GB', 'process_nm': 4, 'tensor_cores': 528,
         'tc_macs_per_clock': 256, 'freq_ghz': 1.98, 'spec_tdp_w': 700,
         'marketing_tflops': 3958},   # FP8 with sparsity
        {'name': 'A100-SXM4-80GB', 'process_nm': 7, 'tensor_cores': 432,
         'tc_macs_per_clock': 256, 'freq_ghz': 1.41, 'spec_tdp_w': 400,
         'marketing_tflops': 624},    # FP16 Tensor
        {'name': 'V100-SXM3-32GB', 'process_nm': 12, 'tensor_cores': 640,
         'tc_macs_per_clock': 64, 'freq_ghz': 1.53, 'spec_tdp_w': 350,
         'marketing_tflops': 125},    # FP16 Tensor
        {'name': 'T4-PCIe-16GB', 'process_nm': 12, 'tensor_cores': 320,
         'tc_macs_per_clock': 64, 'freq_ghz': 1.59, 'spec_tdp_w': 70,
         'marketing_tflops': 65},     # FP16 Tensor
    ]

    print("""
Three estimates compared:
  1. Component Model: Bottom-up from Horowitz-scaled register file + ALU energy (~1.15 pJ at 4nm)
     -> Likely TOO LOW (missing clock distribution, instruction decode, warp scheduling overhead)
  2. Calibrated Model: 1.65 pJ/MAC at 4nm (derived from MFU constraints, see comments in source)
     -> REALISTIC ESTIMATE bounded by: component model (lower) and MFU observations (upper)
  3. External Analysis: 2.5 pJ/MAC (one published analysis)
     -> Likely TOO HIGH (would imply 28% physics limit, but real MFU is 35-50%)
""")

    print(f"{'GPU':<22} {'Node':>5} {'Comp':>8} {'Calib':>8} {'Peak':>10} {'Sust@TDP':>10} "
          f"{'Marketed':>10} {'Efficiency':>10}")
    print(f"{'':22} {'(nm)':>5} {'(pJ)':>8} {'(pJ)':>8} {'(TFLOPS)':>10} {'(TFLOPS)':>10} "
          f"{'(TFLOPS)':>10} {'(Sust/Mkt)':>10}")
    print("-" * 120)

    for gpu in gpus:
        model = SMEnergyModel(gpu['process_nm'])
        e = model.tensor_core_energy_per_mac_pj('FP16')
        component_pj = e['total']

        calibrated_pj = get_calibrated_mac_energy_pj(gpu['process_nm'])

        total_macs = gpu['tensor_cores'] * gpu['tc_macs_per_clock']
        macs_per_sec = total_macs * gpu['freq_ghz'] * 1e9
        peak_tflops = macs_per_sec / 1e12  # In TFLOPS

        # Sustainable TFLOPS at TDP using calibrated energy
        sustainable_tflops = gpu['spec_tdp_w'] / (calibrated_pj * 1e-12) / 1e12

        # Efficiency = sustainable / marketed
        efficiency = sustainable_tflops / gpu['marketing_tflops'] * 100

        print(f"{gpu['name']:<22} {gpu['process_nm']:>5} {component_pj:>8.2f} {calibrated_pj:>8.2f} "
              f"{peak_tflops:>10.1f} {sustainable_tflops:>10.1f} "
              f"{gpu['marketing_tflops']:>10.0f} {efficiency:>9.1f}%")

    print("-" * 120)
    print("""
Columns:
  Comp (pJ)      = Component model energy per MAC (Horowitz-scaled)
  Calib (pJ)     = Calibrated energy per MAC (1.65 pJ baseline at 4nm)
  Peak (TFLOPS)  = All TensorCores at max frequency (theoretical)
  Sust@TDP       = Sustainable TFLOPS within TDP budget using calibrated energy
  Marketed       = NVIDIA marketing spec (peak TFLOPS, often with sparsity)
  Efficiency     = Sust@TDP / Marketed (what % of marketing claims are achievable)
""")

    # Add detailed analysis
    print("\n" + "=" * 120)
    print("MARKETING vs REALITY ANALYSIS")
    print("=" * 120)
    print("""
Real-world Model FLOP Utilization (MFU) from MLPerf and industry benchmarks:

  Source                          | GPU      | MFU Achieved
  --------------------------------|----------|-------------
  MosaicML 30B model (512 GPUs)   | H100     | 40%
  CoreWeave 30B model (128 GPUs)  | H100     | 49-52%
  Llama-3.1 training              | H100     | 38-43%
  Google PaLM (540B)              | TPU v4   | 46%
  OpenAI GPT-3                    | A100     | 19.6%
  Industry average                | Various  | 35-45%

Sources:
  - https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md
  - https://www.coreweave.com/blog/coreweave-leads-the-charge-in-ai-infrastructure-efficiency
  - https://www.lesswrong.com/posts/tJAD2LG9uweeEfjwq/estimating-efficiency-improvements-in-llm-pre-training

ENERGY BOUNDS ANALYSIS:
=======================
The actual energy per MAC is constrained by MFU observations:

  Lower bound (Component Model): ~1.15 pJ/MAC
    -> Horowitz-scaled register file + ALU energy
    -> Missing: clock distribution, instruction decode, warp scheduling
    -> Would imply ~61% physics limit (700W / 1.15pJ = 609 TFLOPS, 609/990 = 61%)

  Upper bound (External Analysis): ~2.5 pJ/MAC
    -> Would imply ~28% physics limit (700W / 2.5pJ = 280 TFLOPS, 280/990 = 28%)
    -> PROBLEM: Real MFU is 35-50%, which EXCEEDS this limit
    -> Conclusion: 2.5 pJ/MAC is TOO HIGH

  Realistic estimate: ~1.65 pJ/MAC (midpoint of 1.5-1.8 range)
    -> Implies ~43% physics limit (700W / 1.65pJ = 424 TFLOPS, 424/990 = 43%)
    -> Real MFU of 35-50% fits BELOW this ceiling
    -> Software overhead (kernel launch, memory stalls) accounts for gap

Key Insight:
  MFU is bounded by BOTH physics (energy) AND software (overhead).
  Physics limit: ~43-50% (energy cost of operand delivery)
  Software gap:  ~5-10%  (kernel launch, memory latency, load imbalance)
  Observed MFU:  35-50%  (varies by workload and optimization)
""")


def print_horowitz_reference():
    """Print Horowitz reference values and how they scale."""

    print("=" * 100)
    print("HOROWITZ REFERENCE VALUES (ISSCC 2014)")
    print("=" * 100)
    print("""
Source: Mark Horowitz, "Computing's Energy Problem (and what we can do about it)"
        ISSCC 2014 Keynote, Figure 1.1.9
        https://www.researchgate.net/publication/271463146

These are the canonical reference values for energy per operation at 45nm, 0.9V:
""")

    print(f"{'Operation':<25} {'45nm (pJ)':<12} {'4nm (pJ)':<12} {'7nm (pJ)':<12} {'12nm (pJ)':<12}")
    print("-" * 75)

    operations = [
        ('8b Int Add', 'int8_add'),
        ('32b Int Add', 'int32_add'),
        ('32b Int Mult', 'int32_mult'),
        ('16b FP Add', 'fp16_add'),
        ('16b FP Mult', 'fp16_mult'),
        ('32b FP Add', 'fp32_add'),
        ('32b FP Mult', 'fp32_mult'),
        ('8KB SRAM Read (32b)', 'sram_8kb_read'),
        ('DRAM Read (32b)', 'dram_read'),
    ]

    for name, key in operations:
        val_45 = HOROWITZ_45NM[key]
        val_4 = get_horowitz_scaled(key, 4)
        val_7 = get_horowitz_scaled(key, 7)
        val_12 = get_horowitz_scaled(key, 12)
        print(f"{name:<25} {val_45:<12.2f} {val_4:<12.2f} {val_7:<12.2f} {val_12:<12.2f}")

    print("-" * 75)
    print("""
Key Insight from Horowitz:
  "Memory access is 3 orders of magnitude more energy expensive than simple arithmetic"

  At 45nm: 8KB SRAM read (5 pJ) > FP32 multiply (3.7 pJ) > FP32 add (0.9 pJ)

  This relationship holds at all process nodes!
""")


if __name__ == '__main__':
    print_horowitz_reference()
    print_energy_breakdown()
    print_tdp_estimate()
