"""
GPU register file model -- banked SRAM, the key SIMT energy
differentiator.

A SIMT pipeline only works if many warps are in flight concurrently;
that requires hundreds-to-thousands of registers per subpartition,
which can only be provided by SRAM, not flip-flops. The SRAM is
banked so that multiple operands can be read in parallel each cycle
to feed the lane-wide datapath. This banked-SRAM cost IS the GPU's
architectural overhead vs accelerators (TPU / KPU / CGRA) that
either eliminate the general-purpose register file (systolic data
flows bank-to-bank) or replace it with FIFOs (dataflow streams).

This module models that cost as concrete bank reads and writes,
parameterised by a TechnologyProfile so all process-node scaling
flows from the documented source of truth.

The default config matches an Ampere subpartition (Jetson Orin
GA10B): 64 KiB / 4 banks of 16 KiB / 1024-bit wide port (32 threads
x 32 bits = one warp's worth of one source operand per cycle).

Energy model (per wide-bank access):

    bank_read_energy_pj   = (bank_width_bits / 8) x sram_per_byte
    bank_write_energy_pj  = bank_read_energy_pj x write_to_read_ratio

The per-byte coefficient comes from
``get_sram_energy_per_byte_pj(profile.process_node_nm, 'register_file')``
-- a small, fast, multi-ported SRAM cell.

Concurrency:

    reads_per_warp_source = ceil(threads_per_warp x bits_per_thread
                                  / bank_width_bits)

For Ampere with bank_width_bits = 1024 and a 32-thread warp at
32 bits per thread, this is 1 wide-bank read per warp source operand
-- a "perfect-fit" bank width. Narrower banks would multiply this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from graphs.hardware.technology_profile import (
    TechnologyProfile,
    get_sram_energy_per_byte_pj,
)


# Ampere SM defaults (Jetson Orin GA10B).
DEFAULT_BYTES_PER_SUBPART = 64 * 1024     # 64 KiB
DEFAULT_NUM_BANKS = 4                      # banks per subpartition
DEFAULT_BANK_WIDTH_BITS = 1024             # one warp's source operand
DEFAULT_THREADS_PER_WARP = 32
DEFAULT_BITS_PER_THREAD = 32
DEFAULT_WRITE_TO_READ_RATIO = 1.25         # writes pay precharge


@dataclass
class GPURegisterFileBankModel:
    """Banked SRAM register file for one GPU subpartition.

    Sized at the subpartition level because each subpartition has its
    own private RF (no cross-subpart RF traffic in the SIMT pipeline).
    """
    bytes_per_subpartition: int = DEFAULT_BYTES_PER_SUBPART
    num_banks: int = DEFAULT_NUM_BANKS
    bank_width_bits: int = DEFAULT_BANK_WIDTH_BITS

    # Per-byte SRAM dynamic energy (set when the model is built from
    # a TechnologyProfile; can also be supplied directly).
    sram_energy_per_byte_pj: float = 0.0

    write_to_read_ratio: float = DEFAULT_WRITE_TO_READ_RATIO

    @classmethod
    def for_profile(
        cls,
        profile: TechnologyProfile,
        bytes_per_subpartition: int = DEFAULT_BYTES_PER_SUBPART,
        num_banks: int = DEFAULT_NUM_BANKS,
        bank_width_bits: int = DEFAULT_BANK_WIDTH_BITS,
    ) -> "GPURegisterFileBankModel":
        """Build a bank model whose per-byte SRAM cost comes from the
        TechnologyProfile's process node (canonical path for the
        SIMT report).

        Uses ``get_sram_energy_per_byte_pj(node, 'register_file')`` --
        the small/fast/multi-port flavour of SRAM, which is the right
        family for a GPU RF cell. The 'scratchpad' and 'l1_cache'
        flavours are larger and slower; using them here would
        over-estimate.
        """
        per_byte = get_sram_energy_per_byte_pj(
            profile.process_node_nm, 'register_file',
        )
        return cls(
            bytes_per_subpartition=bytes_per_subpartition,
            num_banks=num_banks,
            bank_width_bits=bank_width_bits,
            sram_energy_per_byte_pj=per_byte,
        )

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------
    @property
    def bank_size_bytes(self) -> int:
        return self.bytes_per_subpartition // self.num_banks

    @property
    def bytes_per_bank_access(self) -> int:
        return self.bank_width_bits // 8

    def reads_per_warp_source(
        self,
        threads_per_warp: int = DEFAULT_THREADS_PER_WARP,
        bits_per_thread: int = DEFAULT_BITS_PER_THREAD,
    ) -> int:
        """How many wide-bank reads to gather one source operand for
        one warp.

        For Ampere's 1024-bit wide bank and a 32-thread warp at
        32 bits per thread, this is 1 (perfect-fit). Narrower banks
        would require multiple reads.
        """
        warp_bits = threads_per_warp * bits_per_thread
        return max(1, math.ceil(warp_bits / self.bank_width_bits))

    # ------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------
    def bank_read_energy_pj(self) -> float:
        """Per-access dynamic energy of one wide-bank read.

        Models the cell-array + wire energy as bytes-accessed times
        the profile's per-byte SRAM coefficient. The decoder /
        sense-amp overhead is implicitly amortised into the per-byte
        figure (which itself is an averaged number from
        ``get_sram_energy_per_byte_pj``); we deliberately do NOT add
        a separate fixed overhead -- it would double-count what's
        already in the per-byte.
        """
        return self.bytes_per_bank_access * self.sram_energy_per_byte_pj

    def bank_write_energy_pj(self) -> float:
        """Per-access dynamic energy of one wide-bank write.

        Slightly more expensive than a read (precharge + write-back
        cycle on the bit lines). We use a fixed read-to-write ratio
        rather than a separate per-byte write energy; no public
        TechnologyProfile field distinguishes the two.
        """
        return self.bank_read_energy_pj() * self.write_to_read_ratio

    # ------------------------------------------------------------
    # Activity counts at SM level
    # ------------------------------------------------------------
    def sm_bank_reads_per_instruction(
        self,
        sm_subpartitions: int,
        sources_per_op: int,
        threads_per_warp: int = DEFAULT_THREADS_PER_WARP,
        bits_per_thread: int = DEFAULT_BITS_PER_THREAD,
    ) -> int:
        """Total wide-bank reads issued at the SM level for ONE
        SIMT instruction (one warp-instruction issuing on each of
        ``sm_subpartitions`` subpartitions).

        At Ampere defaults: subparts=4, sources=3 (FMA),
        reads_per_source=1 -> 12 wide-bank reads per cycle.
        """
        per_warp = self.reads_per_warp_source(
            threads_per_warp, bits_per_thread,
        )
        return sm_subpartitions * sources_per_op * per_warp

    def sm_bank_writes_per_instruction(
        self,
        sm_subpartitions: int,
        threads_per_warp: int = DEFAULT_THREADS_PER_WARP,
        bits_per_thread: int = DEFAULT_BITS_PER_THREAD,
    ) -> int:
        """Total wide-bank writes for one SIMT instruction (one
        destination operand per warp-instruction). Same wide-bank
        accounting as reads."""
        per_warp = self.reads_per_warp_source(
            threads_per_warp, bits_per_thread,
        )
        return sm_subpartitions * per_warp


def default_ampere_subpartition_rf(
    profile: TechnologyProfile,
) -> GPURegisterFileBankModel:
    """Convenience: Ampere subpartition RF parameterised on a
    profile."""
    return GPURegisterFileBankModel.for_profile(
        profile,
        bytes_per_subpartition=DEFAULT_BYTES_PER_SUBPART,
        num_banks=DEFAULT_NUM_BANKS,
        bank_width_bits=DEFAULT_BANK_WIDTH_BITS,
    )


__all__ = [
    "DEFAULT_BYTES_PER_SUBPART",
    "DEFAULT_NUM_BANKS",
    "DEFAULT_BANK_WIDTH_BITS",
    "DEFAULT_THREADS_PER_WARP",
    "DEFAULT_BITS_PER_THREAD",
    "DEFAULT_WRITE_TO_READ_RATIO",
    "GPURegisterFileBankModel",
    "default_ampere_subpartition_rf",
]
