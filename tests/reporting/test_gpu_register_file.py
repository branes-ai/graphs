"""Tests for the GPU register-file (banked SRAM) model.

Verifies that:
- Default Ampere geometry: 64 KiB / 4 banks of 16 KiB / 1024-bit wide.
- Per-bank read / write energy uses the profile's RF SRAM coefficient.
- Wide-bank Ampere config gives 1 read per warp source operand.
- Narrower banks correctly multiply the read count.
- Bank writes cost ``write_to_read_ratio`` x bank reads.
- SM-level activity counts (sources x subparts x reads_per_source).
"""
from __future__ import annotations

import pytest

from graphs.hardware.technology_profile import (
    EDGE_8NM_LPDDR5,
    DATACENTER_4NM_HBM3,
)
from graphs.reporting.gpu_register_file import (
    DEFAULT_BANK_WIDTH_BITS,
    DEFAULT_BYTES_PER_SUBPART,
    DEFAULT_NUM_BANKS,
    GPURegisterFileBankModel,
    default_ampere_subpartition_rf,
)


class TestGeometry:
    def test_default_ampere_geometry(self):
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        assert rf.bytes_per_subpartition == DEFAULT_BYTES_PER_SUBPART
        assert rf.num_banks == DEFAULT_NUM_BANKS
        assert rf.bank_width_bits == DEFAULT_BANK_WIDTH_BITS
        assert rf.bytes_per_subpartition == 64 * 1024
        assert rf.num_banks == 4
        assert rf.bank_width_bits == 1024

    def test_bank_size_bytes(self):
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        # 64 KiB / 4 banks = 16 KiB per bank
        assert rf.bank_size_bytes == 16 * 1024

    def test_bytes_per_bank_access(self):
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        # 1024 bits / 8 = 128 bytes per access
        assert rf.bytes_per_bank_access == 128


class TestConcurrency:
    def test_perfect_fit_one_read_per_source(self):
        """Ampere bank width matches a 32-thread x 32-bit warp
        operand exactly -> 1 wide-bank read per source."""
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        assert rf.reads_per_warp_source(
            threads_per_warp=32, bits_per_thread=32,
        ) == 1

    def test_narrow_bank_multiplies_reads(self):
        """If the bank is half as wide, you need 2 reads per source."""
        rf = GPURegisterFileBankModel.for_profile(
            EDGE_8NM_LPDDR5,
            bank_width_bits=512,
        )
        assert rf.reads_per_warp_source(32, 32) == 2

    def test_quarter_bank_quadruples_reads(self):
        rf = GPURegisterFileBankModel.for_profile(
            EDGE_8NM_LPDDR5,
            bank_width_bits=256,
        )
        assert rf.reads_per_warp_source(32, 32) == 4

    def test_minimum_one_read(self):
        """Even with a comically wide bank, reads_per_warp_source
        should be at least 1."""
        rf = GPURegisterFileBankModel.for_profile(
            EDGE_8NM_LPDDR5,
            bank_width_bits=8192,
        )
        assert rf.reads_per_warp_source(32, 32) == 1


class TestEnergy:
    def test_bank_read_energy_uses_profile_sram(self):
        """Bank read energy = bytes_per_access x sram_per_byte."""
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        expected = rf.bytes_per_bank_access * rf.sram_energy_per_byte_pj
        assert rf.bank_read_energy_pj() == pytest.approx(expected)

    def test_bank_write_costlier_than_read(self):
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        assert rf.bank_write_energy_pj() > rf.bank_read_energy_pj()

    def test_bank_write_to_read_ratio(self):
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        ratio = rf.bank_write_energy_pj() / rf.bank_read_energy_pj()
        assert ratio == pytest.approx(rf.write_to_read_ratio)

    def test_smaller_node_lower_energy(self):
        rf_8nm = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        rf_4nm = default_ampere_subpartition_rf(DATACENTER_4NM_HBM3)
        # SRAM scales linearly with node; smaller is cheaper.
        assert rf_4nm.bank_read_energy_pj() < rf_8nm.bank_read_energy_pj()

    def test_wider_bank_higher_energy_per_access(self):
        """A wider bank moves more bytes per access, costing more."""
        narrow = GPURegisterFileBankModel.for_profile(
            EDGE_8NM_LPDDR5, bank_width_bits=256,
        )
        wide = GPURegisterFileBankModel.for_profile(
            EDGE_8NM_LPDDR5, bank_width_bits=1024,
        )
        assert wide.bank_read_energy_pj() > narrow.bank_read_energy_pj()
        # 4x wider = 4x energy (linear-with-bytes model)
        assert (wide.bank_read_energy_pj()
                / narrow.bank_read_energy_pj()) == pytest.approx(4.0)


class TestSMLevelActivityCounts:
    def test_sm_bank_reads_default_ampere(self):
        """Default Ampere: 4 subparts x sources x 1 read-per-source."""
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        assert rf.sm_bank_reads_per_instruction(4, 2) == 8   # FMUL/FADD
        assert rf.sm_bank_reads_per_instruction(4, 3) == 12  # FMA

    def test_sm_bank_writes_default_ampere(self):
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        # 1 dest per warp x 4 subpartitions x 1 read-per-source = 4
        assert rf.sm_bank_writes_per_instruction(4) == 4

    def test_narrow_bank_doubles_sm_reads(self):
        rf = GPURegisterFileBankModel.for_profile(
            EDGE_8NM_LPDDR5, bank_width_bits=512,
        )
        # 4 subparts x 3 sources x 2 reads-per-source = 24
        assert rf.sm_bank_reads_per_instruction(4, 3) == 24


class TestPlausibility:
    def test_8nm_bank_read_in_range(self):
        """For Ampere wide-bank reads at 8nm, expect 30-80 pJ.
        Below = under-counts the SIMT cost; above = primitive bug."""
        rf = default_ampere_subpartition_rf(EDGE_8NM_LPDDR5)
        e = rf.bank_read_energy_pj()
        assert 30 <= e <= 80, (
            f"per-bank-read at 8nm = {e:.1f} pJ, outside [30, 80]"
        )
