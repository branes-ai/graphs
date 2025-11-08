"""
KPU-T64 Resource Model - Edge AI Accelerator

Stillwater KPU-T64 (Knowledge Processing Unit)
Domain Flow Architecture (DFA)

Target: Edge AI / IoT applications
Process: 22nm
TDP: 5-15W
Product line: Edge inference

Architecture:
- 64 tiles (8×8 mesh)
- 16 PEs per tile = 1,024 total PEs
- Programmable SURE execution
- 4-stage memory hierarchy (DRAM → L3 → L2 → L1)
- Token-based spatial dataflow
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ClockDomain,
    ComputeResource,
    TileSpecialization,
    KPUComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
)
from ...architectural_energy import KPUTileEnergyModel


def kpu_t64_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T64 resource model.

    Domain Flow Architecture (DFA) - Edge AI accelerator.

    Key characteristics:
    - 64 tiles, 1,024 PEs (16 PEs/tile)
    - 8×8 2D mesh topology
    - Programmable SURE execution (all BLAS operators)
    - Token-based spatial dataflow
    - 4-stage memory hierarchy
    - 5-15W TDP (edge/IoT)
    """

    # Thermal operating point (edge deployment)
    thermal_low_power = ThermalOperatingPoint(
        name="low_power",
        tdp_watts=5.0,  # Minimal power mode
        cooling_solution="passive",
        performance_specs={}
    )

    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=10.0,  # Standard operation
        cooling_solution="passive",
        performance_specs={}
    )

    thermal_turbo = ThermalOperatingPoint(
        name="turbo",
        tdp_watts=15.0,  # Maximum performance
        cooling_solution="active-fan",
        performance_specs={}
    )

    # KPU-T64 tile energy model
    tile_energy_model = KPUTileEnergyModel(
        # Processing element configuration
        num_tiles=64,
        pes_per_tile=16,
        tile_mesh_dimensions=(8, 8),  # 8×8 mesh

        # Memory hierarchy (4-stage, DDR4)
        dram_bandwidth_gb_s=25.6,  # 25.6 GB/s (DDR4-3200)
        l3_size_per_tile=256 * 1024,  # 256 KiB per tile (16 MiB total)
        l2_size_per_tile=32 * 1024,  # 32 KiB per tile (2 MiB total)
        l1_size_per_pe=4 * 1024,  # 4 KiB per PE (64 KiB per tile)

        # Token-based execution
        token_payload_bytes=64,  # 64-byte tokens
        token_signature_bytes=16,  # 16-byte signature
        max_tokens_in_flight=1024,  # Per tile

        # SURE program configuration
        sure_program_size_bytes=4096,  # 4 KiB typical
        sure_program_cache_size=16,  # 16 programs cached

        # Data movement engines
        dma_engines_per_tile=4,  # DRAM ↔ L3
        blockmover_engines_per_tile=2,  # L3 ↔ L2 (inter-tile)
        streamer_engines_per_tile=4,  # L2 ↔ L1 (intra-tile)

        # Clock frequency (22nm process)
        clock_frequency_hz=800e6,  # 800 MHz

        # Memory hierarchy energy (22nm, DDR4)
        dram_read_energy_per_byte=10.0e-12,  # 10 pJ (DDR4)
        dram_write_energy_per_byte=12.0e-12,  # 12 pJ (DDR4)
        l3_read_energy_per_byte=2.0e-12,  # 2 pJ (distributed SRAM)
        l3_write_energy_per_byte=2.5e-12,  # 2.5 pJ
        l2_read_energy_per_byte=0.8e-12,  # 0.8 pJ (tile-local)
        l2_write_energy_per_byte=1.0e-12,  # 1.0 pJ
        l1_read_energy_per_byte=0.3e-12,  # 0.3 pJ (PE-local)
        l1_write_energy_per_byte=0.4e-12,  # 0.4 pJ

        # Data movement engine energy
        dma_transfer_energy_per_byte=1.5e-12,  # 1.5 pJ
        blockmover_energy_per_byte=0.8e-12,  # 0.8 pJ
        streamer_energy_per_byte=0.3e-12,  # 0.3 pJ

        # Token routing energy (UNIQUE TO KPU)
        token_signature_matching_energy=0.6e-12,  # 0.6 pJ (CAM-like)
        token_handshake_energy=0.2e-12,  # 0.2 pJ
        token_routing_per_hop=0.15e-12,  # 0.15 pJ per hop

        # SURE program management
        sure_program_load_energy=50e-12,  # 50 pJ broadcast
        sure_program_cache_hit_energy=1e-12,  # 1 pJ cache hit

        # L3 distributed scratchpad routing
        l3_routing_distance_factor=1.2,
        l3_noc_energy_per_hop=0.5e-12,  # 0.5 pJ per NoC hop

        # Computation energy (22nm, slightly higher than advanced nodes)
        mac_energy_int8=0.35e-12,  # 0.35 pJ (INT8)
        mac_energy_bf16=0.52e-12,  # 0.52 pJ (BF16, 1.5× INT8)
        mac_energy_fp32=1.05e-12,  # 1.05 pJ (FP32, 3× INT8)

        # Operator fusion benefits
        fusion_l2_traffic_reduction=0.7,  # 70% reduction
        fusion_coordination_overhead=5e-12,  # 5 pJ per boundary
    )

    # Performance calculation (1,024 PEs × 800 MHz × 2 ops/cycle)
    # INT8: 1,638 GOPS
    # BF16: 1,638 GFLOPS
    # FP32: 819 GFLOPS (half throughput)
    peak_ops_int8 = 1024 * 800e6 * 2  # 1,638 GOPS
    peak_ops_bf16 = 1024 * 800e6 * 2  # 1,638 GFLOPS
    peak_ops_fp32 = 1024 * 800e6  # 819 GFLOPS (1 FP32 op/cycle)

    model = HardwareResourceModel(
        name="KPU-T64",
        hardware_type=HardwareType.KPU,
        compute_units=64,  # 64 tiles
        threads_per_unit=16,  # 16 PEs per tile
        warps_per_unit=4,  # 4 PE clusters per tile
        warp_size=4,  # 4 PEs per cluster

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_ops_fp32,
                tensor_core_supported=False,  # Programmable PEs
                relative_speedup=0.5,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_ops_bf16,
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_ops_int8,
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=25.6e9,  # 25.6 GB/s (DDR4-3200)
        l1_cache_per_unit=64 * 1024,  # 64 KiB per tile (16 PEs × 4 KiB)
        l2_cache_total=2 * 1024 * 1024,  # 2 MiB total (64 tiles × 32 KiB)
        main_memory=2 * 1024**3,  # 2 GB DDR4 (typical edge config)
        energy_per_flop_fp32=1.05e-12,  # 1.05 pJ per FP32 MAC
        energy_per_byte=10e-12,  # 10 pJ per byte (DDR4)
        min_occupancy=0.6,  # Spatial dataflow benefits from high utilization
        max_concurrent_kernels=4,  # Can run 4 independent SURE programs
        wave_quantization=1,  # Spatial architecture, no wave quantization

        # Thermal profiles
        thermal_operating_points={
            "low_power": thermal_low_power,
            "default": thermal_default,
            "turbo": thermal_turbo,
        },
        default_thermal_profile="default",
    )

    # Attach tile energy model
    model.tile_energy_model = tile_energy_model

    return model
