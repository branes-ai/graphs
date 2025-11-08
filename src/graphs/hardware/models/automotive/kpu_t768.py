"""
KPU-T768 Resource Model - Automotive/Datacenter AI Accelerator

Stillwater KPU-T768 (Knowledge Processing Unit)
Domain Flow Architecture (DFA)

Target: Autonomous vehicles / Edge datacenter
Process: 16nm / 7nm / 4nm variants
TDP: 75-250W
Product line: Automotive L4/L5, edge servers

Architecture:
- 768 tiles (24×32 mesh)
- 16 PEs per tile = 12,288 total PEs
- Programmable SURE execution
- 4-stage memory hierarchy (DRAM/HBM → L3 → L2 → L1)
- Token-based spatial dataflow
- High bandwidth (HBM2)
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


def kpu_t768_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T768 resource model.

    Domain Flow Architecture (DFA) - Automotive/Datacenter AI accelerator.

    Key characteristics:
    - 768 tiles, 12,288 PEs (16 PEs/tile)
    - 24×32 2D mesh topology
    - Programmable SURE execution (all BLAS operators)
    - Token-based spatial dataflow
    - 4-stage memory hierarchy
    - 75-250W TDP (automotive/datacenter)
    - 7nm or 4nm process
    - HBM2 memory
    """

    # Thermal operating point (automotive/datacenter)
    thermal_efficient = ThermalOperatingPoint(
        name="efficient",
        tdp_watts=75.0,  # Eco mode
        cooling_solution="active-liquid",
        performance_specs={}
    )

    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=150.0,  # Standard operation
        cooling_solution="active-liquid",
        performance_specs={}
    )

    thermal_performance = ThermalOperatingPoint(
        name="performance",
        tdp_watts=250.0,  # Maximum performance
        cooling_solution="active-liquid",
        performance_specs={}
    )

    # KPU-T768 tile energy model
    tile_energy_model = KPUTileEnergyModel(
        # Processing element configuration
        num_tiles=768,
        pes_per_tile=16,
        tile_mesh_dimensions=(24, 32),  # 24×32 mesh

        # Memory hierarchy (4-stage, HBM2)
        dram_bandwidth_gb_s=204.8,  # 204.8 GB/s (HBM2)
        l3_size_per_tile=256 * 1024,  # 256 KiB per tile (192 MiB total)
        l2_size_per_tile=32 * 1024,  # 32 KiB per tile (24 MiB total)
        l1_size_per_pe=4 * 1024,  # 4 KiB per PE (768 KiB per tile)

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

        # Clock frequency (7nm/4nm process)
        clock_frequency_hz=1500e6,  # 1.5 GHz

        # Memory hierarchy energy (7nm/4nm, HBM2)
        dram_read_energy_per_byte=5.0e-12,  # 5 pJ (HBM2, lower than DDR4)
        dram_write_energy_per_byte=6.0e-12,  # 6 pJ
        l3_read_energy_per_byte=1.2e-12,  # 1.2 pJ (advanced node)
        l3_write_energy_per_byte=1.5e-12,  # 1.5 pJ
        l2_read_energy_per_byte=0.5e-12,  # 0.5 pJ
        l2_write_energy_per_byte=0.6e-12,  # 0.6 pJ
        l1_read_energy_per_byte=0.2e-12,  # 0.2 pJ
        l1_write_energy_per_byte=0.3e-12,  # 0.3 pJ

        # Data movement engine energy
        dma_transfer_energy_per_byte=1.0e-12,  # 1.0 pJ (optimized for HBM2)
        blockmover_energy_per_byte=0.5e-12,  # 0.5 pJ
        streamer_energy_per_byte=0.2e-12,  # 0.2 pJ

        # Token routing energy
        token_signature_matching_energy=0.4e-12,  # 0.4 pJ (advanced node)
        token_handshake_energy=0.12e-12,  # 0.12 pJ
        token_routing_per_hop=0.1e-12,  # 0.1 pJ per hop

        # SURE program management
        sure_program_load_energy=35e-12,  # 35 pJ broadcast
        sure_program_cache_hit_energy=0.6e-12,  # 0.6 pJ cache hit

        # L3 distributed scratchpad routing
        l3_routing_distance_factor=1.3,  # Larger mesh, longer average distance
        l3_noc_energy_per_hop=0.35e-12,  # 0.35 pJ per NoC hop

        # Computation energy (7nm/4nm)
        mac_energy_int8=0.25e-12,  # 0.25 pJ (INT8)
        mac_energy_bf16=0.38e-12,  # 0.38 pJ (BF16, 1.5× INT8)
        mac_energy_fp32=0.75e-12,  # 0.75 pJ (FP32, 3× INT8)

        # Operator fusion benefits
        fusion_l2_traffic_reduction=0.75,  # 75% reduction (better than smaller variants)
        fusion_coordination_overhead=3.5e-12,  # 3.5 pJ per boundary
    )

    # Performance calculation (12,288 PEs × 1.5 GHz × 2 ops/cycle)
    # INT8: 36,864 GOPS
    # BF16: 36,864 GFLOPS
    # FP32: 18,432 GFLOPS
    peak_ops_int8 = 12288 * 1500e6 * 2  # 36,864 GOPS
    peak_ops_bf16 = 12288 * 1500e6 * 2  # 36,864 GFLOPS
    peak_ops_fp32 = 12288 * 1500e6  # 18,432 GFLOPS

    model = HardwareResourceModel(
        name="KPU-T768",
        hardware_type=HardwareType.KPU,
        compute_units=768,  # 768 tiles
        threads_per_unit=16,  # 16 PEs per tile
        warps_per_unit=4,  # 4 PE clusters per tile
        warp_size=4,  # 4 PEs per cluster

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_ops_fp32,
                tensor_core_supported=False,
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
        default_precision=Precision.BF16,

        peak_bandwidth=204.8e9,  # 204.8 GB/s (HBM2)
        l1_cache_per_unit=768 * 1024,  # 768 KiB per tile
        l2_cache_total=24 * 1024 * 1024,  # 24 MiB total
        main_memory=16 * 1024**3,  # 16 GB HBM2 (automotive config)
        energy_per_flop_fp32=0.75e-12,  # 0.75 pJ per FP32 MAC
        energy_per_byte=5e-12,  # 5 pJ per byte (HBM2)
        min_occupancy=0.70,  # Spatial dataflow, larger mesh needs higher utilization
        max_concurrent_kernels=16,  # Can run 16 independent SURE programs
        wave_quantization=1,

        # Thermal profiles
        thermal_operating_points={
            "efficient": thermal_efficient,
            "default": thermal_default,
            "performance": thermal_performance,
        },
        default_thermal_profile="default",
    )

    # Attach tile energy model
    model.tile_energy_model = tile_energy_model

    return model
