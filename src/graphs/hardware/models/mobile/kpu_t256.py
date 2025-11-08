"""
KPU-T256 Resource Model - Mobile/Robotics AI Accelerator

Stillwater KPU-T256 (Knowledge Processing Unit)
Domain Flow Architecture (DFA)

Target: Mobile / Robotics / Autonomous systems
Process: 16nm / 7nm variants
TDP: 25-75W
Product line: Mobile inference, edge servers

Architecture:
- 256 tiles (16×16 mesh)
- 16 PEs per tile = 4,096 total PEs
- Programmable SURE execution
- 4-stage memory hierarchy (DRAM → L3 → L2 → L1)
- Token-based spatial dataflow
- Enhanced bandwidth (DDR4/LPDDR5)
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


def kpu_t256_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T256 resource model.

    Domain Flow Architecture (DFA) - Mobile/Robotics AI accelerator.

    Key characteristics:
    - 256 tiles, 4,096 PEs (16 PEs/tile)
    - 16×16 2D mesh topology
    - Programmable SURE execution (all BLAS operators)
    - Token-based spatial dataflow
    - 4-stage memory hierarchy
    - 25-75W TDP (mobile/robotics)
    - 16nm or 7nm process
    """

    # Thermal operating point (mobile/robotics deployment)
    thermal_efficient = ThermalOperatingPoint(
        name="efficient",
        tdp_watts=25.0,  # Power-efficient mode
        cooling_solution="active-fan",
        performance_specs={}
    )

    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=50.0,  # Standard operation
        cooling_solution="active-fan",
        performance_specs={}
    )

    thermal_performance = ThermalOperatingPoint(
        name="performance",
        tdp_watts=75.0,  # Maximum performance
        cooling_solution="active-vapor-chamber",
        performance_specs={}
    )

    # KPU-T256 tile energy model
    tile_energy_model = KPUTileEnergyModel(
        # Processing element configuration
        num_tiles=256,
        pes_per_tile=16,
        tile_mesh_dimensions=(16, 16),  # 16×16 mesh

        # Memory hierarchy (4-stage, DDR4 or LPDDR5)
        dram_bandwidth_gb_s=102.4,  # 102.4 GB/s (LPDDR5-6400)
        l3_size_per_tile=256 * 1024,  # 256 KiB per tile (64 MiB total)
        l2_size_per_tile=32 * 1024,  # 32 KiB per tile (8 MiB total)
        l1_size_per_pe=4 * 1024,  # 4 KiB per PE (256 KiB per tile)

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

        # Clock frequency (16nm/7nm process)
        clock_frequency_hz=1200e6,  # 1.2 GHz

        # Memory hierarchy energy (16nm/7nm, DDR4/LPDDR5)
        dram_read_energy_per_byte=10.0e-12,  # 10 pJ (DDR4)
        dram_write_energy_per_byte=12.0e-12,  # 12 pJ
        l3_read_energy_per_byte=1.5e-12,  # 1.5 pJ (advanced node)
        l3_write_energy_per_byte=1.8e-12,  # 1.8 pJ
        l2_read_energy_per_byte=0.6e-12,  # 0.6 pJ
        l2_write_energy_per_byte=0.8e-12,  # 0.8 pJ
        l1_read_energy_per_byte=0.25e-12,  # 0.25 pJ
        l1_write_energy_per_byte=0.35e-12,  # 0.35 pJ

        # Data movement engine energy
        dma_transfer_energy_per_byte=1.2e-12,  # 1.2 pJ (optimized)
        blockmover_energy_per_byte=0.6e-12,  # 0.6 pJ
        streamer_energy_per_byte=0.25e-12,  # 0.25 pJ

        # Token routing energy
        token_signature_matching_energy=0.5e-12,  # 0.5 pJ (advanced node)
        token_handshake_energy=0.15e-12,  # 0.15 pJ
        token_routing_per_hop=0.12e-12,  # 0.12 pJ per hop

        # SURE program management
        sure_program_load_energy=40e-12,  # 40 pJ broadcast
        sure_program_cache_hit_energy=0.8e-12,  # 0.8 pJ cache hit

        # L3 distributed scratchpad routing
        l3_routing_distance_factor=1.2,
        l3_noc_energy_per_hop=0.4e-12,  # 0.4 pJ per NoC hop

        # Computation energy (16nm/7nm)
        mac_energy_int8=0.28e-12,  # 0.28 pJ (INT8)
        mac_energy_bf16=0.42e-12,  # 0.42 pJ (BF16, 1.5× INT8)
        mac_energy_fp32=0.84e-12,  # 0.84 pJ (FP32, 3× INT8)

        # Operator fusion benefits
        fusion_l2_traffic_reduction=0.7,  # 70% reduction
        fusion_coordination_overhead=4e-12,  # 4 pJ per boundary
    )

    # Performance calculation (4,096 PEs × 1.2 GHz × 2 ops/cycle)
    # INT8: 9,830 GOPS
    # BF16: 9,830 GFLOPS
    # FP32: 4,915 GFLOPS
    peak_ops_int8 = 4096 * 1200e6 * 2  # 9,830 GOPS
    peak_ops_bf16 = 4096 * 1200e6 * 2  # 9,830 GFLOPS
    peak_ops_fp32 = 4096 * 1200e6  # 4,915 GFLOPS

    model = HardwareResourceModel(
        name="KPU-T256",
        hardware_type=HardwareType.KPU,
        compute_units=256,  # 256 tiles
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
        default_precision=Precision.INT8,

        peak_bandwidth=102.4e9,  # 102.4 GB/s (LPDDR5)
        l1_cache_per_unit=256 * 1024,  # 256 KiB per tile
        l2_cache_total=8 * 1024 * 1024,  # 8 MiB total
        main_memory=8 * 1024**3,  # 8 GB LPDDR5 (typical mobile config)
        energy_per_flop_fp32=0.84e-12,  # 0.84 pJ per FP32 MAC
        energy_per_byte=10e-12,  # 10 pJ per byte
        min_occupancy=0.65,  # Spatial dataflow
        max_concurrent_kernels=8,  # Can run 8 independent SURE programs
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
