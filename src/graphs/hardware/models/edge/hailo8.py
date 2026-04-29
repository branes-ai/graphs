"""
Hailo-8 Resource Model - Computer Vision AI Accelerator

Dataflow architecture optimized for convolutional neural networks.
Target: Edge AI cameras, drones, robots, embedded vision systems.

Configuration:
- 26 TOPS INT8 (dense workloads, not sparse)
- Dataflow architecture with distributed on-chip memory
- All-on-chip design (no external DRAM)
- 16nm process
- 2.5W typical power consumption

Competitor to: KPU-T64, Google Coral Edge TPU, Qualcomm QCS6490
"""

from ...fabric_model import SoCFabricModel, Topology
from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ComputeFabric,
    get_base_alu_energy,
    ClockDomain,
    ComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
    BOMCostProfile,
)


def _get_hailo8_fabric() -> SoCFabricModel:
    """M6 Layer 6 fabric for Hailo-8.

    Hailo does not publish NoC details. Per the issue M6 constraint
    we ship with ``low_confidence=True`` and document the assumption:
    32 dataflow units laid out as an 8x4 mesh, hop coefficients
    estimated from the 16nm process baseline.
    """
    return SoCFabricModel(
        topology=Topology.MESH_2D,
        hop_latency_ns=1.5,
        pj_per_flit_per_hop=1.5,
        bisection_bandwidth_gbps=64.0,
        controller_count=32,
        flit_size_bytes=16,
        mesh_dimensions=(8, 4),
        routing_distance_factor=1.1,
        low_confidence=True,
        provenance=("Hailo-8: no public NoC topology data; assumed "
                    "8x4 mesh estimated from compute_units=32 layout"),
    )


def hailo8_resource_model() -> HardwareResourceModel:
    """
    Hailo-8 Computer Vision AI Accelerator.

    ARCHITECTURE:
    - Structure-driven graph mapping architecture 
    - Distributed on-chip memory fabric (no Von Neumann bottleneck)
    - Network-specific compilation (custom dataflow per model)
    - All computations on-chip (minimizes external memory access)
    - 16nm TSMC process

    PERFORMANCE:
    - 26 TOPS INT8 (marketed, achievable)
    - Realistic: ~22 TOPS INT8 sustained (85% efficiency)
    - Excellent efficiency due to dataflow architecture
    - No DVFS throttling (low power, excellent thermal design)

    POWER PROFILE:
    - 2.5W typical (single operating point, no DVFS needed)
    - Passive cooling sufficient
    - Best power efficiency in class (10.4 TOPS/W)

    USE CASES:
    - Edge AI cameras (YOLOv5, YOLOv8 detection)
    - Drones (real-time object tracking)
    - Industrial inspection (defect detection)
    - Automotive ADAS (parking assist, lane detection)

    CALIBRATION STATUS: ✅ WELL-DOCUMENTED
    - Hailo publishes detailed benchmarks
    - 85% efficiency typical for CNNs
    - Real-world deployments confirm performance
    """
    # Physical hardware
    num_dataflow_units = 32  # Estimated dataflow processing elements
    int8_ops_per_unit_per_clock = 500  # High ops/clock for dataflow
    sustained_clock_hz = 1.6e9  # 1.6 GHz sustained (no throttling)

    # ========================================================================
    # Dataflow Fabric Architecture (All-on-chip spatial dataflow)
    # ========================================================================
    dataflow_fabric = ComputeFabric(
        fabric_type="dataflow_architecture",
        circuit_type="standard_cell",   # Custom ASIC with standard cells
        num_units=num_dataflow_units,   # 32 dataflow processing elements
        ops_per_unit_per_clock={
            Precision.INT8: 500,         # 500 INT8 ops/cycle per unit
            Precision.INT4: 1000,        # 2× for INT4
        },
        core_frequency_hz=sustained_clock_hz,  # 1.6 GHz sustained
        process_node_nm=16,              # 16nm TSMC
        energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell'),  # 2.7 pJ
        energy_scaling={
            Precision.INT8: 0.125,       # INT8 is very efficient
            Precision.INT4: 0.0625,      # INT4 even more efficient
        }
    )

    # Total INT8: 32 units × 500 ops/cycle × 1.6 GHz = 25.6 TOPS ≈ 26 TOPS ✓

    # ========================================================================
    # 2.5W MODE: Single operating point (no DVFS, well-designed thermal)
    # ========================================================================
    clock_2_5w = ClockDomain(
        base_clock_hz=1.6e9,        # 1.6 GHz (estimated)
        max_boost_clock_hz=1.6e9,   # No boost, constant frequency
        sustained_clock_hz=1.6e9,   # No throttling
        dvfs_enabled=False,          # Fixed frequency, excellent thermal
    )

    compute_resource_2_5w = ComputeResource(
        resource_type="Hailo-Dataflow-Architecture",
        num_units=num_dataflow_units,
        ops_per_unit_per_clock={
            Precision.INT8: 500,  # 32 units × 500 ops/clock × 1.6 GHz ≈ 25.6 TOPS
            Precision.INT4: 1000,  # 2× for INT4 (not primary use case)
        },
        clock_domain=clock_2_5w,
    )

    # Sustained INT8: 32 × 500 × 1.6 GHz = 25.6 TOPS ≈ 26 TOPS ✓

    thermal_2_5w = ThermalOperatingPoint(
        name="2.5W-passive",
        tdp_watts=2.5,
        cooling_solution="passive-heatsink-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_2_5w,
                instruction_efficiency=0.95,  # Dataflow is very efficient
                memory_bottleneck_factor=0.90,  # On-chip memory minimizes bottleneck
                efficiency_factor=0.85,  # 85% → ~22 TOPS effective (excellent!)
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource_2_5w,
                efficiency_factor=0.80,  # Slightly lower for INT4
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM COST PROFILE (Estimated @ 10K units)
    # ========================================================================
    bom_cost = BOMCostProfile(
        silicon_die_cost=25.0,       # 16nm die (small, efficient)
        package_cost=8.0,             # Standard BGA package
        memory_cost=0.0,              # All on-chip SRAM (no external DRAM)
        pcb_assembly_cost=4.0,        # Minimal external components
        thermal_solution_cost=1.0,    # Tiny heatsink (2.5W)
        other_costs=2.0,              # Testing, connectors
        total_bom_cost=0,             # Auto-calculated: $40
        margin_multiplier=4.0,        # High margin for specialized product
        retail_price=160.0,           # Known retail: $150-180 for M.2 module
        volume_tier="10K+",
        process_node="16nm",
        year=2025,
        notes="Ultra-efficient edge AI accelerator. Low BOM due to all-on-chip design. "
              "Highest TOPS/$ and TOPS/W in entry-level segment. M.2 module retails at $160."
    )

    # BOM: $25 + $8 + $0 + $4 + $1 + $2 = $40
    # Retail: $160 (actual market pricing)
    # Cost structure: $40 BOM, $120 margin (75% gross margin)

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================
    model = HardwareResourceModel(
        name="Hailo-8",
        hardware_type=HardwareType.KPU,  # Dataflow architecture (similar to KPU)

        # NEW: Compute fabric (single dataflow fabric)
        compute_fabrics=[dataflow_fabric],

        compute_units=num_dataflow_units,
        threads_per_unit=128,  # Dataflow "threads" per unit
        warps_per_unit=1,
        warp_size=1,

        # Thermal operating points (single profile, no DVFS needed)
        thermal_operating_points={
            "2.5W": thermal_2_5w,
        },
        default_thermal_profile="2.5W",

        # Legacy precision profiles
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=26e12,  # 26 TOPS INT8 (marketed, achievable)
                tensor_core_supported=True,  # Dataflow acts like specialized hardware
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=52e12,  # 52 TOPS INT4 (theoretical)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=0.5,
            ),
        },
        default_precision=Precision.INT8,

        # Memory hierarchy (all on-chip)
        peak_bandwidth=200e9,  # ~200 GB/s on-chip bandwidth (estimated)
        l1_cache_per_unit=512 * 1024,  # 512 KB per unit (on-chip SRAM)
        l2_cache_total=8 * 1024 * 1024,  # 8 MB total on-chip (estimated)
        main_memory=0,  # No external DRAM (uses host memory for I/O only)

        # Energy (use dataflow fabric energy)
        energy_per_flop_fp32=dataflow_fabric.energy_per_flop_fp32,  # 2.7 pJ (16nm, standard cell)
        energy_per_byte=2e-12,          # 2 pJ/byte (on-chip SRAM)

        # Scheduling
        min_occupancy=0.8,  # Dataflow compiler ensures high occupancy
        max_concurrent_kernels=1,  # Single model execution
        wave_quantization=1,

        # BOM cost
        bom_cost_profile=bom_cost,

        # M3 Layer 3: software-managed on-chip SRAM partitioned per
        # dataflow unit. Hailo's compiler statically allocates the
        # working set; no hardware cache.
        l1_storage_kind="scratchpad",

        # M4 Layer 4: Hailo-8's inter-unit shared SRAM acts as the
        # L2 layer above per-unit L1 partitions. 8 MB shared / 32
        # units = 256 KiB per-unit share.
        l2_cache_per_unit=(8 * 1024 * 1024) // 32,  # 256 KiB per unit
        l2_topology="shared-llc",

        # M5 Layer 5: Hailo dataflow has no inter-cluster cache.
        l3_present=False,
        l3_cache_total=0,
        coherence_protocol="none",

        # M7 Layer 7: Hailo-8 deployments load model weights once at
        # initialization, then run inference entirely from on-chip
        # SRAM. The legacy energy_per_byte=2 pJ/B reflects on-chip
        # SRAM access, not DRAM. Model the host-side DRAM here as
        # LPDDR4 for the cold-start / weight-load path.
        memory_technology="LPDDR4 (host) + on-chip SRAM (steady-state)",
        memory_read_energy_per_byte_pj=22.0,
        memory_write_energy_per_byte_pj=27.0,

        # M6 Layer 6: dataflow mesh between 32 PE units. Hailo does
        # NOT publish per-hop NoC details, so this entry ships with
        # low_confidence=True per issue M6 constraint. Mesh
        # dimensions are estimated as a near-square 8x4 layout of
        # the 32 dataflow units.
        soc_fabric=_get_hailo8_fabric(),
    )

    # M3 Layer 3 provenance
    from graphs.core.confidence import EstimationConfidence
    model.set_provenance(
        "l1_cache_per_unit",
        EstimationConfidence.theoretical(
            score=0.75,
            source=("Hailo-8 Architecture Brief: ~16 MB total on-chip "
                    "SRAM partitioned across 32 dataflow units (~512 KB/unit)"),
        ),
    )
    model.set_provenance(
        "l1_storage_kind",
        EstimationConfidence.theoretical(
            score=0.95,
            source="Hailo dataflow architecture: software-managed, no L1 cache",
        ),
    )

    # M4 Layer 4 provenance for the shared L2 SRAM layer
    model.set_provenance(
        "l2_cache_per_unit",
        EstimationConfidence.theoretical(
            score=0.70,
            source=("Hailo-8 architecture brief: ~8 MB inter-unit "
                    "shared SRAM (256 KiB per-unit share). LLC by "
                    "design; no DRAM-side cache."),
        ),
    )
    model.set_provenance(
        "l2_topology",
        EstimationConfidence.theoretical(
            score=0.90,
            source="Hailo dataflow architecture: shared LLC over per-unit L1",
        ),
    )

    # M5 Layer 5 provenance
    model.set_provenance(
        "l3_present",
        EstimationConfidence.theoretical(
            score=0.95,
            source="Hailo dataflow: no inter-cluster cache layer",
        ),
    )
    model.set_provenance(
        "l3_cache_total",
        EstimationConfidence.theoretical(
            score=0.95,
            source=("Hailo dataflow: Layer 5 cache absent by design, "
                    "capacity fixed at 0 bytes"),
        ),
    )
    model.set_provenance(
        "coherence_protocol",
        EstimationConfidence.theoretical(
            score=0.95,
            source="Hailo dataflow: no inter-unit coherence (compiler-routed)",
        ),
    )

    # M6 Layer 6 provenance (low confidence - thin datasheet)
    model.set_provenance(
        "soc_fabric",
        EstimationConfidence.theoretical(
            score=0.40,
            source=("Hailo-8 NoC topology not publicly documented; "
                    "panel ships with low_confidence=True flag"),
        ),
    )

    # M7 Layer 7 provenance: Hailo deploys with weights resident
    # in on-chip SRAM; DRAM only on the cold-start path.
    for key in ("memory_technology",
                "memory_read_energy_per_byte_pj",
                "memory_write_energy_per_byte_pj"):
        model.set_provenance(
            key,
            EstimationConfidence.theoretical(
                score=0.55,
                source=("Hailo-8 host-side LPDDR4 for weight load; "
                        "steady-state inference is on-chip SRAM "
                        "(legacy 2 pJ/B value reflects SRAM, not DRAM)"),
            ),
        )

    return model
