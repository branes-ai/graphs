#  Assessment: Hardware Database vs Calibration Data Architecture

Based on my analysis of the CHANGELOG and session logs (especially 2025-11-17_hardware_database_implementation.md), we have the following state of the implementation of the hardware database and the calibration database:

  1. Hardware Database (hardware_database/) was created to:
    - Replace bug-prone hardcoded PRESETS in the mappers
    - Provide a "single source of truth" for hardware specifications
    - Store theoretical peaks from datasheets
    - Support hardware auto-detection for calibration
  2. Calibration System (src/graphs/hardware/calibration/) was created to:
    - Run micro-benchmarks to measure empirical performance
    - Calculate percent-of-peak efficiency statistics
    - Generate profiles for energy/performance model accuracy

  What Went Wrong: The Two Systems Drifted

  The session log from 2025-11-17 explicitly identified this as deferred work:

  Phase 5 Planning (Next Steps):
  - Design calibration result export to database
  - Define historical tracking schema
  - Plan multi-run averaging algorithm

  Phase 5 was never implemented. The systems were left in limbo:

  | System                | What it has                                               | What's missing                              |
  |-----------------------|-----------------------------------------------------------|---------------------------------------------|
  | hardware_database/    | Theoretical peaks, architecture specs, detection patterns | No empirical data, no efficiency metrics    |
  | calibration/profiles/ | Measured GFLOPS, efficiency %, latency by operation       | No correlation to hardware_database entries |

  How Hardware Database JSON Gets Populated

  Currently, there are two paths:

  1. Manual Creation: JSON files were hand-written based on datasheet values (e.g., Jetson specs)
  2. Migration from PRESETS: scripts/hardware_db/migrate_presets.py migrated the old hardcoded PRESETS
  3. Interactive Wizard: scripts/hardware_db/add_hardware.py --from-detection creates entries from auto-detected hardware

  Missing Path: There is NO mapper → hardware_database export. The mappers have hardcoded specs in src/graphs/hardware/models/ and
  src/graphs/hardware/mappers/ that are duplicated from the hardware_database JSON.

  ---
  Proposed Plan: Aligning Theoretical and Empirical Databases

  Architecture Vision

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        HARDWARE SPECIFICATION                           │
  │                     hardware_database/<type>/<vendor>/<id>.json         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  theoretical_peaks:     From datasheet (fp32, fp16, int8, etc.)         │
  │  peak_bandwidth_gbps:   From datasheet                                  │
  │  architecture specs:    cores, SMs, frequency, etc.                     │
  │                                                                         │
  │  calibration_summary:   ← NEW: Aggregated empirical data                │
  │    ├── measured_peaks:  Best measured GFLOPS per precision              │
  │    ├── efficiency:      measured/theoretical ratios                     │
  │    ├── bandwidth_efficiency: measured/theoretical bandwidth             │
  │    └── last_calibrated: Timestamp                                       │
  │                                                                         │
  │  calibration_profiles:  ← NEW: Links to detailed profiles               │
  │    ├── default: "profiles/nvidia/jetson_orin_nano/25W/...json"          │
  │    └── power_modes: { "7W": "...", "15W": "...", "25W": "..." }         │
  └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                       CALIBRATION PROFILES                              │
  │          src/graphs/hardware/calibration/profiles/<path>.json           │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Detailed micro-benchmark results:                                      │
  │    - Memory copy bandwidth by size                                      │
  │    - Matmul performance by matrix size and precision                    │
  │    - Per-operation latency distributions                                │
  │    - DNN workload measurements (future)                                 │
  └─────────────────────────────────────────────────────────────────────────┘

  Implementation Plan

  Phase 1: Schema Extension (1-2 hours)

  Add calibration_summary and calibration_profiles fields to HardwareSpec:

  # In schema.py
  @dataclass
  class CalibrationSummary:
      """Aggregated empirical performance from calibration."""
      measured_peaks: Dict[str, float]        # precision -> best GFLOPS
      efficiency: Dict[str, float]            # precision -> measured/theoretical
      bandwidth_efficiency: float             # measured/theoretical BW
      avg_efficiency: float                   # overall average
      calibration_date: str                   # ISO timestamp
      power_mode: Optional[str]               # "25W", "7W", etc.
      framework: str                          # "pytorch", "numpy"
      profile_path: str                       # relative path to full profile

  Phase 2: Export Tool (2-3 hours)

  Create scripts/hardware_db/export_calibration.py:

  # After running calibration, export summary to hardware_database
  python scripts/hardware_db/export_calibration.py \
      --calibration profiles/nvidia/jetson_orin_nano/25W/...json \
      --hardware-id jetson_orin_nano_gpu \
      --power-mode 25W

  This would:
  1. Load calibration profile
  2. Extract best measured peaks, efficiencies
  3. Update hardware_database JSON with calibration_summary
  4. Link to full profile via calibration_profiles

  Phase 3: Unified Summary Tool (1-2 hours)

  Update scripts/hardware_db/summarize_database.py to show:

  HARDWARE DATABASE SUMMARY
  ================================================================================

  NVIDIA Jetson Orin Nano (GPU)
    Theoretical FP32:  1280 GFLOPS    Measured: 1199 GFLOPS (93.7% eff)
    Theoretical FP16:  7600 GFLOPS    Measured: 7325 GFLOPS (96.4% eff)
    Theoretical BW:    68.0 GB/s      Measured: 57.8 GB/s   (85.0% eff)

    Power Modes Calibrated: 7W, 15W, 25W
    Last Calibrated: 2025-11-16

  Phase 4: Mapper Integration (3-4 hours)

  Create tool to sync hardware_database → mappers:

  # Generate/update mapper code from hardware_database
  python scripts/hardware_db/sync_mappers.py --id jetson_orin_nano_gpu

  This ensures mappers use the same values as hardware_database (single source of truth).

  Phase 5: DNN Workload Calibration (Future)

  Extend calibration to measure complete DNN workloads:
  - ResNet-18, MobileNetV2, etc.
  - Measure end-to-end latency and energy
  - Calculate "stack efficiency" (measured / roofline prediction)

  ---
  Key Design Decisions

  1. Keep calibration profiles separate: They contain detailed per-operation data that's too granular for hardware_database
  2. Add summary to hardware_database: Aggregated efficiency metrics for quick comparison
  3. Link via paths: calibration_profiles field links to detailed data
  4. Power mode awareness: Jetson has 7W/15W/25W modes with different performance
  5. Single source of truth: hardware_database feeds both mappers AND calibration tools

  Immediate Action Items

  1. Extend HardwareSpec schema with calibration_summary and calibration_profiles fields
  2. Create export_calibration.py to push calibration results into hardware_database
  3. Update summarize_database.py to show theoretical vs empirical side-by-side
  4. Document the workflow in hardware_database/README.md

