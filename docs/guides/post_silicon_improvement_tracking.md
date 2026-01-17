# Post-Silicon Performance Improvement Tracking

## Executive Summary

This document describes the post-silicon dynamics tracking system: a methodology for measuring and analyzing how hardware efficiency improves over time as software stacks mature. This capability is critical for accurate design guidance, vendor assessment, and regression detection.

## The Problem

### Hardware Performance is Not Static

When a new accelerator or processor is released, its measured efficiency is typically far below theoretical peak. Over the following months and years, software stack improvements (drivers, compilers, frameworks, runtimes) progressively unlock more of the hardware's potential:

```
Typical Product Maturity Curve:

    100% |                                          ___________
         |                                    _____/
     90% |                              _____/
         |                        _____/
     80% |                  _____/
         |            _____/
     70% |      _____/
         | ____/
     60% |/
         +----+----+----+----+----+----+----+----+----+----+-->
           Day 0  30   60   90   120  150  180  210  240  270
                            Days Since Launch
```

This progression varies dramatically between vendors and products:
- **Mature ecosystems** (NVIDIA CUDA): Often reach 85%+ efficiency within weeks
- **New architectures** (Intel Gaudi, AMD MI300): May take 6-12 months
- **Edge devices** (Jetson, Coral): Often plateau at 60-70%

### Why This Matters for Design Guidance

When designing a new system, architects need to answer questions like:
- "What efficiency can we realistically expect from Product X?"
- "Is Product Y's current low efficiency a temporary software issue or fundamental?"
- "How long until Product Z reaches production-ready efficiency?"

Without temporal tracking, a single-point calibration can be misleading:
- A measurement taken at Day 30 may show 65% efficiency
- The same hardware at Day 180 may show 92% efficiency
- Design decisions based on Day 30 data would be fundamentally wrong

### Previous Limitations

The original `benchmark_sweep.py` had critical design flaws:

1. **User-provided hardware ID**: Error-prone, inconsistent naming across runs
2. **Upsert semantics**: Overwrote history, destroying temporal data
3. **No software stack versioning**: Couldn't distinguish HW vs SW issues
4. **No temporal queries**: Couldn't analyze improvement trajectories

## The Solution: Versioned, Self-Organizing Schema

### Core Principles

1. **Append-only storage**: Every calibration run creates a NEW record. No updates. No deletions.
2. **Self-organizing identity**: Hardware and software fingerprints are auto-detected, never user-provided.
3. **Complete provenance**: Every result is traceable to specific hardware + software + timestamp.

### Schema Design

The `CalibrationRun` schema captures complete context for each measurement:

```
CalibrationRun
==============

Identity (auto-detected, immutable):
------------------------------------
run_id                 UUID        Unique identifier for this run
hardware_fingerprint   TEXT        SHA256[:16] of hardware identity
software_fingerprint   TEXT        SHA256[:16] of software stack
timestamp              ISO 8601    When this run occurred
timestamp_unix         INTEGER     Unix epoch for efficient queries

Hardware Details (denormalized for query efficiency):
-----------------------------------------------------
cpu_model              TEXT        "Intel(R) Core(TM) i7-12700K"
cpu_vendor             TEXT        "GenuineIntel"
cpu_stepping           INTEGER     Silicon revision
cpu_microcode          TEXT        "0x2c"
cpu_cores_physical     INTEGER
cpu_cores_logical      INTEGER
gpu_model              TEXT        "NVIDIA GeForce RTX 4090" or "N/A"
gpu_pci_id             TEXT        "10de:2684"
gpu_vbios              TEXT        "95.02.18.40.84"
gpu_memory_mb          INTEGER
memory_total_gb        REAL

Software Stack (the key to temporal analysis):
----------------------------------------------
os_kernel              TEXT        "6.8.0-90-generic"
os_distro              TEXT        "Ubuntu 24.04"
gpu_driver_version     TEXT        "560.35.03"
cuda_version           TEXT        "12.4"
pytorch_version        TEXT        "2.7.1"
numpy_version          TEXT        "2.2.6"
blas_library           TEXT        "OpenBLAS" or "MKL"

Environmental Context:
----------------------
power_mode             TEXT        "TDP", "30W", "performance"
cpu_governor           TEXT        "performance", "powersave"
thermal_state          TEXT        "cool", "warm", "throttled"

Calibration Results:
--------------------
precision              TEXT        "fp32", "fp16", "int8"
device                 TEXT        "cpu", "cuda"
stream_best_gbps       REAL        Memory bandwidth
blas3_gops             REAL        GEMM throughput
peak_measured_gops     REAL        Best measured compute
efficiency             REAL        measured / theoretical
```

### Fingerprint Generation

Hardware fingerprints are deterministic and stable across reboots:

```python
def generate_hardware_fingerprint() -> str:
    """
    Fingerprint is stable across:
    - Reboots
    - OS reinstalls
    - Driver updates

    Fingerprint changes when:
    - Different physical hardware
    - BIOS/VBIOS update
    - Microcode update
    """
    components = [
        cpu_model,           # "Intel(R) Core(TM) i7-12700K"
        cpu_vendor,          # "GenuineIntel"
        cpu_stepping,        # "4"
        cpu_microcode,       # "0x2c"
        cpu_cores_physical,  # "12"
        gpu_pci_id,          # "10de:2684" or "no_gpu"
        gpu_vbios,           # "95.02.18.40.84" or "no_gpu"
        memory_total_gb,     # "32"
    ]
    return sha256("|".join(components))[:16]
```

Software fingerprints change with any stack update:

```python
def generate_software_fingerprint() -> str:
    """Changes when any software component is updated."""
    components = [
        os_kernel,           # "6.8.0-90-generic"
        gpu_driver_version,  # "560.35.03"
        cuda_version,        # "12.4"
        pytorch_version,     # "2.7.1"
        numpy_version,       # "2.2.6"
    ]
    return sha256("|".join(components))[:16]
```

## Data Capture Methodology

### Calibration Workflow

```
+------------------+     +------------------+     +------------------+
|  Auto-Detect HW  | --> |  Auto-Detect SW  | --> |  Run Benchmarks  |
|  - CPU model     |     |  - Kernel        |     |  - STREAM        |
|  - GPU PCI ID    |     |  - Driver        |     |  - BLAS L1/L2/L3 |
|  - Memory        |     |  - PyTorch       |     |  - Model proxies |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
+------------------+     +------------------+     +------------------+
|  Query/Analyze   | <-- |  Store Record    | <-- |  Compute Metrics |
|  - Trajectories  |     |  - Append-only   |     |  - Peak GOPS     |
|  - Regressions   |     |  - Full context  |     |  - Bandwidth     |
|  - Comparisons   |     |  - Timestamped   |     |  - Efficiency    |
+------------------+     +------------------+     +------------------+
```

### Running Calibration

```bash
# Basic calibration (auto-detects everything)
./cli/benchmark_sweep.py --quick

# Full calibration with all benchmark layers
./cli/benchmark_sweep.py --layers micro proxy models

# Force run even if preflight checks fail
./cli/benchmark_sweep.py --force

# Show current context without running benchmarks
./cli/benchmark_sweep.py --show-context
```

### Recommended Calibration Schedule

For tracking post-silicon dynamics effectively:

| Event | Action |
|-------|--------|
| New hardware arrival | Full calibration (`--layers micro proxy models`) |
| Driver update | Quick calibration (`--quick`) |
| Framework update (PyTorch, etc.) | Quick calibration |
| Monthly maintenance | Quick calibration to detect drift |
| Performance issue reported | Full calibration + regression check |

### Preflight Checks

The calibration system performs preflight checks to ensure measurement quality:

1. **CPU Governor**: Should be "performance" for consistent results
2. **Thermal State**: Should be "cool" to avoid throttling
3. **Power Mode**: Should match expected configuration
4. **Background Load**: System should be idle

Use `--force` to skip preflight checks when necessary (results are flagged).

## Using the Results

### 1. Efficiency Trajectory Analysis

View how a specific hardware's efficiency has changed over time:

```bash
./cli/analyze_product_trajectory.py --hardware c3f840a080356806
```

Output:
```
==============================================================================
EFFICIENCY TRAJECTORY: c3f840a0...
==============================================================================

    Efficiency vs Time: 12th Gen Intel(R) Core(TM) i7-12700K

   269.0 |                                                            |
   239.1 |                                                   --------*|
   209.3 |                          --------                          |
   179.4 | --------                                                   |
   169.5 |o                                                           |
         +------------------------------------------------------------+
         2025-12-31                                        2025-12-31

------------------------------------------------------------------------------
Date         SW Stack             GOPS    BW (GB/s)        Eff
------------------------------------------------------------------------------
2025-12-31   6286d41799          177.3         81.7     177.3%
2025-12-31   6286d41799          244.5         76.4     244.5%

Milestones:
  70% efficiency: Day 0
  80% efficiency: Day 0
  90% efficiency: Day 0
```

### 2. Time-to-Maturity Analysis

Compare how long different products take to reach efficiency milestones:

```bash
./cli/analyze_product_trajectory.py --maturity-analysis
```

Output:
```
==============================================================================
TIME-TO-MATURITY ANALYSIS
==============================================================================

How long (days) to reach efficiency milestones from first calibration:

Product                             70%      80%      90%    Current
------------------------------------------------------------------------------
NVIDIA H100 SXM5                  Day 0    Day 7   Day 45      94.2%
AMD MI300X                       Day 14   Day 45   Day 90      91.5%
Intel Gaudi 3                    Day 30   Day 90      N/A      78.3%
Qualcomm Cloud AI 100            Day 45  Day 120      N/A      72.1%

Assessment Legend:
  - Day 0-30 to 90%: Excellent software support
  - Day 30-90 to 90%: Good software support
  - Day 90+ to 90%: Slow optimization
  - N/A at 90%: Concerning - may not reach target
```

This analysis helps answer: "Should we wait for Product X to mature, or go with the more mature Product Y?"

### 3. Software Impact Analysis

Understand how driver/framework updates affect performance:

```bash
./cli/analyze_product_trajectory.py --driver-impact --hardware c3f840a080356806
```

Output:
```
==============================================================================
SOFTWARE IMPACT ANALYSIS: c3f840a0...
==============================================================================

SW Fingerprint     Driver       PyTorch          GOPS  BW (GB/s)
------------------------------------------------------------------------------
a1b2c3d4e5f6g7h8   535.86       2.0.1           156.3       72.4
b2c3d4e5f6g7h8i9   545.23       2.1.0           178.9       75.1
c3d4e5f6g7h8i9j0   550.54       2.2.0           195.2       78.3
d4e5f6g7h8i9j0k1   560.35       2.4.0           212.8       81.7

Total improvement from software updates: +36.2%
  First SW stack: 156.3 GOPS
  Latest SW stack: 212.8 GOPS
```

### 4. Regression Detection

Automatically detect when newer software performs worse:

```bash
./cli/analyze_product_trajectory.py --detect-regressions --threshold 3.0
```

Output:
```
==============================================================================
REGRESSION DETECTION (threshold: 3.0%)
==============================================================================

Found 1 regression(s):

--- Regression #1 ---
Hardware:   abc123de... (NVIDIA GeForce RTX 4090)
Precision:  fp32
Metric:     peak_measured_gops

Previous:   1245.6
  Date:     2025-01-10
  SW Stack: def456gh...
  Driver:   560.35.03
  PyTorch:  2.4.0

Current:    1178.2
  Date:     2025-01-15
  SW Stack: ghi789jk...
  Driver:   560.40.01
  PyTorch:  2.4.0

Regression: -5.4%

Likely cause: Driver update 560.35 -> 560.40
Recommendation: Investigate driver change, consider rollback
```

### 5. Cross-Product Comparison

Compare multiple products side-by-side:

```bash
./cli/analyze_product_trajectory.py --compare hw_fp_1 hw_fp_2 hw_fp_3
```

Output:
```
==============================================================================
CROSS-PRODUCT COMPARISON
==============================================================================

Metric                     hw_fp_1      hw_fp_2      hw_fp_3
--------------------------------------------------------------
CPU Model                  i7-12700K    EPYC 9654    Xeon w9-3495X
First GOPS                     156.3        892.4        1245.6
Latest GOPS                    244.5       1456.2        1523.8
First BW (GB/s)                 72.4        384.2         512.3
Latest BW (GB/s)                81.7        412.5         548.9
Calibration Runs                  8           12            15
Improvement (%/month)           12.4          8.2           5.1
```

### 6. Export for External Visualization

Export trajectory data for use in Grafana, Jupyter, or other tools:

```bash
# CSV export
./cli/analyze_product_trajectory.py --export trajectory.csv

# JSON export
./cli/analyze_product_trajectory.py --export trajectory.json
```

CSV format for easy import into spreadsheets and visualization tools:
```csv
run_id,timestamp,hardware_fingerprint,software_fingerprint,cpu_model,...
uuid1,2025-01-15T14:30:00Z,abc123,def456,"Intel i7-12700K",...
uuid2,2025-02-01T10:15:00Z,abc123,ghi789,"Intel i7-12700K",...
```

## Design Guidance Applications

### Use Case 1: New Product Evaluation

When evaluating a newly released accelerator:

1. Run initial calibration to establish baseline
2. Schedule monthly re-calibrations
3. After 3-6 months, analyze trajectory to predict maturity timeline
4. Compare improvement rate against mature products

### Use Case 2: Vendor Assessment

Track multiple vendors' products over time:

```bash
# Compare NVIDIA vs AMD vs Intel trajectories
./cli/analyze_product_trajectory.py --compare nvidia_h100 amd_mi300x intel_gaudi3
```

Metrics to evaluate:
- Time to 90% efficiency (software stack maturity)
- Improvement rate (%/month)
- Regression frequency (software quality)

### Use Case 3: Regression Monitoring

Set up automated regression detection in CI/CD:

```bash
# Check for regressions after any driver/framework update
./cli/analyze_product_trajectory.py --detect-regressions --threshold 3.0
```

Alert if any product regresses more than 3% from previous best.

### Use Case 4: Design Parameter Selection

When designing a new system, use calibrated efficiency factors:

```python
from graphs.hardware.calibration.calibration_db import CalibrationDB

db = CalibrationDB("calibrations_v2.db")

# Get latest calibration for target hardware
latest = db.get_latest(hardware_fingerprint="abc123", precision="fp32")

# Use calibrated efficiency in design calculations
measured_gops = latest.peak_measured_gops
measured_bandwidth = latest.stream_best_gbps
efficiency = latest.efficiency

# Apply to theoretical models
realistic_throughput = theoretical_peak * efficiency
```

## File Locations

| File | Purpose |
|------|---------|
| `src/graphs/hardware/calibration/auto_detect.py` | Hardware/software auto-detection |
| `src/graphs/hardware/calibration/calibration_db.py` | Database schema and queries |
| `cli/benchmark_sweep.py` | Calibration runner |
| `cli/analyze_product_trajectory.py` | Analysis and visualization |
| `results/calibration_db/calibrations_v2.db` | SQLite database (default location) |

## Future Extensions

### Phase 4: Fleet Management (Planned)

- Anonymous telemetry aggregation across multiple systems
- Fleet-wide regression alerts
- Vendor performance dashboards
- Automated calibration scheduling

### Integration with Agentic Tools

The calibration database can be queried by AI agents for design guidance:

```python
# Agent query: "What efficiency should I expect from an H100?"
trajectory = db.get_trajectory("h100_fingerprint")
latest_efficiency = trajectory[-1].efficiency
improvement_rate = db.get_improvement_rate("h100_fingerprint")

# Agent can provide informed guidance based on empirical data
```

## Summary

Post-silicon dynamics tracking transforms calibration from a single-point measurement into a continuous assessment of product maturity. Key benefits:

1. **Accurate design guidance**: Use realistic efficiency expectations, not Day 1 measurements
2. **Vendor assessment**: Objectively compare software stack quality across vendors
3. **Regression detection**: Catch performance issues before they impact production
4. **Forecasting**: Predict when new products will reach production-ready efficiency

The system is fully self-organizing: no manual hardware IDs, no data overwrites, complete provenance for every measurement.
