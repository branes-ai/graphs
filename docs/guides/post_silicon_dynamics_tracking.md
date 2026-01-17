# Post-Silicon Dynamics Tracking

## Problem Statement

Hardware products improve significantly after initial silicon release through
software stack updates (drivers, runtimes, compilers, frameworks). Tracking
this progression is critical for:

1. **Design guidance**: Understanding realistic efficiency expectations
2. **Vendor assessment**: Evaluating product support quality
3. **Regression detection**: Catching performance regressions early
4. **Forecasting**: Predicting when products will reach target efficiency

## Current Limitations

The current `benchmark_sweep.py` has critical design flaws:

1. **User-provided hardware ID**: Error-prone, inconsistent naming
2. **Upsert semantics**: Overwrites history, loses temporal data
3. **No software stack versioning**: Can't distinguish HW vs SW issues
4. **No temporal queries**: Can't analyze improvement trajectories

## Proposed Schema: `CalibrationRun`

### Core Principle: Append-Only, Self-Organizing

Every calibration run creates a **new record**. No updates. No user-provided IDs.
The system auto-detects everything and creates a deterministic fingerprint.

```
CalibrationRun
==============

Primary Key: (hardware_fingerprint, software_fingerprint, timestamp)

Hardware Identity (auto-detected, immutable):
---------------------------------------------
hardware_fingerprint    TEXT    # SHA256 of hardware identity
cpu_model              TEXT    # "Intel(R) Core(TM) i7-12700K"
cpu_stepping           TEXT    # "4" (silicon revision)
cpu_microcode          TEXT    # "0x2c" (microcode version)
cpu_cores_physical     INT
cpu_cores_logical      INT
cpu_base_freq_mhz      INT
cpu_max_freq_mhz       INT

gpu_model              TEXT    # "NVIDIA GeForce RTX 4090" (NULL if no GPU)
gpu_pci_id             TEXT    # "10de:2684" (vendor:device)
gpu_vbios_version      TEXT    # "95.02.18.40.84"
gpu_memory_mb          INT
gpu_compute_capability TEXT    # "8.9"

memory_total_gb        REAL
memory_type            TEXT    # "DDR5-4800", "LPDDR5-6400"

Software Stack (per-run, versioned):
------------------------------------
software_fingerprint   TEXT    # SHA256 of software stack
os_kernel              TEXT    # "6.8.0-90-generic"
os_distro              TEXT    # "Ubuntu 24.04"

driver_version         TEXT    # "560.35.03" (NVIDIA) or "6.8.0" (kernel module)
cuda_version           TEXT    # "12.4" (NULL if no CUDA)
cudnn_version          TEXT    # "8.9.7"
rocm_version           TEXT    # NULL or "6.0"

pytorch_version        TEXT    # "2.4.0"
numpy_version          TEXT    # "2.0.1"
mkl_version            TEXT    # "2024.1" (NULL if not using MKL)
openblas_version       TEXT    # "0.3.27" (NULL if not using OpenBLAS)

compiler_version       TEXT    # "nvcc 12.4" or "gcc 13.2"

Temporal Context:
-----------------
run_id                 TEXT    # UUID for this specific run
timestamp              TEXT    # ISO 8601: "2025-01-15T14:30:00Z"
timestamp_unix         INT     # Unix epoch seconds (for efficient queries)

Environmental Context:
----------------------
power_mode             TEXT    # "TDP", "15W", "30W", "MAXN"
thermal_state          TEXT    # "cool", "warm", "throttled"
cpu_governor           TEXT    # "performance", "powersave"
gpu_power_limit_w      INT     # Current power limit
ambient_temp_c         REAL    # If available

Calibration Results:
--------------------
precision              TEXT    # "fp32", "fp16", "int8"

# STREAM results
stream_copy_gbps       REAL
stream_scale_gbps      REAL
stream_add_gbps        REAL
stream_triad_gbps      REAL
stream_best_gbps       REAL

# BLAS results (per precision)
blas1_gops             REAL    # dot, axpy
blas2_gops             REAL    # gemv
blas3_gops             REAL    # gemm (most important)

# Derived metrics
peak_measured_gops     REAL
theoretical_peak_gops  REAL    # From hardware spec
efficiency             REAL    # measured / theoretical
arithmetic_intensity   REAL    # ops / byte at peak

# Quality indicators
preflight_passed       BOOL    # Were all preflight checks OK?
forced                 BOOL    # Was --force used?
notes                  TEXT    # Any warnings or issues
```

### Fingerprint Generation

```python
def generate_hardware_fingerprint() -> str:
    """
    Generate deterministic hardware fingerprint from auto-detected info.

    Fingerprint is stable across:
    - Reboots
    - OS reinstalls
    - Driver updates

    Fingerprint changes when:
    - Different physical hardware
    - BIOS/VBIOS update (silicon-level change)
    - Microcode update
    """
    components = [
        get_cpu_model(),           # "Intel(R) Core(TM) i7-12700K"
        get_cpu_stepping(),        # "4"
        get_cpu_microcode(),       # "0x2c"
        str(get_cpu_cores()),      # "12"
        get_gpu_pci_id(),          # "10de:2684" or "none"
        get_gpu_vbios(),           # "95.02.18.40.84" or "none"
        str(get_memory_gb()),      # "32"
    ]

    fingerprint_str = "|".join(components)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]


def generate_software_fingerprint() -> str:
    """
    Generate deterministic software stack fingerprint.

    Changes when any software component is updated.
    """
    components = [
        get_kernel_version(),      # "6.8.0-90-generic"
        get_driver_version(),      # "560.35.03"
        get_cuda_version(),        # "12.4"
        get_pytorch_version(),     # "2.4.0"
        get_numpy_version(),       # "2.0.1"
    ]

    fingerprint_str = "|".join(components)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
```

## Migration Plan

### Phase 1: Schema Extension (Non-Breaking)

Add new fields to existing `CalibrationPoint` without removing old ones:

```sql
ALTER TABLE calibrations ADD COLUMN hardware_fingerprint TEXT;
ALTER TABLE calibrations ADD COLUMN software_fingerprint TEXT;
ALTER TABLE calibrations ADD COLUMN run_id TEXT;
ALTER TABLE calibrations ADD COLUMN timestamp_unix INTEGER;
-- ... add all new columns ...

-- Remove UNIQUE constraint (allow multiple records per hardware)
-- This requires table recreation in SQLite
```

### Phase 2: Auto-Detection Module

Create `src/graphs/hardware/calibration/auto_detect.py`:

```python
@dataclass
class HardwareIdentity:
    """Auto-detected hardware identity."""
    fingerprint: str
    cpu_model: str
    cpu_stepping: str
    cpu_microcode: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    gpu_model: Optional[str]
    gpu_pci_id: Optional[str]
    gpu_vbios: Optional[str]
    memory_gb: float

    @classmethod
    def detect(cls) -> 'HardwareIdentity':
        """Auto-detect hardware from system."""
        ...


@dataclass
class SoftwareStack:
    """Auto-detected software stack versions."""
    fingerprint: str
    os_kernel: str
    os_distro: str
    driver_version: str
    cuda_version: Optional[str]
    pytorch_version: str
    numpy_version: str

    @classmethod
    def detect(cls) -> 'SoftwareStack':
        """Auto-detect software versions."""
        ...
```

### Phase 3: Remove User-Provided Hardware ID

Update `benchmark_sweep.py`:

```python
# OLD (error-prone)
parser.add_argument("--hardware-id", ...)

# NEW (self-organizing)
# No --hardware-id argument at all!
# Hardware identity is always auto-detected

def main():
    # Auto-detect everything
    hw = HardwareIdentity.detect()
    sw = SoftwareStack.detect()

    print(f"Hardware: {hw.cpu_model}")
    print(f"Fingerprint: {hw.fingerprint}")
    print(f"Software stack: {sw.fingerprint}")

    # Run calibration and store with auto-detected identity
    ...
```

### Phase 4: Temporal Query Tools

Create `cli/analyze_product_trajectory.py`:

```bash
# Show efficiency progression for a specific hardware fingerprint
./cli/analyze_product_trajectory.py --hardware abc123def456

# Compare two products' maturity curves
./cli/analyze_product_trajectory.py --compare hw1 hw2

# Show driver impact on specific hardware
./cli/analyze_product_trajectory.py --hardware abc123 --group-by driver_version

# Detect regressions (new run worse than previous)
./cli/analyze_product_trajectory.py --detect-regressions

# Export trajectory data for visualization
./cli/analyze_product_trajectory.py --export trajectory.csv
```

## Analysis Tools

### 1. Efficiency Trajectory Plot

```
                Efficiency vs Time: NVIDIA H100 SXM5
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

    Driver versions: 535.86 -> 545.23 -> 550.54 -> 560.35
```

### 2. Time-to-Maturity Analysis

```
                Time to Reach Efficiency Milestones

    Product              70%     80%     90%     Current
    ---------------------------------------------------------
    H100 SXM5           Day 0   Day 45  Day 180   92.1%
    MI300X              Day 30  Day 90  Day 270   88.5%
    Gaudi 3             Day 60  Day 180   N/A     78.2%

    Assessment:
    - H100: Excellent support, rapid optimization
    - MI300X: Good support, slower optimization
    - Gaudi 3: Concerning - not reaching 90% threshold
```

### 3. Regression Detection

```
    REGRESSION ALERT: Intel i7-12700K

    Previous best:  513.0 GFLOPS (2025-01-10, driver 560.35)
    Current:        489.2 GFLOPS (2025-01-15, driver 560.40)

    Regression: -4.6%

    Likely cause: Driver update 560.35 -> 560.40
    Recommendation: Report to vendor, pin to 560.35
```

### 4. Cross-Product Comparison

```python
def compare_maturity_trajectories(
    hw_fingerprints: List[str],
    db: CalibrationDB
) -> pd.DataFrame:
    """
    Compare how different products mature over time.

    Returns DataFrame with:
    - days_since_first_calibration
    - efficiency
    - improvement_rate (% per month)
    """
    ...
```

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. Create `auto_detect.py` with hardware/software fingerprinting
2. Update schema to append-only with timestamps
3. Remove `--hardware-id` from CLI
4. Add `--force` for non-performance-mode runs (with warning flags)

### Phase 2: Temporal Queries (Week 2)
1. Add time-series query methods to `CalibrationDB`
2. Create `analyze_product_trajectory.py` CLI
3. Implement efficiency trajectory plotting

### Phase 3: Analysis Tools (Week 3)
1. Regression detection
2. Time-to-maturity analysis
3. Cross-product comparison
4. Export for external visualization (Grafana, etc.)

### Phase 4: Fleet Management (Future)
1. Anonymous telemetry aggregation
2. Fleet-wide regression alerts
3. Vendor performance dashboards

## API Design

### New CalibrationDB Methods

```python
class CalibrationDB:
    # Existing methods remain...

    # NEW: Temporal queries
    def get_trajectory(
        self,
        hardware_fingerprint: str,
        precision: str = "fp32",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[CalibrationRun]:
        """Get all calibration runs for hardware, sorted by time."""

    def get_latest(
        self,
        hardware_fingerprint: str,
        precision: str = "fp32"
    ) -> Optional[CalibrationRun]:
        """Get most recent calibration for hardware."""

    def detect_regressions(
        self,
        threshold_pct: float = 5.0
    ) -> List[RegressionAlert]:
        """Find cases where newer calibration is worse than previous."""

    def get_software_impact(
        self,
        hardware_fingerprint: str,
        group_by: str = "driver_version"
    ) -> Dict[str, List[CalibrationRun]]:
        """Group calibrations by software component to see impact."""

    def get_time_to_milestone(
        self,
        hardware_fingerprint: str,
        efficiency_threshold: float = 0.9
    ) -> Optional[timedelta]:
        """How long to reach efficiency threshold from first calibration?"""
```

## Data Retention Policy

- **Raw calibration data**: Keep indefinitely (append-only)
- **Aggregated metrics**: Compute on-demand or cache for 24h
- **Export format**: Parquet for analytics, JSON for interchange

## Success Metrics

1. **Self-organization**: Zero user-provided hardware IDs in database
2. **Coverage**: Calibration runs for 20+ distinct hardware fingerprints
3. **Temporal depth**: 6+ months of history for key products
4. **Regression detection**: <24h to detect 5%+ regression
5. **Analysis adoption**: Trajectory analysis used in design decisions
