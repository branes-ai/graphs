"""
Calibration Database with Versioned, Append-Only Schema.

Provides:
1. Append-only storage - every run creates a new record (no updates)
2. Self-organizing hardware identity via fingerprints
3. Complete software stack versioning for regression analysis
4. Temporal queries for post-silicon dynamics tracking
5. Similarity matching for design guidance

Key Principles:
- NO user-provided hardware IDs - everything is auto-detected
- NO upserts - every calibration run creates a new record
- Complete provenance - every result is traceable to specific HW+SW

Usage:
    from graphs.hardware.calibration.calibration_db import CalibrationDB
    from graphs.hardware.calibration.auto_detect import CalibrationContext

    # Detect context and run calibration
    context = CalibrationContext.detect()
    db = CalibrationDB("calibrations.db")

    # Store results (creates new record, never updates)
    db.add_run(context, results)

    # Query trajectory over time
    trajectory = db.get_trajectory(context.hardware.fingerprint)

    # Find similar hardware
    similar = db.find_comparable(target_gops=100, target_bandwidth=200)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import sqlite3
import math

from .auto_detect import CalibrationContext, HardwareIdentity, SoftwareStack

# Try to import numpy for vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# =============================================================================
# CALIBRATION RUN RECORD
# =============================================================================

@dataclass
class CalibrationRun:
    """
    A single calibration run with complete provenance.

    Each run is immutable and uniquely identified by:
    (hardware_fingerprint, software_fingerprint, timestamp, precision)

    This enables:
    - Time-series analysis of hardware performance
    - Driver/framework impact analysis
    - Regression detection
    - Cross-product comparisons
    """

    # Identity (from auto-detection)
    run_id: str                     # UUID for this specific run
    hardware_fingerprint: str       # SHA256[:16] of hardware identity
    software_fingerprint: str       # SHA256[:16] of software stack
    timestamp: str                  # ISO 8601 timestamp
    timestamp_unix: int             # Unix epoch for efficient queries

    # Hardware details (denormalized for query efficiency)
    cpu_model: str
    cpu_vendor: str
    cpu_stepping: int
    cpu_microcode: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    gpu_model: str                  # "N/A" if no GPU
    gpu_pci_id: str
    gpu_vbios: str
    gpu_memory_mb: int
    memory_total_gb: float

    # Software stack (denormalized)
    os_kernel: str
    os_distro: str
    gpu_driver_version: str
    cuda_version: str
    pytorch_version: str
    numpy_version: str
    blas_library: str

    # Environmental context
    power_mode: str
    cpu_governor: str
    cpu_freq_mhz: int
    gpu_freq_mhz: int
    thermal_state: str

    # Calibration configuration
    precision: str                  # fp32, fp16, bf16, int8, etc.
    device: str                     # cpu, cuda

    # STREAM memory bandwidth results
    stream_copy_gbps: float = 0.0
    stream_scale_gbps: float = 0.0
    stream_add_gbps: float = 0.0
    stream_triad_gbps: float = 0.0
    stream_best_gbps: float = 0.0

    # BLAS compute results
    blas1_gops: float = 0.0         # Vector-vector (dot, axpy)
    blas2_gops: float = 0.0         # Matrix-vector (gemv)
    blas3_gops: float = 0.0         # Matrix-matrix (gemm)
    peak_measured_gops: float = 0.0

    # Derived metrics
    theoretical_peak_gops: float = 0.0
    efficiency: float = 0.0         # measured / theoretical
    arithmetic_intensity: float = 0.0  # ops/byte at peak

    # Quality indicators
    preflight_passed: bool = True
    forced: bool = False
    notes: str = ""

    @classmethod
    def from_context_and_results(
        cls,
        context: CalibrationContext,
        precision: str,
        device: str,
        stream_results: Dict[str, float],
        blas_results: Dict[str, float],
        theoretical_peak: float = 0.0,
        preflight_passed: bool = True,
        forced: bool = False,
        notes: str = ""
    ) -> 'CalibrationRun':
        """
        Create a CalibrationRun from auto-detected context and benchmark results.
        """
        hw = context.hardware
        sw = context.software
        env = context.environment

        # Extract peak measured GOPS
        peak_gops = max(
            blas_results.get('blas3_gops', 0),
            blas_results.get('blas2_gops', 0),
            blas_results.get('blas1_gops', 0),
        )

        # Calculate efficiency
        efficiency = peak_gops / theoretical_peak if theoretical_peak > 0 else 0.0

        # Calculate arithmetic intensity at peak
        best_bw = stream_results.get('stream_best_gbps', 0) or stream_results.get('stream_copy_gbps', 0)
        ai = peak_gops / best_bw if best_bw > 0 else 0.0

        return cls(
            # Identity
            run_id=context.run_id,
            hardware_fingerprint=hw.fingerprint,
            software_fingerprint=sw.fingerprint,
            timestamp=context.timestamp.isoformat(),
            timestamp_unix=int(context.timestamp.timestamp()),

            # Hardware
            cpu_model=hw.cpu.model,
            cpu_vendor=hw.cpu.vendor,
            cpu_stepping=hw.cpu.stepping,
            cpu_microcode=hw.cpu.microcode,
            cpu_cores_physical=hw.cpu.cores_physical,
            cpu_cores_logical=hw.cpu.cores_logical,
            gpu_model=hw.gpu.model if hw.gpu else "N/A",
            gpu_pci_id=hw.gpu.pci_id if hw.gpu else "N/A",
            gpu_vbios=hw.gpu.vbios_version if hw.gpu else "N/A",
            gpu_memory_mb=hw.gpu.memory_mb if hw.gpu else 0,
            memory_total_gb=hw.memory.total_gb,

            # Software
            os_kernel=sw.os_release,
            os_distro=sw.os_distro,
            gpu_driver_version=sw.gpu_driver_version,
            cuda_version=sw.cuda_version,
            pytorch_version=sw.pytorch_version,
            numpy_version=sw.numpy_version,
            blas_library=sw.blas_library,

            # Environment
            power_mode=env.power_mode,
            cpu_governor=env.cpu_governor,
            cpu_freq_mhz=env.cpu_freq_mhz,
            gpu_freq_mhz=env.gpu_freq_mhz,
            thermal_state=env.thermal_state,

            # Configuration
            precision=precision,
            device=device,

            # Results
            stream_copy_gbps=stream_results.get('copy_gbps', 0),
            stream_scale_gbps=stream_results.get('scale_gbps', 0),
            stream_add_gbps=stream_results.get('add_gbps', 0),
            stream_triad_gbps=stream_results.get('triad_gbps', 0),
            stream_best_gbps=best_bw,
            blas1_gops=blas_results.get('blas1_gops', 0),
            blas2_gops=blas_results.get('blas2_gops', 0),
            blas3_gops=blas_results.get('blas3_gops', 0),
            peak_measured_gops=peak_gops,
            theoretical_peak_gops=theoretical_peak,
            efficiency=efficiency,
            arithmetic_intensity=ai,

            # Quality
            preflight_passed=preflight_passed,
            forced=forced,
            notes=notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationRun':
        """Create from dictionary."""
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_feature_vector(self, features: Optional[List[str]] = None) -> List[float]:
        """Convert to feature vector for similarity matching."""
        if features is None:
            features = [
                'peak_measured_gops',
                'stream_best_gbps',
                'arithmetic_intensity',
                'efficiency',
            ]

        vector = []
        for feat in features:
            value = getattr(self, feat, 0.0)
            vector.append(float(value) if value is not None else 0.0)

        return vector


@dataclass
class RegressionAlert:
    """Alert for detected performance regression."""
    hardware_fingerprint: str
    precision: str
    previous_run: CalibrationRun
    current_run: CalibrationRun
    metric: str                     # Which metric regressed
    previous_value: float
    current_value: float
    regression_pct: float           # Percentage regression

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"REGRESSION: {self.metric} dropped {self.regression_pct:.1f}%\n"
            f"  Hardware: {self.hardware_fingerprint}\n"
            f"  Precision: {self.precision}\n"
            f"  Previous: {self.previous_value:.2f} ({self.previous_run.software_fingerprint})\n"
            f"  Current:  {self.current_value:.2f} ({self.current_run.software_fingerprint})\n"
            f"  Driver change: {self.previous_run.gpu_driver_version} -> {self.current_run.gpu_driver_version}"
        )


# =============================================================================
# CALIBRATION DATABASE
# =============================================================================

class CalibrationDB:
    """
    Append-only calibration database with temporal queries.

    Key design principles:
    1. Every calibration run creates a NEW record (no updates)
    2. Hardware/software fingerprints are auto-generated (not user-provided)
    3. Temporal queries enable post-silicon dynamics analysis
    """

    # Default features for similarity matching
    DEFAULT_SIMILARITY_FEATURES = [
        'peak_measured_gops',
        'stream_best_gbps',
        'arithmetic_intensity',
    ]

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize calibration database.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        """Create database tables with append-only schema."""
        cursor = self.conn.cursor()

        # Main calibration runs table - append-only, no UNIQUE constraint on update
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Identity (auto-detected, immutable)
                run_id TEXT NOT NULL UNIQUE,
                hardware_fingerprint TEXT NOT NULL,
                software_fingerprint TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                timestamp_unix INTEGER NOT NULL,

                -- Hardware details (denormalized)
                cpu_model TEXT NOT NULL,
                cpu_vendor TEXT NOT NULL,
                cpu_stepping INTEGER,
                cpu_microcode TEXT,
                cpu_cores_physical INTEGER,
                cpu_cores_logical INTEGER,
                gpu_model TEXT,
                gpu_pci_id TEXT,
                gpu_vbios TEXT,
                gpu_memory_mb INTEGER,
                memory_total_gb REAL,

                -- Software stack (denormalized)
                os_kernel TEXT,
                os_distro TEXT,
                gpu_driver_version TEXT,
                cuda_version TEXT,
                pytorch_version TEXT,
                numpy_version TEXT,
                blas_library TEXT,

                -- Environmental context
                power_mode TEXT,
                cpu_governor TEXT,
                cpu_freq_mhz INTEGER,
                gpu_freq_mhz INTEGER,
                thermal_state TEXT,

                -- Calibration configuration
                precision TEXT NOT NULL,
                device TEXT NOT NULL,

                -- STREAM results
                stream_copy_gbps REAL,
                stream_scale_gbps REAL,
                stream_add_gbps REAL,
                stream_triad_gbps REAL,
                stream_best_gbps REAL,

                -- BLAS results
                blas1_gops REAL,
                blas2_gops REAL,
                blas3_gops REAL,
                peak_measured_gops REAL,

                -- Derived metrics
                theoretical_peak_gops REAL,
                efficiency REAL,
                arithmetic_intensity REAL,

                -- Quality indicators
                preflight_passed BOOLEAN DEFAULT 1,
                forced BOOLEAN DEFAULT 0,
                notes TEXT
            )
        """)

        # Indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hw_fingerprint
            ON calibration_runs(hardware_fingerprint)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sw_fingerprint
            ON calibration_runs(software_fingerprint)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON calibration_runs(timestamp_unix)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_precision
            ON calibration_runs(precision)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hw_time
            ON calibration_runs(hardware_fingerprint, timestamp_unix)
        """)

        self.conn.commit()

    # =========================================================================
    # WRITE OPERATIONS (Append-Only)
    # =========================================================================

    def add_run(self, run: CalibrationRun) -> int:
        """
        Add a calibration run to the database.

        This always creates a NEW record. Never updates existing records.

        Args:
            run: CalibrationRun to add.

        Returns:
            Row ID of the inserted record.
        """
        cursor = self.conn.cursor()
        data = run.to_dict()

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])

        cursor.execute(f"""
            INSERT INTO calibration_runs ({columns})
            VALUES ({placeholders})
        """, list(data.values()))

        self.conn.commit()
        return cursor.lastrowid

    def add_runs(self, runs: List[CalibrationRun]) -> int:
        """
        Add multiple calibration runs in a single transaction.

        Args:
            runs: List of CalibrationRun objects.

        Returns:
            Number of records inserted.
        """
        if not runs:
            return 0

        cursor = self.conn.cursor()

        for run in runs:
            data = run.to_dict()
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])

            cursor.execute(f"""
                INSERT INTO calibration_runs ({columns})
                VALUES ({placeholders})
            """, list(data.values()))

        self.conn.commit()
        return len(runs)

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    def get_run(self, run_id: str) -> Optional[CalibrationRun]:
        """Get a specific calibration run by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM calibration_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row:
            return CalibrationRun.from_dict(dict(row))
        return None

    def get_latest(
        self,
        hardware_fingerprint: str,
        precision: str = "fp32",
        device: str = "cpu"
    ) -> Optional[CalibrationRun]:
        """
        Get most recent calibration for specific hardware/precision.

        Args:
            hardware_fingerprint: Hardware fingerprint to look up.
            precision: Precision to filter by.
            device: Device type (cpu, cuda).

        Returns:
            Most recent CalibrationRun, or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM calibration_runs
            WHERE hardware_fingerprint = ?
              AND precision = ?
              AND device = ?
            ORDER BY timestamp_unix DESC
            LIMIT 1
        """, (hardware_fingerprint, precision, device))

        row = cursor.fetchone()
        if row:
            return CalibrationRun.from_dict(dict(row))
        return None

    def get_trajectory(
        self,
        hardware_fingerprint: str,
        precision: str = "fp32",
        device: str = "cpu",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[CalibrationRun]:
        """
        Get all calibration runs for hardware over time.

        This is the key method for post-silicon dynamics analysis.

        Args:
            hardware_fingerprint: Hardware to query.
            precision: Precision to filter by.
            device: Device type.
            start_date: Optional start of time range.
            end_date: Optional end of time range.

        Returns:
            List of CalibrationRun objects, sorted by timestamp ascending.
        """
        cursor = self.conn.cursor()

        query = """
            SELECT * FROM calibration_runs
            WHERE hardware_fingerprint = ?
              AND precision = ?
              AND device = ?
        """
        params = [hardware_fingerprint, precision, device]

        if start_date:
            query += " AND timestamp_unix >= ?"
            params.append(int(start_date.timestamp()))

        if end_date:
            query += " AND timestamp_unix <= ?"
            params.append(int(end_date.timestamp()))

        query += " ORDER BY timestamp_unix ASC"

        cursor.execute(query, params)

        return [CalibrationRun.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_all_hardware(self) -> List[str]:
        """Get list of all unique hardware fingerprints in database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT hardware_fingerprint FROM calibration_runs
            ORDER BY hardware_fingerprint
        """)
        return [row[0] for row in cursor.fetchall()]

    def get_hardware_summary(self, hardware_fingerprint: str) -> Dict[str, Any]:
        """
        Get summary of calibrations for specific hardware.

        Returns:
            Dictionary with first/last calibration, run count, etc.
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                MIN(timestamp) as first_calibration,
                MAX(timestamp) as last_calibration,
                COUNT(*) as total_runs,
                COUNT(DISTINCT software_fingerprint) as software_versions,
                COUNT(DISTINCT precision) as precisions_tested,
                cpu_model,
                gpu_model
            FROM calibration_runs
            WHERE hardware_fingerprint = ?
            GROUP BY hardware_fingerprint
        """, (hardware_fingerprint,))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return {}

    # =========================================================================
    # TEMPORAL ANALYSIS
    # =========================================================================

    def detect_regressions(
        self,
        threshold_pct: float = 5.0,
        metric: str = "peak_measured_gops"
    ) -> List[RegressionAlert]:
        """
        Detect performance regressions across all hardware.

        A regression is when a newer calibration is significantly worse
        than a previous calibration on the same hardware/precision.

        Args:
            threshold_pct: Minimum percentage drop to count as regression.
            metric: Which metric to check for regression.

        Returns:
            List of RegressionAlert objects.
        """
        alerts = []

        for hw_fp in self.get_all_hardware():
            for precision in ['fp32', 'fp16', 'int8']:
                trajectory = self.get_trajectory(hw_fp, precision)
                if len(trajectory) < 2:
                    continue

                # Check each consecutive pair
                for i in range(1, len(trajectory)):
                    prev = trajectory[i - 1]
                    curr = trajectory[i]

                    prev_val = getattr(prev, metric, 0)
                    curr_val = getattr(curr, metric, 0)

                    if prev_val > 0 and curr_val < prev_val:
                        regression_pct = (prev_val - curr_val) / prev_val * 100
                        if regression_pct >= threshold_pct:
                            alerts.append(RegressionAlert(
                                hardware_fingerprint=hw_fp,
                                precision=precision,
                                previous_run=prev,
                                current_run=curr,
                                metric=metric,
                                previous_value=prev_val,
                                current_value=curr_val,
                                regression_pct=regression_pct,
                            ))

        return alerts

    def get_software_impact(
        self,
        hardware_fingerprint: str,
        precision: str = "fp32",
        group_by: str = "gpu_driver_version"
    ) -> Dict[str, List[CalibrationRun]]:
        """
        Group calibrations by software component to analyze impact.

        Args:
            hardware_fingerprint: Hardware to analyze.
            precision: Precision to filter by.
            group_by: Software field to group by (e.g., gpu_driver_version).

        Returns:
            Dictionary of {software_version: [CalibrationRun, ...]}.
        """
        trajectory = self.get_trajectory(hardware_fingerprint, precision)

        groups = {}
        for run in trajectory:
            key = getattr(run, group_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(run)

        return groups

    def get_time_to_milestone(
        self,
        hardware_fingerprint: str,
        efficiency_threshold: float = 0.9,
        precision: str = "fp32"
    ) -> Optional[timedelta]:
        """
        Calculate time from first calibration to reaching efficiency threshold.

        Args:
            hardware_fingerprint: Hardware to analyze.
            efficiency_threshold: Target efficiency (e.g., 0.9 = 90%).
            precision: Precision to filter by.

        Returns:
            Time delta from first calibration to threshold, or None if not reached.
        """
        trajectory = self.get_trajectory(hardware_fingerprint, precision)
        if not trajectory:
            return None

        first_time = datetime.fromisoformat(trajectory[0].timestamp)

        for run in trajectory:
            if run.efficiency >= efficiency_threshold:
                milestone_time = datetime.fromisoformat(run.timestamp)
                return milestone_time - first_time

        return None  # Threshold not reached

    def get_improvement_rate(
        self,
        hardware_fingerprint: str,
        precision: str = "fp32",
        metric: str = "efficiency"
    ) -> Optional[float]:
        """
        Calculate improvement rate (% per month) for hardware.

        Args:
            hardware_fingerprint: Hardware to analyze.
            precision: Precision to filter by.
            metric: Metric to track improvement.

        Returns:
            Percentage improvement per month, or None if insufficient data.
        """
        trajectory = self.get_trajectory(hardware_fingerprint, precision)
        if len(trajectory) < 2:
            return None

        first = trajectory[0]
        last = trajectory[-1]

        first_val = getattr(first, metric, 0)
        last_val = getattr(last, metric, 0)

        if first_val == 0:
            return None

        first_time = datetime.fromisoformat(first.timestamp)
        last_time = datetime.fromisoformat(last.timestamp)

        months = (last_time - first_time).days / 30.0
        if months < 0.1:  # Less than 3 days
            return None

        improvement_pct = (last_val - first_val) / first_val * 100
        return improvement_pct / months

    # =========================================================================
    # SIMILARITY MATCHING
    # =========================================================================

    def find_comparable(
        self,
        target_gops: float,
        target_bandwidth: float,
        target_efficiency: float = 0.0,
        device: str = "cpu",
        n: int = 5,
        features: Optional[List[str]] = None
    ) -> List[Tuple[CalibrationRun, float]]:
        """
        Find N most similar calibrated hardware to a target specification.

        Uses cosine similarity on normalized roofline features.

        Args:
            target_gops: Target peak GOPS.
            target_bandwidth: Target memory bandwidth (GB/s).
            target_efficiency: Target efficiency (optional).
            device: Device type filter.
            n: Number of results to return.
            features: Feature names for similarity.

        Returns:
            List of (CalibrationRun, similarity_score) tuples.
        """
        if features is None:
            features = self.DEFAULT_SIMILARITY_FEATURES

        # Get latest calibration for each hardware
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM calibration_runs
            WHERE device = ?
              AND precision = 'fp32'
            GROUP BY hardware_fingerprint
            HAVING timestamp_unix = MAX(timestamp_unix)
        """, (device,))

        candidates = [CalibrationRun.from_dict(dict(row)) for row in cursor.fetchall()]

        if not candidates:
            return []

        # Build target vector
        target_vec = [target_gops, target_bandwidth]
        if 'arithmetic_intensity' in features:
            ai = target_gops / target_bandwidth if target_bandwidth > 0 else 0
            target_vec.append(ai)
        if 'efficiency' in features:
            target_vec.append(target_efficiency)

        # Compute similarities
        results = []
        for cal in candidates:
            cal_vec = cal.to_feature_vector(features)
            similarity = self._cosine_similarity(target_vec, cal_vec)
            results.append((cal, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: -x[1])
        return results[:n]

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if NUMPY_AVAILABLE:
            a = np.array(vec_a)
            b = np.array(vec_b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        else:
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

    # =========================================================================
    # SUMMARY & EXPORT
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get database summary statistics."""
        cursor = self.conn.cursor()

        stats = {
            'total_runs': 0,
            'unique_hardware': 0,
            'unique_software': 0,
            'date_range': {},
            'by_device': {},
            'by_precision': {},
            'by_cpu_vendor': {},
        }

        cursor.execute("SELECT COUNT(*) FROM calibration_runs")
        stats['total_runs'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT hardware_fingerprint) FROM calibration_runs")
        stats['unique_hardware'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT software_fingerprint) FROM calibration_runs")
        stats['unique_software'] = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM calibration_runs")
        row = cursor.fetchone()
        if row[0]:
            stats['date_range'] = {'first': row[0], 'last': row[1]}

        cursor.execute("""
            SELECT device, COUNT(*) FROM calibration_runs GROUP BY device
        """)
        for row in cursor.fetchall():
            stats['by_device'][row[0]] = row[1]

        cursor.execute("""
            SELECT precision, COUNT(*) FROM calibration_runs GROUP BY precision
        """)
        for row in cursor.fetchall():
            stats['by_precision'][row[0]] = row[1]

        cursor.execute("""
            SELECT cpu_vendor, COUNT(*) FROM calibration_runs GROUP BY cpu_vendor
        """)
        for row in cursor.fetchall():
            stats['by_cpu_vendor'][row[0]] = row[1]

        return stats

    def export_json(self, output_path: str) -> bool:
        """Export database to JSON format."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM calibration_runs ORDER BY timestamp_unix")

            runs = [dict(row) for row in cursor.fetchall()]

            data = {
                'runs': runs,
                'summary': self.get_summary(),
                'export_date': datetime.now(timezone.utc).isoformat(),
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

    def export_parquet(self, output_path: str) -> bool:
        """Export database to Parquet format for analytics."""
        try:
            import pandas as pd
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM calibration_runs")

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            df = pd.DataFrame.from_records(rows, columns=columns)
            df.to_parquet(output_path, index=False)
            return True

        except ImportError:
            print("Parquet export requires pandas: pip install pandas pyarrow")
            return False
        except Exception as e:
            print(f"Error exporting to Parquet: {e}")
            return False

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# MIGRATION HELPER
# =============================================================================

def migrate_legacy_database(
    legacy_db_path: str,
    new_db_path: str
) -> int:
    """
    Migrate data from legacy calibration database to new versioned schema.

    This is a one-way migration. The legacy database is not modified.

    Args:
        legacy_db_path: Path to old calibrations.db
        new_db_path: Path for new versioned database

    Returns:
        Number of records migrated.
    """
    # Import legacy schema
    legacy_conn = sqlite3.connect(legacy_db_path)
    legacy_conn.row_factory = sqlite3.Row

    new_db = CalibrationDB(new_db_path)
    migrated = 0

    try:
        cursor = legacy_conn.cursor()
        cursor.execute("SELECT * FROM calibrations")

        for row in cursor.fetchall():
            data = dict(row)

            # Map legacy fields to new schema
            # Generate synthetic run_id and timestamps for legacy data
            import uuid
            run_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc)

            run = CalibrationRun(
                run_id=run_id,
                hardware_fingerprint=data.get('hardware_id', 'legacy')[:16],
                software_fingerprint='legacy_migration',
                timestamp=timestamp.isoformat(),
                timestamp_unix=int(timestamp.timestamp()),
                cpu_model=data.get('hardware_id', 'Unknown'),
                cpu_vendor=data.get('vendor', 'Unknown'),
                cpu_stepping=0,
                cpu_microcode='unknown',
                cpu_cores_physical=0,
                cpu_cores_logical=0,
                gpu_model='N/A',
                gpu_pci_id='N/A',
                gpu_vbios='N/A',
                gpu_memory_mb=0,
                memory_total_gb=0,
                os_kernel='unknown',
                os_distro='unknown',
                gpu_driver_version='N/A',
                cuda_version='N/A',
                pytorch_version='unknown',
                numpy_version='unknown',
                blas_library='unknown',
                power_mode=data.get('power_mode', 'unknown'),
                cpu_governor='unknown',
                cpu_freq_mhz=data.get('clock_mhz', 0),
                gpu_freq_mhz=0,
                thermal_state='unknown',
                precision=data.get('precision', 'fp32'),
                device=data.get('device_type', 'cpu'),
                stream_best_gbps=data.get('bandwidth_gbps', 0),
                blas3_gops=data.get('gemm_peak_gops', 0),
                peak_measured_gops=data.get('gemm_peak_gops', 0),
                theoretical_peak_gops=data.get('theoretical_peak_gops', 0),
                efficiency=data.get('gemm_efficiency', 0),
                notes='Migrated from legacy database',
            )

            new_db.add_run(run)
            migrated += 1

    except Exception as e:
        print(f"Migration error: {e}")

    finally:
        legacy_conn.close()
        new_db.close()

    return migrated
