"""
Calibration Database with Normalized Schema and Similarity Matching.

Provides:
1. Normalized schema for queryable calibration results
2. SQLite storage backend for fast queries
3. Similarity matching to find comparable hardware architectures
4. Export to Parquet for analytics

Usage:
    from graphs.hardware.calibration.calibration_db import CalibrationDB, CalibrationPoint

    # Store calibration results
    db = CalibrationDB("calibrations.db")
    db.add_calibration(point)

    # Query for similar hardware
    similar = db.find_comparable(
        target_gops=100,
        target_bandwidth=200,
        target_tdp=15,
        n=5
    )
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import sqlite3
import math

# Try to import numpy for vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# =============================================================================
# NORMALIZED CALIBRATION SCHEMA
# =============================================================================

@dataclass
class CalibrationPoint:
    """
    Normalized calibration data point for a single hardware configuration.

    This schema captures the essential roofline characteristics that enable
    comparison across different hardware architectures.
    """

    # Hardware Identity
    hardware_id: str              # e.g., "h100_sxm5_80gb"
    vendor: str                   # e.g., "NVIDIA", "Intel", "AMD"
    architecture: str             # e.g., "hopper", "zen4", "systolic"
    device_type: str              # cpu, gpu, tpu, dsp, accelerator

    # Configuration
    precision: str                # fp32, fp16, bf16, int8
    power_mode: str               # "TDP", "15W", "30W", "MAXN"
    clock_mhz: int                # Actual clock during calibration

    # Micro-Kernel Results (comparable across architectures)
    gemm_peak_gops: float         # Peak measured GOPS (generic for float/int)
    gemm_efficiency: float        # Measured / theoretical (0.0 to 1.0+)
    bandwidth_gbps: float         # STREAM triad bandwidth
    bandwidth_efficiency: float   # Measured / theoretical bandwidth

    # Roofline Characteristics (key for similarity matching)
    theoretical_peak_gops: float  # From hardware spec
    theoretical_bandwidth_gbps: float
    arithmetic_intensity_transition: float  # ops/byte where compute-bound starts

    # Derived Characteristics (for similarity matching)
    gops_per_watt: float = 0.0
    bytes_per_op_at_peak: float = 0.0
    memory_bound_fraction: float = 0.0  # Fraction of ops that are memory-bound

    # BLAS Level Performance (normalized)
    blas1_gops: float = 0.0       # Vector-vector ops (dot, axpy)
    blas2_gops: float = 0.0       # Matrix-vector ops (gemv)
    blas3_gops: float = 0.0       # Matrix-matrix ops (gemm)

    # Metadata
    calibration_date: str = ""
    framework: str = ""           # numpy, pytorch
    framework_version: str = ""
    driver_version: str = ""
    notes: str = ""

    def __post_init__(self):
        """Compute derived metrics after initialization."""
        # Compute arithmetic intensity transition point (ridge point in roofline)
        if self.theoretical_bandwidth_gbps > 0 and self.theoretical_peak_gops > 0:
            # Ridge point = peak_gops / bandwidth_gbps (ops/byte)
            self.arithmetic_intensity_transition = (
                self.theoretical_peak_gops / self.theoretical_bandwidth_gbps
            )

        # Bytes per op at peak (inverse of AI transition)
        if self.arithmetic_intensity_transition > 0:
            self.bytes_per_op_at_peak = 1.0 / self.arithmetic_intensity_transition

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/SQLite serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationPoint':
        """Create from dictionary."""
        # Filter to known fields
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_feature_vector(self, features: Optional[List[str]] = None) -> List[float]:
        """
        Convert to feature vector for similarity matching.

        Args:
            features: List of field names to include. Defaults to roofline features.

        Returns:
            List of float values for the specified features.
        """
        if features is None:
            features = [
                'gemm_peak_gops',
                'bandwidth_gbps',
                'arithmetic_intensity_transition',
                'gops_per_watt',
            ]

        vector = []
        for feat in features:
            value = getattr(self, feat, 0.0)
            vector.append(float(value) if value is not None else 0.0)

        return vector


@dataclass
class HypotheticalArchitecture:
    """
    Specification for a hypothetical or new architecture.

    Used to query the database for similar existing hardware.
    """
    peak_gops: float              # Target peak GOPS
    bandwidth_gbps: float         # Target memory bandwidth
    tdp_watts: float = 0.0        # Target TDP (optional)
    device_type: str = ""         # cpu, gpu, accelerator (optional filter)
    precision: str = "fp32"       # Target precision

    def to_feature_vector(self, features: Optional[List[str]] = None) -> List[float]:
        """Convert to feature vector for similarity matching."""
        if features is None:
            features = [
                'gemm_peak_gops',
                'bandwidth_gbps',
                'arithmetic_intensity_transition',
                'gops_per_watt',
            ]

        # Map hypothetical fields to calibration point fields
        field_mapping = {
            'gemm_peak_gops': self.peak_gops,
            'bandwidth_gbps': self.bandwidth_gbps,
            'arithmetic_intensity_transition': (
                self.peak_gops / self.bandwidth_gbps
                if self.bandwidth_gbps > 0 else 0.0
            ),
            'gops_per_watt': (
                self.peak_gops / self.tdp_watts
                if self.tdp_watts > 0 else 0.0
            ),
        }

        return [field_mapping.get(f, 0.0) for f in features]


# =============================================================================
# CALIBRATION DATABASE
# =============================================================================

class CalibrationDB:
    """
    SQLite-backed calibration database with similarity matching.

    Stores normalized calibration points and supports fast queries for:
    - Finding calibrations by hardware ID
    - Finding similar hardware by roofline characteristics
    - Aggregating results across precisions/power modes
    """

    # Default features for similarity matching
    DEFAULT_SIMILARITY_FEATURES = [
        'gemm_peak_gops',
        'bandwidth_gbps',
        'arithmetic_intensity_transition',
        'gops_per_watt',
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
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Hardware Identity
                hardware_id TEXT NOT NULL,
                vendor TEXT NOT NULL,
                architecture TEXT NOT NULL,
                device_type TEXT NOT NULL,

                -- Configuration
                precision TEXT NOT NULL,
                power_mode TEXT NOT NULL,
                clock_mhz INTEGER NOT NULL,

                -- Micro-Kernel Results
                gemm_peak_gops REAL NOT NULL,
                gemm_efficiency REAL NOT NULL,
                bandwidth_gbps REAL NOT NULL,
                bandwidth_efficiency REAL NOT NULL,

                -- Roofline Characteristics
                theoretical_peak_gops REAL NOT NULL,
                theoretical_bandwidth_gbps REAL NOT NULL,
                arithmetic_intensity_transition REAL NOT NULL,

                -- Derived Characteristics
                gops_per_watt REAL DEFAULT 0.0,
                bytes_per_op_at_peak REAL DEFAULT 0.0,
                memory_bound_fraction REAL DEFAULT 0.0,

                -- BLAS Level Performance
                blas1_gops REAL DEFAULT 0.0,
                blas2_gops REAL DEFAULT 0.0,
                blas3_gops REAL DEFAULT 0.0,

                -- Metadata
                calibration_date TEXT,
                framework TEXT,
                framework_version TEXT,
                driver_version TEXT,
                notes TEXT,

                -- Composite unique key
                UNIQUE(hardware_id, precision, power_mode, clock_mhz)
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hardware_id
            ON calibrations(hardware_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_device_type
            ON calibrations(device_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_precision
            ON calibrations(precision)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_gemm_peak
            ON calibrations(gemm_peak_gops)
        """)

        self.conn.commit()

    def add_calibration(self, point: CalibrationPoint) -> int:
        """
        Add or update a calibration point.

        Args:
            point: CalibrationPoint to add.

        Returns:
            Row ID of the inserted/updated record.
        """
        cursor = self.conn.cursor()

        data = point.to_dict()

        # Use INSERT OR REPLACE to handle duplicates
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])

        cursor.execute(f"""
            INSERT OR REPLACE INTO calibrations ({columns})
            VALUES ({placeholders})
        """, list(data.values()))

        self.conn.commit()
        return cursor.lastrowid

    def add_calibrations(self, points: List[CalibrationPoint]) -> int:
        """
        Add multiple calibration points in a single transaction.

        Args:
            points: List of CalibrationPoint objects.

        Returns:
            Number of records inserted.
        """
        if not points:
            return 0

        cursor = self.conn.cursor()

        for point in points:
            data = point.to_dict()
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])

            cursor.execute(f"""
                INSERT OR REPLACE INTO calibrations ({columns})
                VALUES ({placeholders})
            """, list(data.values()))

        self.conn.commit()
        return len(points)

    def get_calibration(
        self,
        hardware_id: str,
        precision: str = "fp32",
        power_mode: Optional[str] = None
    ) -> Optional[CalibrationPoint]:
        """
        Get calibration for specific hardware configuration.

        Args:
            hardware_id: Hardware ID to look up.
            precision: Precision to filter by.
            power_mode: Power mode to filter by (optional).

        Returns:
            CalibrationPoint if found, None otherwise.
        """
        cursor = self.conn.cursor()

        if power_mode:
            cursor.execute("""
                SELECT * FROM calibrations
                WHERE hardware_id = ? AND precision = ? AND power_mode = ?
                ORDER BY clock_mhz DESC
                LIMIT 1
            """, (hardware_id, precision, power_mode))
        else:
            cursor.execute("""
                SELECT * FROM calibrations
                WHERE hardware_id = ? AND precision = ?
                ORDER BY clock_mhz DESC
                LIMIT 1
            """, (hardware_id, precision))

        row = cursor.fetchone()
        if row:
            return CalibrationPoint.from_dict(dict(row))
        return None

    def all_calibrations(
        self,
        device_type: Optional[str] = None,
        precision: Optional[str] = None
    ) -> List[CalibrationPoint]:
        """
        Get all calibration points, optionally filtered.

        Args:
            device_type: Filter by device type (cpu, gpu, etc.).
            precision: Filter by precision (fp32, fp16, etc.).

        Returns:
            List of CalibrationPoint objects.
        """
        cursor = self.conn.cursor()

        conditions = []
        params = []

        if device_type:
            conditions.append("device_type = ?")
            params.append(device_type)
        if precision:
            conditions.append("precision = ?")
            params.append(precision)

        query = "SELECT * FROM calibrations"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)

        return [CalibrationPoint.from_dict(dict(row)) for row in cursor.fetchall()]

    def find_comparable(
        self,
        target: HypotheticalArchitecture,
        n: int = 5,
        features: Optional[List[str]] = None,
        device_type_filter: Optional[str] = None
    ) -> List[Tuple[CalibrationPoint, float]]:
        """
        Find N most similar calibrated hardware to a hypothetical target.

        Uses cosine similarity on normalized roofline features.

        Args:
            target: HypotheticalArchitecture describing the target.
            n: Number of results to return.
            features: Feature names for similarity (default: roofline features).
            device_type_filter: Optional filter by device type.

        Returns:
            List of (CalibrationPoint, similarity_score) tuples, sorted by similarity.
        """
        if features is None:
            features = self.DEFAULT_SIMILARITY_FEATURES

        # Get all calibrations (optionally filtered by device type)
        device_filter = device_type_filter or target.device_type or None
        all_cals = self.all_calibrations(device_type=device_filter)

        if not all_cals:
            return []

        # Get target feature vector
        target_vec = target.to_feature_vector(features)

        # Compute similarities
        results = []
        for cal in all_cals:
            cal_vec = cal.to_feature_vector(features)
            similarity = self._cosine_similarity(target_vec, cal_vec)
            results.append((cal, similarity))

        # Sort by similarity (descending) and return top N
        results.sort(key=lambda x: -x[1])
        return results[:n]

    def find_by_range(
        self,
        min_gops: float = 0.0,
        max_gops: float = float('inf'),
        min_bandwidth: float = 0.0,
        max_bandwidth: float = float('inf'),
        device_type: Optional[str] = None,
        precision: str = "fp32"
    ) -> List[CalibrationPoint]:
        """
        Find hardware within specified performance ranges.

        Args:
            min_gops: Minimum peak GOPS.
            max_gops: Maximum peak GOPS.
            min_bandwidth: Minimum bandwidth (GB/s).
            max_bandwidth: Maximum bandwidth (GB/s).
            device_type: Filter by device type.
            precision: Filter by precision.

        Returns:
            List of matching CalibrationPoint objects.
        """
        cursor = self.conn.cursor()

        conditions = [
            "gemm_peak_gops >= ?",
            "gemm_peak_gops <= ?",
            "bandwidth_gbps >= ?",
            "bandwidth_gbps <= ?",
            "precision = ?"
        ]
        params = [min_gops, max_gops, min_bandwidth, max_bandwidth, precision]

        if device_type:
            conditions.append("device_type = ?")
            params.append(device_type)

        query = "SELECT * FROM calibrations WHERE " + " AND ".join(conditions)
        cursor.execute(query, params)

        return [CalibrationPoint.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_efficiency_estimate(
        self,
        target: HypotheticalArchitecture,
        n_samples: int = 5
    ) -> Tuple[float, float, List[str]]:
        """
        Estimate efficiency factor for a hypothetical architecture.

        Based on average efficiency of similar calibrated hardware.

        Args:
            target: HypotheticalArchitecture to estimate for.
            n_samples: Number of similar hardware to average.

        Returns:
            Tuple of (mean_efficiency, std_efficiency, source_hardware_ids).
        """
        similar = self.find_comparable(target, n=n_samples)

        if not similar:
            # No comparable data - return conservative estimate
            return (0.5, 0.2, [])

        efficiencies = [cal.gemm_efficiency for cal, _ in similar]
        hardware_ids = [cal.hardware_id for cal, _ in similar]

        mean_eff = sum(efficiencies) / len(efficiencies)

        # Compute std deviation
        if len(efficiencies) > 1:
            variance = sum((e - mean_eff) ** 2 for e in efficiencies) / len(efficiencies)
            std_eff = math.sqrt(variance)
        else:
            std_eff = 0.1  # Default uncertainty

        return (mean_eff, std_eff, hardware_ids)

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Handles zero vectors gracefully.
        """
        if NUMPY_AVAILABLE:
            a = np.array(vec_a)
            b = np.array(vec_b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        else:
            # Pure Python fallback
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

    def _normalize_vector(self, vec: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        if NUMPY_AVAILABLE:
            arr = np.array(vec)
            norm = np.linalg.norm(arr)
            if norm == 0:
                return vec
            return (arr / norm).tolist()
        else:
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0:
                return vec
            return [v / norm for v in vec]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get database summary statistics.

        Returns:
            Dictionary with counts and breakdowns.
        """
        cursor = self.conn.cursor()

        stats = {
            'total_calibrations': 0,
            'unique_hardware': 0,
            'by_device_type': {},
            'by_precision': {},
            'by_vendor': {},
        }

        # Total count
        cursor.execute("SELECT COUNT(*) FROM calibrations")
        stats['total_calibrations'] = cursor.fetchone()[0]

        # Unique hardware
        cursor.execute("SELECT COUNT(DISTINCT hardware_id) FROM calibrations")
        stats['unique_hardware'] = cursor.fetchone()[0]

        # By device type
        cursor.execute("""
            SELECT device_type, COUNT(*) as cnt
            FROM calibrations
            GROUP BY device_type
        """)
        for row in cursor.fetchall():
            stats['by_device_type'][row['device_type']] = row['cnt']

        # By precision
        cursor.execute("""
            SELECT precision, COUNT(*) as cnt
            FROM calibrations
            GROUP BY precision
        """)
        for row in cursor.fetchall():
            stats['by_precision'][row['precision']] = row['cnt']

        # By vendor
        cursor.execute("""
            SELECT vendor, COUNT(*) as cnt
            FROM calibrations
            GROUP BY vendor
        """)
        for row in cursor.fetchall():
            stats['by_vendor'][row['vendor']] = row['cnt']

        return stats

    def export_parquet(self, output_path: str) -> bool:
        """
        Export database to Parquet format for analytics.

        Requires pyarrow or pandas to be installed.

        Args:
            output_path: Path for output Parquet file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            import pandas as pd
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM calibrations")

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            df = pd.DataFrame.from_records(rows, columns=columns)
            df.to_parquet(output_path, index=False)
            return True

        except ImportError:
            print("Parquet export requires pandas with pyarrow: pip install pandas pyarrow")
            return False
        except Exception as e:
            print(f"Error exporting to Parquet: {e}")
            return False

    def export_json(self, output_path: str) -> bool:
        """
        Export database to JSON format.

        Args:
            output_path: Path for output JSON file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            all_cals = self.all_calibrations()
            data = {
                'calibrations': [cal.to_dict() for cal in all_cals],
                'summary': self.get_summary(),
                'export_date': datetime.now().isoformat(),
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

    def import_from_hardware_registry(self, registry_path: Path) -> int:
        """
        Import calibrations from existing hardware_registry JSON files.

        Args:
            registry_path: Path to hardware_registry/ directory.

        Returns:
            Number of calibrations imported.
        """
        from .schema import HardwareCalibration

        imported = 0
        registry_path = Path(registry_path)

        # Find all calibration JSON files
        for cal_file in registry_path.rglob("calibrations/*.json"):
            try:
                # Load the calibration
                calibration = HardwareCalibration.load(cal_file)

                # Extract hardware ID from path
                # e.g., hardware_registry/cpu/intel.../calibrations/xxx.json
                parts = cal_file.parts
                hardware_dir = cal_file.parent.parent
                hardware_id = hardware_dir.name

                # Determine device type from path
                device_type = "unknown"
                for part in parts:
                    if part in ['cpu', 'gpu', 'accelerator', 'dsp']:
                        device_type = part
                        break

                # Create CalibrationPoint for each precision result
                for op_key, op_cal in calibration.operation_profiles.items():
                    if op_cal.precision_results:
                        for prec, result in op_cal.precision_results.items():
                            if result.supported and result.measured_gops:
                                point = CalibrationPoint(
                                    hardware_id=hardware_id,
                                    vendor=self._extract_vendor(hardware_id),
                                    architecture=self._extract_arch(hardware_id),
                                    device_type=device_type,
                                    precision=prec,
                                    power_mode=calibration.metadata.gpu_clock.power_mode_name
                                        if calibration.metadata.gpu_clock else "default",
                                    clock_mhz=int(calibration.metadata.cpu_clock.current_freq_mhz)
                                        if calibration.metadata.cpu_clock else 0,
                                    gemm_peak_gops=result.measured_gops or 0.0,
                                    gemm_efficiency=result.efficiency or 0.0,
                                    bandwidth_gbps=calibration.measured_bandwidth_gbps,
                                    bandwidth_efficiency=calibration.bandwidth_efficiency,
                                    theoretical_peak_gops=calibration.theoretical_peak_gflops,
                                    theoretical_bandwidth_gbps=calibration.theoretical_bandwidth_gbps,
                                    arithmetic_intensity_transition=(
                                        calibration.theoretical_peak_gflops /
                                        calibration.theoretical_bandwidth_gbps
                                        if calibration.theoretical_bandwidth_gbps > 0 else 0.0
                                    ),
                                    calibration_date=calibration.metadata.calibration_date,
                                    framework=calibration.metadata.framework,
                                    framework_version=calibration.metadata.numpy_version
                                        or calibration.metadata.pytorch_version or "",
                                )
                                self.add_calibration(point)
                                imported += 1

            except Exception as e:
                print(f"[!] Error importing {cal_file}: {e}")
                continue

        return imported

    def _extract_vendor(self, hardware_id: str) -> str:
        """Extract vendor from hardware ID."""
        id_lower = hardware_id.lower()
        if 'intel' in id_lower or 'i7' in id_lower or 'i9' in id_lower or 'xeon' in id_lower:
            return 'Intel'
        elif 'amd' in id_lower or 'ryzen' in id_lower or 'epyc' in id_lower:
            return 'AMD'
        elif 'nvidia' in id_lower or 'h100' in id_lower or 'a100' in id_lower or 'jetson' in id_lower:
            return 'NVIDIA'
        elif 'ampere' in id_lower:
            return 'Ampere Computing'
        elif 'qualcomm' in id_lower:
            return 'Qualcomm'
        elif 'arm' in id_lower or 'mali' in id_lower:
            return 'ARM'
        elif 'google' in id_lower or 'tpu' in id_lower:
            return 'Google'
        elif 'hailo' in id_lower:
            return 'Hailo'
        return 'Unknown'

    def _extract_arch(self, hardware_id: str) -> str:
        """Extract architecture from hardware ID."""
        id_lower = hardware_id.lower()
        if 'h100' in id_lower:
            return 'hopper'
        elif 'a100' in id_lower:
            return 'ampere'
        elif 'v100' in id_lower:
            return 'volta'
        elif 't4' in id_lower:
            return 'turing'
        elif 'jetson' in id_lower and 'orin' in id_lower:
            return 'ampere'
        elif 'i7' in id_lower or 'i9' in id_lower:
            gen = '12th' if '12' in id_lower else '13th' if '13' in id_lower else 'unknown'
            return f'alder_lake' if '12' in id_lower else f'raptor_lake' if '13' in id_lower else 'intel'
        elif 'ryzen' in id_lower:
            return 'zen4' if '8' in id_lower else 'zen3' if '7' in id_lower else 'zen'
        elif 'xeon' in id_lower:
            return 'sapphire_rapids'
        elif 'epyc' in id_lower:
            return 'zen4'
        elif 'tpu' in id_lower:
            return 'systolic'
        elif 'mali' in id_lower:
            return 'valhall'
        return 'unknown'

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
