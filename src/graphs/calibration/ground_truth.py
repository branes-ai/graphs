"""
Ground-Truth Measurement Schema v2.0 and Loader.

Defines the canonical data model for hardware measurement results.
These measurements serve as ground truth for estimator validation --
collected once on physical hardware, then loaded repeatedly on dev machines.

Schema v2.0 extends v1.0 (from measure_efficiency.py) with:
- batch_size and input_shape (was only in filename)
- model_summary (whole-model aggregates)
- system_state (hardware conditions during measurement)
- run_id / run_index for tracking repeat runs

The GroundTruthLoader provides query access to stored measurement data,
handles v1.0 backward compatibility, and manages manifest generation.

Usage:
    from graphs.calibration.ground_truth import GroundTruthLoader, MeasurementRecord

    loader = GroundTruthLoader()
    record = loader.load('jetson_orin_agx_maxn', 'resnet18', 'fp32', batch_size=1)
    print(record.model_summary.total_latency_ms)

    # List available data
    for hw in loader.list_hardware():
        configs = loader.list_configurations(hw)
        print(f"{hw}: {len(configs)} configurations")
"""

import json
import math
import re
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


# Default data directory (repo_root/calibration_data)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_DATA_DIR = _REPO_ROOT / "calibration_data"


# ============================================================================
# Schema v2.0 dataclasses
# ============================================================================

@dataclass
class LatencyStats:
    """Statistical summary of latency measurements (ms)."""
    mean: float
    std: float
    min: float
    max: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatencyStats':
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class EfficiencyStats:
    """Statistical summary of efficiency measurements (ratio 0-1)."""
    mean: float
    std: float
    min: float
    max: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EfficiencyStats':
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SubgraphMeasurement:
    """Complete measurement data for a single subgraph."""
    subgraph_id: int
    fusion_pattern: str
    operation_type: str
    flops: int
    total_bytes: int
    arithmetic_intensity: float
    theoretical_peak_flops: float
    measured_latency: LatencyStats
    achieved_flops: float
    efficiency: EfficiencyStats
    node_names: List[str]
    source_model: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['measured_latency'] = self.measured_latency.to_dict()
        d['efficiency'] = self.efficiency.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubgraphMeasurement':
        data = data.copy()
        if 'measured_latency' in data and isinstance(data['measured_latency'], dict):
            data['measured_latency'] = LatencyStats.from_dict(data['measured_latency'])
        if 'efficiency' in data and isinstance(data['efficiency'], dict):
            data['efficiency'] = EfficiencyStats.from_dict(data['efficiency'])
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModelSummary:
    """Whole-model aggregate metrics, computed from subgraph measurements."""
    total_flops: int
    total_bytes: int
    total_latency_ms: float        # sum of subgraph mean latencies
    total_latency_std_ms: float    # propagated std (sqrt of sum of variances)
    num_subgraphs: int
    throughput_fps: float          # batch_size / (total_latency_ms / 1000)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelSummary':
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_subgraphs(
        cls,
        subgraphs: List[SubgraphMeasurement],
        batch_size: int = 1,
    ) -> 'ModelSummary':
        """Compute summary from a list of subgraph measurements."""
        total_flops = sum(sg.flops for sg in subgraphs)
        total_bytes = sum(sg.total_bytes for sg in subgraphs)
        total_latency_ms = sum(sg.measured_latency.mean for sg in subgraphs)
        # Propagate std: sqrt(sum(var_i)) assuming independent subgraphs
        total_var = sum(sg.measured_latency.std ** 2 for sg in subgraphs)
        total_latency_std_ms = math.sqrt(total_var) if total_var > 0 else 0.0
        num_subgraphs = len(subgraphs)
        if total_latency_ms > 0:
            throughput_fps = batch_size / (total_latency_ms / 1000.0)
        else:
            throughput_fps = 0.0
        return cls(
            total_flops=total_flops,
            total_bytes=total_bytes,
            total_latency_ms=total_latency_ms,
            total_latency_std_ms=total_latency_std_ms,
            num_subgraphs=num_subgraphs,
            throughput_fps=throughput_fps,
        )


@dataclass
class SystemState:
    """Hardware conditions captured during measurement."""
    gpu_clock_mhz: Optional[int] = None
    mem_clock_mhz: Optional[int] = None
    power_mode: Optional[str] = None
    temperature_c: Optional[float] = None
    cpu_freq_mhz: Optional[float] = None
    cpu_governor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        if data is None:
            return cls()
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class MeasurementRecord:
    """
    Complete measurement record for one (model, hardware, precision, batch_size).

    Schema v2.0 -- backward-compatible with v1.0 JSON files.
    """
    # --- v1.0 fields (unchanged) ---
    schema_version: str                    # "1.0" or "2.0"
    measurement_type: str                  # "efficiency"
    model: str                             # "resnet18"
    hardware_id: str                       # "jetson_orin_agx_maxn"
    device: str                            # "cuda" or "cpu"
    precision: str                         # "FP32", "FP16", "BF16"
    thermal_profile: Optional[str]         # "MAXN", "15W", etc.
    theoretical_peak_flops: float
    measurement_date: str                  # ISO 8601
    tool_version: str
    subgraphs: List[SubgraphMeasurement]

    # --- v2.0 fields ---
    batch_size: int = 1
    input_shape: Optional[List[int]] = None  # e.g., [1, 3, 224, 224]
    model_summary: Optional[ModelSummary] = None
    system_state: Optional[SystemState] = None
    run_id: Optional[str] = None           # UUID for tracking repeat runs
    run_index: Optional[int] = None        # 1, 2, 3 for repeat sweeps

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "measurement_type": self.measurement_type,
            "model": self.model,
            "hardware_id": self.hardware_id,
            "device": self.device,
            "precision": self.precision,
            "thermal_profile": self.thermal_profile,
            "theoretical_peak_flops": self.theoretical_peak_flops,
            "measurement_date": self.measurement_date,
            "tool_version": self.tool_version,
            "batch_size": self.batch_size,
            "subgraphs": [sg.to_dict() for sg in self.subgraphs],
        }
        if self.input_shape is not None:
            result["input_shape"] = self.input_shape
        if self.model_summary is not None:
            result["model_summary"] = self.model_summary.to_dict()
        if self.system_state is not None:
            result["system_state"] = self.system_state.to_dict()
        if self.run_id is not None:
            result["run_id"] = self.run_id
        if self.run_index is not None:
            result["run_index"] = self.run_index
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], filename: Optional[str] = None) -> 'MeasurementRecord':
        """Create from dictionary. Handles v1.0 backward compatibility.

        Args:
            data: Parsed JSON dictionary.
            filename: Optional filename for inferring batch_size from v1.0 files.
        """
        data = data.copy()

        # Parse subgraphs
        subgraphs = []
        for sg_data in data.get('subgraphs', []):
            subgraphs.append(SubgraphMeasurement.from_dict(sg_data))

        # Infer batch_size for v1.0 files
        batch_size = data.get('batch_size', 1)
        if data.get('schema_version') == '1.0' and 'batch_size' not in data:
            batch_size = _infer_batch_size_from_filename(filename)

        # Parse model_summary or compute from subgraphs
        model_summary = None
        if 'model_summary' in data and data['model_summary'] is not None:
            model_summary = ModelSummary.from_dict(data['model_summary'])
        elif subgraphs:
            model_summary = ModelSummary.from_subgraphs(subgraphs, batch_size)

        # Parse system_state
        system_state = None
        if 'system_state' in data and data['system_state'] is not None:
            system_state = SystemState.from_dict(data['system_state'])

        return cls(
            schema_version=data.get('schema_version', '1.0'),
            measurement_type=data.get('measurement_type', 'efficiency'),
            model=data.get('model', ''),
            hardware_id=data.get('hardware_id', ''),
            device=data.get('device', 'cpu'),
            precision=data.get('precision', 'FP32'),
            thermal_profile=data.get('thermal_profile'),
            theoretical_peak_flops=data.get('theoretical_peak_flops', 0.0),
            measurement_date=data.get('measurement_date', ''),
            tool_version=data.get('tool_version', ''),
            subgraphs=subgraphs,
            batch_size=batch_size,
            input_shape=data.get('input_shape'),
            model_summary=model_summary,
            system_state=system_state,
            run_id=data.get('run_id'),
            run_index=data.get('run_index'),
        )

    def save(self, filepath: Path) -> None:
        """Save measurement record to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'MeasurementRecord':
        """Load measurement record from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, filename=filepath.name)


# ============================================================================
# Manifest schema
# ============================================================================

@dataclass
class ManifestEntry:
    """One entry in a manifest file."""
    model: str
    precision: str
    batch_size: int
    file: str                     # relative path from hardware_id dir
    total_latency_ms: float
    total_flops: int
    num_subgraphs: int
    measurement_date: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManifestEntry':
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ManifestCoverage:
    """Coverage summary in a manifest."""
    models: List[str]
    precisions: List[str]
    batch_sizes: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManifestCoverage':
        return cls(
            models=data.get('models', []),
            precisions=data.get('precisions', []),
            batch_sizes=data.get('batch_sizes', []),
        )


@dataclass
class Manifest:
    """Index of all measurement data for a hardware target."""
    schema_version: str           # "1.0"
    hardware_id: str
    device: str
    generated_date: str
    summary: Dict[str, Any]
    measurements: List[ManifestEntry]
    coverage: ManifestCoverage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hardware_id": self.hardware_id,
            "device": self.device,
            "generated_date": self.generated_date,
            "summary": self.summary,
            "measurements": [m.to_dict() for m in self.measurements],
            "coverage": self.coverage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Manifest':
        measurements = [ManifestEntry.from_dict(m) for m in data.get('measurements', [])]
        coverage = ManifestCoverage.from_dict(data.get('coverage', {}))
        return cls(
            schema_version=data.get('schema_version', '1.0'),
            hardware_id=data.get('hardware_id', ''),
            device=data.get('device', ''),
            generated_date=data.get('generated_date', ''),
            summary=data.get('summary', {}),
            measurements=measurements,
            coverage=coverage,
        )

    def save(self, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'Manifest':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# Helpers
# ============================================================================

def _infer_batch_size_from_filename(filename: Optional[str]) -> int:
    """Extract batch size from filename like 'resnet18_b16.json' or 'resnet18.json'.

    Only recognizes batch >= 1 to avoid confusing model names like
    'efficientnet_b0' with batch=0.
    """
    if filename is None:
        return 1
    match = re.search(r'_b(\d+)\.json$', filename)
    if match:
        batch = int(match.group(1))
        if batch >= 1:
            return batch
    return 1


def _normalize_precision(precision: str) -> str:
    """Normalize precision string to lowercase for path matching."""
    return precision.lower()


# Known model name patterns from torchvision and common DNN families.
# Used to disambiguate batch suffixes from model name suffixes.
_KNOWN_MODEL_PREFIXES = {
    'resnet', 'mobilenet', 'efficientnet', 'densenet', 'vgg',
    'squeezenet', 'shufflenet', 'alexnet', 'inception',
    'vit', 'maxvit', 'swin',
}


def _is_valid_model_name(name: str) -> bool:
    """Check if a name looks like a valid model name (not a truncated one).

    Used to distinguish 'resnet18' (valid model, so '_b4' is batch)
    from 'efficientnet' (truncated model name, so '_b1' is part of name).

    A name is valid if it:
    1. Is a known full model name (ends with digits or version string), or
    2. Matches a known prefix followed by a version/size (e.g., resnet18, vgg16)
    """
    name_lower = name.lower()

    # Check if name matches a pattern like <family><number> or <family>_<variant>_<detail>
    for prefix in _KNOWN_MODEL_PREFIXES:
        if not name_lower.startswith(prefix):
            continue
        rest = name_lower[len(prefix):]

        # Exact prefix match with no suffix is likely truncated
        # e.g., 'efficientnet' from 'efficientnet_b1' -> not valid
        if not rest:
            return False

        # e.g., 'resnet18' -> rest='18', 'vgg16' -> rest='16'
        if rest.isdigit():
            return True

        # e.g., 'mobilenet_v2' -> rest='_v2', 'mobilenet_v3_small' -> rest='_v3_small'
        if rest.startswith('_'):
            return True

        # e.g., 'squeezenet1_0' -> rest='1_0'
        if rest[0].isdigit():
            return True

    # If it doesn't match any known prefix, it's either:
    # - A model name we don't recognize (assume valid)
    # - Or a name we can't disambiguate (assume valid to be safe)
    # But if it exactly matches a known prefix, it's truncated
    if name_lower in _KNOWN_MODEL_PREFIXES:
        return False

    return True


# ============================================================================
# GroundTruthLoader
# ============================================================================

class GroundTruthLoader:
    """Load and query stored measurement data.

    Searches the canonical calibration_data/ directory structure:
        calibration_data/
          <hardware_id>/
            manifest.json
            measurements/
              <precision>/
                <model>_b<batch>.json

    Also supports legacy layouts where measurements are stored under
    calibration_data/<hw_id>/<precision>/measurements/<model>.json
    or measurements/<hw_id>/<precision>/<model>_b<batch>.json.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize loader.

        Args:
            data_dir: Root directory for calibration data.
                      Defaults to <repo_root>/calibration_data/.
        """
        if data_dir is not None:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = DEFAULT_DATA_DIR
        # Cache loaded manifests
        self._manifests: Dict[str, Manifest] = {}

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def list_hardware(self) -> List[str]:
        """Available hardware IDs with measurement data."""
        if not self._data_dir.exists():
            return []
        result = []
        for d in sorted(self._data_dir.iterdir()):
            if d.is_dir() and self._has_measurements(d):
                result.append(d.name)
        return result

    def list_models(
        self,
        hardware_id: str,
        precision: Optional[str] = None,
    ) -> List[str]:
        """Models measured on this hardware.

        Args:
            hardware_id: Hardware target ID.
            precision: Optional precision filter (e.g., 'fp32').
        """
        configs = self.list_configurations(hardware_id)
        models = set()
        for cfg in configs:
            if precision is None or _normalize_precision(cfg['precision']) == _normalize_precision(precision):
                models.add(cfg['model'])
        return sorted(models)

    def list_configurations(self, hardware_id: str) -> List[Dict[str, Any]]:
        """All (model, precision, batch_size) combos available.

        Returns list of dicts with keys: model, precision, batch_size, file.
        """
        hw_dir = self._data_dir / hardware_id
        if not hw_dir.exists():
            return []

        configs = []

        # Check v2.0 canonical layout: measurements/<precision>/<model>_b<batch>.json
        meas_dir = hw_dir / "measurements"
        if meas_dir.exists():
            for prec_dir in sorted(meas_dir.iterdir()):
                if prec_dir.is_dir():
                    for f in sorted(prec_dir.glob("*.json")):
                        model, batch_size = self._parse_measurement_filename(f.name)
                        configs.append({
                            'model': model,
                            'precision': prec_dir.name,
                            'batch_size': batch_size,
                            'file': str(f.relative_to(hw_dir)),
                        })

        # Check legacy layout: <precision>/measurements/<model>.json
        for prec_dir in sorted(hw_dir.iterdir()):
            if prec_dir.is_dir() and prec_dir.name != 'measurements':
                legacy_meas = prec_dir / "measurements"
                if legacy_meas.exists():
                    for f in sorted(legacy_meas.glob("*.json")):
                        model, batch_size = self._parse_measurement_filename(f.name)
                        # Avoid duplicates with canonical layout
                        key = (model, prec_dir.name, batch_size)
                        existing_keys = {
                            (c['model'], c['precision'], c['batch_size'])
                            for c in configs
                        }
                        if key not in existing_keys:
                            configs.append({
                                'model': model,
                                'precision': prec_dir.name,
                                'batch_size': batch_size,
                                'file': str(f.relative_to(hw_dir)),
                            })

        return configs

    def load(
        self,
        hardware_id: str,
        model: str,
        precision: str,
        batch_size: int = 1,
    ) -> MeasurementRecord:
        """Load one measurement record.

        Args:
            hardware_id: Hardware target ID.
            model: Model name (e.g., 'resnet18').
            precision: Precision (e.g., 'fp32', 'FP32').
            batch_size: Batch size (default 1).

        Returns:
            MeasurementRecord

        Raises:
            FileNotFoundError: If no matching measurement file exists.
        """
        filepath = self._find_measurement_file(hardware_id, model, precision, batch_size)
        if filepath is None:
            raise FileNotFoundError(
                f"No measurement found for {hardware_id}/{model}/{precision}/b{batch_size}"
            )
        return MeasurementRecord.load(filepath)

    def load_model_summary(
        self,
        hardware_id: str,
        model: str,
        precision: str,
        batch_size: int = 1,
    ) -> ModelSummary:
        """Load just the model summary.

        Tries manifest first for fast access, falls back to loading full record.
        """
        # Try manifest
        manifest = self._get_manifest(hardware_id)
        if manifest is not None:
            prec_norm = _normalize_precision(precision)
            for entry in manifest.measurements:
                if (entry.model == model and
                        _normalize_precision(entry.precision) == prec_norm and
                        entry.batch_size == batch_size):
                    # Build summary from manifest entry
                    if entry.total_latency_ms > 0:
                        throughput = batch_size / (entry.total_latency_ms / 1000.0)
                    else:
                        throughput = 0.0
                    return ModelSummary(
                        total_flops=entry.total_flops,
                        total_bytes=0,  # not in manifest
                        total_latency_ms=entry.total_latency_ms,
                        total_latency_std_ms=0.0,  # not in manifest
                        num_subgraphs=entry.num_subgraphs,
                        throughput_fps=throughput,
                    )

        # Fall back to full load
        record = self.load(hardware_id, model, precision, batch_size)
        if record.model_summary is not None:
            return record.model_summary
        return ModelSummary.from_subgraphs(record.subgraphs, record.batch_size)

    def load_all(
        self,
        hardware_id: str,
        precision: Optional[str] = None,
    ) -> List[MeasurementRecord]:
        """Load all measurements for a hardware target.

        Args:
            hardware_id: Hardware target ID.
            precision: Optional precision filter.
        """
        configs = self.list_configurations(hardware_id)
        records = []
        hw_dir = self._data_dir / hardware_id
        for cfg in configs:
            if precision is not None and _normalize_precision(cfg['precision']) != _normalize_precision(precision):
                continue
            filepath = hw_dir / cfg['file']
            if filepath.exists():
                records.append(MeasurementRecord.load(filepath))
        return records

    def rebuild_manifest(self, hardware_id: str) -> Manifest:
        """Scan measurement files and regenerate manifest.json.

        Args:
            hardware_id: Hardware target to rebuild manifest for.

        Returns:
            The generated Manifest.
        """
        hw_dir = self._data_dir / hardware_id
        configs = self.list_configurations(hardware_id)

        entries = []
        models_set = set()
        precisions_set = set()
        batch_sizes_set = set()
        device = ""

        for cfg in configs:
            filepath = hw_dir / cfg['file']
            if not filepath.exists():
                continue

            try:
                record = MeasurementRecord.load(filepath)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

            if not device:
                device = record.device

            # Compute summary if needed
            if record.model_summary is not None:
                total_latency = record.model_summary.total_latency_ms
                total_flops = record.model_summary.total_flops
                num_subgraphs = record.model_summary.num_subgraphs
            else:
                total_latency = sum(sg.measured_latency.mean for sg in record.subgraphs)
                total_flops = sum(sg.flops for sg in record.subgraphs)
                num_subgraphs = len(record.subgraphs)

            entries.append(ManifestEntry(
                model=record.model,
                precision=_normalize_precision(record.precision),
                batch_size=record.batch_size,
                file=cfg['file'],
                total_latency_ms=round(total_latency, 4),
                total_flops=total_flops,
                num_subgraphs=num_subgraphs,
                measurement_date=record.measurement_date,
            ))

            models_set.add(record.model)
            precisions_set.add(_normalize_precision(record.precision))
            batch_sizes_set.add(record.batch_size)

        manifest = Manifest(
            schema_version="1.0",
            hardware_id=hardware_id,
            device=device,
            generated_date=datetime.now().isoformat(),
            summary={
                "num_models": len(models_set),
                "num_precisions": len(precisions_set),
                "num_batch_sizes": len(batch_sizes_set),
                "total_measurements": len(entries),
            },
            measurements=entries,
            coverage=ManifestCoverage(
                models=sorted(models_set),
                precisions=sorted(precisions_set),
                batch_sizes=sorted(batch_sizes_set),
            ),
        )

        manifest.save(hw_dir / "manifest.json")
        # Update cache
        self._manifests[hardware_id] = manifest
        return manifest

    # ---- Internal helpers ----

    def _has_measurements(self, hw_dir: Path) -> bool:
        """Check if a hardware directory has any measurement JSON files."""
        # Check canonical: measurements/<prec>/<file>.json
        meas_dir = hw_dir / "measurements"
        if meas_dir.exists():
            for prec_dir in meas_dir.iterdir():
                if prec_dir.is_dir():
                    for _ in prec_dir.glob("*.json"):
                        return True

        # Check legacy: <prec>/measurements/<file>.json
        for prec_dir in hw_dir.iterdir():
            if prec_dir.is_dir() and prec_dir.name != 'measurements':
                legacy_meas = prec_dir / "measurements"
                if legacy_meas.exists():
                    for _ in legacy_meas.glob("*.json"):
                        return True

        return False

    def _parse_measurement_filename(self, filename: str) -> tuple:
        """Parse model name and batch size from filename.

        The run_full_calibration pipeline names files as <model>_b<batch>.json.
        Legacy files (batch=1 only) are just <model>.json.

        To distinguish batch suffixes from model names containing '_b<N>'
        (e.g., 'efficientnet_b1', 'vit_b_16'), we check whether the
        model part (before _b<N>) is a known model name pattern, and
        whether the JSON content confirms the model name.

        Heuristic: if removing _b<N> leaves a name that matches a known
        model family prefix, it's a batch suffix. Otherwise treat the
        whole name as the model.

        Examples:
            'resnet18_b1.json'           -> ('resnet18', 1)
            'resnet18_b16.json'          -> ('resnet18', 16)
            'resnet18.json'              -> ('resnet18', 1)
            'mobilenet_v2_b4.json'       -> ('mobilenet_v2', 4)
            'efficientnet_b0.json'       -> ('efficientnet_b0', 1)
            'efficientnet_b0_b1.json'    -> ('efficientnet_b0', 1)
            'efficientnet_b1.json'       -> ('efficientnet_b1', 1)
            'efficientnet_b1_b4.json'    -> ('efficientnet_b1', 4)
            'vit_b_16.json'              -> ('vit_b_16', 1)
            'vit_b_16_b4.json'           -> ('vit_b_16', 4)
        """
        name = filename.replace('.json', '')
        # Match _b<N> at end where N >= 1
        match = re.match(r'^(.+?)_b(\d+)$', name)
        if match:
            candidate_model = match.group(1)
            batch = int(match.group(2))
            if batch >= 1 and _is_valid_model_name(candidate_model):
                return candidate_model, batch
        return name, 1

    def _find_measurement_file(
        self,
        hardware_id: str,
        model: str,
        precision: str,
        batch_size: int,
    ) -> Optional[Path]:
        """Find measurement file using multiple layout patterns."""
        hw_dir = self._data_dir / hardware_id
        prec = _normalize_precision(precision)

        # Pattern 1: canonical v2.0 layout
        # calibration_data/<hw>/measurements/<prec>/<model>_b<batch>.json
        p1 = hw_dir / "measurements" / prec / f"{model}_b{batch_size}.json"
        if p1.exists():
            return p1

        # Pattern 2: legacy layout (batch in filename)
        # calibration_data/<hw>/<prec>/measurements/<model>_b<batch>.json
        p2 = hw_dir / prec / "measurements" / f"{model}_b{batch_size}.json"
        if p2.exists():
            return p2

        # Pattern 3: legacy layout without batch (implies batch=1)
        if batch_size == 1:
            # calibration_data/<hw>/<prec>/measurements/<model>.json
            p3 = hw_dir / prec / "measurements" / f"{model}.json"
            if p3.exists():
                return p3

        return None

    def _get_manifest(self, hardware_id: str) -> Optional[Manifest]:
        """Load manifest from cache or disk."""
        if hardware_id in self._manifests:
            return self._manifests[hardware_id]

        manifest_path = self._data_dir / hardware_id / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = Manifest.load(manifest_path)
                self._manifests[hardware_id] = manifest
                return manifest
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
        return None
