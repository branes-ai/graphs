"""
Hardware Database Schema

Defines the structure for hardware specifications in the database.
Each hardware entry is a complete specification including identification,
detection hints, performance characteristics, and mapper configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime


@dataclass
class HardwareSpec:
    """
    Complete hardware specification for database.

    This represents a single hardware configuration (CPU, GPU, accelerator)
    with all information needed for detection, mapping, and calibration.
    """

    # =========================================================================
    # IDENTIFICATION
    # =========================================================================

    id: str
    """Unique identifier: 'intel_i7_12700k', 'nvidia_h100_sxm5_80gb'"""

    vendor: str
    """Vendor name: 'Intel', 'NVIDIA', 'AMD', 'Ampere Computing', 'Google'"""

    model: str
    """Full model name: 'Core i7-12700K', 'H100 SXM5 80GB', 'Jetson Orin AGX 64GB'"""

    architecture: str
    """Architecture name: 'Alder Lake', 'Hopper', 'Zen 4', 'Altra', 'Cortex-A78AE'"""

    device_type: str
    """Device type: 'cpu', 'gpu', 'tpu', 'kpu', 'dpu', 'cgra'"""

    platform: str
    """Platform architecture: 'x86_64', 'aarch64', 'arm64'"""

    # =========================================================================
    # DETECTION HINTS (for auto-matching)
    # =========================================================================

    detection_patterns: List[str] = field(default_factory=list)
    """
    Regex patterns for matching detected hardware.

    Examples:
    - "12th Gen Intel.*Core.*i7-12700K"
    - "NVIDIA H100.*80GB"
    - "Cortex-A78AE.*Orin"
    - "AMD Ryzen 9 7950X"

    Used for fuzzy matching during auto-detection.
    """

    os_compatibility: List[str] = field(default_factory=lambda: ["linux", "windows", "macos"])
    """Supported operating systems: 'linux', 'windows', 'macos'"""

    # =========================================================================
    # CORE SPECIFICATIONS
    # =========================================================================

    # CPU/GPU Core Configuration
    cores: Optional[int] = None
    """Physical cores (P-cores for hybrid CPUs)"""

    threads: Optional[int] = None
    """Hardware threads (with SMT/Hyper-Threading)"""

    e_cores: Optional[int] = None
    """Efficiency cores (for hybrid CPUs like Alder Lake)"""

    base_frequency_ghz: Optional[float] = None
    """Base/sustained frequency in GHz"""

    boost_frequency_ghz: Optional[float] = None
    """Maximum boost/turbo frequency in GHz"""

    # GPU-specific
    cuda_cores: Optional[int] = None
    """CUDA cores (NVIDIA GPUs)"""

    tensor_cores: Optional[int] = None
    """Tensor cores for matrix ops (NVIDIA)"""

    sms: Optional[int] = None
    """Streaming Multiprocessors (NVIDIA) or Compute Units (AMD)"""

    rt_cores: Optional[int] = None
    """Ray Tracing cores (NVIDIA RTX)"""

    cuda_capability: Optional[str] = None
    """CUDA compute capability: '8.9', '9.0'"""

    # =========================================================================
    # MEMORY CONFIGURATION
    # =========================================================================

    memory_type: str = "DDR4"
    """Memory type: 'DDR5', 'DDR4', 'HBM3', 'HBM2e', 'LPDDR5', 'GDDR6X'"""

    memory_channels: Optional[int] = None
    """Number of memory channels"""

    memory_bus_width: Optional[int] = None
    """Memory bus width in bits"""

    peak_bandwidth_gbps: float = 0.0
    """Theoretical peak memory bandwidth in GB/s"""

    # =========================================================================
    # ISA & FEATURES
    # =========================================================================

    isa_extensions: List[str] = field(default_factory=list)
    """
    Instruction set extensions.

    Examples:
    - x86_64: ["AVX2", "AVX512F", "AVX512BW", "FMA3", "VNNI"]
    - aarch64: ["NEON", "SVE", "SVE2", "BF16", "FP16"]
    """

    special_features: List[str] = field(default_factory=list)
    """
    Special hardware features.

    Examples:
    - ["Tensor Cores", "RT Cores", "DLSS"]
    - ["Matrix Engine", "Neural Engine"]
    - ["Systolic Array", "TPU Cores"]
    """

    # =========================================================================
    # THEORETICAL PERFORMANCE
    # =========================================================================

    theoretical_peaks: Dict[str, float] = field(default_factory=dict)
    """
    Theoretical peak performance for each precision (GFLOPS for float, GIOPS for int).

    Example:
    {
        "fp64": 360.0,
        "fp32": 720.0,
        "fp16": 1440.0,
        "int64": 360.0,
        "int32": 360.0,
        "int16": 720.0,
        "int8": 1440.0
    }

    Note: These are theoretical maximums based on:
    - Cores × Operations/Cycle × Frequency
    - May not account for memory bottlenecks
    - Actual performance will be lower (measured via calibration)
    """

    # =========================================================================
    # CACHE (CPU-specific)
    # =========================================================================

    l1_cache_kb: Optional[int] = None
    """L1 cache size in KB (per core or total)"""

    l2_cache_kb: Optional[int] = None
    """L2 cache size in KB"""

    l3_cache_kb: Optional[int] = None
    """L3 cache size in KB (shared)"""

    # =========================================================================
    # POWER
    # =========================================================================

    tdp_watts: Optional[float] = None
    """Thermal Design Power in watts"""

    max_power_watts: Optional[float] = None
    """Maximum power consumption in watts"""

    # =========================================================================
    # METADATA
    # =========================================================================

    release_date: Optional[str] = None
    """Release date: 'Q4 2021', '2022-11', 'November 2021'"""

    end_of_life: Optional[str] = None
    """End of life / discontinuation date"""

    manufacturer_url: Optional[str] = None
    """Official product page URL"""

    notes: Optional[str] = None
    """Additional notes, quirks, known issues"""

    data_source: str = "manufacturer"
    """Source of specifications: 'manufacturer', 'measured', 'estimated', 'community'"""

    last_updated: Optional[str] = None
    """ISO 8601 timestamp of last update: '2025-01-17T12:00:00Z'"""

    # =========================================================================
    # MAPPER CONFIGURATION
    # =========================================================================

    mapper_class: str = "CPUMapper"
    """
    Mapper class name for this hardware.

    Examples:
    - 'CPUMapper' (x86_64, aarch64 CPUs)
    - 'GPUMapper' (NVIDIA, AMD GPUs)
    - 'TPUMapper' (Google TPU)
    - 'KPUMapper' (Kendryte K210)
    - 'DPUMapper' (Xilinx Vitis AI)
    - 'CGRAMapper' (Spatial dataflow accelerators)
    """

    mapper_config: Dict[str, Any] = field(default_factory=dict)
    """
    Mapper-specific configuration parameters.

    Examples:
    - CPUMapper: {"simd_width": 256, "cores_to_use": 12}
    - GPUMapper: {"waves_per_sm": 32, "warps_per_block": 32}
    - TPUMapper: {"systolic_array_dim": 128}
    """

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'HardwareSpec':
        """Create from dictionary (from JSON)"""
        return cls(**data)

    def to_json(self, filepath: Path):
        """
        Save to JSON file with pretty formatting.

        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        # Update timestamp
        data['last_updated'] = datetime.utcnow().isoformat() + 'Z'

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, filepath: Path) -> 'HardwareSpec':
        """
        Load from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            HardwareSpec instance
        """
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> List[str]:
        """
        Validate hardware spec for completeness and correctness.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        if not self.id:
            errors.append("Missing required field: id")
        if not self.vendor:
            errors.append("Missing required field: vendor")
        if not self.model:
            errors.append("Missing required field: model")
        if not self.device_type:
            errors.append("Missing required field: device_type")
        if self.device_type not in ['cpu', 'gpu', 'tpu', 'kpu', 'dpu', 'cgra']:
            errors.append(f"Invalid device_type: {self.device_type}")
        if not self.platform:
            errors.append("Missing required field: platform")
        if self.platform not in ['x86_64', 'aarch64', 'arm64']:
            errors.append(f"Invalid platform: {self.platform}")

        # Theoretical peaks should have at least fp32
        if not self.theoretical_peaks or 'fp32' not in self.theoretical_peaks:
            errors.append("theoretical_peaks must include 'fp32'")

        # Peak bandwidth should be positive
        if self.peak_bandwidth_gbps <= 0:
            errors.append("peak_bandwidth_gbps must be positive")

        # Mapper class should be specified
        if not self.mapper_class:
            errors.append("Missing required field: mapper_class")

        return errors


@dataclass
class HardwareDetectionResult:
    """
    Result of hardware auto-detection.

    Contains detected properties and confidence score for matching
    against database entries.
    """

    # Detected properties
    cpu_model: Optional[str] = None
    cpu_vendor: Optional[str] = None
    cpu_cores: Optional[int] = None
    cpu_threads: Optional[int] = None
    cpu_frequency_mhz: Optional[float] = None

    gpu_vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_cuda_capability: Optional[str] = None

    platform: Optional[str] = None
    os_type: Optional[str] = None  # 'linux', 'windows', 'macos'

    isa_extensions: List[str] = field(default_factory=list)

    # Matching
    matched_spec: Optional[HardwareSpec] = None
    confidence_score: float = 0.0  # 0.0 - 1.0
    match_reason: Optional[str] = None

    # Raw detection data (for debugging)
    raw_data: Dict[str, Any] = field(default_factory=dict)
