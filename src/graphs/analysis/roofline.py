"""
Roofline Model Analysis

Analyzes computational graphs using the roofline model to determine
performance bottlenecks (compute-bound vs memory-bound).

The Roofline Model:
- Compute time = FLOPs / peak_FLOPS
- Memory time = bytes / peak_bandwidth
- Actual latency = max(compute_time, memory_time) + overhead
- Bottleneck = which time is larger

Arithmetic Intensity (AI) is the key metric:
- AI = FLOPs / bytes
- AI_breakpoint = peak_FLOPS / peak_bandwidth
- If AI < AI_breakpoint: memory-bound
- If AI > AI_breakpoint: compute-bound

Calibration Integration:
- When calibration data is available, use empirical peaks instead of theoretical
- This gives more realistic latency estimates based on actual hardware behavior
- Falls back to theoretical peaks when no calibration exists
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from enum import Enum

from ..ir.structures import SubgraphDescriptor, PartitionReport, BottleneckType
from ..hardware.resource_model import HardwareResourceModel, Precision

if TYPE_CHECKING:
    from ..hardware.calibration.registry_sync import HardwareEntry


@dataclass
class LatencyDescriptor:
    """
    Latency analysis for a single subgraph.

    Breaks down execution time into compute and memory components
    to identify the bottleneck.
    """

    # Identity
    subgraph_id: str
    subgraph_name: str

    # Roofline components (seconds)
    compute_time: float  # Time limited by FLOPs
    memory_time: float   # Time limited by bandwidth
    actual_latency: float  # max(compute_time, memory_time) + overhead
    overhead: float = 0.0  # Kernel launch, etc.

    # Bottleneck analysis
    bottleneck: BottleneckType = BottleneckType.BALANCED
    bottleneck_ratio: float = 1.0  # How much slower bottleneck is vs other

    # Arithmetic intensity
    arithmetic_intensity: float = 0.0  # FLOPs/byte
    arithmetic_intensity_breakpoint: float = 0.0  # Hardware breakpoint

    # Performance metrics
    attained_flops: float = 0.0  # Actual FLOPs/sec achieved
    peak_flops: float = 0.0  # Hardware peak
    flops_utilization: float = 0.0  # attained / peak

    attained_bandwidth: float = 0.0  # Actual bytes/sec
    peak_bandwidth: float = 0.0  # Hardware peak
    bandwidth_utilization: float = 0.0  # attained / peak

    # Explanation
    explanation: str = ""

    def __str__(self) -> str:
        """Short summary"""
        return (f"Latency({self.subgraph_name}: "
                f"{self.actual_latency*1e6:.1f}μs, {self.bottleneck.value})")

    def format_summary(self) -> str:
        """Detailed multi-line summary"""
        lines = []
        lines.append(f"Subgraph: {self.subgraph_name}")
        lines.append(f"  Latency: {self.actual_latency * 1e6:.1f} μs")
        lines.append(f"    Compute time: {self.compute_time * 1e6:.1f} μs")
        lines.append(f"    Memory time:  {self.memory_time * 1e6:.1f} μs")
        if self.overhead > 0:
            lines.append(f"    Overhead:     {self.overhead * 1e6:.1f} μs")
        lines.append(f"  Bottleneck: {self.bottleneck.value} ({self.bottleneck_ratio:.1f}× slower)")
        lines.append(f"  Arithmetic Intensity: {self.arithmetic_intensity:.2f} FLOPs/byte")
        lines.append(f"  FLOP Utilization: {self.flops_utilization * 100:.1f}%")
        lines.append(f"  Bandwidth Utilization: {self.bandwidth_utilization * 100:.1f}%")
        return "\n".join(lines)


@dataclass
class RooflinePoint:
    """
    A single point on the roofline plot.

    x = arithmetic intensity (FLOPs/byte)
    y = attained performance (FLOPs/sec or GFLOPs/sec)
    """
    arithmetic_intensity: float
    attained_flops: float
    subgraph_name: str = ""
    is_compute_bound: bool = False


@dataclass
class RooflineReport:
    """
    Complete roofline analysis for a partition.

    Contains per-subgraph latency descriptors and summary statistics.
    """

    # Hardware characteristics
    peak_flops: float  # FLOPs/sec
    peak_bandwidth: float  # bytes/sec
    arithmetic_intensity_breakpoint: float  # FLOPs/byte

    # Per-subgraph latency
    latencies: List[LatencyDescriptor] = field(default_factory=list)

    # Aggregate statistics
    total_latency: float = 0.0  # seconds
    total_compute_time: float = 0.0
    total_memory_time: float = 0.0
    total_overhead: float = 0.0

    # Bottleneck distribution
    num_compute_bound: int = 0
    num_memory_bound: int = 0
    num_balanced: int = 0

    # Utilization
    average_flops_utilization: float = 0.0
    average_bandwidth_utilization: float = 0.0

    # Roofline points (for visualization)
    roofline_points: List[RooflinePoint] = field(default_factory=list)

    # Critical path
    critical_path_latency: float = 0.0
    critical_path_subgraphs: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Short summary"""
        return (f"RooflineReport(total={self.total_latency*1e3:.2f}ms, "
                f"compute_bound={self.num_compute_bound}, "
                f"memory_bound={self.num_memory_bound})")

    def format_report(self, show_per_subgraph: bool = False, max_subgraphs: int = 10) -> str:
        """Generate human-readable report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ROOFLINE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Hardware characteristics
        lines.append("Hardware Characteristics:")
        lines.append(f"  Peak Performance: {self.peak_flops / 1e12:.2f} TFLOPS")
        lines.append(f"  Peak Bandwidth:   {self.peak_bandwidth / 1e9:.2f} GB/s")
        lines.append(f"  AI Breakpoint:    {self.arithmetic_intensity_breakpoint:.2f} FLOPs/byte")
        lines.append("")

        # Total latency
        lines.append("Total Latency:")
        lines.append(f"  Total:   {self.total_latency * 1e3:.2f} ms ({self.total_latency * 1e6:.0f} μs)")
        lines.append(f"  Compute: {self.total_compute_time * 1e3:.2f} ms ({self.total_compute_time / self.total_latency * 100:.1f}%)")
        lines.append(f"  Memory:  {self.total_memory_time * 1e3:.2f} ms ({self.total_memory_time / self.total_latency * 100:.1f}%)")
        if self.total_overhead > 0:
            lines.append(f"  Overhead: {self.total_overhead * 1e3:.2f} ms ({self.total_overhead / self.total_latency * 100:.1f}%)")
        lines.append("")

        # Bottleneck distribution
        total_ops = self.num_compute_bound + self.num_memory_bound + self.num_balanced
        lines.append("Bottleneck Distribution:")
        if total_ops > 0:
            lines.append(f"  Compute-bound: {self.num_compute_bound} ops ({self.num_compute_bound / total_ops * 100:.1f}%)")
            lines.append(f"  Memory-bound:  {self.num_memory_bound} ops ({self.num_memory_bound / total_ops * 100:.1f}%)")
            lines.append(f"  Balanced:      {self.num_balanced} ops ({self.num_balanced / total_ops * 100:.1f}%)")
        lines.append("")

        # Utilization
        lines.append("Hardware Utilization:")
        lines.append(f"  Average FLOP Utilization:      {self.average_flops_utilization * 100:.1f}%")
        lines.append(f"  Average Bandwidth Utilization: {self.average_bandwidth_utilization * 100:.1f}%")
        lines.append("")

        # Critical path
        if self.critical_path_latency > 0:
            lines.append("Critical Path:")
            lines.append(f"  Latency: {self.critical_path_latency * 1e3:.2f} ms")
            lines.append(f"  Operations: {len(self.critical_path_subgraphs)}")
            lines.append("")

        # Per-subgraph details
        if show_per_subgraph and self.latencies:
            lines.append("=" * 80)
            lines.append(f"TOP {max_subgraphs} SUBGRAPHS BY LATENCY")
            lines.append("=" * 80)
            lines.append("")

            # Sort by latency
            sorted_latencies = sorted(self.latencies, key=lambda l: l.actual_latency, reverse=True)[:max_subgraphs]

            for i, lat in enumerate(sorted_latencies, 1):
                lines.append(f"{i}. {lat.subgraph_name}")
                lines.append(f"   Latency: {lat.actual_latency * 1e6:.1f} μs")
                lines.append(f"   Bottleneck: {lat.bottleneck.value} ({lat.bottleneck_ratio:.1f}×)")
                lines.append(f"   AI: {lat.arithmetic_intensity:.2f} FLOPs/byte")
                lines.append(f"   FLOP util: {lat.flops_utilization * 100:.1f}%")
                lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


class RooflineAnalyzer:
    """
    Analyzes computational graphs using the roofline model.

    The roofline model determines whether each operation is limited by
    compute (FLOPs) or memory bandwidth, and calculates realistic latency.

    Algorithm:
    1. For each subgraph, compute:
       - Compute time = FLOPs / peak_FLOPS
       - Memory time = bytes / peak_bandwidth
       - Actual latency = max(compute_time, memory_time) + overhead
    2. Identify bottleneck (which time is larger)
    3. Calculate utilization (actual / peak)
    4. Generate roofline points for visualization

    Calibration Support:
    - Pass efficiency_factor (0-1) to scale theoretical peaks to empirical
    - Or use create_calibrated_analyzer() factory for automatic calibration
    """

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        precision: Precision = Precision.FP32,
        efficiency_factor: Optional[float] = None,
        calibrated_peak_flops: Optional[float] = None,
        calibrated_bandwidth: Optional[float] = None,
        thermal_profile: Optional[str] = None
    ):
        """
        Initialize roofline analyzer.

        Args:
            resource_model: Hardware specifications
            precision: Precision to use for analysis
            efficiency_factor: Optional efficiency factor (0-1) to scale peaks.
                If provided, multiplies theoretical peaks by this factor.
            calibrated_peak_flops: Optional calibrated peak FLOPs/sec.
                If provided, overrides resource_model peak.
            calibrated_bandwidth: Optional calibrated bandwidth (bytes/sec).
                If provided, overrides resource_model bandwidth.
            thermal_profile: Optional thermal/power profile (e.g., '15W', '30W').
                If provided, uses thermal_operating_points for realistic performance.
                If None, uses default_thermal_profile or falls back to precision_profiles.
        """
        self.resource_model = resource_model
        self.precision = precision
        self.efficiency_factor = efficiency_factor
        self.thermal_profile = thermal_profile
        self.is_calibrated = (
            efficiency_factor is not None or
            calibrated_peak_flops is not None or
            calibrated_bandwidth is not None
        )

        # Get hardware characteristics for this precision
        # Priority: thermal_operating_points > precision_profiles
        theoretical_peak_flops = self._get_effective_peak_ops(
            resource_model, precision, thermal_profile
        )

        theoretical_bandwidth = resource_model.peak_bandwidth

        # Apply calibration
        if calibrated_peak_flops is not None:
            # Direct calibrated value (in FLOPs/sec)
            self.peak_flops = calibrated_peak_flops * 1e9  # Convert from GFLOPS
        elif efficiency_factor is not None:
            self.peak_flops = theoretical_peak_flops * efficiency_factor
        else:
            self.peak_flops = theoretical_peak_flops

        if calibrated_bandwidth is not None:
            # Direct calibrated value (in bytes/sec)
            self.peak_bandwidth = calibrated_bandwidth * 1e9  # Convert from GB/s
        elif efficiency_factor is not None:
            self.peak_bandwidth = theoretical_bandwidth * efficiency_factor
        else:
            self.peak_bandwidth = theoretical_bandwidth

        # Store theoretical for reference
        self.theoretical_peak_flops = theoretical_peak_flops
        self.theoretical_bandwidth = theoretical_bandwidth

        # Calculate arithmetic intensity breakpoint
        # AI_breakpoint = peak_FLOPS / peak_bandwidth
        self.ai_breakpoint = self.peak_flops / self.peak_bandwidth if self.peak_bandwidth > 0 else 0.0

    @staticmethod
    def _get_effective_peak_ops(
        resource_model: HardwareResourceModel,
        precision: Precision,
        thermal_profile: Optional[str] = None
    ) -> float:
        """
        Get effective peak operations per second, respecting thermal constraints.

        This method implements a fallback chain:
        1. thermal_operating_points with specified thermal_profile
        2. thermal_operating_points with default_thermal_profile
        3. Legacy precision_profiles (theoretical peaks)

        Args:
            resource_model: Hardware specifications
            precision: Target precision
            thermal_profile: Specific thermal profile to use (e.g., '15W')

        Returns:
            Effective peak operations per second
        """
        # Try 1: thermal_operating_points with explicit thermal_profile
        if (hasattr(resource_model, 'thermal_operating_points') and
            resource_model.thermal_operating_points and thermal_profile):
            thermal_point = resource_model.thermal_operating_points.get(thermal_profile)
            if thermal_point and precision in thermal_point.performance_specs:
                perf_spec = thermal_point.performance_specs[precision]
                effective_ops = perf_spec.effective_ops_per_sec
                if effective_ops > 0:
                    return effective_ops

        # Try 2: thermal_operating_points with default profile
        if (hasattr(resource_model, 'thermal_operating_points') and
            resource_model.thermal_operating_points and
            hasattr(resource_model, 'default_thermal_profile') and
            resource_model.default_thermal_profile):
            thermal_point = resource_model.thermal_operating_points.get(
                resource_model.default_thermal_profile
            )
            if thermal_point and precision in thermal_point.performance_specs:
                perf_spec = thermal_point.performance_specs[precision]
                effective_ops = perf_spec.effective_ops_per_sec
                if effective_ops > 0:
                    return effective_ops

        # Try 3: Legacy precision_profiles
        if precision in resource_model.precision_profiles:
            profile = resource_model.precision_profiles[precision]
            return profile.peak_ops_per_sec

        # Try 4: Default precision fallback
        if resource_model.default_precision in resource_model.precision_profiles:
            default_profile = resource_model.precision_profiles[resource_model.default_precision]
            return default_profile.peak_ops_per_sec

        # Last resort: return 0 (will cause issues, but signals problem)
        return 0.0

    def analyze(
        self,
        subgraphs: List[SubgraphDescriptor],
        partition_report: Optional[PartitionReport] = None
    ) -> RooflineReport:
        """
        Analyze latency for all subgraphs using roofline model.

        Args:
            subgraphs: List of computational subgraphs
            partition_report: Optional partition report (for critical path)

        Returns:
            RooflineReport with latency analysis
        """

        latencies = []
        roofline_points = []

        # Analyze each subgraph
        for sg in subgraphs:
            latency = self._analyze_subgraph(sg)
            latencies.append(latency)

            # Create roofline point
            point = RooflinePoint(
                arithmetic_intensity=latency.arithmetic_intensity,
                attained_flops=latency.attained_flops,
                subgraph_name=sg.node_name,
                is_compute_bound=(latency.bottleneck == BottleneckType.COMPUTE_BOUND)
            )
            roofline_points.append(point)

        # Aggregate statistics
        total_latency = sum(l.actual_latency for l in latencies)
        total_compute_time = sum(l.compute_time for l in latencies)
        total_memory_time = sum(l.memory_time for l in latencies)
        total_overhead = sum(l.overhead for l in latencies)

        # Bottleneck distribution
        num_compute_bound = sum(1 for l in latencies if l.bottleneck == BottleneckType.COMPUTE_BOUND)
        num_memory_bound = sum(1 for l in latencies if l.bottleneck == BottleneckType.BANDWIDTH_BOUND)
        num_balanced = sum(1 for l in latencies if l.bottleneck == BottleneckType.BALANCED)

        # Average utilization
        avg_flops_util = sum(l.flops_utilization for l in latencies) / len(latencies) if latencies else 0.0
        avg_bw_util = sum(l.bandwidth_utilization for l in latencies) / len(latencies) if latencies else 0.0

        # Critical path
        critical_path_latency = 0.0
        critical_path_subgraphs = []
        if partition_report and partition_report.critical_path_subgraphs:
            critical_path_subgraphs = partition_report.critical_path_subgraphs
            # Sum latency for critical path
            critical_path_latency = sum(
                l.actual_latency for l in latencies
                if l.subgraph_id in critical_path_subgraphs
            )

        return RooflineReport(
            peak_flops=self.peak_flops,
            peak_bandwidth=self.peak_bandwidth,
            arithmetic_intensity_breakpoint=self.ai_breakpoint,
            latencies=latencies,
            total_latency=total_latency,
            total_compute_time=total_compute_time,
            total_memory_time=total_memory_time,
            total_overhead=total_overhead,
            num_compute_bound=num_compute_bound,
            num_memory_bound=num_memory_bound,
            num_balanced=num_balanced,
            average_flops_utilization=avg_flops_util,
            average_bandwidth_utilization=avg_bw_util,
            roofline_points=roofline_points,
            critical_path_latency=critical_path_latency,
            critical_path_subgraphs=critical_path_subgraphs,
        )

    def _analyze_subgraph(self, sg: SubgraphDescriptor) -> LatencyDescriptor:
        """Analyze latency for a single subgraph"""

        # Compute time = FLOPs / peak_FLOPS
        compute_time = sg.flops / self.peak_flops if self.peak_flops > 0 else 0.0

        # Memory time = bytes / peak_bandwidth
        total_bytes = sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
        memory_time = total_bytes / self.peak_bandwidth if self.peak_bandwidth > 0 else 0.0

        # Apply discrete resource correction for accelerators with few compute units
        # TPUs, KPUs can't fractionally utilize their arrays - adjust for realistic allocation
        correction_factor = self._get_discrete_resource_correction(sg)
        compute_time *= correction_factor
        memory_time *= correction_factor

        # Actual latency = max(compute_time, memory_time)
        bottleneck_time = max(compute_time, memory_time)

        # Determine bottleneck
        if compute_time > memory_time * 1.1:  # 10% threshold
            bottleneck = BottleneckType.COMPUTE_BOUND
            bottleneck_ratio = compute_time / memory_time if memory_time > 0 else 1.0
        elif memory_time > compute_time * 1.1:
            bottleneck = BottleneckType.BANDWIDTH_BOUND
            bottleneck_ratio = memory_time / compute_time if compute_time > 0 else 1.0
        else:
            bottleneck = BottleneckType.BALANCED
            bottleneck_ratio = 1.0

        # Add overhead (kernel launch for GPUs, etc.)
        overhead = self._estimate_overhead(sg)
        actual_latency = bottleneck_time + overhead

        # Arithmetic intensity
        ai = sg.flops / total_bytes if total_bytes > 0 else 0.0

        # Attained performance
        attained_flops = sg.flops / actual_latency if actual_latency > 0 else 0.0
        attained_bandwidth = total_bytes / actual_latency if actual_latency > 0 else 0.0

        # Utilization
        flops_util = attained_flops / self.peak_flops if self.peak_flops > 0 else 0.0
        bw_util = attained_bandwidth / self.peak_bandwidth if self.peak_bandwidth > 0 else 0.0

        # Generate explanation
        explanation = self._explain_latency(sg, compute_time, memory_time, bottleneck, bottleneck_ratio)

        return LatencyDescriptor(
            subgraph_id=sg.node_id,
            subgraph_name=sg.node_name,
            compute_time=compute_time,
            memory_time=memory_time,
            actual_latency=actual_latency,
            overhead=overhead,
            bottleneck=bottleneck,
            bottleneck_ratio=bottleneck_ratio,
            arithmetic_intensity=ai,
            arithmetic_intensity_breakpoint=self.ai_breakpoint,
            attained_flops=attained_flops,
            peak_flops=self.peak_flops,
            flops_utilization=flops_util,
            attained_bandwidth=attained_bandwidth,
            peak_bandwidth=self.peak_bandwidth,
            bandwidth_utilization=bw_util,
            explanation=explanation,
        )

    def _get_discrete_resource_correction(self, sg: SubgraphDescriptor) -> float:
        """
        Apply correction factor for hardware with discrete, non-divisible compute units.

        Problem: Naive calculation assumes fractional utilization of peak performance.
        Reality: Can't use 0.3 TensorCores or 2.7 SMs - resources are discrete!

        For TPU (2 TensorCores):
        - Small kernels (< 100M FLOPs): use 1 TensorCore → 2× slower than peak
        - Large kernels (>= 100M FLOPs): use 2 TensorCores → use peak

        For other hardware: no correction (GPUs have many SMs, CPUs have many cores)
        """
        hw_type = self.resource_model.hardware_type.name

        if hw_type == 'TPU':
            # TPU v4: 2 MXUs (Matrix Multiplier Units), each 128×128 systolic array
            # Small kernels suffer from:
            # 1. Can only use 1 MXU (2× penalty)
            # 2. Matrix dimensions < 128×128 (poor array utilization)
            # 3. Sequential execution overhead

            if sg.flops < 10e6:  # < 10M FLOPs (very small kernels)
                # Tiny kernels: 1 MXU, ~20% utilization → 10× penalty
                return 10.0
            elif sg.flops < 100e6:  # < 100M FLOPs (small kernels like ResNet18)
                # Small kernels: 1 MXU, ~50% utilization → 4× penalty
                return 4.0
            elif sg.flops < 500e6:  # < 500M FLOPs (medium kernels)
                # Medium kernels: can start using both MXUs → 2× penalty
                return 2.0
            else:
                # Large kernels: both MXUs, good utilization
                return 1.0

        elif hw_type == 'KPU':
            # KPU: 256 tiles, but small kernels don't use all of them
            # Already handled by KPU mapper, so no correction needed here
            return 1.0

        else:
            # GPU (132 SMs), CPU (many cores): enough units that fractional is reasonable
            return 1.0

    def _estimate_overhead(self, sg: SubgraphDescriptor) -> float:
        """Estimate overhead (kernel launch, etc.)"""
        # GPU kernel launch overhead
        if self.resource_model.hardware_type.name == 'GPU':
            # Typical kernel launch: 5-10 μs
            return 5e-6  # 5 microseconds

        # TPU systolic array setup overhead
        if self.resource_model.hardware_type.name == 'TPU':
            # Systolic array pipeline fill/drain: ~64 ns
            return 64e-9  # 64 nanoseconds

        # Most other hardware has negligible overhead
        return 0.0

    def _explain_latency(
        self,
        sg: SubgraphDescriptor,
        compute_time: float,
        memory_time: float,
        bottleneck: BottleneckType,
        ratio: float
    ) -> str:
        """Generate human-readable explanation of bottleneck"""

        op_name = sg.node_name

        if bottleneck == BottleneckType.BANDWIDTH_BOUND:
            return (f"{op_name}: Memory-bound (bandwidth limit) - "
                   f"memory time {memory_time*1e6:.1f}μs vs "
                   f"compute time {compute_time*1e6:.1f}μs ({ratio:.1f}× slower)")
        elif bottleneck == BottleneckType.COMPUTE_BOUND:
            return (f"{op_name}: Compute-bound (FLOPs limit) - "
                   f"compute time {compute_time*1e6:.1f}μs vs "
                   f"memory time {memory_time*1e6:.1f}μs ({ratio:.1f}× slower)")
        else:
            return (f"{op_name}: Balanced - "
                   f"compute time {compute_time*1e6:.1f}μs, "
                   f"memory time {memory_time*1e6:.1f}μs")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_calibrated_analyzer(
    resource_model: HardwareResourceModel,
    hardware_id: str,
    precision: str = "fp32",
    registry_path: Optional[str] = None
) -> RooflineAnalyzer:
    """
    Create a RooflineAnalyzer with calibrated peak values from the hardware registry.

    This factory function looks up calibration data for the specified hardware
    and creates an analyzer with empirical peaks instead of theoretical.

    Args:
        resource_model: Hardware specifications (for structure/overhead modeling)
        hardware_id: Registry ID (e.g., "h100_sxm5", "i7_12700k")
        precision: Precision for peaks ("fp32", "fp16", etc.)
        registry_path: Optional custom registry path

    Returns:
        RooflineAnalyzer configured with calibrated peaks (if available)
        or theoretical peaks (as fallback)

    Example:
        >>> from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
        >>> mapper = create_h100_sxm5_80gb_mapper()
        >>> analyzer = create_calibrated_analyzer(
        ...     mapper.resource_model,
        ...     hardware_id="h100_sxm5",
        ...     precision="fp32"
        ... )
        >>> # analyzer now uses calibrated peaks if available
    """
    from pathlib import Path

    # Import registry (deferred to avoid circular imports)
    try:
        from ..hardware.calibration.registry_sync import HardwareRegistry
    except ImportError:
        # Registry not available, use theoretical
        prec = Precision.FP32 if precision == "fp32" else Precision(precision.upper())
        return RooflineAnalyzer(resource_model, precision=prec)

    # Load registry
    reg_path = Path(registry_path) if registry_path else None
    registry = HardwareRegistry(registry_path=reg_path)

    # Look up hardware
    entry = registry.get_hardware(hardware_id)

    if entry is None:
        # Hardware not in registry, use theoretical
        prec = Precision.FP32 if precision == "fp32" else Precision(precision.upper())
        return RooflineAnalyzer(resource_model, precision=prec)

    # Get roofline params from registry
    peak_gflops, bandwidth_gbps, _ = entry.get_roofline_params(precision)

    # Map precision string to enum
    prec = Precision.FP32 if precision == "fp32" else Precision(precision.upper())

    # Create analyzer with calibrated values
    return RooflineAnalyzer(
        resource_model=resource_model,
        precision=prec,
        calibrated_peak_flops=peak_gflops,
        calibrated_bandwidth=bandwidth_gbps
    )


def get_roofline_params_for_hardware(
    hardware_id: str,
    precision: str = "fp32",
    use_calibrated: bool = True
) -> Tuple[float, float, float]:
    """
    Get roofline parameters for a hardware target from the registry.

    Convenience function that returns (peak_gflops, bandwidth_gbps, ridge_point)
    for a hardware target, using calibrated values if available.

    Args:
        hardware_id: Registry ID (e.g., "h100_sxm5", "nvidia_a100_sxm4_80gb")
        precision: Precision ("fp32", "fp16", etc.)
        use_calibrated: Whether to prefer calibrated over theoretical

    Returns:
        Tuple of (peak_gflops, bandwidth_gbps, ridge_point)
        Returns (0, 0, 0) if hardware not found

    Example:
        >>> peak, bw, ridge = get_roofline_params_for_hardware("h100_sxm5", "fp32")
        >>> print(f"H100 peak: {peak:.0f} GFLOPS, ridge: {ridge:.1f}")
        H100 peak: 67000 GFLOPS, ridge: 20.0
    """
    try:
        from ..hardware.calibration.registry_sync import HardwareRegistry
    except ImportError:
        return (0.0, 0.0, 0.0)

    registry = HardwareRegistry()
    entry = registry.get_hardware(hardware_id)

    if entry is None:
        return (0.0, 0.0, 0.0)

    return entry.get_roofline_params(precision, use_calibrated)
