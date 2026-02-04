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

from graphs.core.structures import SubgraphDescriptor, PartitionReport, BottleneckType
from graphs.core.confidence import ConfidenceLevel, EstimationConfidence
from graphs.hardware.resource_model import HardwareResourceModel, Precision

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

    # Confidence (NEW - Phase 7)
    confidence: EstimationConfidence = field(
        default_factory=EstimationConfidence.unknown
    )

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
        lines.append(f"  Latency: {self.actual_latency * 1e6:.1f} us")
        lines.append(f"    Compute time: {self.compute_time * 1e6:.1f} us")
        lines.append(f"    Memory time:  {self.memory_time * 1e6:.1f} us")
        if self.overhead > 0:
            lines.append(f"    Overhead:     {self.overhead * 1e6:.1f} us")
        lines.append(f"  Bottleneck: {self.bottleneck.value} ({self.bottleneck_ratio:.1f}x slower)")
        lines.append(f"  Arithmetic Intensity: {self.arithmetic_intensity:.2f} FLOPs/byte")
        lines.append(f"  FLOP Utilization: {self.flops_utilization * 100:.1f}%")
        lines.append(f"  Bandwidth Utilization: {self.bandwidth_utilization * 100:.1f}%")
        if self.confidence.level != ConfidenceLevel.UNKNOWN:
            lines.append(f"  Confidence: {self.confidence}")
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

        # Get per-operation efficiency scaling based on operation granularity.
        # Large operations achieve higher efficiency (better occupancy, amortized overhead).
        # Small operations suffer from kernel launch overhead and low occupancy.
        # IMPORTANT: Compute and bandwidth have DIFFERENT efficiency characteristics:
        # - Compute efficiency: heavily impacted by kernel launch overhead for small ops
        # - Bandwidth efficiency: limited by DRAM physics, less impacted by kernel size
        compute_efficiency_scale = self._get_compute_efficiency_scale(sg)
        bandwidth_efficiency_scale = self._get_bandwidth_efficiency_scale(sg)

        # Effective peak for this operation = theoretical peak * operation efficiency
        effective_peak_flops = self.peak_flops * compute_efficiency_scale
        effective_peak_bandwidth = self.peak_bandwidth * bandwidth_efficiency_scale

        # Compute time = FLOPs / effective_peak_FLOPS
        compute_time = sg.flops / effective_peak_flops if effective_peak_flops > 0 else 0.0

        # Memory time = bytes / effective_peak_bandwidth
        total_bytes = sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
        memory_time = total_bytes / effective_peak_bandwidth if effective_peak_bandwidth > 0 else 0.0

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

    def _get_compute_efficiency_scale(self, sg: SubgraphDescriptor) -> float:
        """
        Compute efficiency scaling factor based on operation type and granularity.

        This affects COMPUTE performance (FLOPs/sec).

        Large operations achieve higher efficiency due to:
        - Better GPU occupancy (more threads, better SM utilization)
        - Amortized kernel launch overhead
        - Better cache utilization (streaming access patterns)

        Small operations suffer from:
        - Kernel launch overhead (5-10 us dominates sub-millisecond ops)
        - Low occupancy (not enough threads to fill all SMs)
        - Memory latency hiding (not enough compute to hide latency)

        Depthwise convolutions are fundamentally different:
        - Very low arithmetic intensity (few FLOPs per byte)
        - Severely memory-bound regardless of size
        - Measured 3-80 GFLOPS vs 400-4000 GFLOPS for standard convs (50x slower)

        Empirical calibration (Jetson Orin AGX, 50W, FP32):
        - Standard Conv2D: 968 GFLOPS average (26% efficiency)
        - Depthwise Conv2D: 3-80 GFLOPS (0.1-2% efficiency)
        - MobileNetV2: heavily uses depthwise -> 17ms measured
        - ResNet-18:   standard convs only -> 7ms measured
        - ViT-B/16:    558M FLOPs/sg avg -> 67% eff -> scale = 1.91

        Model: Piecewise linear interpolation in log-space with three regimes:
        - Small ops (< 10M): scale = 0.4 (kernel launch dominated)
        - Medium ops (10M - 200M): linear 0.4 -> 0.8
        - Large ops (> 200M): linear 0.8 -> 2.0 (saturates at 2.0)

        Returns:
            Efficiency scale factor (0.4 to 2.0 range)
        """
        import math
        from graphs.core.structures import OperationType

        hw_type = self.resource_model.hardware_type.name
        flops = sg.flops

        if flops <= 0:
            return 1.0  # No compute, efficiency doesn't matter

        # Check for depthwise convolution (severely inefficient on GPUs)
        # CALIBRATION (Jetson Orin AGX, 50W, FP32):
        # - Depthwise 3x3: 3-80 GFLOPS measured vs 968 GFLOPS for standard conv
        # - This is ~0.3-8% of standard conv efficiency -> use 0.03-0.08 scale
        is_depthwise = (
            # Check operation_types list (fused subgraphs may have multiple ops)
            (hasattr(sg, 'operation_types') and OperationType.CONV2D_DEPTHWISE in sg.operation_types) or
            # Check is_depthwise flag if set
            getattr(sg, 'is_depthwise', False) or
            # Fallback to node name detection
            (hasattr(sg, 'node_names') and any('dw' in n.lower() for n in sg.node_names)) or
            (hasattr(sg, 'node_names') and any('depthwise' in n.lower() for n in sg.node_names))
        )

        # Check for Conv2d patterns and BatchNorm presence
        # CALIBRATION (Jetson Orin AGX, 50W, FP32, TF32 ENABLED - cuDNN default):
        # - Conv2d+BN+ReLU (231M): 720 GFLOPS
        # - Conv2d+ReLU no BN (231M): 1083 GFLOPS (1.5x faster without BN overhead)
        # - Conv2d+ReLU no BN (3.7G, VGG): 5374 GFLOPS (large convs more efficient)
        # Key insight: cuDNN uses TF32 by default, achieving ~1.5-2.5x FP32 theoretical
        fusion_pattern = getattr(sg, 'fusion_pattern', '') if hasattr(sg, 'fusion_pattern') else ''
        has_batchnorm = (
            'BatchNorm' in fusion_pattern or
            'batchnorm' in fusion_pattern.lower() or
            (hasattr(sg, 'operation_types') and any(
                'BATCHNORM' in str(op) for op in sg.operation_types
            ))
        )
        is_conv_pattern = (
            'Conv2d' in fusion_pattern or
            'conv2d' in fusion_pattern.lower() or
            (hasattr(sg, 'operation_types') and OperationType.CONV2D in sg.operation_types)
        )
        is_conv_bn_pattern = has_batchnorm and is_conv_pattern
        is_conv_only_pattern = is_conv_pattern and not has_batchnorm

        # GPU efficiency model (CALIBRATED on Jetson Orin AGX 50W, 2026-02-04)
        # Note: cuDNN uses TF32 by default, which achieves higher throughput than pure FP32
        # The 958 GFLOPS base is the 50W thermal profile for FP32
        # Actual achieved GFLOPS with TF32:
        #   - Conv2d+BN+ReLU (231M): 720 GFLOPS = 0.75x of base
        #   - Conv2d+ReLU (231M, no BN): 1083 GFLOPS = 1.13x of base
        #   - Conv2d+ReLU (3.7G, VGG-style): 5374 GFLOPS = 5.6x of base
        if hw_type == 'GPU':
            # Depthwise convolutions: dramatically lower efficiency
            if is_depthwise:
                # Calibrated: depthwise gets 3-80 GFLOPS vs 968 GFLOPS standard
                # Average ~30 GFLOPS = 3% of standard efficiency
                return 0.03

            if flops < 1e6:
                # Tiny ops (< 1M FLOPs): dominated by launch overhead
                return 0.01

            log_flops = math.log10(flops)

            if flops < 10e6:
                # Very small ops (1M-10M): severely overhead-dominated
                # MobileNet-V2 (9M avg): 38 GFLOPS / 958 peak = 4% scale
                t = (log_flops - 6.0) / 1.0
                t = max(0.0, min(1.0, t))
                return 0.02 + 0.04 * t  # 0.02 -> 0.06

            if flops < 50e6:
                # Small ops (10M-50M): improving
                t = (log_flops - 7.0) / 0.7  # log10(50M) = 7.7
                t = max(0.0, min(1.0, t))
                return 0.06 + 0.14 * t  # 0.06 -> 0.20

            if flops < 200e6:
                # Medium ops (50M-200M): good efficiency
                # CALIBRATED from ResNet-18 Conv2d+BN+ReLU benchmark:
                #   - 231M FLOPs layer: 0.330ms = 700 GFLOPS
                #   - Peak 958 GFLOPS -> scale = 0.73
                t = (log_flops - 7.7) / 0.6  # log10(200M) = 8.3
                t = max(0.0, min(1.0, t))
                if is_conv_bn_pattern:
                    # Conv2d+BN: BatchNorm adds memory traffic overhead
                    # bn_fusion_factor = 0.67 applied to base scale
                    base_scale = 0.35 + 0.45 * t  # 0.35 -> 0.80
                    return base_scale * 0.67  # 0.23 -> 0.54
                elif is_conv_only_pattern:
                    # Conv2d+ReLU (no BN): 1083 GFLOPS / 958 = 1.13 at 231M
                    # Higher than Conv2d+BN because no BatchNorm memory overhead
                    return 0.60 + 0.53 * t  # 0.60 -> 1.13
                else:
                    # MatMul/other
                    return 0.35 + 0.45 * t  # 0.35 -> 0.80

            else:
                # Large ops (> 200M)
                # IMPORTANT: Efficiency scales with operation size but NOT linearly
                # Use a curved scaling to match empirical observations:
                #
                # CALIBRATION (Jetson Orin AGX, 50W, TF32 enabled, Conv2d+BN+ReLU):
                # - 231M FLOPs (ResNet 56x56): 720 GFLOPS = 0.75x of 958 base
                # - 925M FLOPs (28x28 spatial): 2475 GFLOPS = 2.58x
                # - 3699M FLOPs (28x28 spatial): 3210 GFLOPS = 3.35x
                # - 7399M FLOPs (large spatial): ~5000 GFLOPS = 5.2x
                #
                # Key insight: Efficiency doesn't climb much from 200M-500M,
                # then ramps up significantly for truly large ops (>1G).
                #
                # For extremely large ops (>10G), efficiency plateaus or drops
                # due to memory pressure (can't fit everything in L2).
                #
                # Three-phase model:
                # Phase 1 (200M-500M): 0.73 -> 0.80 (slow growth)
                # Phase 2 (500M-5G): 0.80 -> 4.50 (fast growth with TF32/occupancy)
                # Phase 3 (>5G): 4.50 -> 5.00 (plateau, memory-limited)

                if flops < 500e6:
                    # Phase 1: 200M-500M, slow efficiency growth
                    # ResNets mostly live here (max 236M)
                    t = (log_flops - 8.3) / 0.4  # log10(500M) = 8.7
                    t = max(0.0, min(1.0, t))
                    if is_conv_bn_pattern:
                        return 0.73 + 0.07 * t  # 0.73 -> 0.80
                    elif is_conv_only_pattern:
                        return 1.13 + 0.17 * t  # 1.13 -> 1.30
                    else:
                        return 0.85 + 0.05 * t  # 0.85 -> 0.90

                elif flops < 5e9:
                    # Phase 2: 500M-5G, fast efficiency growth
                    # VGG and segmentation models benefit here
                    t = (log_flops - 8.7) / 1.0  # log10(5G) = 9.7
                    t = max(0.0, min(1.0, t))
                    if is_conv_bn_pattern:
                        return 0.80 + 3.70 * t  # 0.80 -> 4.50
                    elif is_conv_only_pattern:
                        return 1.30 + 4.10 * t  # 1.30 -> 5.40
                    else:
                        return 0.90 + 0.50 * t  # 0.90 -> 1.40

                else:
                    # Phase 3: >5G, efficiency plateaus
                    # Extremely large ops may hit memory bandwidth limits
                    t = (log_flops - 9.7) / 0.6  # log10(20G) = 10.3
                    t = max(0.0, min(1.0, t))
                    if is_conv_bn_pattern:
                        # Slight increase then plateau at 5.0
                        return 4.50 + 0.50 * t  # 4.50 -> 5.00
                    elif is_conv_only_pattern:
                        return 5.40 + 0.20 * t  # 5.40 -> 5.60
                    else:
                        return 1.40 + 0.10 * t  # 1.40 -> 1.50

        # CPU efficiency model
        # CPUs are much more sensitive to operation size than GPUs because:
        # 1. No SIMT - parallelism is at SIMD and thread level only
        # 2. Function call overhead per operation is significant
        # 3. Cache hierarchy effects are pronounced for small working sets
        # 4. Memory latency is harder to hide with limited parallelism
        #
        # Empirical calibration (i7-12700K, FP32):
        # - MobileNet (9M avg): measured 10ms, need ~0.17 scale to match
        # - ResNet (113M avg): measured 12ms, scale ~0.85 works
        # - ViT (225M avg): measured 97ms, scale ~1.0 works
        if hw_type == 'CPU':
            if flops < 1e6:
                # Tiny ops: dominated by overhead
                return 0.15

            log_flops = math.log10(flops)

            if flops < 10e6:
                # Very small ops (1M - 10M): still heavily overhead-dominated
                # Linear from 0.15 to 0.25
                t = (log_flops - 6.0) / 1.0
                t = max(0.0, min(1.0, t))
                return 0.15 + 0.10 * t
            elif flops < 100e6:
                # Small-medium ops (10M - 100M): improving efficiency
                # Linear from 0.25 to 0.70
                t = (log_flops - 7.0) / 1.0
                t = max(0.0, min(1.0, t))
                return 0.25 + 0.45 * t
            else:
                # Large ops (> 100M): good efficiency, approaching peak
                # Linear from 0.70 to 1.0
                t = (log_flops - 8.0) / 1.0
                t = max(0.0, min(1.0, t))
                return 0.70 + 0.30 * t

        # TPU/KPU: handled by _get_discrete_resource_correction
        # Other hardware: no operation-size efficiency scaling
        return 1.0

    def _get_bandwidth_efficiency_scale(self, sg: SubgraphDescriptor) -> float:
        """
        Bandwidth efficiency scaling factor based on operation characteristics.

        This affects MEMORY BANDWIDTH performance (bytes/sec).

        CRITICAL: Bandwidth efficiency is fundamentally different from compute efficiency:
        - Memory bandwidth is limited by DRAM physics, not kernel launch overhead
        - Small operations can still achieve good bandwidth if access is coalesced
        - GPU memory controller typically achieves 60-80% of peak bandwidth

        EXCEPTION: Depthwise convolutions have poor bandwidth efficiency due to:
        - Scattered access patterns (each filter only touches one channel)
        - Poor L2 cache utilization
        - Many small memory transactions instead of coalesced streaming

        CALIBRATION (Jetson Orin AGX 50W, 204 GB/s peak):
        - add_ReLU (residual connections): measured ~80 GB/s effective = 40%
        - Depthwise Conv2D: measured ~15 GB/s effective = 7%
        - Standard Conv2D: measured ~60 GB/s effective = 30% (compute-bound)

        The key factors affecting bandwidth efficiency:
        1. Access pattern (coalesced vs random): 0.3-0.9x
        2. Memory controller overhead: ~0.8x
        3. Cache effects (L2 hit rate): can improve effective bandwidth

        Model: Conservative floor with size-based improvement
        - Depthwise convolutions: very low efficiency (0.07)
        - All other operations: minimum 0.3 (30% of peak)
        - Large tensor operations: up to 0.7 (70% of peak)
        """
        import math
        from graphs.core.structures import OperationType

        hw_type = self.resource_model.hardware_type.name

        if hw_type == 'GPU':
            # Check for depthwise convolution (poor bandwidth efficiency)
            # CALIBRATION (Jetson Orin AGX, 50W, FP32):
            # - Depthwise Conv2D achieves very low effective bandwidth
            # - Due to scattered access patterns and poor cache utilization
            is_depthwise = (
                # Check operation_types list (fused subgraphs may have multiple ops)
                (hasattr(sg, 'operation_types') and OperationType.CONV2D_DEPTHWISE in sg.operation_types) or
                # Check is_depthwise flag if set
                getattr(sg, 'is_depthwise', False) or
                # Fallback to node name detection
                (hasattr(sg, 'node_names') and any('dw' in n.lower() for n in sg.node_names)) or
                (hasattr(sg, 'node_names') and any('depthwise' in n.lower() for n in sg.node_names))
            )

            if is_depthwise:
                # Depthwise convolutions have VERY poor bandwidth utilization
                # Due to scattered access patterns and lack of data reuse
                # CALIBRATION (Jetson Orin AGX 50W, MobileNet-V2):
                # - 17 depthwise subgraphs should account for ~6.2ms of 15.5ms total
                # - With 0.07 efficiency: 1.9ms estimated (3x underestimate)
                # - Need ~0.02 efficiency to match measured (~4 GB/s effective)
                return 0.02  # ~4 GB/s on 204 GB/s peak

            # GPU memory bandwidth is relatively consistent regardless of kernel size
            # The main factors are access patterns and memory controller efficiency

            # Total bytes transferred (approximation of working set size)
            total_bytes = sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes

            if total_bytes <= 0:
                return 0.5  # Default moderate efficiency

            # Larger transfers achieve better bandwidth efficiency due to:
            # - Better pipelining in memory controller
            # - Amortized DRAM row activation overhead
            # - Better L2 cache behavior

            log_bytes = math.log10(max(total_bytes, 1))

            if total_bytes < 1e4:  # < 10 KB
                # Very small transfers: memory latency dominates
                return 0.3

            if total_bytes < 1e6:  # 10 KB - 1 MB
                # Small transfers: improving efficiency
                # Linear from 0.3 to 0.5
                t = (log_bytes - 4.0) / 2.0  # log10(1MB) = 6.0
                t = max(0.0, min(1.0, t))
                return 0.3 + 0.2 * t

            if total_bytes < 10e6:  # 1 MB - 10 MB
                # Medium transfers: good efficiency
                # Linear from 0.5 to 0.6
                t = (log_bytes - 6.0) / 1.0
                t = max(0.0, min(1.0, t))
                return 0.5 + 0.1 * t

            else:  # > 10 MB
                # Large transfers: best efficiency (streaming)
                return 0.7

        elif hw_type == 'CPU':
            # CPU memory bandwidth is more variable due to cache hierarchy
            # but still doesn't have the kernel launch penalty
            return 0.5  # Conservative estimate

        # Other hardware: default to moderate efficiency
        return 0.5

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
        """
        Estimate overhead (kernel launch, etc.)

        CALIBRATION (Jetson Orin AGX, 50W):
        - Simple kernel launch: ~5-10 us
        - Hardswish/Hardsigmoid: ~100 us (non-fused activation)
        - SE block (avgpool + 2 FC + activations): ~575 us
        - Pointwise conv (tiny 1x1): ~100-200 us

        MobileNet-V3 has many small operations that are dominated by
        kernel launch overhead rather than actual compute/memory time.
        """
        if self.resource_model.hardware_type.name == 'GPU':
            # Base kernel launch overhead
            base_overhead = 5e-6  # 5 microseconds

            # Check operation patterns for additional overhead
            fusion_pattern = getattr(sg, 'fusion_pattern', '') if hasattr(sg, 'fusion_pattern') else ''
            node_name = sg.node_name if hasattr(sg, 'node_name') else ''

            # Get operation types list
            op_types_str = '_'.join(str(op).split('.')[-1] for op in sg.operation_types) if hasattr(sg, 'operation_types') else ''

            # Hardswish/Hardsigmoid activations have high overhead (~100us)
            # These are often separate kernels on GPU
            if 'HARDSWISH' in op_types_str.upper() or 'hardswish' in fusion_pattern.lower():
                base_overhead = 100e-6  # 100 microseconds

            if 'hardsigmoid' in node_name.lower() or 'scale_activation' in node_name.lower():
                # Hardsigmoid in SE blocks
                base_overhead = 100e-6

            # Squeeze-Excitation pattern: avgpool -> fc -> relu -> fc -> sigmoid -> mul
            # Very expensive due to multiple sequential tiny kernels
            if 'ADAPTIVEAVGPOOL' in op_types_str and 'CONV2D_POINTWISE' in op_types_str:
                # SE block internal path
                base_overhead = 200e-6  # 200 microseconds for SE FC path

            # UNKNOWN operations - need to distinguish MobileNet-V3 patterns from others
            # MobileNet-V3 UNKNOWN: hardsigmoid, mul (SE scaling) - slow activations
            # ViT UNKNOWN: gelu, softmax, add, reshape - typically faster
            if 'UNKNOWN' in op_types_str:
                # Check node name for specific slow patterns
                node_lower = node_name.lower()
                if ('hardsigmoid' in node_lower or
                    'scale_activation' in node_lower or
                    ('mul' in node_lower and sg.total_flops < 100000)):
                    # MobileNet-V3 style slow activations/SE scaling
                    base_overhead = 100e-6  # 100 microseconds
                else:
                    # Other UNKNOWN ops (GELU, softmax, add, etc.)
                    # These have lower overhead
                    base_overhead = 10e-6  # 10 microseconds

            # Pointwise convolutions (1x1) with very few FLOPs
            # These are common in MobileNet-V3 and are overhead-dominated
            if 'POINTWISE' in op_types_str:
                if sg.total_flops < 1e6:
                    # Very tiny pointwise convs (MobileNet-V3-Small style)
                    base_overhead = max(base_overhead, 150e-6)
                elif sg.total_flops < 5e6:
                    # Small pointwise convs
                    base_overhead = max(base_overhead, 100e-6)

            # Depthwise convolutions also have high per-op overhead
            if 'DEPTHWISE' in op_types_str:
                if sg.total_flops < 2e6:
                    # Tiny depthwise (MobileNet-V3-Small)
                    base_overhead = max(base_overhead, 150e-6)
                else:
                    base_overhead = max(base_overhead, 75e-6)

            return base_overhead

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
