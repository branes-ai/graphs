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

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from enum import Enum

from graphs.core.structures import SubgraphDescriptor, PartitionReport, BottleneckType
from graphs.core.confidence import ConfidenceLevel, EstimationConfidence
from graphs.hardware.resource_model import HardwareResourceModel, Precision

if TYPE_CHECKING:
    from ..hardware.calibration.registry_sync import HardwareEntry


@dataclass(frozen=True)
class MemoryExplanation:
    """V5-4 explainability: structured breakdown of where memory_time
    came from when the V5-3b tier-aware path fired.

    Populated on ``LatencyDescriptor.memory_explanation`` whenever
    ``RooflineAnalyzer(use_tier_aware_memory=True)`` successfully routed
    a subgraph through ``pick_binding_tier``. ``None`` on the scalar
    ``bw_efficiency_scale`` path -- callers that need to render a
    binding-tier column should treat ``None`` as "not applicable" and
    fall back to displaying nothing rather than a placeholder.

    Field semantics:
      * ``binding_tier_name`` -- the tier whose bandwidth gates kernel
        throughput (the streaming source one outward from the residency
        tier). Typical values: 'L1', 'L2', 'L3', 'DRAM'.
      * ``residency_tier_name`` -- the tier whose capacity holds the
        per-op residency window. Equal to ``binding_tier_name`` when
        the kernel is already at the outermost tier (DRAM).
      * ``tile_dims`` -- op-specific (vector_add: ``(N,)``; matmul /
        linear: ``(Mt, Nt)`` C-tile dims).
      * ``residency_bytes`` -- the working set the chosen tile occupies
        in the residency tier (always <= residency tier's aggregate
        capacity, except for the DRAM-overflow fallthrough).
      * ``bytes_loaded`` -- bytes streamed from the binding tier per
        kernel exec; the numerator of ``memory_time`` in the new path.
      * ``effective_bandwidth_bps`` -- the binding tier's calibrated
        BW (peak * achievable_fraction); the denominator of memory_time.
    """

    binding_tier_name: str
    residency_tier_name: str
    tile_dims: Tuple[int, ...]
    residency_bytes: int
    bytes_loaded: int
    effective_bandwidth_bps: float


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
    memory_time: float  # Time limited by bandwidth
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

    # V5-4: structured breakdown of the tier-aware memory_time path.
    # Populated by RooflineAnalyzer when use_tier_aware_memory=True
    # and the subgraph passed the V5-3b eligibility predicate. Stays
    # None on the scalar bw_efficiency_scale path.
    memory_explanation: Optional[MemoryExplanation] = None

    def __str__(self) -> str:
        """Short summary"""
        return (
            f"Latency({self.subgraph_name}: "
            f"{self.actual_latency*1e6:.1f}μs, {self.bottleneck.value})"
        )

    def format_summary(self) -> str:
        """Detailed multi-line summary"""
        lines = []
        lines.append(f"Subgraph: {self.subgraph_name}")
        lines.append(f"  Latency: {self.actual_latency * 1e6:.1f} us")
        lines.append(f"    Compute time: {self.compute_time * 1e6:.1f} us")
        lines.append(f"    Memory time:  {self.memory_time * 1e6:.1f} us")
        if self.overhead > 0:
            lines.append(f"    Overhead:     {self.overhead * 1e6:.1f} us")
        lines.append(
            f"  Bottleneck: {self.bottleneck.value} ({self.bottleneck_ratio:.1f}x slower)"
        )
        lines.append(
            f"  Arithmetic Intensity: {self.arithmetic_intensity:.2f} FLOPs/byte"
        )
        lines.append(f"  FLOP Utilization: {self.flops_utilization * 100:.1f}%")
        lines.append(
            f"  Bandwidth Utilization: {self.bandwidth_utilization * 100:.1f}%"
        )
        if self.confidence.level != ConfidenceLevel.UNKNOWN:
            lines.append(f"  Confidence: {self.confidence}")
        if self.memory_explanation is not None:
            me = self.memory_explanation
            lines.append(
                f"  Memory binding: tier={me.binding_tier_name} "
                f"(residency={me.residency_tier_name}, "
                f"tile={me.tile_dims}, "
                f"residency_bytes={me.residency_bytes}, "
                f"bytes_loaded={me.bytes_loaded}, "
                f"eff_bw={me.effective_bandwidth_bps / 1e9:.1f} GB/s)"
            )
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
        return (
            f"RooflineReport(total={self.total_latency*1e3:.2f}ms, "
            f"compute_bound={self.num_compute_bound}, "
            f"memory_bound={self.num_memory_bound})"
        )

    def format_report(
        self, show_per_subgraph: bool = False, max_subgraphs: int = 10
    ) -> str:
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
        lines.append(
            f"  AI Breakpoint:    {self.arithmetic_intensity_breakpoint:.2f} FLOPs/byte"
        )
        lines.append("")

        # Total latency
        lines.append("Total Latency:")
        lines.append(
            f"  Total:   {self.total_latency * 1e3:.2f} ms ({self.total_latency * 1e6:.0f} μs)"
        )
        lines.append(
            f"  Compute: {self.total_compute_time * 1e3:.2f} ms ({self.total_compute_time / self.total_latency * 100:.1f}%)"
        )
        lines.append(
            f"  Memory:  {self.total_memory_time * 1e3:.2f} ms ({self.total_memory_time / self.total_latency * 100:.1f}%)"
        )
        if self.total_overhead > 0:
            lines.append(
                f"  Overhead: {self.total_overhead * 1e3:.2f} ms ({self.total_overhead / self.total_latency * 100:.1f}%)"
            )
        lines.append("")

        # Bottleneck distribution
        total_ops = self.num_compute_bound + self.num_memory_bound + self.num_balanced
        lines.append("Bottleneck Distribution:")
        if total_ops > 0:
            lines.append(
                f"  Compute-bound: {self.num_compute_bound} ops ({self.num_compute_bound / total_ops * 100:.1f}%)"
            )
            lines.append(
                f"  Memory-bound:  {self.num_memory_bound} ops ({self.num_memory_bound / total_ops * 100:.1f}%)"
            )
            lines.append(
                f"  Balanced:      {self.num_balanced} ops ({self.num_balanced / total_ops * 100:.1f}%)"
            )
        lines.append("")

        # Utilization
        lines.append("Hardware Utilization:")
        lines.append(
            f"  Average FLOP Utilization:      {self.average_flops_utilization * 100:.1f}%"
        )
        lines.append(
            f"  Average Bandwidth Utilization: {self.average_bandwidth_utilization * 100:.1f}%"
        )
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
            sorted_latencies = sorted(
                self.latencies, key=lambda l: l.actual_latency, reverse=True
            )[:max_subgraphs]

            for i, lat in enumerate(sorted_latencies, 1):
                lines.append(f"{i}. {lat.subgraph_name}")
                lines.append(f"   Latency: {lat.actual_latency * 1e6:.1f} μs")
                lines.append(
                    f"   Bottleneck: {lat.bottleneck.value} ({lat.bottleneck_ratio:.1f}×)"
                )
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
        thermal_profile: Optional[str] = None,
        use_tier_aware_memory: bool = False,
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
            use_tier_aware_memory: V5-3b opt-in flag. When True, single-op
                MATMUL/LINEAR subgraphs route memory_time through the
                tier-aware path (tier_picker + per-op reuse models +
                MemoryTier.effective_bandwidth_bps) instead of the scalar
                bw_efficiency_scale * peak_bandwidth. Defaults to False so
                V4 floors hold until V5-5 calibrates per-tier
                achievable_fraction. Only takes effect on hardware whose
                memory_hierarchy has >=2 tiers; everything else falls
                through to the scalar path even when the flag is True.
        """
        self.resource_model = resource_model
        self.precision = precision
        self.efficiency_factor = efficiency_factor
        self.thermal_profile = thermal_profile
        self.use_tier_aware_memory = use_tier_aware_memory
        # Stashes for the V5-3b/V5-4 tier-aware path: when
        # _try_tier_aware_memory_time succeeds, it writes the binding
        # tier bytes_loaded + structured MemoryExplanation here so
        # _analyze_subgraph can use the tier-aware byte count for
        # downstream attained_bandwidth math AND attach the
        # explanation to the LatencyDescriptor it returns. Reset by
        # the helper itself; only read after a successful call.
        self._last_tier_bytes_loaded: int = 0
        self._last_memory_explanation: Optional["MemoryExplanation"] = None
        self.is_calibrated = (
            efficiency_factor is not None
            or calibrated_peak_flops is not None
            or calibrated_bandwidth is not None
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
        self.ai_breakpoint = (
            self.peak_flops / self.peak_bandwidth if self.peak_bandwidth > 0 else 0.0
        )

    @staticmethod
    def _get_effective_peak_ops(
        resource_model: HardwareResourceModel,
        precision: Precision,
        thermal_profile: Optional[str] = None,
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
        if (
            hasattr(resource_model, "thermal_operating_points")
            and resource_model.thermal_operating_points
            and thermal_profile
        ):
            thermal_point = resource_model.thermal_operating_points.get(thermal_profile)
            if thermal_point and precision in thermal_point.performance_specs:
                perf_spec = thermal_point.performance_specs[precision]
                effective_ops = perf_spec.effective_ops_per_sec
                if effective_ops > 0:
                    return effective_ops

        # Try 2: thermal_operating_points with default profile
        if (
            hasattr(resource_model, "thermal_operating_points")
            and resource_model.thermal_operating_points
            and hasattr(resource_model, "default_thermal_profile")
            and resource_model.default_thermal_profile
        ):
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
            default_profile = resource_model.precision_profiles[
                resource_model.default_precision
            ]
            return default_profile.peak_ops_per_sec

        # Last resort: return 0 (will cause issues, but signals problem)
        return 0.0

    def analyze(
        self,
        subgraphs: List[SubgraphDescriptor],
        partition_report: Optional[PartitionReport] = None,
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
                is_compute_bound=(latency.bottleneck == BottleneckType.COMPUTE_BOUND),
            )
            roofline_points.append(point)

        # Aggregate statistics
        total_latency = sum(l.actual_latency for l in latencies)
        total_compute_time = sum(l.compute_time for l in latencies)
        total_memory_time = sum(l.memory_time for l in latencies)
        total_overhead = sum(l.overhead for l in latencies)

        # Bottleneck distribution
        num_compute_bound = sum(
            1 for l in latencies if l.bottleneck == BottleneckType.COMPUTE_BOUND
        )
        num_memory_bound = sum(
            1 for l in latencies if l.bottleneck == BottleneckType.BANDWIDTH_BOUND
        )
        num_balanced = sum(
            1 for l in latencies if l.bottleneck == BottleneckType.BALANCED
        )

        # Average utilization
        avg_flops_util = (
            sum(l.flops_utilization for l in latencies) / len(latencies)
            if latencies
            else 0.0
        )
        avg_bw_util = (
            sum(l.bandwidth_utilization for l in latencies) / len(latencies)
            if latencies
            else 0.0
        )

        # Critical path
        critical_path_latency = 0.0
        critical_path_subgraphs = []
        if partition_report and partition_report.critical_path_subgraphs:
            critical_path_subgraphs = partition_report.critical_path_subgraphs
            # Sum latency for critical path
            critical_path_latency = sum(
                l.actual_latency
                for l in latencies
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
        compute_time = (
            sg.flops / effective_peak_flops if effective_peak_flops > 0 else 0.0
        )

        # Memory time = bytes / effective_peak_bandwidth.
        # ``_dram_traffic_bytes`` is hardware-aware: on weight-stationary
        # accelerators (KPU) it amortizes weight loads across the on-chip
        # tile fabric instead of treating weights as re-fetched per layer
        # (issue #51). For other hardware the result is the naive sum.
        total_bytes = self._dram_traffic_bytes(sg)
        memory_time = (
            total_bytes / effective_peak_bandwidth
            if effective_peak_bandwidth > 0
            else 0.0
        )

        # V5-3b: opt-in tier-aware memory_time. When the analyzer was
        # constructed with use_tier_aware_memory=True AND the subgraph
        # is a single-op MATMUL/LINEAR with a clean 2D shape AND the
        # hardware's memory_hierarchy has >=2 tiers, replace the scalar
        # memory_time with the tier-picker output. Default False -> no
        # behavior change vs pre-V5-3b. See _try_tier_aware_memory_time
        # for the eligibility predicate.
        if self.use_tier_aware_memory:
            # Reset the explanation stash before the attempt so that a
            # decline (returning None) doesn't leave a stale explanation
            # from a prior subgraph attached to this descriptor.
            self._last_memory_explanation = None
            tier_memory_time = self._try_tier_aware_memory_time(sg)
            if tier_memory_time is not None:
                memory_time = tier_memory_time
                total_bytes = self._last_tier_bytes_loaded

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

        # CPU dispatch floor: PyTorch's per-op runtime overhead that the
        # roofline math doesn't capture. Apply as a floor rather than
        # additive overhead so it only kicks in for ops too small for
        # the kernel time to dominate -- avoids over-correcting medium
        # ops where the overhead amortizes naturally.
        #
        # Op-aware: see _get_cpu_dispatch_floor for the first-principles
        # decomposition (base ATen dispatch + BLAS thread launch +
        # stride/layout normalization + parameter access + bias add).
        # Empirical floors on i7-12700K (#69 set 5 us across the board;
        # this PR splits per op based on smallest-shape measurements):
        #   vector_add: 2 us  (base ATen dispatch only)
        #   matmul:     6 us  (+ BLAS thread launch, stride norm)
        #   linear:     9 us  (+ parameter access, bias epilogue)
        if self.resource_model.hardware_type.name == "CPU":
            actual_latency = max(actual_latency, self._get_cpu_dispatch_floor(sg))

        # Arithmetic intensity
        ai = sg.flops / total_bytes if total_bytes > 0 else 0.0

        # Attained performance
        attained_flops = sg.flops / actual_latency if actual_latency > 0 else 0.0
        attained_bandwidth = total_bytes / actual_latency if actual_latency > 0 else 0.0

        # Utilization
        flops_util = attained_flops / self.peak_flops if self.peak_flops > 0 else 0.0
        bw_util = (
            attained_bandwidth / self.peak_bandwidth if self.peak_bandwidth > 0 else 0.0
        )

        # Generate explanation
        explanation = self._explain_latency(
            sg, compute_time, memory_time, bottleneck, bottleneck_ratio
        )

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
            memory_explanation=self._last_memory_explanation,
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
        #
        # IMPORTANT: MBConv blocks (EfficientNet, MobileNetV3) fuse pointwise + depthwise
        # In these fused subgraphs, pointwise convolutions dominate FLOPs (~95%+)
        # Don't penalize the entire subgraph for containing a depthwise op.
        has_depthwise = (
            (
                hasattr(sg, "operation_types")
                and OperationType.CONV2D_DEPTHWISE in sg.operation_types
            )
            or getattr(sg, "is_depthwise", False)
            or (
                hasattr(sg, "node_names")
                and any("dw" in n.lower() for n in sg.node_names)
            )
            or (
                hasattr(sg, "node_names")
                and any("depthwise" in n.lower() for n in sg.node_names)
            )
        )
        has_pointwise = (
            hasattr(sg, "operation_types")
            and OperationType.CONV2D_POINTWISE in sg.operation_types
        )
        has_standard_conv = (
            hasattr(sg, "operation_types")
            and OperationType.CONV2D in sg.operation_types
        )
        # Only flag as "pure depthwise" if it has depthwise but NOT pointwise/standard conv
        # MBConv blocks have both -> use pointwise-dominated efficiency
        is_depthwise = has_depthwise and not has_pointwise and not has_standard_conv

        # Check for Conv2d patterns and BatchNorm presence
        # CALIBRATION (Jetson Orin AGX, 50W, FP32, TF32 ENABLED - cuDNN default):
        # - Conv2d+BN+ReLU (231M): 720 GFLOPS
        # - Conv2d+ReLU no BN (231M): 1083 GFLOPS (1.5x faster without BN overhead)
        # - Conv2d+ReLU no BN (3.7G, VGG): 5374 GFLOPS (large convs more efficient)
        # Key insight: cuDNN uses TF32 by default, achieving ~1.5-2.5x FP32 theoretical
        fusion_pattern = (
            getattr(sg, "fusion_pattern", "") if hasattr(sg, "fusion_pattern") else ""
        )
        has_batchnorm = (
            "BatchNorm" in fusion_pattern
            or "batchnorm" in fusion_pattern.lower()
            or (
                hasattr(sg, "operation_types")
                and any("BATCHNORM" in str(op) for op in sg.operation_types)
            )
        )
        is_conv_pattern = (
            "Conv2d" in fusion_pattern
            or "conv2d" in fusion_pattern.lower()
            or (
                hasattr(sg, "operation_types")
                and OperationType.CONV2D in sg.operation_types
            )
        )
        is_conv_bn_pattern = has_batchnorm and is_conv_pattern
        is_conv_only_pattern = is_conv_pattern and not has_batchnorm

        # Check for MBConv-style fused blocks (pointwise + depthwise)
        # These are used in EfficientNet, MobileNetV2/V3
        # FLOPs are dominated by pointwise, but overall efficiency is lower than
        # pure Conv2d+BN due to:
        # 1. Multiple sequential kernel launches within the fused block
        # 2. Large intermediate tensors (expansion ratio 4-6x)
        # 3. SiLU/Swish activation more expensive than ReLU
        is_mbconv_pattern = has_depthwise and has_pointwise

        # GPU efficiency model (CALIBRATED on Jetson Orin AGX 50W, 2026-02-04)
        # Note: cuDNN uses TF32 by default, which achieves higher throughput than pure FP32
        # The 958 GFLOPS base is the 50W thermal profile for FP32
        # Actual achieved GFLOPS with TF32:
        #   - Conv2d+BN+ReLU (231M): 720 GFLOPS = 0.75x of base
        #   - Conv2d+ReLU (231M, no BN): 1083 GFLOPS = 1.13x of base
        #   - Conv2d+ReLU (3.7G, VGG-style): 5374 GFLOPS = 5.6x of base
        #   - MBConv blocks: ~40-60 GFLOPS = 0.04-0.06x (much lower efficiency)
        if hw_type == "GPU":
            # Pure depthwise convolutions: dramatically lower efficiency
            if is_depthwise:
                # Calibrated: depthwise gets 3-80 GFLOPS vs 968 GFLOPS standard
                # Average ~30 GFLOPS = 3% of standard efficiency
                return 0.03

            # MBConv-style fused blocks (EfficientNet, MobileNet)
            # These achieve much lower efficiency than standard Conv2d+BN
            # despite pointwise convolutions dominating the FLOPs
            #
            # CALIBRATION (Jetson Orin AGX, 50W):
            # - EfficientNet-B0: avg 26M MBConv, measured 22.2ms
            # - EfficientNet-B1: avg 33M MBConv, measured 32.9ms
            # - EfficientNet-B2: avg 48M MBConv, measured 32.4ms
            #
            # Smaller MBConv blocks are LESS efficient due to:
            # 1. Higher kernel launch overhead relative to compute
            # 2. Less opportunity for memory latency hiding
            # 3. Worse GPU occupancy with smaller tensors
            if is_mbconv_pattern:
                log_flops_mbconv = math.log10(flops) if flops > 0 else 0
                if flops < 10e6:
                    return 0.025  # Very small MBConv: ~24 GFLOPS
                elif flops < 30e6:
                    # Small MBConv (B0 range: 15-30M)
                    t = (log_flops_mbconv - 7.0) / 0.48  # log10(30M) = 7.48
                    t = max(0.0, min(1.0, t))
                    return 0.025 + 0.010 * t  # 0.025 -> 0.035
                elif flops < 50e6:
                    # Medium MBConv (B1/B2 range: 30-50M)
                    t = (log_flops_mbconv - 7.48) / 0.22  # log10(50M) = 7.70
                    t = max(0.0, min(1.0, t))
                    return 0.035 + 0.020 * t  # 0.035 -> 0.055
                else:
                    # Large MBConv (B2 range: >50M)
                    t = (log_flops_mbconv - 7.70) / 0.60  # up to ~200M
                    t = max(0.0, min(1.0, t))
                    return 0.055 + 0.045 * t  # 0.055 -> 0.10

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
                    # ViT attention varies: B=244M, L=430M
                    t = (log_flops - 8.3) / 0.4  # log10(500M) = 8.7
                    t = max(0.0, min(1.0, t))
                    if is_conv_bn_pattern:
                        return 0.73 + 0.07 * t  # 0.73 -> 0.80
                    elif is_conv_only_pattern:
                        return 1.13 + 0.17 * t  # 1.13 -> 1.30
                    else:
                        # MatMul/Attention: efficiency scales strongly with size
                        # Smaller ViT (B/32, 244M) has lower efficiency (~0.7)
                        # Larger ViT (L/32, 430M) has higher efficiency (~1.1)
                        # This is because larger matrices utilize GPU better
                        return 0.60 + 0.55 * t  # 0.60 -> 1.15

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

        # CPU efficiency model -- single-kernel calibration.
        #
        # CALIBRATED against V4 single-kernel matmul measurements on
        # i7-12700K with Intel MKL, fp32 (committed at
        # validation/model_v4/results/baselines/i7_12700k_matmul.csv).
        # 78 measurements; medians per flops decade:
        #   10^5..10^6: scale ~0.34 (124 GFLOPS / 360 effective)
        #   10^6..10^7: scale ~0.48 (174 GFLOPS / 360)
        #   10^7..10^8: scale ~1.10 (395 GFLOPS / 360, MKL packing wins)
        #   10^9..10^10: scale ~0.91 (327 GFLOPS / 360, varies with shape)
        #
        # NOTE: This is a SINGLE-KERNEL curve. Full-model CNN inference is
        # SLOWER per layer than this curve predicts because of Python /
        # PyTorch dispatch overhead, activation tensor allocation, and
        # poor BLAS reuse across small layers. The previous curve
        # (0.15..1.0) was anchored to full-model measurements
        # (MobileNet/ResNet/ViT) and systematically over-predicted
        # single-kernel latency by 2-3x (issue #67).
        #
        # Until a separate "in-model overhead" term is modeled (#69 is
        # the linear counterpart, both point at the same architectural
        # gap), full-model latency will under-predict by 2-3x for
        # small-layer models like MobileNet. Use single-kernel V4 to
        # validate this curve; use a future full-model V4 sweep to
        # validate the per-call overhead model.
        if hw_type == "CPU":
            if flops < 1e6:
                # Tiny ops: ~0.34 of effective peak (median of V4 baseline
                # 10^5..10^6 bucket; 124 GFLOPS achieved).
                return 0.30

            log_flops = math.log10(flops)

            if flops < 10e6:
                # 1M..10M: steep ramp at MKL packing threshold.
                # 1.6M shapes hit ~0.55, 10M shapes approach ~0.95.
                t = (log_flops - 6.0) / 1.0
                t = max(0.0, min(1.0, t))
                return 0.30 + 0.65 * t
            elif flops < 100e6:
                # 10M..100M: peak per-kernel efficiency.
                # 19M shapes show median 1.10, individual cubes hit 1.37.
                t = (log_flops - 7.0) / 1.0
                t = max(0.0, min(1.0, t))
                return 0.95 + 0.25 * t
            else:
                # > 100M: plateau slightly above peak (1.20). Some cube
                # shapes achieve 1.5+ (compute exceeds the conservative
                # spec), some rectangular shapes drop to 0.3 (poor MKL
                # packing). 1.20 is a defensible median for typical use;
                # see #68 for the underlying peak FP32 spec issue.
                return 1.20

        # TPU/KPU: handled by _get_discrete_resource_correction
        # Other hardware: no operation-size efficiency scaling
        return 1.0

    # ------------------------------------------------------------------
    # CPU dispatch floor (op-aware; #69 set a single 5 us floor, this
    # PR splits per op based on first-principles dispatch decomposition
    # validated against the V4 i7-12700K baseline).
    # ------------------------------------------------------------------

    # Op -> dispatch floor (seconds), keyed by OperationType. The first-
    # principles model breaks dispatch into stackable components; each
    # op pays for the components that actually execute on its path.
    #
    # ## What is "dispatch" on a CPU?
    #
    # Unlike a GPU "kernel launch" (a discrete cudaLaunchKernel call
    # that the device queues), a CPU "dispatch" is the chain of
    # software work that PyTorch performs before any FLOPs execute:
    # Python frame entry, ATen dispatcher key resolution, output
    # tensor allocation, kernel invocation. None of these are physical
    # constraints -- a CPU can context-switch in nanoseconds, take an
    # ISR, return from an RPC -- but they ARE the unavoidable runtime
    # cost of routing a Python-level tensor op through the framework.
    # The roofline math ignores all of this (it's pure compute / BW
    # arithmetic), so we floor the prediction at the empirical
    # smallest-shape latency.
    #
    # ## Per-component cost (first-principles, validated on i7 + Python 3.10 / PyTorch 2.x):
    #
    # 1. Base ATen dispatch (~2 us)        - all ops
    #    Python frame entry + arg parsing (~300-500 ns), ATen
    #    dispatcher key resolution + boxing (~100-300 ns), output
    #    tensor allocation `at::empty` (~500 ns - 1 us), kernel
    #    invocation overhead (~200-500 ns).
    #
    # 2. BLAS thread launch (+2-4 us)      - matmul, linear (any GEMM)
    #    Once we route to oneMKL `cblas_sgemm`, the BLAS implementation
    #    forks worker threads via OpenMP / TBB to parallelize the GEMM.
    #    For tiny matrices the thread launch cost can EXCEED the actual
    #    compute time; this is the dominant non-base contribution to
    #    matmul/linear floor on x86.
    #
    # 3. Stride/layout normalization (+0.5-1 us) - matmul, linear
    #    Inputs may need contiguity / transpose for the BLAS backend
    #    (`is_contiguous` checks, optional copy via `at::empty +
    #    at::contiguous`). Vector add doesn't need this -- it's
    #    elementwise on identical-shape tensors.
    #
    # 4. nn.Module parameter access (+0.5-1 us) - linear, conv2d (parametric layers)
    #    `self.weight`, `self.bias` go through `nn.Module.__getattr__`
    #    overrides + `_parameters` OrderedDict lookups. Vector add and
    #    raw matmul take no parameters; this cost only hits parametric
    #    layers.
    #
    # 5. Bias epilogue (+0.5-2 us) - linear (and any addmm/fused-bias path)
    #    After `x @ W.T`, the bias add is either fused (`at::addmm`) or
    #    a separate dispatch hop (`at::add`). Either way, an additional
    #    chunk of dispatch + tiny memory traffic.
    #
    # ## Summary by op (i7-12700K, fp32, smallest measured shape):
    #
    #   vector_add: components 1 only         -> ~2 us empirical (1.84-1.97 us)
    #   matmul:     components 1+2+3          -> ~6 us empirical (6.2-6.8 us)
    #   linear:     components 1+2+3+4+5      -> ~9 us empirical (8.8-9.1 us)
    #
    # Pre-PR: a single 5 us floor over-predicted vector_add at small N
    # by ~150% and under-predicted linear at small shapes by ~40%.
    _CPU_DISPATCH_FLOOR_SECONDS: dict = {
        # ELEMENTWISE covers vector_add, ReLU, sigmoid, etc. They share
        # the simplest dispatch path (no BLAS, no params, no bias).
        "ELEMENTWISE": 2e-6,
        "MATMUL": 6e-6,
        "LINEAR": 9e-6,
    }
    _CPU_DISPATCH_FLOOR_DEFAULT_SECONDS: float = 5e-6

    def _get_cpu_dispatch_floor(self, sg: SubgraphDescriptor) -> float:
        """Op-aware CPU dispatch floor in seconds.

        Looks up the floor by ``sg.operation_type`` against
        ``_CPU_DISPATCH_FLOOR_SECONDS``. Unknown / fused / unset op
        types fall back to ``_CPU_DISPATCH_FLOOR_DEFAULT_SECONDS``
        (the legacy 5 us value), so this change is conservative for
        any op kind not explicitly covered.
        """
        op_name = sg.operation_type.name if sg.operation_type else None
        return self._CPU_DISPATCH_FLOOR_SECONDS.get(
            op_name, self._CPU_DISPATCH_FLOOR_DEFAULT_SECONDS
        )

    # ------------------------------------------------------------------
    # V5-3b: tier-aware memory_time path (opt-in; gated by
    # use_tier_aware_memory). Eligible only on single-op MATMUL/LINEAR
    # subgraphs with a clean 2D shape and a multi-tier hierarchy.
    # ------------------------------------------------------------------

    def _try_tier_aware_memory_time(self, sg: SubgraphDescriptor) -> Optional[float]:
        """Return the tier-aware memory_time for ``sg`` or None if the
        subgraph isn't eligible (caller falls back to scalar path).

        Eligibility predicate is intentionally conservative for V5-3b:
          * single-operator subgraph (``num_operators == 1``)
          * op type is MATMUL or LINEAR (the V5-3a reuse models cover
            these; vector_add tier-picking is wired but not yet routed
            from the analyzer because elementwise op detection is
            broader than 1-D vector add)
          * tensor info is populated and yields a clean 2D extraction
          * hardware ``memory_hierarchy`` has at least 2 tiers (mappers
            without on-chip BW peaks return DRAM-only; routing those
            through the new path with achievable_fraction defaulting
            to 1.0 would regress floors vs the scalar derate)

        On success, also stashes the bytes_loaded number on
        ``self._last_tier_bytes_loaded`` so the caller can use it for
        downstream attained_bandwidth math. This avoids returning a
        tuple and complicating the existing analyzer flow.
        """
        # Local imports keep the module-level import set unchanged for
        # callers that don't use the V5-3b path (e.g. legacy notebooks
        # importing RooflineAnalyzer in a pinned-deps env).
        from graphs.core.structures import OperationType
        from graphs.estimation.reuse_models import (
            REUSE_MODELS,
            bytes_per_element as _bytes_per_element,
        )
        from graphs.estimation.tier_picker import normalize_dtype, pick_binding_tier

        if sg.num_operators != 1:
            return None

        op_type = sg.operation_type
        if op_type == OperationType.MATMUL:
            op_kind = "matmul"
            shape = self._extract_matmul_shape(sg)
        elif op_type == OperationType.LINEAR:
            op_kind = "linear"
            shape = self._extract_linear_shape(sg)
        elif op_type == OperationType.ELEMENTWISE:
            # Tighter predicate than just OperationType.ELEMENTWISE:
            # vector_add specifically (a + b -> c). ReLU / sigmoid /
            # other elementwise ops have different operand counts
            # and reuse patterns, so the V5-3a vector_add reuse
            # model would mis-bytes-loaded for them. The shape
            # extractor returns None unless the SG matches the
            # strict 3-tensor 1-D vector-add layout.
            op_kind = "vector_add"
            shape = self._extract_vector_add_shape(sg)
        else:
            return None
        if shape is None:
            return None

        hierarchy = self.resource_model.memory_hierarchy
        if len(hierarchy) < 2:
            return None

        # Resolve dtype from the first input tensor; all tensors of a
        # well-formed matmul/linear share dtype, so checking the first
        # is sufficient. If the dtype string isn't in the table the
        # bytes_per_element call below raises KeyError -- catch and
        # fall back rather than crash a whole analyze() run.
        if not sg.input_tensors:
            return None
        raw_dtype = sg.input_tensors[0].dtype
        try:
            dtype = normalize_dtype(raw_dtype)
            _bytes_per_element(dtype)  # raises ValueError if unknown
        except (KeyError, ValueError):
            return None

        reuse_model = REUSE_MODELS[op_kind]
        result = pick_binding_tier(reuse_model, shape, dtype, hierarchy)
        if result is None:
            return None

        # Per-op effective BW (V5-3b flag-flip prerequisite): a single
        # ``achievable_fraction`` per tier doesn't capture that matmul /
        # linear achieve different effective BW than vector_add at the
        # same tier (different access patterns -> different cache hit
        # rates). The lookup falls back per-op -> per-tier -> 1.0.
        bw = self._per_op_effective_bw(result.binding_tier, op_kind)
        if bw <= 0:
            return None

        # V5 follow-up: L3/DRAM boundary cliff resolution for
        # vector_add. When the operand overflows the outermost cache
        # but is comparable to it (e.g., 50 MB vs 25 MB LLC), the
        # measured BW lands between cache and DRAM. The binary tier
        # picker says "binding = DRAM" with full DRAM-calibrated BW
        # (35 GB/s on i7), but reality shows ~84 GB/s for those
        # boundary shapes -- partial cache hits the model misses.
        #
        # OVERLAP physical model: cache warmup (filling LLC with the
        # first cap bytes) and DRAM streaming (remaining bytes) happen
        # in parallel via prefetch + memory-controller overlap, not
        # sequentially. memory_time = max(cache_fill_time,
        # dram_stream_time).
        #
        # Scope: vector_add specifically (zero-reuse model where
        # bytes_loaded == operand_footprint). For matmul / linear the
        # bytes_loaded calculation already encodes reuse via the
        # per-op model; the OVERLAP physics differ when data is
        # tile-streamed with reuse, so we don't apply it there.
        memory_time = self._memory_time_with_boundary_overlap(
            op_kind=op_kind,
            bytes_loaded=result.bytes_loaded,
            binding_tier=result.binding_tier,
            hierarchy=hierarchy,
            binding_tier_bw=bw,
        )
        self._last_tier_bytes_loaded = result.bytes_loaded
        self._last_memory_explanation = MemoryExplanation(
            binding_tier_name=result.binding_tier.name,
            residency_tier_name=result.residency_tier.name,
            tile_dims=tuple(result.tile.tile_dims),
            residency_bytes=int(result.tile.residency_bytes),
            bytes_loaded=int(result.bytes_loaded),
            effective_bandwidth_bps=float(bw),
        )
        return memory_time

    def _per_op_effective_bw(self, tier: "MemoryTier", op_kind: str) -> float:
        """V5-3b flag-flip prerequisite: per-op effective BW lookup.

        Returns ``tier.peak_bandwidth_bps * fraction``, where fraction
        falls back through:
          1. ``tier_achievable_fractions_by_op[op_kind][tier.name]``
             -- per-op override
          2. ``tier_achievable_fractions[tier.name]`` -- per-tier
             default (the V5-5 calibrated value)
          3. ``1.0`` -- ideal

        Per-op overrides are a workload-specific calibration knob:
        matmul / linear achieve different effective BW than
        vector_add at the same tier because of structured access
        patterns. The single-fraction-per-tier model can't capture
        this, so the per-op override sits on top.
        """
        op_dict = self.resource_model.tier_achievable_fractions_by_op.get(op_kind, {})
        if tier.name in op_dict:
            fraction = op_dict[tier.name]
        else:
            fraction = self.resource_model.tier_achievable_fractions.get(tier.name, 1.0)
        return tier.peak_bandwidth_bps * fraction

    def _memory_time_with_boundary_overlap(
        self,
        op_kind: str,
        bytes_loaded: int,
        binding_tier: "MemoryTier",
        hierarchy: list,
        binding_tier_bw: float,
    ) -> float:
        """V5-followup boundary cliff model: when a vector_add op
        binds at DRAM but its operand size is comparable to the
        outermost cache, blend cache fill time with DRAM stream
        time via the OVERLAP physics:

            memory_time = max(cache_fill_time, dram_stream_time)

        where:
            cache_fill_time = min(bytes_loaded, last_cache.cap)
                              / last_cache.effective_bandwidth_bps
            dram_stream_time = max(0, bytes_loaded - last_cache.cap)
                              / dram.effective_bandwidth_bps

        For shapes far past LLC (overflow >> cap), the dram_stream_time
        dominates and the model collapses to the current binary
        behavior (memory_time = bytes_loaded / dram_eff_bw, with the
        cache_fill_time contributing only the first ~149 us of L3
        warmup -- negligible at multi-millisecond DRAM streaming).

        For shapes well inside LLC, the picker doesn't bind at DRAM
        in the first place, so this code path doesn't fire.

        For the boundary regime (1x-4x past LLC), the cache_fill_time
        is non-negligible relative to dram_stream_time, and OVERLAP
        gives a more accurate prediction than pure-DRAM-stream. The
        N=4M vector_add case lands at +20% (PASS) vs +140% (FAIL)
        under the previous binary model.
        """
        # Default behavior (current): pure binding-tier streaming
        # using the caller-provided ``binding_tier_bw`` (which is
        # already the per-op effective BW from _per_op_effective_bw).
        default_time = bytes_loaded / binding_tier_bw

        # Scope: vector_add only. The matmul / linear bytes_loaded
        # calculation encodes reuse via tile-streaming reload counts;
        # the OVERLAP physics (cache fills then concurrent DRAM
        # stream) doesn't apply when the data has structured reuse.
        if op_kind != "vector_add":
            return default_time

        # Predicate: only at the outermost-tier boundary (binding == DRAM).
        # For binding at L1 / L2 / L3 the picker has already routed
        # correctly and OVERLAP would be a model-confusion.
        if binding_tier.name != "DRAM":
            return default_time

        # Find the last cache tier (innermost-out, the tier just
        # before DRAM). If there isn't one (DRAM-only hierarchy),
        # OVERLAP is moot.
        cache_tiers = [t for t in hierarchy if t.name != "DRAM"]
        if not cache_tiers:
            return default_time
        last_cache = cache_tiers[-1]
        cache_cap = last_cache.total_capacity_bytes
        # Per-op cache BW (vector_add gets the per-tier default,
        # which is the V5-5 calibrated value for that tier).
        cache_bw = self._per_op_effective_bw(last_cache, op_kind)
        if cache_cap <= 0 or cache_bw <= 0:
            return default_time

        # OVERLAP physics: cache warmup time + DRAM streaming time,
        # parallelized.
        cache_fill_bytes = min(bytes_loaded, cache_cap)
        dram_stream_bytes = max(0, bytes_loaded - cache_cap)
        cache_fill_time = cache_fill_bytes / cache_bw
        dram_stream_time = dram_stream_bytes / binding_tier_bw
        return max(cache_fill_time, dram_stream_time)

    @staticmethod
    def _extract_matmul_shape(sg: SubgraphDescriptor) -> Optional[tuple]:
        """Pull (M, K, N) from a single-op MATMUL subgraph.

        Expects two 2-D input tensors of shapes (M, K) and (K, N) with
        a matching inner dim. Returns None for batched matmul, fused
        subgraphs, or anything that doesn't cleanly match -- the caller
        falls back to the scalar path."""
        if len(sg.input_tensors) != 2:
            return None
        a_shape = sg.input_tensors[0].shape
        b_shape = sg.input_tensors[1].shape
        if len(a_shape) != 2 or len(b_shape) != 2:
            return None
        M, Ka = a_shape
        Kb, N = b_shape
        if Ka != Kb:
            return None
        return (int(M), int(Ka), int(N))

    @staticmethod
    def _extract_linear_shape(sg: SubgraphDescriptor) -> Optional[tuple]:
        """Pull (B, IN, OUT) from a single-op LINEAR subgraph.

        Expects input shape (B, IN), weight shape (OUT, IN) (PyTorch
        nn.Linear convention). Returns None for higher-dim inputs or
        missing weight tensors."""
        if not sg.input_tensors or not sg.weight_tensors:
            return None
        in_shape = sg.input_tensors[0].shape
        w_shape = sg.weight_tensors[0].shape
        if len(in_shape) != 2 or len(w_shape) != 2:
            return None
        B, IN = in_shape
        OUT, IN_w = w_shape
        if IN != IN_w:
            return None
        return (int(B), int(IN), int(OUT))

    @staticmethod
    def _extract_vector_add_shape(sg: SubgraphDescriptor) -> Optional[tuple]:
        """Pull (N,) from a single-op subgraph that's specifically a
        vector add (c = a + b on N-element 1-D tensors).

        Stricter than just OperationType.ELEMENTWISE because the V5-3a
        vector_add reuse model assumes 3 buffers and zero reuse;
        applying it to ReLU (1 in -> 1 out) or fused
        elementwise-with-broadcast would produce wrong bytes_loaded.
        Predicate:
          * exactly 2 input tensors, 1 output tensor
          * all 3 are 1-D
          * all 3 share the same length and dtype

        Returns ``None`` for anything that doesn't match -- caller
        falls through to the scalar bw_efficiency_scale path."""
        if len(sg.input_tensors) != 2 or len(sg.output_tensors) != 1:
            return None
        a, b = sg.input_tensors
        c = sg.output_tensors[0]
        if len(a.shape) != 1 or len(b.shape) != 1 or len(c.shape) != 1:
            return None
        if a.shape != b.shape or a.shape != c.shape:
            return None
        if a.dtype != b.dtype or a.dtype != c.dtype:
            return None
        return (int(a.shape[0]),)

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

        if hw_type == "GPU":
            # Check for depthwise convolution (poor bandwidth efficiency)
            # CALIBRATION (Jetson Orin AGX, 50W, FP32):
            # - Depthwise Conv2D achieves very low effective bandwidth
            # - Due to scattered access patterns and poor cache utilization
            is_depthwise = (
                # Check operation_types list (fused subgraphs may have multiple ops)
                (
                    hasattr(sg, "operation_types")
                    and OperationType.CONV2D_DEPTHWISE in sg.operation_types
                )
                or
                # Check is_depthwise flag if set
                getattr(sg, "is_depthwise", False)
                or
                # Fallback to node name detection
                (
                    hasattr(sg, "node_names")
                    and any("dw" in n.lower() for n in sg.node_names)
                )
                or (
                    hasattr(sg, "node_names")
                    and any("depthwise" in n.lower() for n in sg.node_names)
                )
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
            total_bytes = (
                sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
            )

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

        elif hw_type == "CPU":
            # CPU bandwidth efficiency depends on working set size because
            # of the cache hierarchy. The previous flat 0.5 was reasonable
            # for small/medium ops but 1.8x pessimistic for large
            # GEMM-style streaming workloads (#74): medium B=1 with large
            # IN/OUT linear shapes predicted 100-260% over because
            # memory_time was inflated by the 0.5 derate when real
            # streaming GEMM achieves close to peak DRAM BW.
            #
            # CALIBRATED against V4 baseline measurements on i7-12700K
            # (validation/model_v4/results/baselines/i7_12700k_*.csv).
            # Median achieved BW per WS bucket:
            #   < 1M bytes      -> 0.16-0.20 (cold misses dominate)
            #   1M - 10M        -> 0.63 (cache-resident streaming begins)
            #   10M - 100M      -> 0.88 (sequential GEMM saturates BW)
            #
            # Curve choice: keep the 0.5 floor for sub-1M (where the old
            # constant happened to match the high end of the bucket and
            # was passing predictions), then ramp up for larger WS where
            # the empirical evidence is clear that 0.5 was too pessimistic.
            # This reduces #74 over-prediction without regressing the
            # smaller-WS shapes that were passing.
            total_bytes = (
                sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
            )
            if total_bytes <= 0:
                return 0.5

            if total_bytes < 1e6:
                # Sub-1M: keep the legacy 0.5 (matches old behavior; many
                # shapes in this regime were passing pre-#74)
                return 0.5

            log_bytes = math.log10(max(total_bytes, 1))

            if total_bytes < 10e6:
                # 1M - 10M: cache-resident streaming kicks in, ramp to 0.75
                t = (log_bytes - 6.0) / 1.0
                t = max(0.0, min(1.0, t))
                return 0.5 + 0.25 * t  # 0.5 -> 0.75
            # > 10M: streaming GEMM, plateau at 0.85 (slightly under
            # empirical median 0.88 to stay conservative). Some shapes
            # achieve > 1.0 effective via cache hits (working set
            # partially resident in LLC); modeling that requires
            # explicit cache-hierarchy modeling -- left for future work.
            return 0.85

        # Other hardware: default to moderate efficiency
        return 0.5

    def _dram_traffic_bytes(self, sg: SubgraphDescriptor) -> int:
        """Bytes that actually cross the DRAM boundary for a subgraph.

        For most hardware this is just ``input + output + weight`` -- the
        roofline assumes weights stream through DRAM each call.

        For weight-stationary accelerators (KPU), the architecture
        double-buffers weight prefetch with compute: while layer N runs on
        weights already resident in the tile fabric, layer N+1's weights are
        fetched from DRAM in parallel. As long as a subgraph's weight
        footprint fits in the aggregate on-chip capacity (sum of per-tile
        L1 + shared L2), the weight load fully overlaps the previous
        layer's compute and does NOT contribute to this subgraph's memory
        floor. Per-subgraph DRAM traffic is then just the activation in/out.

        When per-subgraph weights exceed the on-chip budget, the prefetch
        cannot fully hide weight loads -- model that as ``ceil(weight /
        weight_budget)`` outer passes that each load a weight slab.

        Without this hook (issue #51), transformer workloads get scored as
        bandwidth-bound on KPU even though the dataflow is specifically
        designed to keep weights resident.
        """
        base = sg.total_input_bytes + sg.total_output_bytes
        weight_bytes = sg.total_weight_bytes
        hw_type = self.resource_model.hardware_type.name

        if hw_type == "KPU" and weight_bytes > 0:
            # Aggregate on-chip = sum of per-tile L1 + shared L2.
            on_chip = (
                self.resource_model.compute_units
                * self.resource_model.l1_cache_per_unit
                + self.resource_model.l2_cache_total
            )
            # Reserve 20% of on-chip for the activation working set.
            weight_budget = max(1, int(on_chip * 0.8))
            if weight_bytes <= weight_budget:
                # Stationary: weight load overlaps previous layer's compute.
                return base
            # Weights exceed on-chip: outer-tiled. Each weight byte is still
            # loaded *once* total (the slabs are different weight tiles, not
            # the same tile reloaded); what gets repeated is the activation
            # stream, which has to cycle through each weight slab.
            outer_loads = max(1, math.ceil(weight_bytes / weight_budget))
            return base * outer_loads + weight_bytes

        return base + weight_bytes

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

        if hw_type == "TPU":
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

        elif hw_type == "KPU":
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
        if self.resource_model.hardware_type.name == "GPU":
            # Base kernel launch overhead
            base_overhead = 5e-6  # 5 microseconds

            # Check operation patterns for additional overhead
            fusion_pattern = (
                getattr(sg, "fusion_pattern", "")
                if hasattr(sg, "fusion_pattern")
                else ""
            )
            node_name = sg.node_name if hasattr(sg, "node_name") else ""

            # Get operation types list
            op_types_str = (
                "_".join(str(op).split(".")[-1] for op in sg.operation_types)
                if hasattr(sg, "operation_types")
                else ""
            )

            # Hardswish/Hardsigmoid activations have high overhead (~100us)
            # These are often separate kernels on GPU
            if (
                "HARDSWISH" in op_types_str.upper()
                or "hardswish" in fusion_pattern.lower()
            ):
                base_overhead = 100e-6  # 100 microseconds

            if (
                "hardsigmoid" in node_name.lower()
                or "scale_activation" in node_name.lower()
            ):
                # Hardsigmoid in SE blocks
                base_overhead = 100e-6

            # Squeeze-Excitation pattern: avgpool -> fc -> relu -> fc -> sigmoid -> mul
            # Very expensive due to multiple sequential tiny kernels
            if "ADAPTIVEAVGPOOL" in op_types_str and "CONV2D_POINTWISE" in op_types_str:
                # SE block internal path
                base_overhead = 200e-6  # 200 microseconds for SE FC path

            # UNKNOWN operations - need to distinguish MobileNet-V3 patterns from others
            # MobileNet-V3 UNKNOWN: hardsigmoid, mul (SE scaling) - slow activations
            # ViT UNKNOWN: gelu, softmax, add, reshape - typically faster
            # MaxViT UNKNOWN: many reshape/transpose for window attention - moderate overhead
            if "UNKNOWN" in op_types_str:
                # Check node name for specific slow patterns
                node_lower = node_name.lower()
                if (
                    "hardsigmoid" in node_lower
                    or "scale_activation" in node_lower
                    or ("mul" in node_lower and sg.total_flops < 100000)
                ):
                    # MobileNet-V3 style slow activations/SE scaling
                    base_overhead = 100e-6  # 100 microseconds
                elif (
                    "swap" in node_lower
                    or "partition" in node_lower
                    or "window" in node_lower
                    or "grid" in node_lower
                ):
                    # MaxViT window/grid attention partitioning operations
                    # These involve tensor reshaping/transposing which has memory overhead
                    base_overhead = 50e-6  # 50 microseconds
                elif "softmax" in node_lower:
                    # Softmax has moderate overhead
                    base_overhead = 30e-6  # 30 microseconds
                elif "floordiv" in node_lower or "floor_divide" in node_lower:
                    # Integer division ops (MaxViT uses for indexing)
                    base_overhead = 25e-6  # 25 microseconds
                elif "getitem" in node_lower or "getattr" in node_lower:
                    # Tensor indexing operations (MaxViT has hundreds)
                    # Each one triggers memory access pattern changes
                    base_overhead = 25e-6  # 25 microseconds
                elif "chunk" in node_lower or "split" in node_lower:
                    # Tensor splitting operations
                    base_overhead = 30e-6  # 30 microseconds
                else:
                    # Other UNKNOWN ops (GELU, add, etc.)
                    base_overhead = 15e-6  # 15 microseconds

            # Pointwise convolutions (1x1) with very few FLOPs
            # These are common in MobileNet-V3 and are overhead-dominated
            if "POINTWISE" in op_types_str:
                if sg.total_flops < 1e6:
                    # Very tiny pointwise convs (MobileNet-V3-Small style)
                    base_overhead = max(base_overhead, 150e-6)
                elif sg.total_flops < 5e6:
                    # Small pointwise convs
                    base_overhead = max(base_overhead, 100e-6)

            # Depthwise convolutions also have high per-op overhead
            if "DEPTHWISE" in op_types_str:
                if sg.total_flops < 2e6:
                    # Tiny depthwise (MobileNet-V3-Small)
                    base_overhead = max(base_overhead, 150e-6)
                else:
                    base_overhead = max(base_overhead, 75e-6)

            return base_overhead

        # TPU systolic array setup overhead
        if self.resource_model.hardware_type.name == "TPU":
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
        ratio: float,
    ) -> str:
        """Generate human-readable explanation of bottleneck"""

        op_name = sg.node_name

        if bottleneck == BottleneckType.BANDWIDTH_BOUND:
            return (
                f"{op_name}: Memory-bound (bandwidth limit) - "
                f"memory time {memory_time*1e6:.1f}μs vs "
                f"compute time {compute_time*1e6:.1f}μs ({ratio:.1f}× slower)"
            )
        elif bottleneck == BottleneckType.COMPUTE_BOUND:
            return (
                f"{op_name}: Compute-bound (FLOPs limit) - "
                f"compute time {compute_time*1e6:.1f}μs vs "
                f"memory time {memory_time*1e6:.1f}μs ({ratio:.1f}× slower)"
            )
        else:
            return (
                f"{op_name}: Balanced - "
                f"compute time {compute_time*1e6:.1f}μs, "
                f"memory time {memory_time*1e6:.1f}μs"
            )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_calibrated_analyzer(
    resource_model: HardwareResourceModel,
    hardware_id: str,
    precision: str = "fp32",
    registry_path: Optional[str] = None,
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
        calibrated_bandwidth=bandwidth_gbps,
    )


def get_roofline_params_for_hardware(
    hardware_id: str, precision: str = "fp32", use_calibrated: bool = True
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
