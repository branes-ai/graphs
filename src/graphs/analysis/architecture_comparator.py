"""
Architecture Comparator - Multi-Architecture Energy/Performance Comparison

This module provides hierarchical comparison of DNN models across different
hardware architectures, with drill-down capabilities for discovery workflows.

Levels:
- Level 0 (summary): High-level metrics comparison + recommendations
- Level 1 (detailed): Per-architecture energy/latency breakdowns
- Level 2 (subgraph): Per-subgraph comparison across architectures

Educational focus: Not just "what" but "WHY" one architecture is better.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import json
import csv
import io
import math

from graphs.hardware.resource_model import Precision
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, UnifiedAnalysisResult

if TYPE_CHECKING:
    from graphs.hardware.architectural_energy import ArchitecturalEnergyBreakdown


class ComparisonLevel(Enum):
    """Hierarchical comparison levels"""
    SUMMARY = "summary"      # Level 0: Executive summary
    DETAILED = "detailed"    # Level 1: Per-architecture breakdown
    SUBGRAPH = "subgraph"    # Level 2: Per-subgraph comparison


@dataclass
class ArchitectureMetrics:
    """Metrics for a single architecture"""
    name: str
    total_energy_j: float
    total_latency_s: float
    peak_memory_bytes: int
    utilization: float
    energy_per_inference_j: float
    throughput_inferences_per_sec: float

    # Energy breakdown
    compute_energy_j: float
    memory_energy_j: float
    architectural_overhead_j: float

    # Bottleneck analysis
    compute_bound_subgraphs: int
    memory_bound_subgraphs: int
    total_subgraphs: int

    # Full result for detailed analysis
    full_result: UnifiedAnalysisResult

    # Architectural energy breakdown (optional)
    architectural_breakdown: Optional['ArchitecturalEnergyBreakdown'] = None

    # NEW: Hardware utilization metrics
    attained_tops: float = 0.0  # Actual performance achieved (TOPS)
    peak_tops: float = 0.0  # Hardware peak performance (TOPS)
    compute_utilization_pct: float = 0.0  # Compute utilization (%)
    memory_bandwidth_utilization_pct: float = 0.0  # Memory BW utilization (%)
    arithmetic_intensity: float = 0.0  # FLOPs/byte
    compute_units_allocated: int = 0  # Actual compute units used (SMs, MXUs, cores)
    compute_units_total: int = 0  # Total available compute units
    energy_efficiency_tops_per_watt: float = 0.0  # Performance per watt

    # NEW: EDP metrics (Energy-Delay Product)
    edp: float = 0.0  # Energy-Delay Product (J·s)
    edp_normalized: float = 0.0  # Normalized to baseline

    # Energy breakdown for EDP analysis
    compute_edp: float = 0.0  # Compute energy × latency
    memory_edp: float = 0.0   # Memory energy × latency
    architectural_edp: float = 0.0  # Architectural overhead × latency


@dataclass
class ComparisonSummary:
    """Summary of multi-architecture comparison"""
    architectures: Dict[str, ArchitectureMetrics]

    # Winners
    energy_winner: str
    latency_winner: str
    throughput_winner: str
    memory_winner: str
    balance_winner: str  # Best energy-latency product
    edp_winner: str  # Best EDP (Energy-Delay Product)

    # Ratios (all relative to baseline, which is typically GPU or CPU)
    baseline: str
    energy_ratios: Dict[str, float]
    latency_ratios: Dict[str, float]
    edp_ratios: Dict[str, float]  # EDP ratios vs baseline

    # Insights
    insights: List[str] = field(default_factory=list)


@dataclass
class SubgraphEDPDescriptor:
    """
    Per-subgraph EDP breakdown for a single architecture.

    Combines energy and latency from existing analyzers to compute
    subgraph-level EDP, revealing which subgraphs dominate total EDP.
    """

    # Identity
    subgraph_id: str
    subgraph_name: str
    fusion_pattern: str  # e.g., "Conv_BN_ReLU", "Linear_Bias_ReLU"
    num_operators: int   # Number of operators fused (1 if not fused)

    # Energy components (Joules)
    energy_j: float
    compute_energy_j: float
    memory_energy_j: float
    static_energy_j: float

    # Latency components (seconds)
    latency_s: float
    compute_time_s: float
    memory_time_s: float

    # EDP (Energy-Delay Product, J·s)
    edp: float

    # Component EDPs
    compute_edp: float   # compute_energy × latency
    memory_edp: float    # memory_energy × latency
    static_edp: float    # static_energy × latency

    # Contribution to total
    edp_fraction: float = 0.0  # Percentage of total model EDP

    # Bottleneck analysis
    bottleneck: str = "balanced"  # "compute_bound", "memory_bound", "balanced"
    arithmetic_intensity: float = 0.0  # FLOPs/byte

    def __str__(self) -> str:
        """Short summary"""
        return (f"SubgraphEDP({self.subgraph_name}: "
                f"{self.edp * 1e9:.2f} nJ·s, {self.edp_fraction * 100:.1f}% of total)")


@dataclass
class OperatorEDPDescriptor:
    """
    Per-operator EDP breakdown within a subgraph (Phase 2).

    Decomposes subgraph EDP to individual operators, applying architectural
    modifiers to reveal fusion benefits and architecture-specific overhead.

    Example:
        Subgraph: fc1 (Linear_Bias_ReLU, 3 ops)
          Linear:  95.0%  (1.0× modifier - dominates)
          Bias:     2.5%  (0.05× modifier - hidden in dataflow on KPU)
          ReLU:     2.5%  (0.05× modifier - hidden in dataflow on KPU)
    """

    # Identity
    operator_id: str
    operator_type: str  # "Linear", "Conv2d", "ReLU", "Bias", "BatchNorm2d", etc.
    subgraph_id: str    # Parent subgraph
    subgraph_name: str

    # Base EDP (hardware-agnostic, FLOP-proportional allocation)
    base_edp: float  # J·s

    # Architectural EDP (with architecture-specific modifiers)
    architectural_edp: float  # J·s
    architectural_modifier: float  # Multiplier applied (e.g., 0.05 for hidden ReLU, 3.0 for separate kernel)

    # Contribution
    edp_fraction_of_subgraph: float  # Percentage within parent subgraph
    edp_fraction_of_model: float     # Percentage of total model EDP

    # Fusion metadata
    is_fused: bool  # True if this operator is fused with others in subgraph
    fusion_benefit: Optional[float] = None  # EDP savings from fusion (if applicable)

    # Operator characteristics
    flops: float = 0.0  # Floating-point operations
    memory_bytes: float = 0.0  # Memory footprint
    arithmetic_intensity: float = 0.0  # FLOPs/byte

    def __str__(self) -> str:
        """Short summary"""
        modifier_str = f"{self.architectural_modifier:.2f}×"
        return (f"OperatorEDP({self.operator_type}: "
                f"{self.architectural_edp * 1e9:.2f} nJ·s, "
                f"{self.edp_fraction_of_subgraph * 100:.1f}% of subgraph, "
                f"modifier={modifier_str})")


class ArchitectureComparator:
    """
    Multi-architecture comparison with hierarchical drill-down.

    Usage:
        comparator = ArchitectureComparator(
            model_name="resnet18",
            architectures={
                'CPU': cpu_mapper,
                'GPU': gpu_mapper,
                'TPU': tpu_mapper,
                'KPU': kpu_mapper,
            },
            batch_size=1,
            precision=Precision.FP32
        )

        comparator.analyze_all()
        print(comparator.generate_summary())
        print(comparator.generate_detailed('GPU'))
        print(comparator.explain_difference('GPU', 'TPU', 'energy'))
    """

    def __init__(
        self,
        model_name: str,
        architectures: Dict[str, 'HardwareMapper'],
        batch_size: int = 1,
        precision: Precision = Precision.FP32,
        model: Optional[Any] = None,
        input_tensor: Optional[Any] = None
    ):
        """
        Initialize comparator.

        Args:
            model_name: Model to analyze (e.g., 'resnet18')
            architectures: Dict mapping architecture name to HardwareMapper
            batch_size: Batch size for analysis
            precision: Numerical precision
            model: Optional pre-loaded model instance (for custom models)
            input_tensor: Optional input tensor (for custom models)
        """
        self.model_name = model_name
        self.architectures = architectures
        self.batch_size = batch_size
        self.precision = precision
        self.custom_model = model
        self.custom_input_tensor = input_tensor

        # Results storage
        self.results: Dict[str, UnifiedAnalysisResult] = {}
        self.metrics: Dict[str, ArchitectureMetrics] = {}
        self.summary: Optional[ComparisonSummary] = None

    def analyze_all(self):
        """Run analysis on all architectures"""
        import torch
        import torchvision.models as models

        print(f"Analyzing {self.model_name} on {len(self.architectures)} architectures...")
        print()

        # Use custom model if provided, otherwise load from torchvision
        if self.custom_model is not None and self.custom_input_tensor is not None:
            model = self.custom_model
            input_tensor = self.custom_input_tensor
        else:
            # Load model from torchvision
            try:
                model = getattr(models, self.model_name)(pretrained=False)
                model.eval()
            except AttributeError:
                raise ValueError(f"Model '{self.model_name}' not found in torchvision.models")

            # Create input tensor with correct batch size
            # Assume standard ImageNet input shape
            input_shape = (self.batch_size, 3, 224, 224)
            input_tensor = torch.randn(input_shape)

        for arch_name, mapper in self.architectures.items():
            print(f"  Analyzing {arch_name}...")
            analyzer = UnifiedAnalyzer(verbose=False)

            # Disable operator EDP to avoid circular dependency
            # (ArchitectureComparator provides operator EDP directly via get_operator_edp_breakdown)
            from graphs.analysis.unified_analyzer import AnalysisConfig
            config = AnalysisConfig(run_operator_edp=False)

            result = analyzer.analyze_model_with_custom_hardware(
                model=model,
                input_tensor=input_tensor,
                model_name=self.model_name,
                hardware_mapper=mapper,
                precision=self.precision,
                config=config
            )
            self.results[arch_name] = result

            # Extract metrics
            self.metrics[arch_name] = self._extract_metrics(arch_name, result)

        print()
        print("Analysis complete!")
        print()

        # Generate summary
        self.summary = self._generate_summary()

    def _extract_metrics(
        self,
        name: str,
        result: UnifiedAnalysisResult
    ) -> ArchitectureMetrics:
        """Extract key metrics from analysis result"""

        # Energy breakdown
        compute_energy = 0.0
        memory_energy = 0.0

        if result.energy_report:
            compute_energy = result.energy_report.compute_energy_j
            memory_energy = result.energy_report.memory_energy_j

        # Architectural energy breakdown (if available)
        architectural_overhead = 0.0
        architectural_breakdown = None

        # Calculate architectural energy breakdown if mapper has energy model
        mapper = self.architectures.get(name)
        if mapper and mapper.resource_model.architecture_energy_model:
            # Aggregate ops and bytes across all subgraphs
            total_ops = 0
            total_bytes = 0

            if result.partition_report:
                for sg in result.partition_report.subgraphs:
                    total_ops += sg.flops
                    total_bytes += sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes

            # Compute architectural breakdown
            try:
                from graphs.hardware.architectural_energy import ArchitecturalEnergyBreakdown

                # Create execution context (architecture-specific)
                execution_context = self._build_execution_context(name, result)

                # Calculate baseline energy
                compute_baseline = compute_energy
                memory_baseline = memory_energy

                # Check if architecture_energy_model has the compute_architectural_energy method
                # (OLD API - some models may not support it)
                if hasattr(mapper.resource_model.architecture_energy_model, 'compute_architectural_energy'):
                    architectural_breakdown = mapper.resource_model.architecture_energy_model.compute_architectural_energy(
                        ops=total_ops,
                        bytes_transferred=total_bytes,
                        compute_energy_baseline=compute_baseline,
                        memory_energy_baseline=memory_baseline,
                        execution_context=execution_context
                    )

                    architectural_overhead = architectural_breakdown.total_overhead
                else:
                    # Architecture energy model doesn't support OLD API
                    # Skip architectural breakdown (main energy calculation still works)
                    architectural_overhead = 0.0

            except Exception as e:
                # Silently skip architectural breakdown on error
                # (This is not critical - main energy calculation still works)
                architectural_overhead = 0.0

        # Bottleneck analysis
        compute_bound = 0
        memory_bound = 0
        total_subgraphs = 0

        if result.roofline_report:
            # Count bottleneck types from latency descriptors
            for lat_desc in result.roofline_report.latencies:
                total_subgraphs += 1
                if lat_desc.bottleneck.value == "compute_bound":
                    compute_bound += 1
                elif lat_desc.bottleneck.value in ["memory_bound", "bandwidth_bound"]:
                    memory_bound += 1

        # Calculate average utilization from roofline report
        utilization = 0.0
        if result.roofline_report and hasattr(result.roofline_report, 'average_utilization'):
            utilization = result.roofline_report.average_utilization

        # NEW: Calculate hardware utilization metrics
        mapper = self.architectures.get(name)
        hardware = mapper.resource_model if mapper else result.hardware

        # Attained performance (TOPS) = total_ops / latency
        total_ops = 0
        total_bytes = 0
        if result.partition_report:
            for sg in result.partition_report.subgraphs:
                total_ops += sg.flops
                total_bytes += sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes

        latency_s = result.total_latency_ms / 1000.0
        attained_tops = (total_ops / latency_s / 1e12) if latency_s > 0 else 0.0

        # Peak performance (TOPS)
        peak_tops = hardware.get_peak_ops(self.precision) / 1e12

        # Compute utilization (%)
        compute_util_pct = (attained_tops / peak_tops * 100.0) if peak_tops > 0 else 0.0

        # Memory bandwidth utilization (%)
        attained_bandwidth = (total_bytes / latency_s) if latency_s > 0 else 0.0
        peak_bandwidth = hardware.peak_bandwidth
        mem_bw_util_pct = (attained_bandwidth / peak_bandwidth * 100.0) if peak_bandwidth > 0 else 0.0

        # Arithmetic intensity (FLOPs/byte)
        arith_intensity = (total_ops / total_bytes) if total_bytes > 0 else 0.0

        # Compute units allocated/total
        compute_units_allocated = 0
        compute_units_total = hardware.compute_units

        if result.roofline_report and hasattr(result.roofline_report, 'average_compute_units_used'):
            compute_units_allocated = int(result.roofline_report.average_compute_units_used)
        elif result.partition_report and hasattr(result.partition_report, 'average_compute_units_used'):
            compute_units_allocated = int(result.partition_report.average_compute_units_used)
        else:
            # Estimate from compute utilization percentage (calculated above)
            # Use ceil() to ensure at least 1 unit is allocated if there's any activity
            utilization_fraction = compute_util_pct / 100.0
            if utilization_fraction > 0:
                compute_units_allocated = math.ceil(utilization_fraction * compute_units_total)
            else:
                compute_units_allocated = 0

        # Energy efficiency (TOPS/W)
        average_power_w = (result.energy_per_inference_mj / 1000.0) / latency_s if latency_s > 0 else 0.0
        energy_efficiency = (attained_tops / average_power_w) if average_power_w > 0 else 0.0

        # NEW: Calculate EDP (Energy-Delay Product)
        energy_j = result.energy_per_inference_mj / 1000.0  # Convert mJ to J
        edp = energy_j * latency_s  # J·s

        # Component EDPs
        compute_edp = compute_energy * latency_s
        memory_edp = memory_energy * latency_s
        architectural_edp = architectural_overhead * latency_s

        return ArchitectureMetrics(
            name=name,
            total_energy_j=result.energy_per_inference_mj / 1000.0,  # Convert mJ to J
            total_latency_s=result.total_latency_ms / 1000.0,  # Convert ms to s
            peak_memory_bytes=int(result.peak_memory_mb * 1e6),  # Convert MB to bytes
            utilization=utilization,
            energy_per_inference_j=result.energy_per_inference_mj / 1000.0,  # Convert mJ to J
            throughput_inferences_per_sec=result.throughput_fps,
            compute_energy_j=compute_energy,
            memory_energy_j=memory_energy,
            architectural_overhead_j=architectural_overhead,
            compute_bound_subgraphs=compute_bound,
            memory_bound_subgraphs=memory_bound,
            total_subgraphs=total_subgraphs,
            full_result=result,
            architectural_breakdown=architectural_breakdown,
            # NEW: Hardware utilization metrics
            attained_tops=attained_tops,
            peak_tops=peak_tops,
            compute_utilization_pct=compute_util_pct,
            memory_bandwidth_utilization_pct=mem_bw_util_pct,
            arithmetic_intensity=arith_intensity,
            compute_units_allocated=compute_units_allocated,
            compute_units_total=compute_units_total,
            energy_efficiency_tops_per_watt=energy_efficiency,
            # NEW: EDP metrics
            edp=edp,
            edp_normalized=0.0,  # Will be set in _generate_summary()
            compute_edp=compute_edp,
            memory_edp=memory_edp,
            architectural_edp=architectural_edp,
        )

    def _build_execution_context(
        self,
        arch_name: str,
        result: UnifiedAnalysisResult
    ) -> Dict:
        """
        Build architecture-specific execution context for energy modeling.

        This provides additional information needed by architectural energy models,
        such as thread counts for GPU, cache line size for CPU, etc.
        """
        context = {}

        # GPU-specific context
        if 'GPU' in arch_name.upper():
            # Estimate concurrent threads from parallelism analysis
            if result.partition_report:
                max_threads = 0
                for sg in result.partition_report.subgraphs:
                    if sg.parallelism and sg.parallelism.total_threads:
                        max_threads = max(max_threads, sg.parallelism.total_threads)

                if max_threads > 0:
                    context['concurrent_threads'] = max_threads
                    context['warp_size'] = 32  # NVIDIA warp size
                    context['cache_line_size'] = 128  # GPU cache line

        # CPU-specific context
        elif 'CPU' in arch_name.upper() or 'XEON' in arch_name.upper():
            context['cache_line_size'] = 64  # x86 cache line

        # KPU-specific context
        elif 'KPU' in arch_name.upper():
            # Number of schedule changes (could be derived from partition count)
            if result.partition_report:
                context['schedule_changes'] = len(result.partition_report.subgraphs) // 10

        # Add batch size to all contexts
        context['batch_size'] = self.batch_size

        return context

    def _generate_summary(self) -> ComparisonSummary:
        """Generate comparison summary with winners and insights"""

        if not self.metrics:
            raise ValueError("No metrics available. Run analyze_all() first.")

        # Find winners
        energy_winner = min(self.metrics.items(), key=lambda x: x[1].total_energy_j)[0]
        latency_winner = min(self.metrics.items(), key=lambda x: x[1].total_latency_s)[0]
        throughput_winner = max(self.metrics.items(), key=lambda x: x[1].throughput_inferences_per_sec)[0]
        memory_winner = min(self.metrics.items(), key=lambda x: x[1].peak_memory_bytes)[0]

        # Balance winner: minimize energy × latency product
        balance_winner = min(
            self.metrics.items(),
            key=lambda x: x[1].total_energy_j * x[1].total_latency_s
        )[0]

        # NEW: EDP winner: minimize Energy-Delay Product
        edp_winner = min(self.metrics.items(), key=lambda x: x[1].edp)[0]

        # Choose baseline (typically GPU if present, else first architecture)
        baseline = 'GPU' if 'GPU' in self.metrics else list(self.metrics.keys())[0]
        baseline_energy = self.metrics[baseline].total_energy_j
        baseline_latency = self.metrics[baseline].total_latency_s
        baseline_edp = self.metrics[baseline].edp

        # Calculate ratios
        energy_ratios = {
            name: metrics.total_energy_j / baseline_energy
            for name, metrics in self.metrics.items()
        }

        latency_ratios = {
            name: metrics.total_latency_s / baseline_latency
            for name, metrics in self.metrics.items()
        }

        # NEW: Calculate EDP ratios
        edp_ratios = {
            name: metrics.edp / baseline_edp
            for name, metrics in self.metrics.items()
        }

        # Update normalized EDP values in metrics
        for name, metrics in self.metrics.items():
            metrics.edp_normalized = edp_ratios[name]

        # Generate insights
        insights = self._generate_insights(
            energy_winner, latency_winner, balance_winner,
            baseline, energy_ratios, latency_ratios
        )

        return ComparisonSummary(
            architectures=self.metrics,
            energy_winner=energy_winner,
            latency_winner=latency_winner,
            throughput_winner=throughput_winner,
            memory_winner=memory_winner,
            balance_winner=balance_winner,
            edp_winner=edp_winner,
            baseline=baseline,
            energy_ratios=energy_ratios,
            latency_ratios=latency_ratios,
            edp_ratios=edp_ratios,
            insights=insights
        )

    def _generate_insights(
        self,
        energy_winner: str,
        latency_winner: str,
        balance_winner: str,
        baseline: str,
        energy_ratios: Dict[str, float],
        latency_ratios: Dict[str, float]
    ) -> List[str]:
        """Generate actionable insights from comparison"""

        insights = []

        # Energy insight
        if energy_winner != baseline:
            ratio = 1.0 / energy_ratios[energy_winner]
            insights.append(
                f"{energy_winner} is {ratio:.1f}× more energy efficient than {baseline}"
            )

        # Latency insight
        if latency_winner != baseline:
            ratio = 1.0 / latency_ratios[latency_winner]
            insights.append(
                f"{latency_winner} is {ratio:.1f}× faster than {baseline}"
            )

        # Trade-off insight
        if energy_winner != latency_winner:
            insights.append(
                f"Trade-off: {energy_winner} wins energy, {latency_winner} wins latency"
            )

        # Balance recommendation
        if balance_winner not in [energy_winner, latency_winner]:
            insights.append(
                f"{balance_winner} offers the best energy-latency balance"
            )

        # Batch size recommendation
        if baseline == 'GPU' and self.batch_size < 8:
            insights.append(
                f"GPU efficiency improves at larger batch sizes (current: {self.batch_size})"
            )

        return insights

    def generate_summary(self) -> str:
        """Generate Level 0: Executive summary"""

        if not self.summary:
            raise ValueError("No summary available. Run analyze_all() first.")

        lines = []
        lines.append("=" * 80)
        lines.append(f"Architecture Comparison: {self.model_name}")
        lines.append(f"Batch Size: {self.batch_size}, Precision: {self.precision.value}")
        lines.append("=" * 80)
        lines.append("")

        # Recommendations
        lines.append("Recommendations:")
        lines.append(f"  Best for Energy:      {self.summary.energy_winner}")
        lines.append(f"  Best for Latency:     {self.summary.latency_winner}")
        lines.append(f"  Best for Throughput:  {self.summary.throughput_winner}")
        lines.append(f"  Best EDP (E×D):       {self.summary.edp_winner}")
        lines.append(f"  Best Balance:         {self.summary.balance_winner}")
        lines.append("")

        # Comparison table (expanded with utilization metrics)
        lines.append(f"{'Architecture':<12} {'Energy':<11} {'Latency':<11} {'Throughput':<11} {'Attained':<10} {'Compute':<9} {'Mem BW':<8} {'Units':<10} {'vs ' + self.summary.baseline:<12}")
        lines.append(f"{'':12} {'':11} {'':11} {'(FPS)':11} {'(TOPS)':10} {'Util%':9} {'Util%':8} {'Alloc':10} {'':12}")
        lines.append("-" * 120)

        for name in sorted(self.metrics.keys()):
            metrics = self.metrics[name]
            energy_str = self._format_energy(metrics.total_energy_j)
            latency_str = self._format_time(metrics.total_latency_s)
            throughput_str = f"{metrics.throughput_inferences_per_sec:.0f}"

            # NEW: Format new metrics
            attained_str = f"{metrics.attained_tops:.2f}"
            compute_util_str = f"{metrics.compute_utilization_pct:.1f}%"
            mem_bw_util_str = f"{metrics.memory_bandwidth_utilization_pct:.1f}%"
            units_str = f"{metrics.compute_units_allocated}/{metrics.compute_units_total}"

            # Ratio vs baseline
            if name == self.summary.baseline:
                ratio_str = "baseline"
            else:
                energy_ratio = self.summary.energy_ratios[name]
                ratio_str = f"{energy_ratio:.2f}× energy"

            # Highlight winners
            style = ""
            if name == self.summary.energy_winner:
                style = " ⭐ (energy)"
            elif name == self.summary.latency_winner:
                style = " ⭐ (speed)"

            lines.append(
                f"{name:<12} {energy_str:<11} {latency_str:<11} {throughput_str:<11} "
                f"{attained_str:<10} {compute_util_str:<9} {mem_bw_util_str:<8} {units_str:<10} {ratio_str:<12}{style}"
            )

        lines.append("")

        # NEW: EDP Comparison Table
        lines.append("Energy-Delay Product (EDP) Comparison:")
        lines.append(f"{'Architecture':<12} {'EDP (nJ·s)':<15} {'vs ' + self.summary.baseline:<12} {'Breakdown':<30}")
        lines.append("-" * 80)

        for name in sorted(self.metrics.keys()):
            metrics = self.metrics[name]
            edp_nj_s = metrics.edp * 1e9  # Convert J·s to nJ·s
            edp_ratio = self.summary.edp_ratios[name]

            # Breakdown percentages
            total_component_edp = metrics.compute_edp + metrics.memory_edp + metrics.architectural_edp
            if total_component_edp > 0:
                compute_pct = metrics.compute_edp / total_component_edp * 100
                memory_pct = metrics.memory_edp / total_component_edp * 100
                arch_pct = metrics.architectural_edp / total_component_edp * 100
                breakdown = f"C:{compute_pct:.0f}% M:{memory_pct:.0f}% A:{arch_pct:.0f}%"
            else:
                breakdown = "—"

            # Highlight EDP winner
            style = " ⭐ (EDP)" if name == self.summary.edp_winner else ""

            # Format ratio
            if name == self.summary.baseline:
                ratio_str = "baseline"
            else:
                ratio_str = f"{edp_ratio:.2f}×"

            lines.append(f"{name:<12} {edp_nj_s:<15.2f} {ratio_str:<12} {breakdown:<30}{style}")

        lines.append("")

        # Architectural energy comparison (if available)
        has_arch_breakdown = any(m.architectural_breakdown is not None for m in self.metrics.values())
        if has_arch_breakdown:
            lines.append("Architectural Energy Breakdown:")
            lines.append(f"{'Architecture':<15} {'Compute OH':<15} {'Memory OH':<15} {'Control OH':<15} {'Total OH':<15}")
            lines.append("-" * 80)

            for name in sorted(self.metrics.keys()):
                metrics = self.metrics[name]
                if metrics.architectural_breakdown:
                    b = metrics.architectural_breakdown
                    compute_str = self._format_energy(b.compute_overhead) if abs(b.compute_overhead) > 1e-15 else "—"
                    memory_str = self._format_energy(b.memory_overhead) if abs(b.memory_overhead) > 1e-15 else "—"
                    control_str = self._format_energy(b.control_overhead) if abs(b.control_overhead) > 1e-15 else "—"
                    total_str = self._format_energy(b.total_overhead)

                    lines.append(f"{name:<15} {compute_str:<15} {memory_str:<15} {control_str:<15} {total_str:<15}")
                else:
                    lines.append(f"{name:<15} {'Not configured':<60}")

            lines.append("")
            lines.append("Note: Negative values indicate energy savings vs. baseline")
            lines.append("")

        # Key insights
        lines.append("Key Insights:")
        for insight in self.summary.insights:
            lines.append(f"  • {insight}")

        lines.append("")
        lines.append("→ Use --level detailed to see per-architecture breakdowns")
        lines.append("→ Use --level subgraph to see per-layer analysis")
        lines.append("→ Use generate_subgraph_edp_report(<arch>) to see per-subgraph EDP breakdown")
        lines.append("→ Use --explain <arch1> <arch2> to understand differences")
        lines.append("")

        return "\n".join(lines)

    def generate_detailed(self, arch_name: str) -> str:
        """Generate Level 1: Detailed breakdown for one architecture"""

        if arch_name not in self.metrics:
            raise ValueError(f"Architecture '{arch_name}' not found")

        metrics = self.metrics[arch_name]
        result = metrics.full_result

        lines = []
        lines.append("=" * 80)
        lines.append(f"Detailed Analysis: {arch_name}")
        lines.append("=" * 80)
        lines.append("")

        # Energy breakdown
        lines.append("Energy Breakdown:")
        lines.append(f"  Total Energy:          {self._format_energy(metrics.total_energy_j)}")

        if metrics.compute_energy_j > 0 or metrics.memory_energy_j > 0:
            total = metrics.compute_energy_j + metrics.memory_energy_j
            compute_pct = (metrics.compute_energy_j / total * 100) if total > 0 else 0
            memory_pct = (metrics.memory_energy_j / total * 100) if total > 0 else 0

            lines.append(f"    ├─ Compute:          {self._format_energy(metrics.compute_energy_j)} ({compute_pct:.1f}%)")
            lines.append(f"    └─ Memory:           {self._format_energy(metrics.memory_energy_j)} ({memory_pct:.1f}%)")

        if metrics.architectural_overhead_j != 0:
            lines.append(f"  Architectural Overhead: {self._format_energy(metrics.architectural_overhead_j)}")

        lines.append("")

        # Architectural energy breakdown (if available)
        if metrics.architectural_breakdown:
            lines.append("Architectural Energy Events:")
            breakdown = metrics.architectural_breakdown

            # Show component breakdowns
            if abs(breakdown.compute_overhead) > 1e-15:
                sign = "+" if breakdown.compute_overhead > 0 else ""
                lines.append(f"  Compute Overhead:      {sign}{self._format_energy(breakdown.compute_overhead)}")

            if abs(breakdown.memory_overhead) > 1e-15:
                sign = "+" if breakdown.memory_overhead > 0 else ""
                lines.append(f"  Memory Overhead:       {sign}{self._format_energy(breakdown.memory_overhead)}")

            if abs(breakdown.control_overhead) > 1e-15:
                sign = "+" if breakdown.control_overhead > 0 else ""
                lines.append(f"  Control Overhead:      {sign}{self._format_energy(breakdown.control_overhead)}")

            # Show extra details if available
            if breakdown.extra_details:
                lines.append("")
                lines.append("  Detailed Events:")
                for event_name, event_energy in breakdown.extra_details.items():
                    sign = "+" if event_energy > 0 else ""
                    lines.append(f"    {event_name:<25} {sign}{self._format_energy(event_energy)}")

            # Show explanation
            if breakdown.explanation:
                lines.append("")
                lines.append("  Explanation:")
                for line in breakdown.explanation.split('\n'):
                    if line.strip():
                        lines.append(f"    {line}")

            lines.append("")

        # Performance metrics
        lines.append("Performance:")
        lines.append(f"  Latency:               {self._format_time(metrics.total_latency_s)}")
        lines.append(f"  Throughput:            {metrics.throughput_inferences_per_sec:.1f} inferences/sec")
        lines.append(f"  Utilization:           {metrics.utilization*100:.1f}%")
        lines.append("")

        # Memory metrics
        lines.append("Memory:")
        lines.append(f"  Peak Memory:           {self._format_bytes(metrics.peak_memory_bytes)}")

        if result.memory_report:
            lines.append(f"  Activation Memory:     {self._format_bytes(result.memory_report.activation_memory_bytes)}")
            lines.append(f"  Weight Memory:         {self._format_bytes(result.memory_report.weight_memory_bytes)}")

        lines.append("")

        # Bottleneck analysis
        if metrics.total_subgraphs > 0:
            compute_pct = metrics.compute_bound_subgraphs / metrics.total_subgraphs * 100
            memory_pct = metrics.memory_bound_subgraphs / metrics.total_subgraphs * 100

            lines.append("Bottleneck Analysis:")
            lines.append(f"  Compute-bound:         {metrics.compute_bound_subgraphs} subgraphs ({compute_pct:.0f}%)")
            lines.append(f"  Memory-bound:          {metrics.memory_bound_subgraphs} subgraphs ({memory_pct:.0f}%)")
            lines.append(f"  Total subgraphs:       {metrics.total_subgraphs}")
            lines.append("")

        # Roofline summary
        if result.roofline_report:
            lines.append("Roofline Model:")
            if hasattr(result.roofline_report, 'peak_utilization'):
                lines.append(f"  Peak Utilization:      {result.roofline_report.peak_utilization*100:.1f}%")
            if hasattr(result.roofline_report, 'average_utilization'):
                lines.append(f"  Average Utilization:   {result.roofline_report.average_utilization*100:.1f}%")
            lines.append("")

        return "\n".join(lines)

    def get_subgraph_edp_breakdown(self, arch_name: str) -> List[SubgraphEDPDescriptor]:
        """
        Get per-subgraph EDP breakdown for one architecture.

        Combines energy descriptors and latency descriptors from existing
        analyzers to compute EDP at subgraph granularity.

        Args:
            arch_name: Architecture name (e.g., "GPU", "TPU", "KPU")

        Returns:
            List of SubgraphEDPDescriptor sorted by EDP (descending)

        Raises:
            ValueError: If architecture not found or analysis not run
        """
        if arch_name not in self.results:
            raise ValueError(f"Architecture '{arch_name}' not found. Available: {list(self.results.keys())}")

        result = self.results[arch_name]

        # Check that we have the required reports
        if not result.energy_report or not result.roofline_report:
            raise ValueError(f"Missing energy or roofline report for {arch_name}")

        energy_descriptors = result.energy_report.energy_descriptors
        latency_descriptors = result.roofline_report.latencies

        if len(energy_descriptors) != len(latency_descriptors):
            raise ValueError(
                f"Mismatch: {len(energy_descriptors)} energy descriptors vs "
                f"{len(latency_descriptors)} latency descriptors"
            )

        # Extract subgraph information from partition report
        subgraph_info = {}
        if result.partition_report:
            for sg in result.partition_report.subgraphs:
                subgraph_info[sg.node_id] = {
                    'fusion_pattern': sg.fusion_pattern or sg.node_name,
                    'num_operators': 1,  # Default to 1 (will update if fusion report available)
                    'arithmetic_intensity': sg.arithmetic_intensity,
                }

            # If we have fusion report, get operator counts
            if hasattr(result.partition_report, 'fusion_report') and result.partition_report.fusion_report:
                for fused_sg in result.partition_report.fusion_report.fused_subgraphs:
                    sg_id = str(fused_sg.subgraph_id)
                    if sg_id in subgraph_info:
                        subgraph_info[sg_id]['num_operators'] = fused_sg.num_operators
                        subgraph_info[sg_id]['fusion_pattern'] = fused_sg.fusion_pattern

        # Build subgraph EDP descriptors
        subgraph_edps = []

        for e_desc, l_desc in zip(energy_descriptors, latency_descriptors):
            # Ensure IDs match
            if e_desc.subgraph_id != l_desc.subgraph_id:
                print(f"Warning: ID mismatch: {e_desc.subgraph_id} vs {l_desc.subgraph_id}")

            # Calculate EDP
            edp = e_desc.total_energy_j * l_desc.actual_latency

            # Component EDPs
            compute_edp = e_desc.compute_energy_j * l_desc.actual_latency
            memory_edp = e_desc.memory_energy_j * l_desc.actual_latency
            static_edp = e_desc.static_energy_j * l_desc.actual_latency

            # Get subgraph metadata
            sg_info = subgraph_info.get(e_desc.subgraph_id, {})
            fusion_pattern = sg_info.get('fusion_pattern', e_desc.subgraph_name)
            num_operators = sg_info.get('num_operators', 1)
            arithmetic_intensity = sg_info.get('arithmetic_intensity', 0.0)

            # Determine bottleneck
            bottleneck = l_desc.bottleneck.value if hasattr(l_desc.bottleneck, 'value') else str(l_desc.bottleneck)

            subgraph_edps.append(SubgraphEDPDescriptor(
                subgraph_id=e_desc.subgraph_id,
                subgraph_name=e_desc.subgraph_name,
                fusion_pattern=fusion_pattern,
                num_operators=num_operators,
                energy_j=e_desc.total_energy_j,
                compute_energy_j=e_desc.compute_energy_j,
                memory_energy_j=e_desc.memory_energy_j,
                static_energy_j=e_desc.static_energy_j,
                latency_s=l_desc.actual_latency,
                compute_time_s=l_desc.compute_time,
                memory_time_s=l_desc.memory_time,
                edp=edp,
                compute_edp=compute_edp,
                memory_edp=memory_edp,
                static_edp=static_edp,
                edp_fraction=0.0,  # Will be set below
                bottleneck=bottleneck,
                arithmetic_intensity=arithmetic_intensity,
            ))

        # Calculate EDP fractions
        total_edp = sum(sg.edp for sg in subgraph_edps)
        for sg in subgraph_edps:
            sg.edp_fraction = sg.edp / total_edp if total_edp > 0 else 0.0

        # Sort by EDP (descending)
        subgraph_edps.sort(key=lambda x: x.edp, reverse=True)

        return subgraph_edps

    def generate_subgraph_edp_report(self, arch_name: str, top_n: int = 10) -> str:
        """
        Generate subgraph-level EDP breakdown report for one architecture.

        Args:
            arch_name: Architecture name
            top_n: Number of top subgraphs to show (default: 10)

        Returns:
            Human-readable report string
        """
        subgraph_edps = self.get_subgraph_edp_breakdown(arch_name)

        lines = []
        lines.append("=" * 100)
        lines.append(f"Subgraph-Level EDP Breakdown: {self.model_name} on {arch_name}")
        lines.append("=" * 100)
        lines.append("")

        # Summary statistics
        total_edp = sum(sg.edp for sg in subgraph_edps)
        total_subgraphs = len(subgraph_edps)

        lines.append(f"Total Subgraphs: {total_subgraphs}")
        lines.append(f"Total Model EDP: {total_edp * 1e9:.2f} nJ·s")
        lines.append("")

        # Verify against model-level EDP
        if arch_name in self.metrics:
            model_edp = self.metrics[arch_name].edp
            diff_pct = abs(total_edp - model_edp) / model_edp * 100 if model_edp > 0 else 0
            if diff_pct > 1.0:
                lines.append(f"⚠ Warning: Subgraph EDP sum ({total_edp*1e9:.2f} nJ·s) differs from model EDP "
                           f"({model_edp*1e9:.2f} nJ·s) by {diff_pct:.1f}%")
            else:
                lines.append(f"✓ Validation: Subgraph EDPs sum to model EDP (within {diff_pct:.2f}%)")
            lines.append("")

        # Show top N subgraphs
        lines.append(f"Top {min(top_n, len(subgraph_edps))} Subgraphs by EDP:")
        lines.append("")
        lines.append(f"{'Rank':<5} {'Subgraph':<35} {'Energy':<12} {'Latency':<12} {'EDP':<15} {'% Total':<10} {'Pattern (Ops)':<25} {'Bottleneck'}")
        lines.append("-" * 140)

        for i, sg in enumerate(subgraph_edps[:top_n], 1):
            # Highlight top contributor
            marker = " ⭐" if i == 1 else ""

            pattern_str = f"{sg.fusion_pattern}"
            if sg.num_operators > 1:
                pattern_str += f" ({sg.num_operators} ops)"

            lines.append(
                f"{i:<5} "
                f"{sg.subgraph_name:<35} "
                f"{sg.energy_j*1e6:<12.2f} µJ "
                f"{sg.latency_s*1e6:<12.2f} µs "
                f"{sg.edp*1e9:<15.2f} nJ·s "
                f"{sg.edp_fraction*100:<10.1f}% "
                f"{pattern_str:<25} "
                f"{sg.bottleneck:<12}"
                f"{marker}"
            )

        lines.append("")

        # Component breakdown for top subgraph
        if subgraph_edps:
            top_sg = subgraph_edps[0]
            lines.append("Top Subgraph Component Breakdown:")
            lines.append(f"  {top_sg.subgraph_name}")
            lines.append(f"    Compute EDP:  {top_sg.compute_edp*1e9:>10.2f} nJ·s ({top_sg.compute_edp/top_sg.edp*100:.1f}%)")
            lines.append(f"    Memory EDP:   {top_sg.memory_edp*1e9:>10.2f} nJ·s ({top_sg.memory_edp/top_sg.edp*100:.1f}%)")
            lines.append(f"    Static EDP:   {top_sg.static_edp*1e9:>10.2f} nJ·s ({top_sg.static_edp/top_sg.edp*100:.1f}%)")
            lines.append(f"    Total EDP:    {top_sg.edp*1e9:>10.2f} nJ·s")
            lines.append("")

        # Cumulative analysis
        cumulative_pct = 0.0
        lines.append("Cumulative EDP Distribution:")
        for i, threshold in enumerate([50, 80, 90, 95, 99]):
            for j, sg in enumerate(subgraph_edps, 1):
                cumulative_pct += sg.edp_fraction * 100
                if cumulative_pct >= threshold:
                    lines.append(f"  Top {j} subgraphs account for {threshold}% of total EDP")
                    cumulative_pct = 0.0  # Reset for next threshold
                    break

        lines.append("")
        lines.append("Optimization Insight:")
        if subgraph_edps:
            top3_pct = sum(sg.edp_fraction for sg in subgraph_edps[:3]) * 100
            lines.append(f"  → Focus optimization efforts on top 3 subgraphs ({top3_pct:.1f}% of total EDP)")
            lines.append(f"  → Top subgraph: {subgraph_edps[0].subgraph_name} ({subgraph_edps[0].edp_fraction*100:.1f}%)")

        lines.append("")
        return "\n".join(lines)

    # ==========================================================================
    # Phase 2: Operator-Level EDP Breakdown
    # ==========================================================================

    def _get_architecture_class(self, mapper: 'HardwareMapper') -> 'ArchitectureClass':
        """
        Determine architecture class from mapper's energy model.

        Args:
            mapper: Hardware mapper instance

        Returns:
            ArchitectureClass enum value
        """
        from graphs.hardware.architectural_energy import (
            ArchitectureClass,
            StoredProgramEnergyModel,
            DataParallelEnergyModel,
            SystolicArrayEnergyModel,
            DomainFlowEnergyModel,
            DataFlowMachineEnergyModel,
            SpatialPartitionEnergyModel,
            AdaptiveDatapathEnergyModel
        )

        # Get energy model from resource model
        energy_model = mapper.resource_model.architecture_energy_model

        # Map energy model type to architecture class
        if isinstance(energy_model, StoredProgramEnergyModel):
            return ArchitectureClass.STORED_PROGRAM
        elif isinstance(energy_model, DataParallelEnergyModel):
            return ArchitectureClass.DATA_PARALLEL
        elif isinstance(energy_model, SystolicArrayEnergyModel):
            return ArchitectureClass.SYSTOLIC_ARRAY
        elif isinstance(energy_model, DomainFlowEnergyModel):
            return ArchitectureClass.DOMAIN_FLOW
        elif isinstance(energy_model, DataFlowMachineEnergyModel):
            return ArchitectureClass.DATA_FLOW_MACHINE
        elif isinstance(energy_model, SpatialPartitionEnergyModel):
            return ArchitectureClass.SPATIAL_PARTITION
        elif isinstance(energy_model, AdaptiveDatapathEnergyModel):
            return ArchitectureClass.ADAPTIVE_DATAPATH
        else:
            # Default to STORED_PROGRAM if unknown
            return ArchitectureClass.STORED_PROGRAM

    def get_operator_edp_breakdown(
        self,
        arch_name: str,
        subgraph_name: Optional[str] = None
    ) -> List['OperatorEDPDescriptor']:
        """
        Get per-operator EDP breakdown within subgraphs (Phase 2).

        Decomposes subgraph EDP to individual operators using:
        1. FLOP-proportional base allocation
        2. Architectural modifiers (fusion-aware)
        3. Normalization to match subgraph total

        Args:
            arch_name: Architecture name (e.g., 'GPU', 'KPU')
            subgraph_name: Optional subgraph to focus on (all if None)

        Returns:
            List of OperatorEDPDescriptor, sorted by architectural_edp (descending)

        Example:
            >>> operator_edps = comparator.get_operator_edp_breakdown('KPU', 'fc1')
            >>> for op in operator_edps:
            ...     print(f"{op.operator_type}: {op.architectural_edp*1e9:.2f} nJ·s ({op.edp_fraction_of_subgraph*100:.1f}%)")
            Linear: 0.48 nJ·s (95.0%)
            Bias: 0.01 nJ·s (2.5%)
            ReLU: 0.01 nJ·s (2.5%)
        """
        from graphs.analysis.architectural_modifiers import get_architectural_modifier, get_fusion_benefit
        from graphs.hardware.architectural_energy import ArchitectureClass

        if not self.metrics or arch_name not in self.metrics:
            raise ValueError(f"No metrics for architecture '{arch_name}'. Run analyze_all() first.")

        # Get architecture metrics and extract full result
        metrics = self.metrics[arch_name]
        result = metrics.full_result

        if not result.energy_report or not result.roofline_report or not result.partition_report:
            raise ValueError(f"Missing reports for {arch_name}")

        # Get architecture class from the energy model
        mapper = self.architectures[arch_name]
        arch_class = self._get_architecture_class(mapper)

        # Get subgraph EDP breakdown
        subgraph_edps = self.get_subgraph_edp_breakdown(arch_name)

        # Filter to specific subgraph if requested
        if subgraph_name:
            subgraph_edps = [sg for sg in subgraph_edps if sg.subgraph_name == subgraph_name]
            if not subgraph_edps:
                raise ValueError(f"Subgraph '{subgraph_name}' not found for {arch_name}")

        # Decompose each subgraph to operators
        all_operator_edps = []

        for sg_edp in subgraph_edps:
            # Find corresponding subgraph descriptor for operator info
            sg_desc = next((sg for sg in result.partition_report.subgraphs if sg.node_name == sg_edp.subgraph_name), None)
            if not sg_desc:
                continue

            # Check if this is a truly fused subgraph (multiple operations)
            # For now, most subgraphs are single operations, so use operation_type directly
            if hasattr(sg_desc, 'ops') and sg_desc.ops and len(sg_desc.ops) > 1:
                # Truly fused subgraph - parse fusion pattern
                operators = self._parse_fusion_pattern(sg_desc.fusion_pattern, sg_desc.flops)
            else:
                # Single operation subgraph - use operation_type directly
                op_type_str = self._operation_type_to_string(sg_desc.operation_type)
                operators = [{
                    'type': op_type_str,
                    'flops': sg_desc.flops,
                    'memory_bytes': sg_desc.total_input_bytes + sg_desc.total_output_bytes + sg_desc.total_weight_bytes
                }]

            if not operators:
                # Final fallback: use node name
                operators = [{
                    'type': sg_desc.node_name,
                    'flops': sg_desc.flops,
                    'memory_bytes': sg_desc.total_input_bytes + sg_desc.total_output_bytes + sg_desc.total_weight_bytes
                }]

            # Calculate base EDP allocation (FLOP-proportional)
            total_flops = sum(op['flops'] for op in operators)
            if total_flops == 0:
                total_flops = len(operators)  # Equal split if no FLOP info

            is_fused = len(operators) > 1

            for i, op_info in enumerate(operators):
                # Base EDP (FLOP-proportional)
                flop_fraction = op_info['flops'] / total_flops if total_flops > 0 else 1.0 / len(operators)
                base_edp = sg_edp.edp * flop_fraction

                # Get architectural modifier
                modifier = get_architectural_modifier(op_info['type'], arch_class, is_fused)

                # Architectural EDP (before normalization)
                architectural_edp_raw = base_edp * modifier

                # Store for normalization
                op_info['base_edp'] = base_edp
                op_info['modifier'] = modifier
                op_info['architectural_edp_raw'] = architectural_edp_raw
                op_info['index'] = i

            # Normalize architectural EDPs to match subgraph total
            total_arch_edp_raw = sum(op['architectural_edp_raw'] for op in operators)
            normalization = sg_edp.edp / total_arch_edp_raw if total_arch_edp_raw > 0 else 1.0

            # Create OperatorEDPDescriptor for each operator
            for op_info in operators:
                architectural_edp = op_info['architectural_edp_raw'] * normalization

                # Calculate fusion benefit (if fused)
                fusion_benefit_ratio = None
                if is_fused:
                    fusion_benefit_ratio = get_fusion_benefit(op_info['type'], arch_class)

                operator_edp = OperatorEDPDescriptor(
                    operator_id=f"{sg_desc.node_id}_op{op_info['index']}",
                    operator_type=op_info['type'],
                    subgraph_id=sg_desc.node_id,
                    subgraph_name=sg_desc.node_name,
                    base_edp=op_info['base_edp'],
                    architectural_edp=architectural_edp,
                    architectural_modifier=op_info['modifier'],
                    edp_fraction_of_subgraph=architectural_edp / sg_edp.edp if sg_edp.edp > 0 else 0.0,
                    edp_fraction_of_model=sg_edp.edp_fraction * (architectural_edp / sg_edp.edp) if sg_edp.edp > 0 else 0.0,
                    is_fused=is_fused,
                    fusion_benefit=fusion_benefit_ratio,
                    flops=op_info['flops'],
                    memory_bytes=op_info.get('memory_bytes', 0),
                    arithmetic_intensity=op_info['flops'] / op_info.get('memory_bytes', 1) if op_info.get('memory_bytes', 0) > 0 else 0.0
                )

                all_operator_edps.append(operator_edp)

        # Recalculate operator fractions based on energy, not EDP
        # Key insight: At the model level, all operators share the same total latency.
        # Therefore: operator_edp_fraction = operator_energy / model_energy
        #
        # Because: Model_EDP = Model_Energy × Model_Latency
        #          Operator contribution to Model_EDP = Operator_Energy × Model_Latency
        #          Fraction = (Op_Energy × Model_Lat) / (Model_Energy × Model_Lat)
        #                   = Op_Energy / Model_Energy
        #
        # Energy is additive, latency is shared!
        model_energy = metrics.total_energy_j

        if model_energy > 0:
            # Build lookup of subgraph latencies
            sg_latencies = {sg.subgraph_id: sg.latency_s for sg in subgraph_edps}

            for op in all_operator_edps:
                # Extract operator energy from its EDP
                # operator_edp = operator_energy × subgraph_latency
                sg_latency = sg_latencies.get(op.subgraph_id, 1.0)
                if sg_latency > 0:
                    operator_energy = op.architectural_edp / sg_latency
                    op.edp_fraction_of_model = operator_energy / model_energy
                else:
                    op.edp_fraction_of_model = 0.0

        # Sort by architectural EDP (descending)
        all_operator_edps.sort(key=lambda x: x.architectural_edp, reverse=True)

        return all_operator_edps

    def _operation_type_to_string(self, op_type) -> str:
        """
        Convert OperationType enum to human-readable string.

        Args:
            op_type: OperationType enum value

        Returns:
            Human-readable operator name (e.g., "Conv2d", "ReLU", "Linear")
        """
        from graphs.ir.structures import OperationType

        mapping = {
            OperationType.CONV2D: 'Conv2d',
            OperationType.CONV2D_DEPTHWISE: 'Conv2d_Depthwise',
            OperationType.CONV2D_POINTWISE: 'Conv2d_Pointwise',
            OperationType.LINEAR: 'Linear',
            OperationType.MATMUL: 'MatMul',
            OperationType.BATCHNORM: 'BatchNorm2d',
            OperationType.LAYERNORM: 'LayerNorm',
            OperationType.RELU: 'ReLU',
            OperationType.RELU6: 'ReLU6',
            OperationType.SILU: 'SiLU',
            OperationType.SWISH: 'Swish',
            OperationType.GELU: 'GELU',
            OperationType.HARDSWISH: 'Hardswish',
            OperationType.SIGMOID: 'Sigmoid',
            OperationType.MAXPOOL: 'MaxPool2d',
            OperationType.AVGPOOL: 'AvgPool2d',
            OperationType.ADAPTIVEAVGPOOL: 'AdaptiveAvgPool2d',
            OperationType.ELEMENTWISE: 'Elementwise',
            OperationType.SQUEEZE_EXCITE: 'SqueezeExcite',
            OperationType.MULTIHEAD_ATTENTION: 'MultiheadAttention',
            OperationType.DROPOUT: 'Dropout',
            OperationType.STOCHASTIC_DEPTH: 'StochasticDepth',
            OperationType.UNKNOWN: 'Unknown',
        }

        return mapping.get(op_type, str(op_type))

    def _parse_fusion_pattern(self, fusion_pattern: str, total_flops: float) -> List[Dict]:
        """
        Parse fusion pattern to extract operator list with estimated FLOPs.

        Args:
            fusion_pattern: e.g., "conv_bn_relu", "linear", "add"
            total_flops: Total FLOPs for the subgraph

        Returns:
            List of dicts: [{'type': 'Conv2d', 'flops': ...}, {'type': 'BatchNorm2d', 'flops': ...}, ...]
        """
        # Common fusion pattern mappings
        pattern_to_ops = {
            # Convolution patterns
            'conv_bn_relu': [('Conv2d', 0.95), ('BatchNorm2d', 0.04), ('ReLU', 0.01)],
            'conv_relu': [('Conv2d', 0.98), ('ReLU', 0.02)],
            'conv_bn': [('Conv2d', 0.96), ('BatchNorm2d', 0.04)],
            'conv': [('Conv2d', 1.0)],

            # Linear patterns
            'linear_bias_relu': [('Linear', 0.95), ('Bias', 0.03), ('ReLU', 0.02)],
            'linear_relu': [('Linear', 0.98), ('ReLU', 0.02)],
            'linear': [('Linear', 1.0)],

            # Activation
            'relu': [('ReLU', 1.0)],
            'gelu': [('GELU', 1.0)],
            'sigmoid': [('Sigmoid', 1.0)],
            'tanh': [('Tanh', 1.0)],

            # Pooling
            'max_pool2d': [('MaxPool2d', 1.0)],
            'avg_pool2d': [('AvgPool2d', 1.0)],
            'adaptive_avg_pool2d': [('AdaptiveAvgPool2d', 1.0)],

            # Normalization
            'batch_norm': [('BatchNorm2d', 1.0)],
            'layer_norm': [('LayerNorm', 1.0)],

            # Other
            'add': [('add', 1.0)],
            'matmul': [('matmul', 1.0)],
            'bmm': [('bmm', 1.0)],
            'softmax': [('Softmax', 1.0)],
        }

        # Normalize pattern (lowercase, remove underscores at ends)
        pattern_normalized = fusion_pattern.lower().strip('_')

        # Try exact match first
        if pattern_normalized in pattern_to_ops:
            ops_with_fractions = pattern_to_ops[pattern_normalized]
        else:
            # Try splitting by underscore and mapping each part
            parts = pattern_normalized.split('_')
            ops_with_fractions = []

            for part in parts:
                # Map common abbreviations
                if part in ['conv', 'conv2d']:
                    ops_with_fractions.append(('Conv2d', 1.0))
                elif part in ['bn', 'batchnorm']:
                    ops_with_fractions.append(('BatchNorm2d', 1.0))
                elif part == 'relu':
                    ops_with_fractions.append(('ReLU', 1.0))
                elif part in ['linear', 'fc']:
                    ops_with_fractions.append(('Linear', 1.0))
                elif part == 'add':
                    ops_with_fractions.append(('add', 1.0))
                elif part == 'bias':
                    ops_with_fractions.append(('Bias', 1.0))
                else:
                    # Unknown pattern - return as-is
                    ops_with_fractions.append((part.capitalize(), 1.0))

            # Normalize fractions if auto-split
            if len(ops_with_fractions) > 1:
                # Heuristic: compute ops get more FLOPs
                normalized_fractions = []
                for op_type, _ in ops_with_fractions:
                    if op_type in ['Conv2d', 'Linear', 'matmul', 'bmm']:
                        normalized_fractions.append(0.9 / sum(1 for t, _ in ops_with_fractions if t in ['Conv2d', 'Linear', 'matmul', 'bmm']))
                    else:
                        remaining_ops = sum(1 for t, _ in ops_with_fractions if t not in ['Conv2d', 'Linear', 'matmul', 'bmm'])
                        normalized_fractions.append(0.1 / remaining_ops if remaining_ops > 0 else 0.0)

                ops_with_fractions = [(op_type, frac) for (op_type, _), frac in zip(ops_with_fractions, normalized_fractions)]

        # Convert to dict list with absolute FLOPs
        operators = []
        for op_type, frac in ops_with_fractions:
            operators.append({
                'type': op_type,
                'flops': total_flops * frac,
                'memory_bytes': 0  # TODO: could estimate from tensor sizes
            })

        return operators

    def generate_operator_edp_report(
        self,
        arch_name: str,
        subgraph_name: Optional[str] = None,
        top_n: int = 10,
        show_fusion_benefits: bool = True
    ) -> str:
        """
        Generate comprehensive operator-level EDP breakdown report (Phase 2).

        Args:
            arch_name: Architecture name
            subgraph_name: Optional subgraph to focus on (all if None)
            top_n: Number of top operators to show
            show_fusion_benefits: Whether to show fusion benefit analysis

        Returns:
            Formatted report string

        Example Output:
            ==============================================================================
            Operator-Level EDP Breakdown: KPU
            ==============================================================================

            Top 10 Operators by EDP:

            Rank  Operator   Subgraph  EDP (nJ·s)  % Subgraph  % Model  Modifier  Fused
            ------------------------------------------------------------------------------
            1 ⭐  Linear     fc1       0.48        95.0%       76.0%    1.00×     Yes
            2     Linear     fc2       0.12        95.0%       19.0%    1.00×     Yes
            3     Bias       fc1       0.01         2.5%        2.0%    0.05×     Yes ✓
            ...
        """
        from graphs.analysis.architectural_modifiers import explain_modifier

        lines = []
        lines.append("=" * 140)
        lines.append(f"Operator-Level EDP Breakdown: {arch_name}")
        if subgraph_name:
            lines.append(f"Subgraph: {subgraph_name}")
        lines.append("=" * 140)
        lines.append("")

        # Get operator EDPs
        operator_edps = self.get_operator_edp_breakdown(arch_name, subgraph_name)

        if not operator_edps:
            lines.append("No operator data available.")
            return "\n".join(lines)

        # Summary
        total_operators = len(operator_edps)
        unique_subgraphs = len(set(op.subgraph_name for op in operator_edps))
        total_edp_fraction = sum(op.edp_fraction_of_model for op in operator_edps)

        lines.append(f"Total Operators: {total_operators}")
        lines.append(f"Subgraphs: {unique_subgraphs}")
        lines.append(f"Operator EDP Coverage: {total_edp_fraction*100:.1f}% of model energy")
        if total_edp_fraction < 0.99:
            remaining_pct = (1.0 - total_edp_fraction) * 100
            lines.append(f"  (Remaining {remaining_pct:.1f}% is static/leakage energy)")
        lines.append("")

        # Top N operators
        lines.append(f"Top {min(top_n, len(operator_edps))} Operators by EDP:")
        lines.append("")
        lines.append(f"{'Rank':<5} {'Operator':<15} {'Subgraph':<25} {'EDP (nJ·s)':<12} {'% Subgraph':<12} {'% Model':<10} {'Modifier':<10} {'Fused':<8}")
        lines.append("-" * 140)

        for i, op in enumerate(operator_edps[:top_n], 1):
            marker = " ⭐" if i == 1 else ""
            fused_str = "Yes" if op.is_fused else "No"
            if op.is_fused and op.fusion_benefit and op.fusion_benefit > 5.0:
                fused_str += " ✓"  # High fusion benefit

            lines.append(
                f"{i:<5} "
                f"{op.operator_type:<15} "
                f"{op.subgraph_name:<25} "
                f"{op.architectural_edp*1e9:<12.2f} "
                f"{op.edp_fraction_of_subgraph*100:<12.1f}% "
                f"{op.edp_fraction_of_model*100:<10.1f}% "
                f"{op.architectural_modifier:<10.2f}× "
                f"{fused_str:<8}"
                f"{marker}"
            )

        lines.append("")

        # Top operator detailed breakdown
        if operator_edps:
            top_op = operator_edps[0]
            lines.append("Top Operator Detailed Breakdown:")
            lines.append(f"  Operator: {top_op.operator_type}")
            lines.append(f"  Subgraph: {top_op.subgraph_name}")
            lines.append(f"    Base EDP (FLOP-proportional):     {top_op.base_edp*1e9:>10.2f} nJ·s")
            lines.append(f"    Architectural Modifier:           {top_op.architectural_modifier:>10.2f}×")
            lines.append(f"    Architectural EDP (final):        {top_op.architectural_edp*1e9:>10.2f} nJ·s")
            lines.append(f"    Fraction of subgraph:             {top_op.edp_fraction_of_subgraph*100:>10.1f}%")
            lines.append(f"    Fraction of model:                {top_op.edp_fraction_of_model*100:>10.1f}%")
            lines.append(f"    Fused: {top_op.is_fused}")

            # Get architecture class for explanation
            mapper = self.architectures[arch_name]
            arch_class = self._get_architecture_class(mapper)
            explanation = explain_modifier(top_op.operator_type, arch_class, top_op.is_fused)
            lines.append(f"    Modifier explanation: {explanation}")

            lines.append("")

        # Fusion benefit analysis (if requested and applicable)
        if show_fusion_benefits:
            fused_ops = [op for op in operator_edps if op.is_fused and op.fusion_benefit is not None]

            if fused_ops:
                lines.append("Fusion Benefit Analysis:")
                lines.append("  (Shows EDP multiplier: separate_execution / fused_execution)")
                lines.append("")
                lines.append(f"  {'Operator':<15} {'Subgraph':<25} {'Fusion Benefit':<15} {'Interpretation'}")
                lines.append("  " + "-" * 100)

                # Sort by fusion benefit (descending)
                fused_ops_sorted = sorted(fused_ops, key=lambda x: x.fusion_benefit or 0, reverse=True)

                for op in fused_ops_sorted[:10]:  # Top 10 fusion benefits
                    if op.fusion_benefit >= 10.0:
                        interpretation = "High benefit - fusion critical"
                    elif op.fusion_benefit >= 3.0:
                        interpretation = "Moderate benefit - fusion helpful"
                    elif op.fusion_benefit >= 1.5:
                        interpretation = "Low benefit - fusion optional"
                    elif op.fusion_benefit > 1.0:
                        interpretation = "Marginal benefit"
                    elif op.fusion_benefit == 1.0:
                        interpretation = "No benefit (same either way)"
                    else:
                        interpretation = "Fusion hurts (rare)"

                    lines.append(
                        f"  {op.operator_type:<15} "
                        f"{op.subgraph_name:<25} "
                        f"{op.fusion_benefit or 0.0:<15.2f}× "
                        f"{interpretation}"
                    )

                lines.append("")

        # Operator type distribution
        op_type_counts = {}
        op_type_edps = {}
        for op in operator_edps:
            op_type_counts[op.operator_type] = op_type_counts.get(op.operator_type, 0) + 1
            op_type_edps[op.operator_type] = op_type_edps.get(op.operator_type, 0.0) + op.architectural_edp

        lines.append("Operator Type Distribution:")
        lines.append("")
        lines.append(f"  {'Operator Type':<15} {'Count':<8} {'Total EDP (nJ·s)':<20} {'% of Model'}")
        lines.append("  " + "-" * 80)

        # Sort by total EDP (descending)
        total_model_edp = sum(op_type_edps.values())
        sorted_types = sorted(op_type_edps.items(), key=lambda x: x[1], reverse=True)

        for op_type, total_edp in sorted_types[:15]:  # Top 15 operator types
            count = op_type_counts[op_type]
            pct = (total_edp / total_model_edp * 100) if total_model_edp > 0 else 0.0

            lines.append(
                f"  {op_type:<15} "
                f"{count:<8} "
                f"{total_edp*1e9:<20.2f} "
                f"{pct:>6.1f}%"
            )

        lines.append("")

        # Optimization insights
        lines.append("Optimization Insights:")

        # Top operator
        if operator_edps:
            top_op = operator_edps[0]
            lines.append(f"  → Top operator: {top_op.operator_type} in {top_op.subgraph_name} ({top_op.edp_fraction_of_model*100:.1f}% of model)")

        # High fusion benefit ops
        high_benefit_ops = [op for op in operator_edps if op.is_fused and op.fusion_benefit and op.fusion_benefit > 10.0]
        if high_benefit_ops:
            lines.append(f"  → {len(high_benefit_ops)} operators show high fusion benefit (>10×)")
            lines.append(f"    Types: {', '.join(set(op.operator_type for op in high_benefit_ops[:5]))}")

        # Low-modifier ops (architectural optimization opportunities)
        low_modifier_ops = [op for op in operator_edps if op.is_fused and op.architectural_modifier < 0.2]
        if low_modifier_ops:
            lines.append(f"  → {len(low_modifier_ops)} operators hidden in dataflow (modifier < 0.2)")
            total_hidden_pct = sum(op.edp_fraction_of_model for op in low_modifier_ops) * 100
            lines.append(f"    Total EDP: {total_hidden_pct:.1f}% of model (effectively 'free' due to fusion)")

        lines.append("")
        return "\n".join(lines)

    def generate_subgraph_comparison(self, show_details: bool = False) -> str:
        """Generate Level 2: Per-subgraph comparison"""

        if not self.metrics:
            raise ValueError("No metrics available. Run analyze_all() first.")

        lines = []
        lines.append("=" * 80)
        lines.append(f"Subgraph-Level Comparison: {self.model_name}")
        lines.append("=" * 80)
        lines.append("")

        # Get subgraph count from first architecture
        first_arch = list(self.metrics.values())[0]
        num_subgraphs = first_arch.total_subgraphs

        if num_subgraphs == 0:
            lines.append("No subgraph data available.")
            return "\n".join(lines)

        # Header
        arch_names = list(self.metrics.keys())
        header = f"{'ID':<4} {'Operation':<20}"
        for name in arch_names:
            header += f" {name:<12}"
        header += f" {'Winner':<10}"

        lines.append(header)
        lines.append("-" * 80)

        # For each subgraph, compare across architectures
        for sg_idx in range(min(num_subgraphs, 20)):  # Limit to first 20
            row = f"{sg_idx:<4} "

            # Get operation name from first architecture
            first_result = first_arch.full_result
            if first_result.partition_report and sg_idx < len(first_result.partition_report.subgraphs):
                sg = first_result.partition_report.subgraphs[sg_idx]
                op_name = sg.node_names[0] if sg.node_names else "unknown"
                row += f"{op_name[:20]:<20}"
            else:
                row += f"{'subgraph':<20}"

            # Get energy for this subgraph from each architecture
            sg_energies = {}
            for arch_name, metrics in self.metrics.items():
                result = metrics.full_result
                if result.energy_report and sg_idx < len(result.energy_report.energy_descriptors):
                    energy_desc = result.energy_report.energy_descriptors[sg_idx]
                    energy = energy_desc.total_energy_j
                    sg_energies[arch_name] = energy
                    row += f" {self._format_energy(energy):<12}"
                else:
                    row += f" {'N/A':<12}"

            # Find winner
            if sg_energies:
                winner = min(sg_energies.items(), key=lambda x: x[1])[0]
                row += f" {winner:<10}"
            else:
                row += f" {'—':<10}"

            lines.append(row)

        if num_subgraphs > 20:
            lines.append(f"... ({num_subgraphs - 20} more subgraphs)")

        lines.append("")

        # Summary statistics
        lines.append("Summary:")
        for arch_name in arch_names:
            # Count wins
            wins = 0
            # TODO: Count actual wins when we have subgraph data
            lines.append(f"  {arch_name}: {wins} subgraphs (best)")

        lines.append("")

        return "\n".join(lines)

    def explain_difference(
        self,
        arch1: str,
        arch2: str,
        metric: str = "energy"
    ) -> str:
        """
        Generate educational explanation of why arch1 differs from arch2.

        Args:
            arch1: First architecture name
            arch2: Second architecture name
            metric: Metric to explain ('energy', 'latency', 'memory')
        """

        if arch1 not in self.metrics or arch2 not in self.metrics:
            raise ValueError(f"Architecture not found: {arch1} or {arch2}")

        m1 = self.metrics[arch1]
        m2 = self.metrics[arch2]

        lines = []
        lines.append("=" * 80)
        lines.append(f"Why is {arch1} different from {arch2}?")
        lines.append("=" * 80)
        lines.append("")

        if metric == "energy":
            lines.extend(self._explain_energy_difference(arch1, m1, arch2, m2))
        elif metric == "latency":
            lines.extend(self._explain_latency_difference(arch1, m1, arch2, m2))
        elif metric == "memory":
            lines.extend(self._explain_memory_difference(arch1, m1, arch2, m2))
        else:
            lines.append(f"Unknown metric: {metric}")

        return "\n".join(lines)

    def _explain_energy_difference(
        self,
        arch1: str, m1: ArchitectureMetrics,
        arch2: str, m2: ArchitectureMetrics
    ) -> List[str]:
        """Explain energy difference between two architectures"""

        lines = []

        ratio = m1.total_energy_j / m2.total_energy_j

        if ratio > 1.0:
            lines.append(f"{arch1} uses {ratio:.1f}× MORE energy than {arch2}")
            lines.append("")
            lines.append("Energy Breakdown:")
            lines.append(f"  {arch1}: {self._format_energy(m1.total_energy_j)}")
            lines.append(f"    ├─ Compute:  {self._format_energy(m1.compute_energy_j)}")
            lines.append(f"    └─ Memory:   {self._format_energy(m1.memory_energy_j)}")
            lines.append("")
            lines.append(f"  {arch2}: {self._format_energy(m2.total_energy_j)}")
            lines.append(f"    ├─ Compute:  {self._format_energy(m2.compute_energy_j)}")
            lines.append(f"    └─ Memory:   {self._format_energy(m2.memory_energy_j)}")
            lines.append("")

            # Architectural explanation
            lines.append("Architectural Differences:")

            # Use actual architectural breakdown if available
            if m1.architectural_breakdown and m2.architectural_breakdown:
                lines.append(f"  {arch2} is more energy efficient due to:")
                lines.append("")

                # Compare specific architectural events
                b1 = m1.architectural_breakdown
                b2 = m2.architectural_breakdown

                # Control overhead comparison
                if abs(b1.control_overhead - b2.control_overhead) > 1e-12:
                    diff = b1.control_overhead - b2.control_overhead
                    if diff > 0:
                        lines.append(f"    • {self._format_energy(abs(diff))} saved in control overhead")
                        lines.append(f"      ({arch2} eliminates instruction fetch and scheduling)")

                # Memory overhead comparison
                if abs(b1.memory_overhead - b2.memory_overhead) > 1e-12:
                    diff = b1.memory_overhead - b2.memory_overhead
                    if diff > 0:
                        lines.append(f"    • {self._format_energy(abs(diff))} saved in memory overhead")
                        lines.append(f"      ({arch2} uses more efficient memory access patterns)")

                # Compute overhead comparison
                if abs(b1.compute_overhead - b2.compute_overhead) > 1e-12:
                    diff = b1.compute_overhead - b2.compute_overhead
                    if diff > 0:
                        lines.append(f"    • {self._format_energy(abs(diff))} saved in compute overhead")

                lines.append("")
                lines.append(f"  Key architectural advantage of {arch2}:")

                # Extract key insight from explanation
                if b2.explanation:
                    # Get first meaningful line from explanation
                    for line in b2.explanation.split('\n'):
                        if 'Architecture' in line or 'Energy' in line:
                            lines.append(f"    {line.strip()}")
                            break

            else:
                lines.append(f"  {arch2} is more energy efficient because it uses a different")
                lines.append("  resource contention management strategy.")
                lines.append("")
                lines.append("  For detailed architectural energy events, see the architectural")
                lines.append("  energy model breakdown in the detailed view.")
        else:
            lines.append(f"{arch1} uses {1.0/ratio:.1f}× LESS energy than {arch2}")

        lines.append("")

        return lines

    def _explain_latency_difference(
        self,
        arch1: str, m1: ArchitectureMetrics,
        arch2: str, m2: ArchitectureMetrics
    ) -> List[str]:
        """Explain latency difference between two architectures"""

        lines = []

        ratio = m1.total_latency_s / m2.total_latency_s

        if ratio > 1.0:
            lines.append(f"{arch1} is {ratio:.1f}× SLOWER than {arch2}")
        else:
            lines.append(f"{arch1} is {1.0/ratio:.1f}× FASTER than {arch2}")

        lines.append("")
        lines.append("Latency Breakdown:")
        lines.append(f"  {arch1}: {self._format_time(m1.total_latency_s)}")
        lines.append(f"  {arch2}: {self._format_time(m2.total_latency_s)}")
        lines.append("")

        # Bottleneck analysis
        lines.append("Bottleneck Analysis:")
        lines.append(f"  {arch1}:")
        if m1.total_subgraphs > 0:
            compute_pct = m1.compute_bound_subgraphs / m1.total_subgraphs * 100
            memory_pct = m1.memory_bound_subgraphs / m1.total_subgraphs * 100
            lines.append(f"    - Compute-bound: {compute_pct:.0f}%")
            lines.append(f"    - Memory-bound:  {memory_pct:.0f}%")

        lines.append(f"  {arch2}:")
        if m2.total_subgraphs > 0:
            compute_pct = m2.compute_bound_subgraphs / m2.total_subgraphs * 100
            memory_pct = m2.memory_bound_subgraphs / m2.total_subgraphs * 100
            lines.append(f"    - Compute-bound: {compute_pct:.0f}%")
            lines.append(f"    - Memory-bound:  {memory_pct:.0f}%")

        lines.append("")

        return lines

    def _explain_memory_difference(
        self,
        arch1: str, m1: ArchitectureMetrics,
        arch2: str, m2: ArchitectureMetrics
    ) -> List[str]:
        """Explain memory difference between two architectures"""

        lines = []

        ratio = m1.peak_memory_bytes / m2.peak_memory_bytes

        if ratio > 1.0:
            lines.append(f"{arch1} uses {ratio:.1f}× MORE memory than {arch2}")
        else:
            lines.append(f"{arch1} uses {1.0/ratio:.1f}× LESS memory than {arch2}")

        lines.append("")
        lines.append("Memory Usage:")
        lines.append(f"  {arch1}: {self._format_bytes(m1.peak_memory_bytes)}")
        lines.append(f"  {arch2}: {self._format_bytes(m2.peak_memory_bytes)}")
        lines.append("")

        return lines

    # Helper formatting methods

    def _format_energy(self, joules: float) -> str:
        """Format energy in human-readable units"""
        if joules >= 1.0:
            return f"{joules:.2f} J"
        elif joules >= 1e-3:
            return f"{joules*1e3:.2f} mJ"
        elif joules >= 1e-6:
            return f"{joules*1e6:.2f} µJ"
        elif joules >= 1e-9:
            return f"{joules*1e9:.2f} nJ"
        else:
            return f"{joules*1e12:.2f} pJ"

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable units"""
        if seconds >= 1.0:
            return f"{seconds:.2f} s"
        elif seconds >= 1e-3:
            return f"{seconds*1e3:.2f} ms"
        elif seconds >= 1e-6:
            return f"{seconds*1e6:.2f} µs"
        else:
            return f"{seconds*1e9:.2f} ns"

    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes in human-readable units"""
        if bytes_val >= 1e9:
            return f"{bytes_val/1e9:.2f} GB"
        elif bytes_val >= 1e6:
            return f"{bytes_val/1e6:.2f} MB"
        elif bytes_val >= 1e3:
            return f"{bytes_val/1e3:.2f} KB"
        else:
            return f"{bytes_val} B"

    # Export methods

    def export_json(self) -> str:
        """
        Export comparison data as JSON.

        Returns complete structured data including all metrics,
        breakdowns, and insights for programmatic analysis.

        Returns:
            JSON string with complete comparison data
        """
        data = {
            'model': self.model_name,
            'batch_size': self.batch_size,
            'precision': self.precision.value,
            'architectures': {},
            'summary': {
                'best_energy': self.summary.energy_winner,
                'best_latency': self.summary.latency_winner,
                'best_throughput': self.summary.throughput_winner,
                'best_memory': self.summary.memory_winner,
                'best_edp': self.summary.edp_winner,
                'best_balance': self.summary.balance_winner,
                'insights': self.summary.insights,
            }
        }

        # Add per-architecture data
        for name, metrics in self.metrics.items():
            arch_data = {
                'energy': {
                    'total_j': metrics.total_energy_j,
                    'total_mj': metrics.total_energy_j * 1000,
                    'compute_j': metrics.compute_energy_j,
                    'memory_j': metrics.memory_energy_j,
                    'architectural_overhead_j': metrics.architectural_overhead_j,
                    'per_inference_j': metrics.energy_per_inference_j,
                },
                'performance': {
                    'latency_s': metrics.total_latency_s,
                    'latency_ms': metrics.total_latency_s * 1000,
                    'throughput_fps': metrics.throughput_inferences_per_sec,
                    'utilization_percent': metrics.utilization * 100,
                },
                'memory': {
                    'peak_bytes': metrics.peak_memory_bytes,
                    'peak_mb': metrics.peak_memory_bytes / 1e6,
                },
                'bottlenecks': {
                    'compute_bound': metrics.compute_bound_subgraphs,
                    'memory_bound': metrics.memory_bound_subgraphs,
                    'total_subgraphs': metrics.total_subgraphs,
                    'compute_bound_percent': (metrics.compute_bound_subgraphs / metrics.total_subgraphs * 100) if metrics.total_subgraphs > 0 else 0,
                },
                'edp': {
                    'total_j_s': metrics.edp,
                    'total_nj_s': metrics.edp * 1e9,
                    'normalized': metrics.edp_normalized,
                    'compute_j_s': metrics.compute_edp,
                    'memory_j_s': metrics.memory_edp,
                    'architectural_j_s': metrics.architectural_edp,
                }
            }

            # Add architectural energy breakdown if available
            if metrics.architectural_breakdown:
                b = metrics.architectural_breakdown
                arch_data['architectural_energy'] = {
                    'compute_overhead_j': b.compute_overhead,
                    'memory_overhead_j': b.memory_overhead,
                    'control_overhead_j': b.control_overhead,
                    'total_overhead_j': b.total_overhead,
                    'extra_details': b.extra_details if b.extra_details else {},
                    'explanation': b.explanation,
                }

            data['architectures'][name] = arch_data

        return json.dumps(data, indent=2)

    def export_csv(self) -> str:
        """
        Export comparison data as CSV.

        Returns tabular data suitable for spreadsheets and analysis tools.
        Includes summary metrics and per-architecture breakdowns.

        Returns:
            CSV string with comparison data
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Architecture',
            'Energy (J)',
            'Energy (mJ)',
            'Latency (ms)',
            'Throughput (FPS)',
            'EDP (nJ·s)',
            'EDP (normalized)',
            'Compute EDP (nJ·s)',
            'Memory EDP (nJ·s)',
            'Architectural EDP (nJ·s)',
            'Attained (TOPS)',
            'Peak (TOPS)',
            'Compute Util (%)',
            'Mem BW Util (%)',
            'Arith Intensity (FLOPs/byte)',
            'Compute Units Allocated',
            'Compute Units Total',
            'Energy Efficiency (TOPS/W)',
            'Peak Memory (MB)',
            'Compute Energy (J)',
            'Memory Energy (J)',
            'Architectural Overhead (J)',
            'Compute Bound (%)',
            'Memory Bound (%)',
            'Total Subgraphs',
        ])

        # Data rows
        for name in sorted(self.metrics.keys()):
            metrics = self.metrics[name]
            writer.writerow([
                name,
                f"{metrics.total_energy_j:.6f}",
                f"{metrics.total_energy_j * 1000:.3f}",
                f"{metrics.total_latency_s * 1000:.3f}",
                f"{metrics.throughput_inferences_per_sec:.1f}",
                f"{metrics.edp * 1e9:.4f}",  # EDP in nJ·s
                f"{metrics.edp_normalized:.4f}",
                f"{metrics.compute_edp * 1e9:.4f}",  # Compute EDP in nJ·s
                f"{metrics.memory_edp * 1e9:.4f}",   # Memory EDP in nJ·s
                f"{metrics.architectural_edp * 1e9:.4f}",  # Architectural EDP in nJ·s
                f"{metrics.attained_tops:.4f}",
                f"{metrics.peak_tops:.2f}",
                f"{metrics.compute_utilization_pct:.2f}",
                f"{metrics.memory_bandwidth_utilization_pct:.2f}",
                f"{metrics.arithmetic_intensity:.2f}",
                metrics.compute_units_allocated,
                metrics.compute_units_total,
                f"{metrics.energy_efficiency_tops_per_watt:.4f}",
                f"{metrics.peak_memory_bytes / 1e6:.2f}",
                f"{metrics.compute_energy_j:.6f}",
                f"{metrics.memory_energy_j:.6f}",
                f"{metrics.architectural_overhead_j:.6f}",
                f"{(metrics.compute_bound_subgraphs / metrics.total_subgraphs * 100) if metrics.total_subgraphs > 0 else 0:.1f}",
                f"{(metrics.memory_bound_subgraphs / metrics.total_subgraphs * 100) if metrics.total_subgraphs > 0 else 0:.1f}",
                metrics.total_subgraphs,
            ])

        # Add architectural energy breakdown if available
        has_arch_breakdown = any(m.architectural_breakdown is not None for m in self.metrics.values())
        if has_arch_breakdown:
            output.write('\n')
            writer.writerow(['Architectural Energy Breakdown'])
            writer.writerow([
                'Architecture',
                'Compute Overhead (J)',
                'Memory Overhead (J)',
                'Control Overhead (J)',
                'Total Overhead (J)',
            ])

            for name in sorted(self.metrics.keys()):
                metrics = self.metrics[name]
                if metrics.architectural_breakdown:
                    b = metrics.architectural_breakdown
                    writer.writerow([
                        name,
                        f"{b.compute_overhead:.9f}",
                        f"{b.memory_overhead:.9f}",
                        f"{b.control_overhead:.9f}",
                        f"{b.total_overhead:.9f}",
                    ])
                else:
                    writer.writerow([name, 'N/A', 'N/A', 'N/A', 'N/A'])

        return output.getvalue()

    def export_html(self) -> str:
        """
        Export comparison data as interactive HTML with charts.

        Creates a standalone HTML file with:
        - Bar charts for energy, latency, throughput comparison
        - Pie charts for energy breakdown
        - Bottleneck distribution charts
        - Interactive tooltips and legends

        Uses Chart.js for visualization (loaded from CDN).

        Returns:
            HTML string with embedded charts
        """
        # Prepare data for charts
        arch_names = sorted(self.metrics.keys())
        energies = [self.metrics[name].total_energy_j * 1000 for name in arch_names]  # mJ
        latencies = [self.metrics[name].total_latency_s * 1000 for name in arch_names]  # ms
        throughputs = [self.metrics[name].throughput_inferences_per_sec for name in arch_names]
        memories = [self.metrics[name].peak_memory_bytes / 1e6 for name in arch_names]  # MB

        # Energy breakdown for first architecture (as example)
        first_arch = arch_names[0]
        first_metrics = self.metrics[first_arch]
        compute_pct = (first_metrics.compute_energy_j / first_metrics.total_energy_j * 100) if first_metrics.total_energy_j > 0 else 0
        memory_pct = (first_metrics.memory_energy_j / first_metrics.total_energy_j * 100) if first_metrics.total_energy_j > 0 else 0
        overhead_pct = (first_metrics.architectural_overhead_j / first_metrics.total_energy_j * 100) if first_metrics.total_energy_j > 0 else 0

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Architecture Comparison: {self.model_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .metadata {{
            color: #666;
            font-size: 14px;
        }}
        .recommendations {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
        }}
        .recommendations h2 {{
            margin: 0 0 15px 0;
            color: #1976D2;
        }}
        .rec-item {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        h2 {{
            margin: 0 0 20px 0;
            color: #333;
            font-size: 18px;
        }}
        .insights {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .insights ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .insights li {{
            margin: 8px 0;
            color: #555;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .winner {{
            color: #4CAF50;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Architecture Comparison: {self.model_name}</h1>
        <div class="metadata">
            Batch Size: {self.batch_size} | Precision: {self.precision.value}
        </div>
    </div>

    <div class="recommendations">
        <h2>🎯 Recommendations</h2>
        <div class="rec-item">✓ Best for Energy: <strong>{self.summary.energy_winner}</strong></div>
        <div class="rec-item">✓ Best for Latency: <strong>{self.summary.latency_winner}</strong></div>
        <div class="rec-item">✓ Best for Throughput: <strong>{self.summary.throughput_winner}</strong></div>
        <div class="rec-item">✓ Best Balance: <strong>{self.summary.balance_winner}</strong></div>
    </div>

    <div class="chart-row">
        <div class="chart-container">
            <h2>Energy Consumption (mJ)</h2>
            <canvas id="energyChart"></canvas>
        </div>
        <div class="chart-container">
            <h2>Latency (ms)</h2>
            <canvas id="latencyChart"></canvas>
        </div>
    </div>

    <div class="chart-row">
        <div class="chart-container">
            <h2>Throughput (FPS)</h2>
            <canvas id="throughputChart"></canvas>
        </div>
        <div class="chart-container">
            <h2>Peak Memory (MB)</h2>
            <canvas id="memoryChart"></canvas>
        </div>
    </div>

    <div class="chart-container">
        <h2>Energy Breakdown: {first_arch}</h2>
        <canvas id="energyBreakdownChart"></canvas>
    </div>

    <div class="insights">
        <h2>Key Insights</h2>
        <ul>
"""

        for insight in self.summary.insights:
            html += f"            <li>{insight}</li>\n"

        html += """        </ul>
    </div>

    <script>
        // Color schemes
        const colors = [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 206, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)',
        ];

        const borderColors = [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
            'rgba(255, 159, 64, 1)',
        ];

        // Energy Chart
        new Chart(document.getElementById('energyChart'), {
            type: 'bar',
            data: {
                labels: """ + json.dumps(arch_names) + """,
                datasets: [{
                    label: 'Energy (mJ)',
                    data: """ + json.dumps(energies) + """,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Energy (mJ)' }
                    }
                }
            }
        });

        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {
            type: 'bar',
            data: {
                labels: """ + json.dumps(arch_names) + """,
                datasets: [{
                    label: 'Latency (ms)',
                    data: """ + json.dumps(latencies) + """,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Latency (ms)' }
                    }
                }
            }
        });

        // Throughput Chart
        new Chart(document.getElementById('throughputChart'), {
            type: 'bar',
            data: {
                labels: """ + json.dumps(arch_names) + """,
                datasets: [{
                    label: 'Throughput (FPS)',
                    data: """ + json.dumps(throughputs) + """,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Throughput (FPS)' }
                    }
                }
            }
        });

        // Memory Chart
        new Chart(document.getElementById('memoryChart'), {
            type: 'bar',
            data: {
                labels: """ + json.dumps(arch_names) + """,
                datasets: [{
                    label: 'Peak Memory (MB)',
                    data: """ + json.dumps(memories) + """,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Peak Memory (MB)' }
                    }
                }
            }
        });

        // Energy Breakdown Pie Chart
        new Chart(document.getElementById('energyBreakdownChart'), {
            type: 'pie',
            data: {
                labels: ['Compute', 'Memory', 'Architectural Overhead'],
                datasets: [{
                    data: [""" + f"{compute_pct:.2f}, {memory_pct:.2f}, {overhead_pct:.2f}" + """],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>"""

        return html
