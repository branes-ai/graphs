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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import json
import csv
import io

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

    # Ratios (all relative to baseline, which is typically GPU or CPU)
    baseline: str
    energy_ratios: Dict[str, float]
    latency_ratios: Dict[str, float]

    # Insights
    insights: List[str] = field(default_factory=list)


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
        precision: Precision = Precision.FP32
    ):
        """
        Initialize comparator.

        Args:
            model_name: Model to analyze (e.g., 'resnet18')
            architectures: Dict mapping architecture name to HardwareMapper
            batch_size: Batch size for analysis
            precision: Numerical precision
        """
        self.model_name = model_name
        self.architectures = architectures
        self.batch_size = batch_size
        self.precision = precision

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

        # Load model once (shared across all architectures)
        # Use torchvision models
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

            result = analyzer.analyze_model_with_custom_hardware(
                model=model,
                input_tensor=input_tensor,
                model_name=self.model_name,
                hardware_mapper=mapper,
                precision=self.precision
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

                architectural_breakdown = mapper.resource_model.architecture_energy_model.compute_architectural_energy(
                    ops=total_ops,
                    bytes_transferred=total_bytes,
                    compute_energy_baseline=compute_baseline,
                    memory_energy_baseline=memory_baseline,
                    execution_context=execution_context
                )

                architectural_overhead = architectural_breakdown.total_overhead

            except Exception as e:
                print(f"Warning: Failed to compute architectural breakdown for {name}: {e}")

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
            architectural_breakdown=architectural_breakdown
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

        # Choose baseline (typically GPU if present, else first architecture)
        baseline = 'GPU' if 'GPU' in self.metrics else list(self.metrics.keys())[0]
        baseline_energy = self.metrics[baseline].total_energy_j
        baseline_latency = self.metrics[baseline].total_latency_s

        # Calculate ratios
        energy_ratios = {
            name: metrics.total_energy_j / baseline_energy
            for name, metrics in self.metrics.items()
        }

        latency_ratios = {
            name: metrics.total_latency_s / baseline_latency
            for name, metrics in self.metrics.items()
        }

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
            baseline=baseline,
            energy_ratios=energy_ratios,
            latency_ratios=latency_ratios,
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
        lines.append(f"  Best Balance:         {self.summary.balance_winner}")
        lines.append("")

        # Comparison table
        lines.append(f"{'Architecture':<15} {'Energy':<12} {'Latency':<12} {'Memory':<12} {'Util%':<8} {'vs ' + self.summary.baseline:<12}")
        lines.append("-" * 80)

        for name in sorted(self.metrics.keys()):
            metrics = self.metrics[name]
            energy_str = self._format_energy(metrics.total_energy_j)
            latency_str = self._format_time(metrics.total_latency_s)
            memory_str = self._format_bytes(metrics.peak_memory_bytes)
            util_str = f"{metrics.utilization*100:.1f}%"

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
                f"{name:<15} {energy_str:<12} {latency_str:<12} {memory_str:<12} "
                f"{util_str:<8} {ratio_str:<12}{style}"
            )

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
            'Peak Memory (MB)',
            'Utilization (%)',
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
                f"{metrics.peak_memory_bytes / 1e6:.2f}",
                f"{metrics.utilization * 100:.1f}",
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
