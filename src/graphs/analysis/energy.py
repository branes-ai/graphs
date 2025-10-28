"""
Energy Analysis

Analyzes computational graphs for energy consumption with hardware-aware modeling.

Energy Components:
- Compute energy: energy_per_flop × FLOPs
- Memory energy: energy_per_byte × bytes_transferred
- Static energy: idle_power × latency (leakage, always-on circuits)

Energy Efficiency:
- Efficiency = useful_energy / total_energy
- Wasted energy = energy spent on idle resources
- Utilization directly impacts efficiency
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

from ..ir.structures import SubgraphDescriptor, PartitionReport
from ..hardware.resource_model import HardwareResourceModel, Precision


@dataclass
class EnergyDescriptor:
    """
    Energy consumption breakdown for a single subgraph.

    Tracks compute energy, memory energy, and static energy
    to identify optimization opportunities.
    """

    # Identity
    subgraph_id: str
    subgraph_name: str

    # Component-level energy (Joules)
    compute_energy_j: float  # Energy for FLOPs
    memory_energy_j: float   # Energy for data movement
    static_energy_j: float   # Idle power during execution
    total_energy_j: float

    # Operations
    compute_ops: int  # FLOPs
    bytes_transferred: int  # Total bytes

    # Latency (needed for static energy)
    latency_s: float

    # Efficiency metrics
    utilization: float = 0.0  # Fraction of hardware utilized
    wasted_energy_j: float = 0.0  # Energy on idle resources
    efficiency: float = 0.0  # useful / total

    # Comparison
    peak_energy_j: float = 0.0  # If 100% utilized

    # Explanation
    explanation: str = ""

    def __str__(self) -> str:
        """Short summary"""
        return (f"Energy({self.subgraph_name}: "
                f"{self.total_energy_j * 1e6:.2f}μJ, "
                f"eff={self.efficiency * 100:.1f}%)")

    def format_summary(self) -> str:
        """Detailed multi-line summary"""
        lines = []
        lines.append(f"Subgraph: {self.subgraph_name}")
        lines.append(f"  Total Energy: {self.total_energy_j * 1e6:.2f} μJ")
        lines.append(f"    Compute:  {self.compute_energy_j * 1e6:.2f} μJ ({self.compute_energy_j / self.total_energy_j * 100:.1f}%)")
        lines.append(f"    Memory:   {self.memory_energy_j * 1e6:.2f} μJ ({self.memory_energy_j / self.total_energy_j * 100:.1f}%)")
        lines.append(f"    Static:   {self.static_energy_j * 1e6:.2f} μJ ({self.static_energy_j / self.total_energy_j * 100:.1f}%)")
        lines.append(f"  Efficiency: {self.efficiency * 100:.1f}%")
        lines.append(f"  Utilization: {self.utilization * 100:.1f}%")
        if self.wasted_energy_j > 0:
            lines.append(f"  Wasted Energy: {self.wasted_energy_j * 1e6:.2f} μJ")
        return "\n".join(lines)


@dataclass
class EnergyReport:
    """
    Complete energy analysis for a partition.

    Contains per-subgraph energy descriptors and summary statistics.
    """

    # Total energy
    total_energy_j: float  # Joules
    total_energy_mj: float  # Millijoules
    energy_per_inference_j: float  # Same as total for single inference

    # Breakdown (Joules)
    compute_energy_j: float
    memory_energy_j: float
    static_energy_j: float

    # Efficiency
    average_efficiency: float = 0.0
    average_utilization: float = 0.0
    wasted_energy_j: float = 0.0
    wasted_energy_percent: float = 0.0

    # Power analysis (Watts)
    average_power_w: float = 0.0
    peak_power_w: float = 0.0
    total_latency_s: float = 0.0

    # Per-subgraph energy
    energy_descriptors: List[EnergyDescriptor] = field(default_factory=list)

    # Top contributors
    top_energy_consumers: List[Tuple[str, float]] = field(default_factory=list)  # (name, energy_j)

    # Optimization suggestions
    optimization_opportunities: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Short summary"""
        return (f"EnergyReport(total={self.total_energy_mj:.2f}mJ, "
                f"power={self.average_power_w:.2f}W, "
                f"eff={self.average_efficiency * 100:.1f}%)")

    def format_report(self, show_per_subgraph: bool = False, max_subgraphs: int = 10) -> str:
        """Generate human-readable report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENERGY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Total energy
        lines.append("Total Energy:")
        lines.append(f"  Total:   {self.total_energy_mj:.2f} mJ ({self.total_energy_j * 1e6:.0f} μJ)")
        lines.append(f"  Compute: {self.compute_energy_j * 1e6:.0f} μJ ({self.compute_energy_j / self.total_energy_j * 100:.1f}%)")
        lines.append(f"  Memory:  {self.memory_energy_j * 1e6:.0f} μJ ({self.memory_energy_j / self.total_energy_j * 100:.1f}%)")
        lines.append(f"  Static:  {self.static_energy_j * 1e6:.0f} μJ ({self.static_energy_j / self.total_energy_j * 100:.1f}%)")
        lines.append("")

        # Power analysis
        lines.append("Power Analysis:")
        lines.append(f"  Average Power: {self.average_power_w:.2f} W")
        if self.peak_power_w > 0:
            lines.append(f"  Peak Power:    {self.peak_power_w:.2f} W")
        lines.append(f"  Total Latency: {self.total_latency_s * 1e3:.2f} ms")
        lines.append("")

        # Efficiency
        lines.append("Efficiency:")
        lines.append(f"  Average Efficiency:   {self.average_efficiency * 100:.1f}%")
        lines.append(f"  Average Utilization:  {self.average_utilization * 100:.1f}%")
        lines.append(f"  Wasted Energy: {self.wasted_energy_j * 1e6:.0f} μJ ({self.wasted_energy_percent:.1f}%)")
        lines.append("")

        # Top consumers
        if self.top_energy_consumers:
            lines.append("Top Energy Consumers:")
            for i, (name, energy) in enumerate(self.top_energy_consumers[:5], 1):
                pct = energy / self.total_energy_j * 100 if self.total_energy_j > 0 else 0
                lines.append(f"  {i}. {name}: {energy * 1e6:.0f} μJ ({pct:.1f}%)")
            lines.append("")

        # Optimization opportunities
        if self.optimization_opportunities:
            lines.append("Optimization Opportunities:")
            for opp in self.optimization_opportunities:
                lines.append(f"  {opp}")
            lines.append("")

        # Per-subgraph details
        if show_per_subgraph and self.energy_descriptors:
            lines.append("=" * 80)
            lines.append(f"TOP {max_subgraphs} SUBGRAPHS BY ENERGY")
            lines.append("=" * 80)
            lines.append("")

            # Sort by energy
            sorted_descriptors = sorted(
                self.energy_descriptors,
                key=lambda d: d.total_energy_j,
                reverse=True
            )[:max_subgraphs]

            for i, desc in enumerate(sorted_descriptors, 1):
                lines.append(f"{i}. {desc.subgraph_name}")
                lines.append(f"   Energy: {desc.total_energy_j * 1e6:.0f} μJ")
                lines.append(f"   Breakdown: Compute {desc.compute_energy_j / desc.total_energy_j * 100:.1f}%, "
                           f"Memory {desc.memory_energy_j / desc.total_energy_j * 100:.1f}%, "
                           f"Static {desc.static_energy_j / desc.total_energy_j * 100:.1f}%")
                lines.append(f"   Efficiency: {desc.efficiency * 100:.1f}%")
                lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


class EnergyAnalyzer:
    """
    Analyzes computational graphs for energy consumption.

    Algorithm:
    1. For each subgraph, compute:
       - Compute energy = FLOPs × energy_per_flop
       - Memory energy = bytes × energy_per_byte
       - Static energy = idle_power × latency
    2. Calculate efficiency based on utilization
    3. Identify optimization opportunities
    4. Generate energy report

    Energy Model:
    - Compute energy: Depends on operation type and precision
    - Memory energy: Depends on memory hierarchy (DRAM, cache)
    - Static energy: Leakage power, always-on circuits (significant on modern hardware)
    """

    # Idle power fractions (from silicon leakage and always-on circuits)
    IDLE_POWER_FRACTION = 0.3  # GPUs: ~30% of TDP at idle
    CPU_IDLE_POWER_FRACTION = 0.1  # CPUs: ~10% of TDP at idle

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        precision: Precision = Precision.FP32,
        latency_s: Optional[float] = None
    ):
        """
        Initialize energy analyzer.

        Args:
            resource_model: Hardware specifications
            precision: Precision to use for energy calculations
            latency_s: Optional total latency (if not provided, estimated from roofline)
        """
        self.resource_model = resource_model
        self.precision = precision
        self.latency_s = latency_s

        # Get energy coefficients
        self.energy_per_flop = resource_model.energy_per_flop_fp32

        # Scale for precision
        if precision in resource_model.energy_scaling:
            energy_scale = resource_model.energy_scaling[precision]
            self.energy_per_flop *= energy_scale

        self.energy_per_byte = resource_model.energy_per_byte

        # Calculate TDP (thermal design power)
        self._estimate_tdp()

    def _estimate_tdp(self):
        """Estimate TDP from hardware specs"""
        if hasattr(self.resource_model, 'thermal_operating_points') and \
           self.resource_model.thermal_operating_points:
            # Use default thermal profile
            default_profile_name = getattr(self.resource_model, 'default_thermal_profile', None)
            if default_profile_name and default_profile_name in self.resource_model.thermal_operating_points:
                profile = self.resource_model.thermal_operating_points[default_profile_name]
                self.tdp_watts = profile.tdp_watts
            else:
                # Use first available profile
                first_profile = next(iter(self.resource_model.thermal_operating_points.values()))
                self.tdp_watts = first_profile.tdp_watts
        else:
            # Estimate from peak power
            # For GPUs: Peak power ~= 2× average, TDP ~= 1.5× average
            # Rough estimate: TDP = peak_FLOPs × energy_per_flop × 2
            peak_flops = self.resource_model.precision_profiles[self.precision].peak_ops_per_sec
            peak_dynamic_power = peak_flops * self.energy_per_flop
            self.tdp_watts = peak_dynamic_power * 2.0

        # Idle power
        if self.resource_model.hardware_type.name == 'CPU':
            self.idle_power_watts = self.tdp_watts * self.CPU_IDLE_POWER_FRACTION
        else:
            self.idle_power_watts = self.tdp_watts * self.IDLE_POWER_FRACTION

    def analyze(
        self,
        subgraphs: List[SubgraphDescriptor],
        partition_report: Optional[PartitionReport] = None,
        latencies: Optional[List[float]] = None
    ) -> EnergyReport:
        """
        Analyze energy for all subgraphs.

        Args:
            subgraphs: List of computational subgraphs
            partition_report: Optional partition report
            latencies: Optional per-subgraph latencies (from roofline analysis)

        Returns:
            EnergyReport with energy analysis
        """

        energy_descriptors = []

        # If latencies not provided, estimate simple latency
        if latencies is None:
            latencies = self._estimate_latencies(subgraphs)

        # Analyze each subgraph
        for i, sg in enumerate(subgraphs):
            latency = latencies[i] if i < len(latencies) else 0.001  # default 1ms
            descriptor = self._analyze_subgraph(sg, latency)
            energy_descriptors.append(descriptor)

        # Aggregate statistics
        total_energy = sum(d.total_energy_j for d in energy_descriptors)
        compute_energy = sum(d.compute_energy_j for d in energy_descriptors)
        memory_energy = sum(d.memory_energy_j for d in energy_descriptors)
        static_energy = sum(d.static_energy_j for d in energy_descriptors)
        wasted_energy = sum(d.wasted_energy_j for d in energy_descriptors)

        # Efficiency
        avg_efficiency = sum(d.efficiency for d in energy_descriptors) / len(energy_descriptors) if energy_descriptors else 0.0
        avg_utilization = sum(d.utilization for d in energy_descriptors) / len(energy_descriptors) if energy_descriptors else 0.0
        wasted_percent = wasted_energy / total_energy * 100 if total_energy > 0 else 0.0

        # Power analysis
        total_latency = sum(latencies)
        average_power = total_energy / total_latency if total_latency > 0 else 0.0

        # Peak power (max instantaneous power)
        peak_power = max(
            d.total_energy_j / d.latency_s if d.latency_s > 0 else 0.0
            for d in energy_descriptors
        ) if energy_descriptors else 0.0

        # Top energy consumers
        top_consumers = sorted(
            [(d.subgraph_name, d.total_energy_j) for d in energy_descriptors],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Optimization opportunities
        optimizations = self._identify_optimizations(energy_descriptors, total_energy)

        return EnergyReport(
            total_energy_j=total_energy,
            total_energy_mj=total_energy * 1e3,
            energy_per_inference_j=total_energy,
            compute_energy_j=compute_energy,
            memory_energy_j=memory_energy,
            static_energy_j=static_energy,
            average_efficiency=avg_efficiency,
            average_utilization=avg_utilization,
            wasted_energy_j=wasted_energy,
            wasted_energy_percent=wasted_percent,
            average_power_w=average_power,
            peak_power_w=peak_power,
            total_latency_s=total_latency,
            energy_descriptors=energy_descriptors,
            top_energy_consumers=top_consumers,
            optimization_opportunities=optimizations,
        )

    def _analyze_subgraph(self, sg: SubgraphDescriptor, latency: float) -> EnergyDescriptor:
        """Analyze energy for a single subgraph"""

        # Compute energy = FLOPs × energy_per_flop
        compute_energy = sg.flops * self.energy_per_flop

        # Memory energy = bytes × energy_per_byte
        total_bytes = sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
        memory_energy = total_bytes * self.energy_per_byte

        # Static energy = idle_power × latency
        static_energy = self.idle_power_watts * latency

        # Total energy
        dynamic_energy = compute_energy + memory_energy
        total_energy = dynamic_energy + static_energy

        # Utilization (from parallelism descriptor if available)
        utilization = 1.0
        if sg.parallelism and sg.parallelism.total_threads > 0:
            # Estimate utilization based on thread count
            max_threads = self.resource_model.compute_units * self.resource_model.threads_per_unit
            utilization = min(1.0, sg.parallelism.total_threads / max_threads)

        # Wasted energy = energy on idle resources
        # If utilization is 50%, then 50% of static energy is wasted
        wasted_energy = static_energy * (1.0 - utilization)

        # Efficiency = dynamic_energy / total_energy
        # (fraction of energy doing useful work vs leakage)
        efficiency = dynamic_energy / total_energy if total_energy > 0 else 0.0

        # Peak energy (if 100% utilized)
        peak_energy = dynamic_energy / utilization if utilization > 0 else dynamic_energy

        # Explanation
        explanation = self._explain_energy(sg, compute_energy, memory_energy, static_energy)

        return EnergyDescriptor(
            subgraph_id=sg.node_id,
            subgraph_name=sg.node_name,
            compute_energy_j=compute_energy,
            memory_energy_j=memory_energy,
            static_energy_j=static_energy,
            total_energy_j=total_energy,
            compute_ops=sg.flops,
            bytes_transferred=total_bytes,
            latency_s=latency,
            utilization=utilization,
            wasted_energy_j=wasted_energy,
            efficiency=efficiency,
            peak_energy_j=peak_energy,
            explanation=explanation,
        )

    def _estimate_latencies(self, subgraphs: List[SubgraphDescriptor]) -> List[float]:
        """Estimate latencies if not provided (simple roofline)"""
        latencies = []
        peak_flops = self.resource_model.precision_profiles[self.precision].peak_ops_per_sec
        peak_bandwidth = self.resource_model.peak_bandwidth

        for sg in subgraphs:
            # Simple roofline: max(compute_time, memory_time)
            compute_time = sg.flops / peak_flops if peak_flops > 0 else 0.0
            total_bytes = sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
            memory_time = total_bytes / peak_bandwidth if peak_bandwidth > 0 else 0.0
            latency = max(compute_time, memory_time)

            # Add small overhead
            latency += 5e-6  # 5 microseconds
            latencies.append(latency)

        return latencies

    def _explain_energy(
        self,
        sg: SubgraphDescriptor,
        compute_energy: float,
        memory_energy: float,
        static_energy: float
    ) -> str:
        """Generate human-readable explanation"""

        total = compute_energy + memory_energy + static_energy
        if total == 0:
            return f"{sg.node_name}: No energy consumption"

        # Determine dominant component
        if compute_energy > memory_energy and compute_energy > static_energy:
            dominant = "compute"
            pct = compute_energy / total * 100
        elif memory_energy > static_energy:
            dominant = "memory"
            pct = memory_energy / total * 100
        else:
            dominant = "static (leakage)"
            pct = static_energy / total * 100

        return (f"{sg.node_name}: {total * 1e6:.1f}μJ total, "
                f"{dominant} dominant ({pct:.1f}%)")

    def _identify_optimizations(
        self,
        descriptors: List[EnergyDescriptor],
        total_energy: float
    ) -> List[str]:
        """Identify energy optimization opportunities"""

        optimizations = []

        # High static energy percentage
        total_static = sum(d.static_energy_j for d in descriptors)
        static_pct = total_static / total_energy * 100 if total_energy > 0 else 0

        if static_pct > 40:
            optimizations.append(
                f"✓ Reduce latency: {static_pct:.0f}% energy is static (leakage). "
                f"Faster execution saves {total_static * 1e6:.0f}μJ"
            )

        # Low average utilization
        avg_util = sum(d.utilization for d in descriptors) / len(descriptors) if descriptors else 0
        if avg_util < 0.5:
            total_wasted = sum(d.wasted_energy_j for d in descriptors)
            optimizations.append(
                f"✓ Improve utilization: {avg_util * 100:.0f}% average utilization. "
                f"Better batching/fusion could save {total_wasted * 1e6:.0f}μJ"
            )

        # Memory-heavy operations
        total_memory = sum(d.memory_energy_j for d in descriptors)
        memory_pct = total_memory / total_energy * 100 if total_energy > 0 else 0

        if memory_pct > 40:
            optimizations.append(
                f"✓ Reduce data movement: {memory_pct:.0f}% energy is memory transfers. "
                f"Cache reuse could save {total_memory * 0.5 * 1e6:.0f}μJ"
            )

        # Quantization opportunity
        if self.precision == Precision.FP32:
            # FP32 → INT8 typically saves 4× energy for compute
            compute_savings = sum(d.compute_energy_j for d in descriptors) * 0.75
            optimizations.append(
                f"✓ Quantization (FP32→INT8): Could save ~{compute_savings * 1e6:.0f}μJ in compute energy"
            )

        return optimizations
