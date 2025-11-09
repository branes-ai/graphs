"""
Unified Analysis Framework

Single orchestrator for comprehensive model analysis, coordinating all Phase 3 analyzers
with correct data dependencies and validation.

This module provides:
- UnifiedAnalyzer: Orchestrates all Phase 3 analyzers in correct order
- UnifiedAnalysisResult: Single data structure containing all analysis results
- AnalysisConfig: Configurable analysis options

Usage:
    from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
    from graphs.hardware.resource_model import Precision

    analyzer = UnifiedAnalyzer(verbose=True)
    result = analyzer.analyze_model(
        model_name='resnet18',
        hardware_name='H100',
        batch_size=1,
        precision=Precision.FP32
    )

    print(f"Latency: {result.total_latency_ms:.2f} ms")
    print(f"Energy: {result.total_energy_mj:.2f} mJ")
    print(f"Peak Memory: {result.peak_memory_mb:.2f} MB")
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import warnings

import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models

# Graph structures
from graphs.ir.structures import PartitionReport, SubgraphDescriptor

# Partitioning
from graphs.transform.partitioning import GraphPartitioner
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner

# Phase 3 Analyzers
from graphs.analysis.roofline import RooflineAnalyzer, RooflineReport
from graphs.analysis.energy import EnergyAnalyzer, EnergyReport
from graphs.analysis.memory import MemoryEstimator, MemoryReport
from graphs.analysis.concurrency import ConcurrencyAnalyzer
from graphs.ir.structures import ConcurrencyDescriptor

# Phase 2: Operator-level EDP (NEW)
# Note: Import moved to method to avoid circular dependency
# from graphs.analysis.architecture_comparator import (...)

# Hardware models
from graphs.hardware.resource_model import (
    Precision,
    HardwareResourceModel,
    GraphHardwareAllocation,
    HardwareAllocation,
)

# Hardware mapper imports
from graphs.hardware.mappers.gpu import (
    create_h100_mapper,
    create_a100_mapper,
    create_v100_mapper,
    create_jetson_orin_agx_mapper,
    create_jetson_orin_nano_mapper,
    create_jetson_thor_mapper,
)
from graphs.hardware.mappers.accelerators.tpu import (
    create_tpu_v4_mapper,
    create_coral_edge_tpu_mapper,
)
from graphs.hardware.mappers.accelerators.kpu import (
    create_kpu_t64_mapper,
    create_kpu_t256_mapper,
    create_kpu_t768_mapper,
)
from graphs.hardware.mappers.cpu import (
    create_amd_epyc_9754_mapper,
    create_intel_xeon_platinum_8490h_mapper,
    create_ampere_ampereone_192_mapper,
    create_i7_12700k_mapper,
    create_amd_cpu_mapper,
)
from graphs.hardware.mappers.dsp import (
    create_qrb5165_mapper,
    create_ti_tda4vm_mapper,
    create_qualcomm_snapdragon_ride_mapper,
    create_qualcomm_sa8775p_mapper,
    create_ti_tda4vh_mapper,
    create_ti_tda4vl_mapper,
    create_ti_tda4al_mapper,
)
from graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper
from graphs.hardware.mappers.accelerators.cgra import create_plasticine_v2_mapper
from graphs.hardware.mappers.accelerators.hailo import (
    create_hailo10h_mapper,
    create_hailo8_mapper,
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Configuration for unified analysis.

    Allows fine-grained control over which analyses to run,
    useful for performance optimization.
    """
    # Which analyses to run
    run_roofline: bool = True
    run_energy: bool = True
    run_memory: bool = True
    run_concurrency: bool = False  # Optional, expensive
    run_hardware_mapping: bool = True  # NEW: Use hardware mapper for allocation

    # Partitioning strategy
    use_fusion_partitioning: bool = False  # Reserved for future use (FusionBasedPartitioner needs integration work)

    # Analysis options
    estimate_checkpointing_savings: bool = False
    estimate_quantization_savings: bool = False
    detailed_memory_timeline: bool = True

    # Power management (Phase 2)
    power_gating_enabled: bool = False  # NEW: Model power gating of unused units

    # Operator-level EDP (Phase 2)
    run_operator_edp: bool = True  # NEW: Generate operator-level EDP breakdown

    # Validation
    validate_consistency: bool = True  # Check reports agree with each other


@dataclass
class UnifiedAnalysisResult:
    """
    Complete analysis results from all Phase 3 analyzers.

    This is the single source of truth for analysis results,
    ensuring consistency across all tools and use cases.
    """
    # Model metadata
    model_name: str
    display_name: str
    batch_size: int
    precision: Precision

    # Hardware metadata
    hardware_name: str
    hardware_display_name: str
    hardware: HardwareResourceModel

    # Graph structures
    fx_graph: GraphModule
    partition_report: PartitionReport

    # Phase 2: Hardware mapping (NEW)
    hardware_allocation: Optional['GraphHardwareAllocation'] = None

    # Phase 3 analysis results
    roofline_report: Optional[RooflineReport] = None
    energy_report: Optional[EnergyReport] = None
    memory_report: Optional[MemoryReport] = None
    concurrency_report: Optional[ConcurrencyDescriptor] = None

    # Phase 2: Operator-level EDP breakdown (NEW)
    # Using Any to avoid circular import (actual types: SubgraphEDPDescriptor, OperatorEDPDescriptor)
    subgraph_edp_breakdown: Optional[List[Any]] = None
    operator_edp_breakdown: Optional[List[Any]] = None

    # Derived metrics (computed from above)
    total_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    total_energy_mj: float = 0.0
    energy_per_inference_mj: float = 0.0
    peak_memory_mb: float = 0.0
    average_utilization_pct: float = 0.0

    # Validation warnings
    validation_warnings: List[str] = field(default_factory=list)

    # Timestamps
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_executive_summary(self) -> Dict[str, Any]:
        """
        Generate executive summary dict for quick overview.

        Returns:
            Dict with key metrics and recommendations
        """
        summary = {
            'model': self.display_name,
            'hardware': self.hardware_display_name,
            'batch_size': self.batch_size,
            'precision': self.precision.name,
            'performance': {
                'latency_ms': round(self.total_latency_ms, 2),
                'throughput_fps': round(self.throughput_fps, 1),
                'utilization_pct': round(self.average_utilization_pct, 1),
            },
            'energy': {
                'total_mj': round(self.total_energy_mj, 2),
                'per_inference_mj': round(self.energy_per_inference_mj, 2),
            },
            'memory': {
                'peak_mb': round(self.peak_memory_mb, 1),
            },
            'recommendations': self._generate_recommendations(),
        }

        if self.validation_warnings:
            summary['warnings'] = self.validation_warnings

        return summary

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []

        # Batch size recommendation
        if self.batch_size == 1 and self.energy_report:
            static_pct = (self.energy_report.static_energy_j /
                         (self.energy_report.compute_energy_j +
                          self.energy_report.memory_energy_j +
                          self.energy_report.static_energy_j)) * 100
            if static_pct > 50:
                recommendations.append(
                    f"Increase batch size to amortize static energy ({static_pct:.0f}% overhead)"
                )

        # Precision recommendation
        if self.precision == Precision.FP32 and self.average_utilization_pct < 20:
            recommendations.append(
                "Consider FP16 for 2× speedup with minimal accuracy loss"
            )

        # Bottleneck recommendation
        if self.roofline_report:
            memory_bound_count = sum(1 for lat in self.roofline_report.latencies
                                    if lat.bottleneck == 'memory')
            if memory_bound_count > len(self.roofline_report.latencies) / 2:
                recommendations.append(
                    "Optimize data layout - majority of operations are memory-bound"
                )

        # Memory recommendation
        if self.memory_report and not self.memory_report.fits_in_l2_cache:
            recommendations.append(
                "Consider tiling or model partitioning to improve cache locality"
            )

        return recommendations

    def validate(self) -> List[str]:
        """
        Validate consistency between reports.

        Returns:
            List of warning messages (empty if all consistent)
        """
        warnings = []

        # Check that latency from roofline matches energy latency
        if self.roofline_report and self.energy_report:
            roofline_latency = sum(lat.actual_latency for lat in self.roofline_report.latencies)
            # Energy report doesn't store total latency directly, so we check descriptors
            if len(self.energy_report.energy_descriptors) != len(self.roofline_report.latencies):
                warnings.append(
                    f"Mismatch in subgraph count: roofline={len(self.roofline_report.latencies)}, "
                    f"energy={len(self.energy_report.energy_descriptors)}"
                )

        # Check that partition report subgraph count matches
        if self.partition_report:
            expected_count = len(self.partition_report.subgraphs)
            if self.roofline_report and len(self.roofline_report.latencies) != expected_count:
                warnings.append(
                    f"Roofline report has {len(self.roofline_report.latencies)} entries, "
                    f"expected {expected_count}"
                )
            if self.energy_report and len(self.energy_report.energy_descriptors) != expected_count:
                warnings.append(
                    f"Energy report has {len(self.energy_report.energy_descriptors)} entries, "
                    f"expected {expected_count}"
                )

        # Check memory constraints for hardware without external DRAM (e.g., Hailo-8)
        if self.hardware.main_memory == 0 and self.fx_graph is not None:
            available_memory_mb = self.hardware.l2_cache_total / (1024 ** 2)  # Bytes to MB
            # Calculate model size based on precision
            model_params = sum(p.numel() for p in self.fx_graph.parameters())
            bytes_per_param = {
                Precision.FP32: 4,
                Precision.FP16: 2,
                Precision.BF16: 2,
                Precision.INT8: 1,
                Precision.INT4: 0.5,
            }.get(self.precision, 4)
            model_size_mb = (model_params * bytes_per_param) / (1024 ** 2)

            if model_size_mb > available_memory_mb:
                warnings.append(
                    f"MEMORY CONSTRAINT VIOLATION: Model size ({model_size_mb:.1f} MB) "
                    f"exceeds available on-chip memory ({available_memory_mb:.1f} MB). "
                    f"Hardware {self.hardware_name} requires full model to fit on-chip for "
                    f"standalone automotive deployment (no external DRAM available)."
                )

        return warnings


# =============================================================================
# Unified Analyzer
# =============================================================================

class UnifiedAnalyzer:
    """
    Unified orchestrator for comprehensive model analysis.

    Coordinates Phase 3 analyzers with correct data dependencies:
    1. Trace model with FX + shape propagation
    2. Partition graph into subgraphs
    3. Run roofline analysis (latency, bottlenecks)
    4. Run energy analysis (uses latencies from roofline)
    5. Run memory analysis (peak memory, timeline)
    6. Optionally run concurrency analysis

    Usage:
        analyzer = UnifiedAnalyzer()
        result = analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=AnalysisConfig()
        )
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize analyzer.

        Args:
            verbose: Print progress messages during analysis
        """
        self.verbose = verbose

    def analyze_model(
        self,
        model_name: str,
        hardware_name: str,
        batch_size: int = 1,
        precision: Precision = Precision.FP32,
        config: Optional[AnalysisConfig] = None,
    ) -> UnifiedAnalysisResult:
        """
        Run comprehensive analysis on a model.

        This is the main entry point. It orchestrates all sub-analyses
        in the correct order with proper data dependencies.

        Args:
            model_name: Model to analyze (e.g., 'resnet18')
            hardware_name: Target hardware (e.g., 'H100')
            batch_size: Batch size for analysis
            precision: Precision to use (FP32, FP16, INT8)
            config: Analysis configuration (uses defaults if None)

        Returns:
            UnifiedAnalysisResult with all analysis reports

        Raises:
            ValueError: If model or hardware not found
            RuntimeError: If analysis fails
        """
        if config is None:
            config = AnalysisConfig()

        # Create model and hardware
        if self.verbose:
            print(f"Creating model: {model_name} (batch_size={batch_size})...")
        model, input_tensor, display_name = self._create_model(model_name, batch_size)

        if self.verbose:
            print(f"Creating hardware mapper: {hardware_name} (precision={precision.name})...")
        hardware_mapper = self._create_hardware_mapper(hardware_name)
        hardware = hardware_mapper.resource_model
        hardware_display_name = hardware.name

        # Analyze with custom model/hardware
        return self.analyze_model_with_custom_hardware(
            model=model,
            input_tensor=input_tensor,
            model_name=display_name,
            hardware_mapper=hardware_mapper,
            precision=precision,
            config=config,
            original_model_name=model_name,  # NEW: Keep original for ArchitectureComparator
        )

    def analyze_model_with_custom_hardware(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        model_name: str,
        hardware_mapper: Any,  # Any mapper type (GPUMapper, CPUMapper, etc.) with resource_model attribute
        precision: Precision = Precision.FP32,
        config: Optional[AnalysisConfig] = None,
        original_model_name: Optional[str] = None,  # NEW: Original model name for ArchitectureComparator
    ) -> UnifiedAnalysisResult:
        """
        Analyze with custom model/hardware (for advanced users).

        Args:
            model: PyTorch model to analyze
            input_tensor: Input tensor for shape propagation
            model_name: Display name for model
            hardware_mapper: Hardware mapper with resource model
            precision: Precision to use
            config: Analysis configuration

        Returns:
            UnifiedAnalysisResult with all analysis reports
        """
        if config is None:
            config = AnalysisConfig()

        hardware = hardware_mapper.resource_model
        batch_size = input_tensor.shape[0]

        # Step 1: Trace and partition
        if self.verbose:
            print("Tracing model with FX...")
        fx_graph, partition_report = self._trace_and_partition(model, input_tensor, config)

        if self.verbose:
            print(f"Partitioned into {len(partition_report.subgraphs)} subgraphs")
            print(f"Total FLOPs: {partition_report.total_flops / 1e9:.2f} GFLOPs")
            print(f"Total memory traffic: {partition_report.total_memory_traffic / 1e6:.2f} MB")

        # Initialize result
        result = UnifiedAnalysisResult(
            model_name=model_name,
            display_name=model_name,
            batch_size=batch_size,
            precision=precision,
            hardware_name=hardware.name,
            hardware_display_name=hardware.name,
            hardware=hardware,
            fx_graph=fx_graph,
            partition_report=partition_report,
        )

        # Step 2: Run hardware mapping (NEW)
        if config.run_hardware_mapping:
            if self.verbose:
                print("Running hardware mapping...")
            result.hardware_allocation = self._run_hardware_mapping(
                partition_report, hardware_mapper, batch_size, precision, fx_graph
            )
            if self.verbose:
                print(f"  Peak units allocated: {result.hardware_allocation.peak_compute_units_used}/{hardware.compute_units}")
                print(f"  Average utilization: {result.hardware_allocation.average_utilization * 100:.1f}%")

        # Step 3: Run roofline analysis
        if config.run_roofline:
            if self.verbose:
                print("Running roofline analysis...")
            result.roofline_report = self._run_roofline_analysis(
                partition_report, hardware, precision
            )

        # Step 4: Run energy analysis (depends on roofline for latencies, hardware mapping for allocation)
        if config.run_energy:
            if self.verbose:
                print("Running energy analysis...")
            result.energy_report = self._run_energy_analysis(
                partition_report, hardware, result.roofline_report,
                result.hardware_allocation, precision, config
            )

        # Step 4: Run memory analysis
        if config.run_memory:
            if self.verbose:
                print("Running memory analysis...")
            result.memory_report = self._run_memory_analysis(
                partition_report, hardware, config
            )

        # Step 5: Optionally run concurrency analysis
        if config.run_concurrency:
            if self.verbose:
                print("Running concurrency analysis...")
            result.concurrency_report = self._run_concurrency_analysis(partition_report)

        # Step 5.5: Optionally run operator-level EDP breakdown (NEW)
        if config.run_operator_edp and result.energy_report and result.roofline_report:
            if self.verbose:
                print("Running operator-level EDP breakdown...")
            # Use original_model_name for ArchitectureComparator (it needs lowercase 'resnet18', not 'ResNet-18')
            model_name_for_comparator = original_model_name if original_model_name else model_name
            subgraph_edps, operator_edps = self._run_operator_edp_analysis(
                model_name_for_comparator, hardware_mapper, batch_size, precision,
                model=model, input_tensor=input_tensor
            )
            result.subgraph_edp_breakdown = subgraph_edps
            result.operator_edp_breakdown = operator_edps
            if self.verbose:
                print(f"  Found {len(operator_edps)} operators across {len(subgraph_edps)} subgraphs")

        # Step 6: Compute derived metrics
        self._compute_derived_metrics(result)

        # Step 7: Validate consistency
        if config.validate_consistency:
            result.validation_warnings = self._validate_result(result)
            if result.validation_warnings and self.verbose:
                print("\nValidation warnings:")
                for warning in result.validation_warnings:
                    print(f"  ⚠ {warning}")

        if self.verbose:
            print("\nAnalysis complete!")

        return result

    # =========================================================================
    # Private Methods - Orchestration
    # =========================================================================

    def _trace_and_partition(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        config: AnalysisConfig
    ) -> Tuple[GraphModule, PartitionReport]:
        """
        Trace model using PyTorch Dynamo export (state-of-the-art) and partition into subgraphs.

        Dynamo is more reliable than symbolic_trace for complex models like YOLO, transformers, etc.
        """
        if self.verbose:
            print("  Tracing model with PyTorch Dynamo export...")

        # Set model to evaluation mode (CRITICAL for BatchNorm with batch=1)
        # This prevents "Expected more than 1 value per channel" errors in models like DeepLabV3
        model.eval()

        # Warm-up model (important for lazy initialization)
        with torch.no_grad():
            try:
                _ = model(input_tensor)
            except Exception as e:
                if self.verbose:
                    print(f"    Note: Warm-up failed ({e}), continuing anyway...")

        # Export with Dynamo (state-of-the-art tracing)
        try:
            exported_program = torch.export.export(model, (input_tensor,))
            fx_graph = exported_program.module()
            if self.verbose:
                print("    ✓ Dynamo export successful")
        except Exception as e:
            if self.verbose:
                print(f"    ✗ Dynamo export failed: {e}")
            raise RuntimeError(f"Failed to trace model with Dynamo: {e}")

        # Shape propagation
        shape_prop = ShapeProp(fx_graph)
        shape_prop.propagate(input_tensor)

        # Partition - use FusionBasedPartitioner (required for Dynamo export)
        # FusionBasedPartitioner works better with Dynamo's flattened graph structure
        if self.verbose:
            print("  Running fusion-based partitioning...")

        partitioner = FusionBasedPartitioner()
        fusion_report = partitioner.partition(fx_graph)

        if self.verbose:
            print(f"    Partitioned into {len(fusion_report.fused_subgraphs)} fused subgraphs")
            print(f"    Total FLOPs: {fusion_report.total_flops / 1e9:.2f} GFLOPs")

        return fx_graph, fusion_report

    def _run_hardware_mapping(
        self,
        partition_report: PartitionReport,
        hardware_mapper: 'HardwareMapper',
        batch_size: int,
        precision: Precision,
        fx_graph: 'GraphModule' = None
    ) -> GraphHardwareAllocation:
        """
        Map subgraphs to hardware resources (NEW - Phase 1).

        This integrates the hardware mapper to get actual allocation decisions
        (compute_units_allocated) which feed into energy calculation for
        accurate idle power accounting.

        Args:
            partition_report: Partitioned graph
            hardware_mapper: Hardware mapper (GPU, CPU, TPU, etc.)
            batch_size: Batch size for scaling
            precision: Numerical precision
            fx_graph: FX graph module (optional, for model size calculation)

        Returns:
            GraphHardwareAllocation with per-subgraph allocations
        """
        # Check if mapper has map_graph() method (for cross-subgraph analysis like PCIe streaming)
        if hasattr(hardware_mapper, 'map_graph') and callable(getattr(hardware_mapper, 'map_graph')):
            # Use mapper's map_graph() which can handle cross-subgraph optimizations
            # partition_report is already a FusionReport (they have the same structure)

            # Create execution stages (all sequential for now)
            execution_stages = [[i] for i in range(len(partition_report.subgraphs))]

            # Calculate model size (for PCIe streaming overhead in Hailo-8)
            model_size_bytes = 0
            if fx_graph is not None:
                model_params = sum(p.numel() for p in fx_graph.parameters())
                bytes_per_param = {
                    Precision.FP32: 4,
                    Precision.FP16: 2,
                    Precision.BF16: 2,
                    Precision.INT8: 1,
                    Precision.INT4: 0.5,
                }.get(precision, 4)
                model_size_bytes = int(model_params * bytes_per_param)

            # Call mapper's map_graph()
            # Note: Not all mappers accept model_size_bytes, so check signature
            import inspect
            sig = inspect.signature(hardware_mapper.map_graph)
            if 'model_size_bytes' in sig.parameters:
                return hardware_mapper.map_graph(
                    fusion_report=partition_report,
                    execution_stages=execution_stages,
                    batch_size=batch_size,
                    precision=precision,
                    model_size_bytes=model_size_bytes
                )
            else:
                return hardware_mapper.map_graph(
                    fusion_report=partition_report,
                    execution_stages=execution_stages,
                    batch_size=batch_size,
                    precision=precision
                )

        # Fallback: Use per-subgraph mapping (original implementation)
        hardware = hardware_mapper.resource_model
        allocations = []

        # Map each subgraph
        for sg in partition_report.subgraphs:
            # Simple defaults for Phase 1 (no concurrency analysis)
            execution_stage = 0  # All sequential
            concurrent_subgraphs = 1  # No parallelism

            # Map subgraph to hardware
            allocation = hardware_mapper.map_subgraph(
                subgraph=sg,
                execution_stage=execution_stage,
                concurrent_subgraphs=concurrent_subgraphs,
                precision=precision
            )
            allocations.append(allocation)

        # Compute aggregate stats
        total_subgraphs = len(allocations)
        peak_units = max((a.compute_units_allocated for a in allocations), default=0)
        avg_units = sum(a.compute_units_allocated for a in allocations) / max(1, total_subgraphs)
        peak_util = peak_units / hardware.compute_units
        avg_util = avg_units / hardware.compute_units

        # Compute total latency (sequential for now)
        total_latency = sum(a.estimated_latency for a in allocations)
        total_energy = sum(a.total_energy for a in allocations)

        # Bottleneck counts
        compute_bound = sum(1 for a in allocations if a.bottleneck == 'compute')
        memory_bound = sum(1 for a in allocations if a.bottleneck == 'memory')
        bandwidth_bound = sum(1 for a in allocations if a.bottleneck == 'bandwidth')
        balanced = total_subgraphs - compute_bound - memory_bound - bandwidth_bound

        return GraphHardwareAllocation(
            model_name=partition_report.model_name if hasattr(partition_report, 'model_name') else "model",
            hardware_name=hardware.name,
            batch_size=batch_size,
            model_precision=precision,
            subgraph_allocations=allocations,
            total_subgraphs=total_subgraphs,
            total_execution_stages=1,  # Sequential for Phase 1
            peak_compute_units_used=peak_units,
            average_compute_units_used=avg_units,
            peak_utilization=peak_util,
            average_utilization=avg_util,
            total_latency=total_latency,
            latency_breakdown={0: total_latency},  # Single stage
            total_energy=total_energy,
            naive_latency=total_latency,  # Will compute properly later
            latency_correction_factor=1.0,  # Placeholder
            compute_bound_count=compute_bound,
            memory_bound_count=memory_bound,
            bandwidth_bound_count=bandwidth_bound,
            balanced_count=balanced,
        )

    def _run_roofline_analysis(
        self,
        partition_report: PartitionReport,
        hardware: HardwareResourceModel,
        precision: Precision
    ) -> RooflineReport:
        """Run roofline analysis for latency and bottlenecks"""
        analyzer = RooflineAnalyzer(hardware, precision=precision)
        return analyzer.analyze(partition_report.subgraphs, partition_report)

    def _run_energy_analysis(
        self,
        partition_report: PartitionReport,
        hardware: HardwareResourceModel,
        roofline_report: Optional[RooflineReport],
        hardware_allocation: Optional[GraphHardwareAllocation],
        precision: Precision,
        config: AnalysisConfig
    ) -> EnergyReport:
        """Run energy analysis using roofline latencies and hardware allocation (NEW)"""
        analyzer = EnergyAnalyzer(
            hardware,
            precision=precision,
            power_gating_enabled=config.power_gating_enabled  # NEW
        )

        # Extract latencies from roofline if available
        latencies = None
        if roofline_report:
            latencies = [lat.actual_latency for lat in roofline_report.latencies]

        return analyzer.analyze(
            partition_report.subgraphs,
            partition_report,
            latencies=latencies,
            hardware_allocation=hardware_allocation  # NEW
        )

    def _run_memory_analysis(
        self,
        partition_report: PartitionReport,
        hardware: HardwareResourceModel,
        config: AnalysisConfig
    ) -> MemoryReport:
        """Run memory analysis for peak usage and timeline"""
        estimator = MemoryEstimator(hardware)
        return estimator.estimate_memory(partition_report.subgraphs, partition_report)

    def _run_concurrency_analysis(
        self,
        partition_report: PartitionReport
    ) -> ConcurrencyDescriptor:
        """Run concurrency analysis for parallelism opportunities"""
        analyzer = ConcurrencyAnalyzer()
        return analyzer.analyze(partition_report)

    def _run_operator_edp_analysis(
        self,
        model_name: str,
        hardware_mapper: 'HardwareMapper',
        batch_size: int,
        precision: Precision,
        model: Optional[Any] = None,
        input_tensor: Optional[Any] = None
    ) -> Tuple[List[Any], List[Any]]:  # Returns (SubgraphEDPDescriptor list, OperatorEDPDescriptor list)
        """
        Run operator-level EDP breakdown analysis (Phase 2).

        Uses ArchitectureComparator to compute subgraph and operator-level
        EDP breakdowns with architectural modifiers.

        Args:
            model_name: Model name (e.g., 'resnet18')
            hardware_mapper: Hardware mapper instance
            batch_size: Batch size for analysis
            precision: Numerical precision
            model: Optional pre-loaded model instance (for custom models)
            input_tensor: Optional input tensor (for custom models)

        Returns:
            Tuple of (subgraph_edp_breakdown, operator_edp_breakdown)
        """
        # Local import to avoid circular dependency
        from graphs.analysis.architecture_comparator import ArchitectureComparator

        # Extract hardware name from mapper
        hardware_name = hardware_mapper.resource_model.name

        # Create ArchitectureComparator with single architecture
        architectures = {hardware_name: hardware_mapper}

        comparator = ArchitectureComparator(
            model_name=model_name,
            architectures=architectures,
            batch_size=batch_size,
            precision=precision,
            model=model,
            input_tensor=input_tensor
        )

        # Run analysis
        comparator.analyze_all()

        # Extract breakdowns
        subgraph_edps = comparator.get_subgraph_edp_breakdown(hardware_name)
        operator_edps = comparator.get_operator_edp_breakdown(hardware_name)

        return subgraph_edps, operator_edps

    def _compute_derived_metrics(self, result: UnifiedAnalysisResult) -> None:
        """Compute derived metrics from analysis reports"""
        # Latency and throughput
        if result.roofline_report:
            result.total_latency_ms = sum(
                lat.actual_latency for lat in result.roofline_report.latencies
            ) * 1000  # s to ms
            result.throughput_fps = (result.batch_size / result.total_latency_ms) * 1000 if result.total_latency_ms > 0 else 0
            result.average_utilization_pct = result.roofline_report.average_flops_utilization * 100

        # Energy
        if result.energy_report:
            result.total_energy_mj = (
                result.energy_report.compute_energy_j +
                result.energy_report.memory_energy_j +
                result.energy_report.static_energy_j
            ) * 1000  # J to mJ
            result.energy_per_inference_mj = result.total_energy_mj / result.batch_size

        # Add PCIe streaming overhead for Hailo-8 (no external DRAM)
        if result.hardware.main_memory == 0 and result.fx_graph is not None:
            # Calculate model size
            model_params = sum(p.numel() for p in result.fx_graph.parameters())
            bytes_per_param = {
                Precision.FP32: 4,
                Precision.FP16: 2,
                Precision.BF16: 2,
                Precision.INT8: 1,
                Precision.INT4: 0.5,
            }.get(result.precision, 4)
            model_size_bytes = model_params * bytes_per_param
            on_chip_memory_bytes = result.hardware.l2_cache_total

            # Check if PCIe streaming is required
            if model_size_bytes > on_chip_memory_bytes:
                # PCIe Gen3 x4 parameters
                pcie_bandwidth_bytes_per_sec = 4e9  # 4 GB/s
                pcie_energy_per_byte = 25e-12  # 25 pJ/byte

                # Calculate overhead
                pcie_transfer_time_s = model_size_bytes / pcie_bandwidth_bytes_per_sec
                pcie_transfer_energy_j = model_size_bytes * pcie_energy_per_byte

                # Add to totals
                result.total_latency_ms += pcie_transfer_time_s * 1000
                result.total_energy_mj += pcie_transfer_energy_j * 1000
                result.energy_per_inference_mj = result.total_energy_mj / result.batch_size
                result.throughput_fps = (result.batch_size / result.total_latency_ms) * 1000 if result.total_latency_ms > 0 else 0

        # Memory
        if result.memory_report:
            result.peak_memory_mb = result.memory_report.peak_memory_mb

            # Check memory constraints for hardware without external DRAM
            # Hailo-8 has only on-chip SRAM (main_memory=0), so model must fit in L2 cache
            if result.hardware.main_memory == 0:  # No external DRAM
                available_memory_mb = result.hardware.l2_cache_total / (1024 ** 2)  # Bytes to MB
                # Calculate model size based on precision
                model_params = sum(p.numel() for p in result.fx_graph.parameters())
                bytes_per_param = {
                    Precision.FP32: 4,
                    Precision.FP16: 2,
                    Precision.BF16: 2,
                    Precision.INT8: 1,
                    Precision.INT4: 0.5,
                }.get(result.precision, 4)
                model_size_mb = (model_params * bytes_per_param) / (1024 ** 2)

                if model_size_mb > available_memory_mb:
                    warning_msg = (
                        f"MEMORY CONSTRAINT VIOLATION: Model size ({model_size_mb:.1f} MB) "
                        f"exceeds available on-chip memory ({available_memory_mb:.1f} MB). "
                        f"Hardware {result.hardware_name} requires full model to fit on-chip for "
                        f"standalone automotive deployment (no external DRAM available)."
                    )
                    result.validation_warnings.append(warning_msg)

    def _validate_result(self, result: UnifiedAnalysisResult) -> List[str]:
        """Validate consistency between reports"""
        return result.validate()

    # =========================================================================
    # Private Methods - Model and Hardware Creation
    # =========================================================================

    def _create_model(self, model_name: str, batch_size: int) -> Tuple[nn.Module, torch.Tensor, str]:
        """
        Create PyTorch model and input tensor.

        Args:
            model_name: Name of model
            batch_size: Batch size for input

        Returns:
            (model, input_tensor, display_name)
        """
        model_name_lower = model_name.lower()

        # ResNet family
        if model_name_lower == 'resnet18':
            return models.resnet18(weights=None), torch.randn(batch_size, 3, 224, 224), "ResNet-18"
        elif model_name_lower == 'resnet34':
            return models.resnet34(weights=None), torch.randn(batch_size, 3, 224, 224), "ResNet-34"
        elif model_name_lower == 'resnet50':
            return models.resnet50(weights=None), torch.randn(batch_size, 3, 224, 224), "ResNet-50"
        elif model_name_lower == 'resnet101':
            return models.resnet101(weights=None), torch.randn(batch_size, 3, 224, 224), "ResNet-101"
        elif model_name_lower == 'resnet152':
            return models.resnet152(weights=None), torch.randn(batch_size, 3, 224, 224), "ResNet-152"

        # MobileNet family
        elif model_name_lower in ['mobilenet', 'mobilenet_v2', 'mobilenetv2']:
            return models.mobilenet_v2(weights=None), torch.randn(batch_size, 3, 224, 224), "MobileNet-V2"
        elif model_name_lower in ['mobilenet_v3_small', 'mobilenetv3_small']:
            return models.mobilenet_v3_small(weights=None), torch.randn(batch_size, 3, 224, 224), "MobileNet-V3-Small"
        elif model_name_lower in ['mobilenet_v3_large', 'mobilenetv3_large']:
            return models.mobilenet_v3_large(weights=None), torch.randn(batch_size, 3, 224, 224), "MobileNet-V3-Large"

        # EfficientNet family
        elif model_name_lower in ['efficientnet_b0', 'efficientnetb0']:
            return models.efficientnet_b0(weights=None), torch.randn(batch_size, 3, 224, 224), "EfficientNet-B0"
        elif model_name_lower in ['efficientnet_b1', 'efficientnetb1']:
            return models.efficientnet_b1(weights=None), torch.randn(batch_size, 3, 240, 240), "EfficientNet-B1"
        elif model_name_lower in ['efficientnet_b2', 'efficientnetb2']:
            return models.efficientnet_b2(weights=None), torch.randn(batch_size, 3, 260, 260), "EfficientNet-B2"

        # VGG family
        elif model_name_lower == 'vgg11':
            return models.vgg11(weights=None), torch.randn(batch_size, 3, 224, 224), "VGG-11"
        elif model_name_lower == 'vgg16':
            return models.vgg16(weights=None), torch.randn(batch_size, 3, 224, 224), "VGG-16"
        elif model_name_lower == 'vgg19':
            return models.vgg19(weights=None), torch.randn(batch_size, 3, 224, 224), "VGG-19"

        # ViT
        elif model_name_lower in ['vit', 'vit_b_16']:
            return models.vit_b_16(weights=None), torch.randn(batch_size, 3, 224, 224), "ViT-B/16"

        # Segmentation models
        elif model_name_lower in ['deeplabv3', 'deeplabv3_resnet50']:
            return models.segmentation.deeplabv3_resnet50(weights=None), torch.randn(batch_size, 3, 224, 224), "DeepLabV3-ResNet50"
        elif model_name_lower in ['fcn', 'fcn_resnet50']:
            return models.segmentation.fcn_resnet50(weights=None), torch.randn(batch_size, 3, 224, 224), "FCN-ResNet50"

        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported: resnet18/34/50/101/152, "
                f"mobilenet_v2/v3_small/v3_large, efficientnet_b0/b1/b2, vgg11/16/19, vit_b_16, "
                f"deeplabv3_resnet50, fcn_resnet50"
            )

    def _create_hardware_mapper(self, hardware_name: str) -> Any:
        """
        Create hardware mapper by name.

        Args:
            hardware_name: Hardware name (e.g., 'H100', 'Jetson-Orin-AGX')

        Returns:
            HardwareMapper instance
        """
        hardware_map = {
            # GPUs - Datacenter
            'h100': create_h100_mapper,
            'h100-pcie': create_h100_mapper,
            'a100': create_a100_mapper,
            'v100': create_v100_mapper,

            # GPUs - Edge
            'jetson-orin-agx': create_jetson_orin_agx_mapper,
            'jetson-orin': create_jetson_orin_agx_mapper,
            'jetson-orin-nano': create_jetson_orin_nano_mapper,
            'jetson-nano': create_jetson_orin_nano_mapper,
            'jetson-thor': create_jetson_thor_mapper,

            # TPUs
            'tpu-v4': create_tpu_v4_mapper,
            'tpu': create_tpu_v4_mapper,
            'coral': create_coral_edge_tpu_mapper,
            'coral-tpu': create_coral_edge_tpu_mapper,

            # KPUs
            'kpu-t64': create_kpu_t64_mapper,
            'kpu-t256': create_kpu_t256_mapper,
            'kpu-t768': create_kpu_t768_mapper,

            # CPUs - Datacenter
            'epyc': create_amd_epyc_9754_mapper,
            'amd-epyc': create_amd_epyc_9754_mapper,
            'xeon': create_intel_xeon_platinum_8490h_mapper,
            'intel-xeon': create_intel_xeon_platinum_8490h_mapper,
            'ampere-one': create_ampere_ampereone_192_mapper,

            # CPUs - Consumer
            'i7-12700k': create_i7_12700k_mapper,
            'ryzen-7-5800x': create_amd_cpu_mapper,
            'ryzen': create_amd_cpu_mapper,

            # DSPs
            'qrb5165': create_qrb5165_mapper,
            'qualcomm-qrb5165': create_qrb5165_mapper,

            # Automotive DSPs (Qualcomm)
            'snapdragon-ride': create_qualcomm_snapdragon_ride_mapper,
            'qualcomm-snapdragon-ride': create_qualcomm_snapdragon_ride_mapper,
            'sa8775p': create_qualcomm_sa8775p_mapper,
            'qualcomm-sa8775p': create_qualcomm_sa8775p_mapper,

            # Automotive DSPs (TI)
            'ti-tda4vm': create_ti_tda4vm_mapper,
            'tda4vm': create_ti_tda4vm_mapper,
            'ti-tda4vh': create_ti_tda4vh_mapper,
            'tda4vh': create_ti_tda4vh_mapper,
            'ti-tda4vl': create_ti_tda4vl_mapper,
            'tda4vl': create_ti_tda4vl_mapper,
            'ti-tda4al': create_ti_tda4al_mapper,
            'tda4al': create_ti_tda4al_mapper,

            # Automotive Accelerators (Hailo)
            'hailo-10h': create_hailo10h_mapper,
            'hailo-8': create_hailo8_mapper,

            # Accelerators
            'dpu': create_dpu_vitis_ai_mapper,
            'xilinx-dpu': create_dpu_vitis_ai_mapper,
            'cgra': create_plasticine_v2_mapper,
            'plasticine': create_plasticine_v2_mapper,
        }

        hardware_key = hardware_name.lower()
        if hardware_key not in hardware_map:
            available = ', '.join(sorted(hardware_map.keys()))
            raise ValueError(f"Unknown hardware: {hardware_name}. Available: {available}")

        return hardware_map[hardware_key]()
