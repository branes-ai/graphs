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
from torch.fx import symbolic_trace, GraphModule
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

# Hardware models
from graphs.hardware.resource_model import Precision, HardwareResourceModel

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
)
from graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper
from graphs.hardware.mappers.accelerators.cgra import create_plasticine_v2_mapper


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

    # Partitioning strategy
    use_fusion_partitioning: bool = False  # Reserved for future use (FusionBasedPartitioner needs integration work)

    # Analysis options
    estimate_checkpointing_savings: bool = False
    estimate_quantization_savings: bool = False
    detailed_memory_timeline: bool = True

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

    # Phase 3 analysis results
    roofline_report: Optional[RooflineReport] = None
    energy_report: Optional[EnergyReport] = None
    memory_report: Optional[MemoryReport] = None
    concurrency_report: Optional[ConcurrencyDescriptor] = None

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
        )

    def analyze_model_with_custom_hardware(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        model_name: str,
        hardware_mapper: Any,  # Any mapper type (GPUMapper, CPUMapper, etc.) with resource_model attribute
        precision: Precision = Precision.FP32,
        config: Optional[AnalysisConfig] = None,
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

        # Step 2: Run roofline analysis
        if config.run_roofline:
            if self.verbose:
                print("Running roofline analysis...")
            result.roofline_report = self._run_roofline_analysis(
                partition_report, hardware, precision
            )

        # Step 3: Run energy analysis (depends on roofline for latencies)
        if config.run_energy:
            if self.verbose:
                print("Running energy analysis...")
            result.energy_report = self._run_energy_analysis(
                partition_report, hardware, result.roofline_report, precision
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
        """Trace model and partition into subgraphs"""
        # FX trace
        fx_graph = symbolic_trace(model)

        # Shape propagation
        shape_prop = ShapeProp(fx_graph)
        shape_prop.propagate(input_tensor)

        # Partition - always use GraphPartitioner for now
        # FusionBasedPartitioner returns FusionReport which has different structure
        if config.use_fusion_partitioning:
            warnings.warn(
                "Fusion partitioning not yet supported in UnifiedAnalyzer. "
                "Using GraphPartitioner instead.",
                UserWarning
            )

        partitioner = GraphPartitioner()
        partition_report = partitioner.partition(fx_graph)

        return fx_graph, partition_report

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
        precision: Precision
    ) -> EnergyReport:
        """Run energy analysis using roofline latencies"""
        analyzer = EnergyAnalyzer(hardware, precision=precision)

        # Extract latencies from roofline if available
        latencies = None
        if roofline_report:
            latencies = [lat.actual_latency for lat in roofline_report.latencies]

        return analyzer.analyze(
            partition_report.subgraphs,
            partition_report,
            latencies=latencies
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

        # Memory
        if result.memory_report:
            result.peak_memory_mb = result.memory_report.peak_memory_mb

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

        else:
            raise ValueError(
                f"Unknown model: {model_name}. Supported: resnet18/34/50/101/152, "
                f"mobilenet_v2/v3_small/v3_large, efficientnet_b0/b1/b2, vgg11/16/19, vit_b_16"
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
            'ti-tda4vm': create_ti_tda4vm_mapper,
            'tda4vm': create_ti_tda4vm_mapper,

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
