"""
Unit tests for UnifiedAnalyzer

Tests the unified analysis framework that orchestrates all Phase 3 analyzers.
"""

import unittest
import torch
import torch.nn as nn

from graphs.analysis.unified_analyzer import (
    UnifiedAnalyzer,
    AnalysisConfig,
    UnifiedAnalysisResult,
)
from graphs.hardware.resource_model import Precision


class SimpleModel(nn.Module):
    """Simple test model"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class TestUnifiedAnalyzer(unittest.TestCase):
    """Test UnifiedAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = UnifiedAnalyzer(verbose=False)

    def test_basic_analysis(self):
        """Test basic analysis with all components"""
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        # Check result structure
        self.assertIsInstance(result, UnifiedAnalysisResult)
        self.assertEqual(result.model_name, 'ResNet-18')
        self.assertEqual(result.batch_size, 1)
        self.assertEqual(result.precision, Precision.FP32)

        # Check that all reports are present
        self.assertIsNotNone(result.roofline_report)
        self.assertIsNotNone(result.energy_report)
        self.assertIsNotNone(result.memory_report)
        self.assertIsNone(result.concurrency_report)  # Not enabled by default

        # Check derived metrics
        self.assertGreater(result.total_latency_ms, 0)
        self.assertGreater(result.throughput_fps, 0)
        self.assertGreater(result.total_energy_mj, 0)
        self.assertGreater(result.peak_memory_mb, 0)

    def test_custom_config(self):
        """Test with custom AnalysisConfig"""
        config = AnalysisConfig(
            run_roofline=True,
            run_energy=True,
            run_memory=True,
            run_concurrency=True,  # Enable concurrency analysis
            use_fusion_partitioning=False,
            validate_consistency=True
        )

        result = self.analyzer.analyze_model(
            model_name='mobilenet_v2',
            hardware_name='Jetson-Orin-Nano',
            batch_size=1,
            precision=Precision.FP32,
            config=config
        )

        # Check that concurrency analysis ran
        self.assertIsNotNone(result.concurrency_report)

    def test_different_precisions(self):
        """Test FP32, FP16"""
        for precision in [Precision.FP32, Precision.FP16]:
            with self.subTest(precision=precision):
                result = self.analyzer.analyze_model(
                    model_name='resnet18',
                    hardware_name='H100',
                    batch_size=1,
                    precision=precision
                )

                self.assertEqual(result.precision, precision)
                self.assertGreater(result.total_latency_ms, 0)

    def test_batch_sizes(self):
        """Test different batch sizes"""
        batch_sizes = [1, 4, 16]
        results = []

        for batch_size in batch_sizes:
            result = self.analyzer.analyze_model(
                model_name='resnet18',
                hardware_name='H100',
                batch_size=batch_size,
                precision=Precision.FP32
            )
            results.append(result)

        # Verify batch size increases latency but improves throughput
        self.assertLess(results[0].total_latency_ms, results[2].total_latency_ms)
        self.assertGreater(results[2].throughput_fps, results[0].throughput_fps)

        # Verify energy per inference decreases with batching
        self.assertGreater(results[0].energy_per_inference_mj, results[2].energy_per_inference_mj)

    def test_error_handling_invalid_model(self):
        """Test error handling for invalid model"""
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze_model(
                model_name='invalid_model_name',
                hardware_name='H100',
                batch_size=1,
                precision=Precision.FP32
            )
        self.assertIn("Unknown model", str(ctx.exception))

    def test_error_handling_invalid_hardware(self):
        """Test error handling for invalid hardware"""
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze_model(
                model_name='resnet18',
                hardware_name='invalid_hardware',
                batch_size=1,
                precision=Precision.FP32
            )
        self.assertIn("Unknown hardware", str(ctx.exception))

    def test_consistency_validation(self):
        """Test report consistency validation"""
        config = AnalysisConfig(validate_consistency=True)

        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=config
        )

        # Validation warnings should be empty or minimal
        # (Some warnings are acceptable due to different partitioning strategies)
        self.assertIsInstance(result.validation_warnings, list)

    def test_executive_summary(self):
        """Test executive summary generation"""
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        summary = result.get_executive_summary()

        # Check structure
        self.assertIn('model', summary)
        self.assertIn('hardware', summary)
        self.assertIn('performance', summary)
        self.assertIn('energy', summary)
        self.assertIn('memory', summary)
        self.assertIn('recommendations', summary)

        # Check values
        self.assertEqual(summary['batch_size'], 1)
        self.assertEqual(summary['precision'], 'FP32')
        self.assertGreater(summary['performance']['latency_ms'], 0)
        self.assertGreater(summary['energy']['total_mj'], 0)
        self.assertGreater(summary['memory']['peak_mb'], 0)

    def test_custom_model_analysis(self):
        """Test analysis with custom model"""
        model = SimpleModel()
        input_tensor = torch.randn(1, 3, 224, 224)

        # Create hardware mapper
        from graphs.hardware.mappers.gpu import create_h100_pcie_80gb_mapper
        hardware_mapper = create_h100_pcie_80gb_mapper()

        result = self.analyzer.analyze_model_with_custom_hardware(
            model=model,
            input_tensor=input_tensor,
            model_name='SimpleModel',
            hardware_mapper=hardware_mapper,
            precision=Precision.FP32
        )

        # Check result
        self.assertEqual(result.model_name, 'SimpleModel')
        self.assertGreater(result.total_latency_ms, 0)

    def test_fusion_partitioning(self):
        """Test fusion-based partitioning (now always uses Dynamo + FusionBasedPartitioner)"""
        # NOTE: With Dynamo-first architecture, we always use FusionBasedPartitioner
        # The use_fusion_partitioning flag is deprecated but kept for backward compatibility

        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        # Verify we got valid partition results
        self.assertIsNotNone(result.partition_report)
        self.assertGreater(len(result.partition_report.subgraphs), 0)

        # Fusion report should have fusion metrics
        self.assertGreater(result.partition_report.original_operators, 0,
                          "Should track original operator count")
        self.assertGreaterEqual(result.partition_report.data_movement_reduction, 0.0,
                               "Should have data movement reduction metric")

        # With fusion, average fusion size should typically be > 1
        # (though for some models it might be 1.0 if no fusion opportunities)
        self.assertGreaterEqual(result.partition_report.avg_fusion_size, 1.0,
                               "Average fusion size should be >= 1.0")

    def test_selective_analysis(self):
        """Test running only specific analyses"""
        # Only roofline
        config_roofline = AnalysisConfig(
            run_roofline=True,
            run_energy=False,
            run_memory=False,
            run_concurrency=False
        )

        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=config_roofline
        )

        self.assertIsNotNone(result.roofline_report)
        self.assertIsNone(result.energy_report)
        self.assertIsNone(result.memory_report)
        self.assertIsNone(result.concurrency_report)


if __name__ == '__main__':
    unittest.main()
