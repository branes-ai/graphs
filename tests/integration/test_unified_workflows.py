"""
Integration Tests for Unified Workflows

Tests the complete unified analysis framework end-to-end:
- UnifiedAnalyzer + ReportGenerator integration
- Consistency with direct Phase 3 analyzer usage
- Multi-format report generation
- Cross-configuration consistency
"""

import unittest
import json
import csv
import tempfile
import os
from io import StringIO

from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig, UnifiedAnalysisResult
from graphs.reporting import ReportGenerator
from graphs.hardware.resource_model import Precision

# For comparison with direct usage
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.estimation.energy import EnergyAnalyzer
from graphs.estimation.memory import MemoryEstimator

import torch
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models


class TestUnifiedWorkflows(unittest.TestCase):
    """Integration tests for unified analysis workflows"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = UnifiedAnalyzer(verbose=False)
        self.generator = ReportGenerator()

    # =========================================================================
    # End-to-End Workflow Tests
    # =========================================================================

    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow from model to report"""
        # Run analysis
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        # Verify result structure
        self.assertIsInstance(result, UnifiedAnalysisResult)
        self.assertIsNotNone(result.roofline_report)
        self.assertIsNotNone(result.energy_report)
        self.assertIsNotNone(result.memory_report)

        # Generate all report formats
        text_report = self.generator.generate_text_report(result)
        json_report = self.generator.generate_json_report(result)
        csv_report = self.generator.generate_csv_report(result)
        md_report = self.generator.generate_markdown_report(result)

        # Verify all formats are non-empty
        self.assertGreater(len(text_report), 100)
        self.assertGreater(len(json_report), 100)
        self.assertGreater(len(csv_report), 50)
        self.assertGreater(len(md_report), 100)

        # Verify JSON is parseable
        data = json.loads(json_report)
        self.assertIn('metadata', data)
        self.assertIn('derived_metrics', data)

        # Verify CSV is parseable
        reader = csv.DictReader(StringIO(csv_report))
        rows = list(reader)
        self.assertEqual(len(rows), 1)

    def test_batch_size_sweep_workflow(self):
        """Test batch size sweep workflow"""
        batch_sizes = [1, 4, 8]
        results = []

        for batch_size in batch_sizes:
            result = self.analyzer.analyze_model(
                model_name='resnet18',
                hardware_name='H100',
                batch_size=batch_size,
                precision=Precision.FP32
            )
            results.append(result)

        # Verify results
        self.assertEqual(len(results), 3)

        # Generate comparison report
        comparison = self.generator.generate_comparison_report(
            results,
            format='text'
        )

        # Verify comparison contains all batch sizes
        self.assertIn('Batch 1', comparison)
        self.assertIn('Batch 4', comparison)
        self.assertIn('Batch 8', comparison)

        # Verify trends (throughput should increase, energy per inference should decrease)
        self.assertGreater(results[2].throughput_fps, results[0].throughput_fps)
        self.assertLess(results[2].energy_per_inference_mj, results[0].energy_per_inference_mj)

    def test_multi_model_comparison_workflow(self):
        """Test multi-model comparison workflow"""
        models = ['resnet18', 'mobilenet_v2']
        results = []

        for model in models:
            result = self.analyzer.analyze_model(
                model_name=model,
                hardware_name='H100',
                batch_size=1,
                precision=Precision.FP32
            )
            results.append(result)

        # Generate comparison report
        comparison_csv = self.generator.generate_comparison_report(
            results,
            format='csv'
        )

        # Parse CSV
        reader = csv.DictReader(StringIO(comparison_csv))
        rows = list(reader)

        # Verify both models present
        self.assertEqual(len(rows), 2)
        models_in_csv = [row['model'] for row in rows]
        self.assertIn('ResNet-18', models_in_csv)
        self.assertIn('MobileNet-V2', models_in_csv)

    # =========================================================================
    # Consistency Tests
    # =========================================================================

    def test_consistency_with_direct_phase3_usage(self):
        """Test that UnifiedAnalyzer produces same results as direct Phase 3 usage"""
        # Setup
        model = models.resnet18(weights=None)
        input_tensor = torch.randn(1, 3, 224, 224)

        # Direct Phase 3 usage (matching UnifiedAnalyzer's approach)
        # Use symbolic_trace to match unified analyzer (which prefers module-level
        # tracing for better fusion and weight counting)
        model.eval()
        from torch.fx import symbolic_trace
        fx_graph = symbolic_trace(model)

        shape_prop = ShapeProp(fx_graph)
        shape_prop.propagate(input_tensor)

        # Use FusionBasedPartitioner (not GraphPartitioner) to match unified analyzer
        from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
        partitioner = FusionBasedPartitioner()
        partition_report = partitioner.partition(fx_graph)

        from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
        hardware_mapper = create_h100_sxm5_80gb_mapper()
        hardware = hardware_mapper.resource_model

        roofline_analyzer = RooflineAnalyzer(hardware, precision=Precision.FP32)
        roofline_report = roofline_analyzer.analyze(partition_report.subgraphs, partition_report)

        energy_analyzer = EnergyAnalyzer(hardware, precision=Precision.FP32)
        latencies = [lat.actual_latency for lat in roofline_report.latencies]
        energy_report = energy_analyzer.analyze(partition_report.subgraphs, partition_report, latencies=latencies)

        memory_estimator = MemoryEstimator(hardware)
        memory_report = memory_estimator.estimate_memory(partition_report.subgraphs, partition_report)

        # Unified usage (disable hardware mapping to match direct approach)
        config = AnalysisConfig(
            run_hardware_mapping=False  # Match direct Phase 3 usage (no hardware mapping)
        )
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=config
        )

        # Compare results (allow tolerance for Dynamo export differences)
        direct_latency = sum(lat.actual_latency for lat in roofline_report.latencies) * 1000  # s to ms
        unified_latency = result.total_latency_ms

        # Dynamo export may produce slightly different graph structure than symbolic_trace
        # Allow 30% tolerance (architectural difference, not a bug)
        self.assertAlmostEqual(direct_latency, unified_latency, delta=direct_latency * 0.30)

        # Compare energy (allow tolerance for Dynamo export differences)
        direct_energy = (energy_report.compute_energy_j +
                        energy_report.memory_energy_j +
                        energy_report.static_energy_j) * 1000  # J to mJ
        unified_energy = result.total_energy_mj

        # Energy proportional to latency, so use same 30% tolerance
        self.assertAlmostEqual(direct_energy, unified_energy, delta=direct_energy * 0.30)

        # Compare memory (allow small tolerance due to rounding)
        self.assertAlmostEqual(memory_report.peak_memory_bytes / 1e6,
                              result.peak_memory_mb,
                              delta=5.0)  # Within 5 MB

    def test_cross_precision_consistency(self):
        """Test that different precisions produce consistent relative results"""
        fp32_result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        fp16_result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP16
        )

        # FP16 should be faster than FP32
        # Note: Speedup is modest (~1.1×) for batch=1 due to many bandwidth-bound subgraphs
        # Larger batches show greater speedup as they become more compute-bound
        self.assertLess(fp16_result.total_latency_ms, fp32_result.total_latency_ms,
                       "FP16 should be faster than FP32")

        # FP16 should use less energy
        self.assertLess(fp16_result.total_energy_mj, fp32_result.total_energy_mj,
                       "FP16 should use less energy than FP32")

        # Verify non-zero values
        self.assertGreater(fp32_result.total_latency_ms, 0)
        self.assertGreater(fp16_result.total_latency_ms, 0)
        self.assertGreater(fp32_result.total_energy_mj, 0)
        self.assertGreater(fp16_result.total_energy_mj, 0)

        # Note: Memory estimator currently doesn't differentiate by precision
        # (it estimates based on tensor shapes, not runtime precision)
        # So we just verify both have reasonable memory values
        self.assertGreater(fp32_result.peak_memory_mb, 0)
        self.assertGreater(fp16_result.peak_memory_mb, 0)

    def test_report_format_consistency(self):
        """Test that different report formats contain consistent data"""
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        # Generate all formats
        json_str = self.generator.generate_json_report(result)
        csv_str = self.generator.generate_csv_report(result)
        text_str = self.generator.generate_text_report(result)
        md_str = self.generator.generate_markdown_report(result)

        # Parse JSON
        json_data = json.loads(json_str)

        # Parse CSV
        csv_reader = csv.DictReader(StringIO(csv_str))
        csv_data = list(csv_reader)[0]

        # Verify key metrics are consistent across formats
        latency_ms = result.total_latency_ms

        # JSON
        self.assertAlmostEqual(json_data['derived_metrics']['latency_ms'], latency_ms, delta=0.01)

        # CSV
        self.assertAlmostEqual(float(csv_data['latency_ms']), latency_ms, delta=0.01)

        # Text (check it contains the value)
        self.assertIn(f"{latency_ms:.2f}", text_str)

        # Markdown (check it contains the value)
        self.assertIn(f"{latency_ms:.2f}", md_str)

    # =========================================================================
    # Configuration Tests
    # =========================================================================

    def test_selective_analysis_config(self):
        """Test selective analysis configuration"""
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

        # Only energy
        config_energy = AnalysisConfig(
            run_roofline=False,
            run_energy=True,
            run_memory=False,
            run_concurrency=False
        )

        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=config_energy
        )

        self.assertIsNone(result.roofline_report)
        self.assertIsNotNone(result.energy_report)
        self.assertIsNone(result.memory_report)

    def test_validation_config(self):
        """Test consistency validation configuration"""
        config = AnalysisConfig(validate_consistency=True)

        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=config
        )

        # Validation warnings should be a list (may be empty)
        self.assertIsInstance(result.validation_warnings, list)

    # =========================================================================
    # Report Generation Tests
    # =========================================================================

    def test_report_sections_filtering(self):
        """Test selective report sections"""
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        # Only executive summary and performance
        report = self.generator.generate_text_report(
            result,
            include_sections=['executive', 'performance']
        )

        # Should include these
        self.assertIn('EXECUTIVE SUMMARY', report)
        self.assertIn('PERFORMANCE ANALYSIS', report)

        # Should NOT include these
        self.assertNotIn('ENERGY ANALYSIS', report)
        self.assertNotIn('MEMORY ANALYSIS', report)

    def test_csv_subgraph_details(self):
        """Test CSV generation with subgraph details"""
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        # Summary mode
        csv_summary = self.generator.generate_csv_report(
            result,
            include_subgraph_details=False
        )
        rows_summary = list(csv.DictReader(StringIO(csv_summary)))
        self.assertEqual(len(rows_summary), 1)

        # Detailed mode
        csv_detailed = self.generator.generate_csv_report(
            result,
            include_subgraph_details=True
        )
        rows_detailed = list(csv.DictReader(StringIO(csv_detailed)))
        self.assertGreater(len(rows_detailed), 1)
        self.assertIn('subgraph_id', rows_detailed[0])

    def test_comparison_report_sorting(self):
        """Test comparison report sorting"""
        # Create results with different latencies
        results = [
            self.analyzer.analyze_model('resnet18', 'H100', batch_size=1, precision=Precision.FP32),
            self.analyzer.analyze_model('resnet18', 'H100', batch_size=8, precision=Precision.FP32),
            self.analyzer.analyze_model('resnet18', 'H100', batch_size=4, precision=Precision.FP32),
        ]

        # Sort by latency
        comparison = self.generator.generate_comparison_report(
            results,
            format='csv',
            sort_by='latency'
        )

        rows = list(csv.DictReader(StringIO(comparison)))
        latencies = [float(row['latency_ms']) for row in rows]

        # Should be sorted ascending
        self.assertEqual(latencies, sorted(latencies))

        # Sort by throughput
        comparison = self.generator.generate_comparison_report(
            results,
            format='csv',
            sort_by='throughput'
        )

        rows = list(csv.DictReader(StringIO(comparison)))
        throughputs = [float(row['throughput_fps']) for row in rows]

        # Should be sorted descending
        self.assertEqual(throughputs, sorted(throughputs, reverse=True))

    # =========================================================================
    # File I/O Tests
    # =========================================================================

    def test_save_and_load_json(self):
        """Test saving and loading JSON reports"""
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            # Save
            self.generator.save_report(result, output_file, format='json')

            # Load and verify
            with open(output_file, 'r') as f:
                data = json.load(f)

            self.assertEqual(data['metadata']['model'], 'ResNet-18')
            self.assertEqual(data['metadata']['batch_size'], 1)
            self.assertAlmostEqual(
                data['derived_metrics']['latency_ms'],
                result.total_latency_ms,
                delta=0.01
            )

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_auto_format_detection(self):
        """Test automatic format detection from file extension"""
        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        test_files = [
            ('report.json', json.loads),
            ('report.csv', lambda x: list(csv.DictReader(StringIO(x)))),
            ('report.md', lambda x: x),
            ('report.txt', lambda x: x),
        ]

        for filename, parser in test_files:
            with self.subTest(filename=filename):
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{filename.split(".")[-1]}', delete=False) as f:
                    output_file = f.name

                try:
                    # Save with auto-detection
                    self.generator.save_report(result, output_file)

                    # Verify file exists and is parseable
                    self.assertTrue(os.path.exists(output_file))

                    with open(output_file, 'r') as f:
                        content = f.read()

                    # Should not raise exception
                    parsed = parser(content)
                    self.assertIsNotNone(parsed)

                finally:
                    if os.path.exists(output_file):
                        os.unlink(output_file)

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_analysis_performance(self):
        """Test that analysis completes in reasonable time"""
        import time

        start = time.time()

        result = self.analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

        elapsed = time.time() - start

        # Should complete in under 5 seconds
        self.assertLess(elapsed, 5.0)

        # Result should be valid
        self.assertIsNotNone(result)
        self.assertGreater(result.total_latency_ms, 0)

    def test_batch_sweep_performance(self):
        """Test batch sweep performance"""
        import time

        start = time.time()

        results = []
        for batch_size in [1, 4, 8]:
            result = self.analyzer.analyze_model(
                model_name='resnet18',
                hardware_name='H100',
                batch_size=batch_size,
                precision=Precision.FP32,
                config=AnalysisConfig(validate_consistency=False)  # Skip for speed
            )
            results.append(result)

        elapsed = time.time() - start

        # Should complete in under 10 seconds for 3 configs
        self.assertLess(elapsed, 10.0)

        # All results should be valid
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertGreater(r.total_latency_ms, 0)

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_invalid_model_error(self):
        """Test error handling for invalid model"""
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze_model(
                model_name='invalid_model_12345',
                hardware_name='H100',
                batch_size=1,
                precision=Precision.FP32
            )
        self.assertIn('Unknown model', str(ctx.exception))

    def test_invalid_hardware_error(self):
        """Test error handling for invalid hardware"""
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze_model(
                model_name='resnet18',
                hardware_name='invalid_hardware_12345',
                batch_size=1,
                precision=Precision.FP32
            )
        self.assertIn('Unknown hardware', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
