"""
Unit tests for ReportGenerator

Tests the report generation framework that transforms analysis results into various formats.
"""

import unittest
import json
import csv
import tempfile
import os
from io import StringIO

from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.reporting import ReportGenerator
from graphs.hardware.resource_model import Precision


class TestReportGenerator(unittest.TestCase):
    """Test ReportGenerator"""

    def setUp(self):
        """Set up test fixtures"""
        self.generator = ReportGenerator(style='default')

        # Create a sample analysis result
        analyzer = UnifiedAnalyzer(verbose=False)
        self.result = analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32
        )

    def test_text_report(self):
        """Test text report generation"""
        report = self.generator.generate_text_report(self.result)

        # Check that report contains expected sections
        self.assertIn("COMPREHENSIVE ANALYSIS REPORT", report)
        self.assertIn("EXECUTIVE SUMMARY", report)
        self.assertIn("PERFORMANCE ANALYSIS", report)
        self.assertIn("ENERGY ANALYSIS", report)
        self.assertIn("MEMORY ANALYSIS", report)

        # Check that key metrics are present
        self.assertIn("Latency", report)
        self.assertIn("Energy", report)
        self.assertIn("Memory", report)
        self.assertIn("fps", report)

    def test_text_report_selective_sections(self):
        """Test text report with selective sections"""
        report = self.generator.generate_text_report(
            self.result,
            include_sections=['executive', 'performance']
        )

        # Should include these sections
        self.assertIn("EXECUTIVE SUMMARY", report)
        self.assertIn("PERFORMANCE ANALYSIS", report)

        # Should NOT include these sections
        self.assertNotIn("ENERGY ANALYSIS", report)
        self.assertNotIn("MEMORY ANALYSIS", report)

    def test_json_report(self):
        """Test JSON report generation and parsing"""
        json_str = self.generator.generate_json_report(self.result)

        # Parse JSON
        data = json.loads(json_str)

        # Check structure
        self.assertIn('metadata', data)
        self.assertIn('executive_summary', data)
        self.assertIn('derived_metrics', data)
        self.assertIn('recommendations', data)

        # Check metadata
        self.assertEqual(data['metadata']['model'], 'ResNet-18')
        self.assertEqual(data['metadata']['batch_size'], 1)
        self.assertEqual(data['metadata']['precision'], 'FP32')

        # Check derived metrics
        self.assertIn('latency_ms', data['derived_metrics'])
        self.assertIn('throughput_fps', data['derived_metrics'])
        self.assertIn('total_energy_mj', data['derived_metrics'])
        self.assertIn('peak_memory_mb', data['derived_metrics'])

        # Check that values are numeric
        self.assertIsInstance(data['derived_metrics']['latency_ms'], (int, float))
        self.assertGreater(data['derived_metrics']['latency_ms'], 0)

    def test_json_report_compact(self):
        """Test compact JSON output"""
        json_str = self.generator.generate_json_report(
            self.result,
            pretty_print=False
        )

        # Should not contain indentation
        self.assertNotIn('\n  ', json_str)

        # But should still be valid JSON
        data = json.loads(json_str)
        self.assertIn('metadata', data)

    def test_csv_report_summary(self):
        """Test CSV report generation (summary mode)"""
        csv_str = self.generator.generate_csv_report(self.result, include_subgraph_details=False)

        # Parse CSV
        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)

        # Should have exactly 1 row (summary)
        self.assertEqual(len(rows), 1)

        row = rows[0]

        # Check columns
        self.assertIn('model', row)
        self.assertIn('hardware', row)
        self.assertIn('latency_ms', row)
        self.assertIn('throughput_fps', row)
        self.assertIn('energy_mj', row)
        self.assertIn('peak_mem_mb', row)

        # Check values
        self.assertEqual(row['model'], 'ResNet-18')
        self.assertEqual(row['batch_size'], '1')
        self.assertGreater(float(row['latency_ms']), 0)
        self.assertGreater(float(row['throughput_fps']), 0)

    def test_csv_report_detailed(self):
        """Test CSV report with subgraph details"""
        csv_str = self.generator.generate_csv_report(self.result, include_subgraph_details=True)

        # Parse CSV
        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)

        # Should have multiple rows (one per subgraph)
        self.assertGreater(len(rows), 1)

        # Check first row
        row = rows[0]
        self.assertIn('subgraph_id', row)
        self.assertIn('flops', row)
        self.assertIn('memory_bytes', row)
        self.assertIn('bottleneck', row)

    def test_markdown_report(self):
        """Test Markdown report generation"""
        md = self.generator.generate_markdown_report(self.result)

        # Check Markdown syntax
        self.assertIn('#', md)  # Headers
        self.assertIn('**', md)  # Bold

        # Check sections
        self.assertIn('# Analysis Report:', md)
        self.assertIn('## Executive Summary', md)
        self.assertIn('## Performance Analysis', md)
        self.assertIn('## Energy Analysis', md)

    def test_markdown_report_with_tables(self):
        """Test Markdown report with tables"""
        md = self.generator.generate_markdown_report(
            self.result,
            include_tables=True
        )

        # Check for table syntax
        self.assertIn('|', md)
        self.assertIn('Metric | Value', md)
        self.assertIn('Latency', md)

    def test_comparison_report_text(self):
        """Test comparison report generation (text format)"""
        # Create multiple results
        analyzer = UnifiedAnalyzer(verbose=False)

        results = [
            analyzer.analyze_model('resnet18', 'H100', batch_size=1, precision=Precision.FP32),
            analyzer.analyze_model('resnet18', 'H100', batch_size=4, precision=Precision.FP32),
            analyzer.analyze_model('resnet18', 'H100', batch_size=16, precision=Precision.FP32),
        ]

        comparison = self.generator.generate_comparison_report(
            results,
            format='text'
        )

        # Check structure
        self.assertIn("COMPARISON REPORT", comparison)
        self.assertIn("Batch 1", comparison)
        self.assertIn("Batch 4", comparison)
        self.assertIn("Batch 16", comparison)

        # Check metrics
        self.assertIn("Latency", comparison)
        self.assertIn("Throughput", comparison)
        self.assertIn("Energy", comparison)

    def test_comparison_report_csv(self):
        """Test comparison report CSV format"""
        analyzer = UnifiedAnalyzer(verbose=False)

        results = [
            analyzer.analyze_model('resnet18', 'H100', batch_size=1, precision=Precision.FP32),
            analyzer.analyze_model('resnet18', 'H100', batch_size=4, precision=Precision.FP32),
        ]

        csv_str = self.generator.generate_comparison_report(
            results,
            format='csv'
        )

        # Parse CSV
        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)

        # Should have 2 rows
        self.assertEqual(len(rows), 2)

        # Check columns
        self.assertIn('name', rows[0])
        self.assertIn('latency_ms', rows[0])
        self.assertIn('throughput_fps', rows[0])

    def test_comparison_report_markdown(self):
        """Test comparison report Markdown format"""
        analyzer = UnifiedAnalyzer(verbose=False)

        results = [
            analyzer.analyze_model('resnet18', 'H100', batch_size=1, precision=Precision.FP32),
            analyzer.analyze_model('resnet18', 'H100', batch_size=4, precision=Precision.FP32),
        ]

        md = self.generator.generate_comparison_report(
            results,
            format='markdown'
        )

        # Check Markdown table
        self.assertIn('# Comparison Report', md)
        self.assertIn('| Name |', md)
        self.assertIn('Batch 1', md)
        self.assertIn('Batch 4', md)

    def test_save_report_json(self):
        """Test saving report to JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            self.generator.save_report(self.result, output_file)

            # Check file exists and is valid JSON
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, 'r') as f:
                data = json.load(f)

            self.assertIn('metadata', data)
            self.assertEqual(data['metadata']['model'], 'ResNet-18')

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_save_report_csv(self):
        """Test saving report to CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            self.generator.save_report(self.result, output_file)

            # Check file exists and is valid CSV
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertIn('model', rows[0])

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_save_report_markdown(self):
        """Test saving report to Markdown file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_file = f.name

        try:
            self.generator.save_report(self.result, output_file)

            # Check file exists
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, 'r') as f:
                content = f.read()

            self.assertIn('# Analysis Report:', content)
            self.assertIn('## Executive Summary', content)

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_save_comparison_report(self):
        """Test saving comparison report"""
        analyzer = UnifiedAnalyzer(verbose=False)

        results = [
            analyzer.analyze_model('resnet18', 'H100', batch_size=1, precision=Precision.FP32),
            analyzer.analyze_model('resnet18', 'H100', batch_size=4, precision=Precision.FP32),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            self.generator.save_comparison_report(results, output_file)

            # Check file exists and is valid CSV
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_format_auto_detection(self):
        """Test automatic format detection from file extension"""
        test_cases = [
            ('.json', 'metadata'),
            ('.csv', 'model,hardware'),
            ('.md', '# Analysis Report'),
            ('.txt', 'COMPREHENSIVE ANALYSIS'),
        ]

        for ext, expected_content in test_cases:
            with self.subTest(extension=ext):
                with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                    output_file = f.name

                try:
                    self.generator.save_report(self.result, output_file)

                    with open(output_file, 'r') as f:
                        content = f.read()

                    self.assertIn(expected_content, content)

                finally:
                    if os.path.exists(output_file):
                        os.unlink(output_file)


if __name__ == '__main__':
    unittest.main()
