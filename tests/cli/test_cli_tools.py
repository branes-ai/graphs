#!/usr/bin/env python
"""
Integration Tests for Phase 4.1 CLI Tools

Tests the three main CLI tools:
1. analyze_comprehensive.py - Flagship deep-dive analysis tool
2. analyze_batch.py - Batch sweep and comparison tool
3. analyze_graph_mapping.py - Enhanced with Phase 3 analysis flags

These are integration tests that run the actual CLI tools and verify:
- Tools execute successfully
- Output formats are correct
- Analysis modes work
- Error handling is appropriate
"""

import subprocess
import json
import csv
import os
import tempfile
import pytest
from pathlib import Path


# Path to CLI tools
CLI_DIR = Path(__file__).parent.parent.parent / "cli"
ANALYZE_COMPREHENSIVE = CLI_DIR / "analyze_comprehensive.py"
ANALYZE_BATCH = CLI_DIR / "analyze_batch.py"
ANALYZE_GRAPH_MAPPING = CLI_DIR / "analyze_graph_mapping.py"


# =============================================================================
# Helper Functions
# =============================================================================

def run_cli(script, args, timeout=120):
    """
    Run a CLI script and return output

    Args:
        script: Path to script
        args: List of command-line arguments
        timeout: Timeout in seconds

    Returns:
        (returncode, stdout, stderr)
    """
    cmd = ["python", str(script)] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def check_json_output(json_str):
    """Validate JSON output and return parsed data"""
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON output: {e}")


def check_csv_output(csv_file):
    """Validate CSV output and return rows"""
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows
    except Exception as e:
        pytest.fail(f"Invalid CSV output: {e}")


# =============================================================================
# Test analyze_comprehensive.py
# =============================================================================

class TestAnalyzeComprehensive:
    """Tests for analyze_comprehensive.py"""

    def test_basic_execution(self):
        """Test basic execution with default settings"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_COMPREHENSIVE,
            ["--model", "resnet18", "--hardware", "H100", "--quiet"]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "COMPREHENSIVE ANALYSIS" in stdout
        assert "ResNet-18" in stdout
        assert "H100" in stdout

    def test_json_output(self):
        """Test JSON output format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            returncode, stdout, stderr = run_cli(
                ANALYZE_COMPREHENSIVE,
                [
                    "--model", "resnet18",
                    "--hardware", "H100",
                    "--output-format", "json",
                    "--output-file", output_file,
                    "--quiet"
                ]
            )

            assert returncode == 0, f"Tool failed: {stderr}"

            # Load and validate JSON
            with open(output_file, 'r') as f:
                data = json.load(f)

            # Check required fields
            assert 'configuration' in data
            assert 'model_info' in data
            assert 'performance' in data
            assert 'roofline' in data
            assert 'energy' in data
            assert 'memory' in data

            # Check configuration
            assert data['configuration']['model'] == 'ResNet-18'
            assert 'H100' in data['configuration']['hardware']
            assert data['configuration']['precision'] == 'fp32'

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_csv_output(self):
        """Test CSV output format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            returncode, stdout, stderr = run_cli(
                ANALYZE_COMPREHENSIVE,
                [
                    "--model", "resnet18",
                    "--hardware", "H100",
                    "--output-format", "csv",
                    "--output-file", output_file,
                    "--quiet"
                ]
            )

            assert returncode == 0, f"Tool failed: {stderr}"

            # Load and validate CSV
            rows = check_csv_output(output_file)
            assert len(rows) == 1  # Single row for single configuration

            row = rows[0]
            assert row['Model'] == 'ResNet-18'
            assert 'H100' in row['Hardware']
            assert row['Precision'] == 'fp32'
            assert 'Latency_ms' in row
            assert 'Energy_mJ' in row
            assert 'PeakMemory_MB' in row

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_markdown_output(self):
        """Test Markdown output format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_file = f.name

        try:
            returncode, stdout, stderr = run_cli(
                ANALYZE_COMPREHENSIVE,
                [
                    "--model", "resnet18",
                    "--hardware", "H100",
                    "--output-format", "markdown",
                    "--output-file", output_file,
                    "--quiet"
                ]
            )

            assert returncode == 0, f"Tool failed: {stderr}"

            # Load and validate Markdown
            with open(output_file, 'r') as f:
                content = f.read()

            assert "# Comprehensive Analysis" in content
            assert "## Executive Summary" in content
            assert "## Model Information" in content
            assert "## Energy Analysis" in content

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_different_precision(self):
        """Test with different precision settings"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_COMPREHENSIVE,
            [
                "--model", "resnet18",
                "--hardware", "H100",
                "--precision", "fp16",
                "--quiet"
            ]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "fp16" in stdout

    def test_different_model(self):
        """Test with different model"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_COMPREHENSIVE,
            [
                "--model", "mobilenet_v2",
                "--hardware", "H100",
                "--quiet"
            ]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "MobileNet" in stdout

    def test_invalid_model(self):
        """Test error handling for invalid model"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_COMPREHENSIVE,
            ["--model", "invalid_model", "--hardware", "H100", "--quiet"]
        )

        assert returncode != 0, "Tool should fail for invalid model"
        assert "Error" in stderr or "Error" in stdout


# =============================================================================
# Test analyze_batch.py
# =============================================================================

class TestAnalyzeBatch:
    """Tests for analyze_batch.py"""

    def test_batch_size_sweep(self):
        """Test batch size sweep"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            returncode, stdout, stderr = run_cli(
                ANALYZE_BATCH,
                [
                    "--model", "resnet18",
                    "--hardware", "H100",
                    "--batch-size", "1", "2", "4",
                    "--output", output_file,
                    "--quiet"
                ],
                timeout=300
            )

            assert returncode == 0, f"Tool failed: {stderr}"

            # Load and validate CSV
            rows = check_csv_output(output_file)
            assert len(rows) == 3  # 3 batch sizes

            # Check batch sizes
            batch_sizes = [int(row['batch_size']) for row in rows]
            assert batch_sizes == [1, 2, 4]

            # Check that energy per inference decreases with batch size
            energies = [float(row['energy_per_inference_mj']) for row in rows]
            assert energies[0] > energies[1] > energies[2], \
                "Energy per inference should decrease with batch size"

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_model_comparison(self):
        """Test model comparison"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            returncode, stdout, stderr = run_cli(
                ANALYZE_BATCH,
                [
                    "--models", "resnet18", "mobilenet_v2",
                    "--hardware", "H100",
                    "--output", output_file,
                    "--quiet"
                ],
                timeout=300
            )

            assert returncode == 0, f"Tool failed: {stderr}"

            # Load and validate CSV
            rows = check_csv_output(output_file)
            assert len(rows) == 2  # 2 models

            models = [row['model'] for row in rows]
            assert "ResNet-18" in models
            assert "MobileNet-V2" in models

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_hardware_comparison(self):
        """Test hardware comparison"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            returncode, stdout, stderr = run_cli(
                ANALYZE_BATCH,
                [
                    "--model", "resnet18",
                    "--hardware", "H100", "Jetson-Orin",
                    "--output", output_file,
                    "--quiet"
                ],
                timeout=300
            )

            assert returncode == 0, f"Tool failed: {stderr}"

            # Load and validate CSV
            rows = check_csv_output(output_file)
            assert len(rows) == 2  # 2 hardware targets

            hardware = [row['hardware'] for row in rows]
            assert any('H100' in hw for hw in hardware)
            assert any('Jetson' in hw for hw in hardware)

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_json_output_format(self):
        """Test JSON output format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            returncode, stdout, stderr = run_cli(
                ANALYZE_BATCH,
                [
                    "--model", "resnet18",
                    "--hardware", "H100",
                    "--batch-size", "1", "2",
                    "--output", output_file,
                    "--format", "json",
                    "--quiet"
                ],
                timeout=300
            )

            assert returncode == 0, f"Tool failed: {stderr}"

            # Load and validate JSON
            with open(output_file, 'r') as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 2  # 2 batch sizes

            # Check fields
            for item in data:
                assert 'model' in item
                assert 'batch_size' in item
                assert 'latency_ms' in item
                assert 'energy_per_inference_mj' in item

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_show_insights(self):
        """Test batch size insights display"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            returncode, stdout, stderr = run_cli(
                ANALYZE_BATCH,
                [
                    "--model", "resnet18",
                    "--hardware", "H100",
                    "--batch-size", "1", "2", "4", "8",
                    "--output", output_file,
                    "--show-insights"
                ],
                timeout=400
            )

            assert returncode == 0, f"Tool failed: {stderr}"
            assert "BATCH SIZE ANALYSIS" in stdout
            assert "Throughput improvement" in stdout
            assert "Energy/inference improvement" in stdout
            assert "Recommended batch size" in stdout

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


# =============================================================================
# Test analyze_graph_mapping.py (Enhanced)
# =============================================================================

class TestAnalyzeGraphMappingEnhanced:
    """Tests for enhanced analyze_graph_mapping.py"""

    def test_basic_mode_backward_compatibility(self):
        """Test that basic mode works (backward compatibility)"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_GRAPH_MAPPING,
            ["--model", "resnet18", "--hardware", "H100"]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "Graph Mapping Analysis" in stdout
        assert "EXECUTION SUMMARY" in stdout
        # Should NOT have Phase 3 analysis in basic mode
        assert "PHASE 3" not in stdout

    def test_energy_analysis_mode(self):
        """Test energy analysis mode"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_GRAPH_MAPPING,
            [
                "--model", "resnet18",
                "--hardware", "H100",
                "--analysis", "energy",
                "--show-energy-breakdown"
            ]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "ENERGY ANALYSIS" in stdout
        assert "Total Energy" in stdout
        assert "Compute Energy" in stdout
        assert "Static Energy" in stdout
        assert "Energy Breakdown" in stdout

    def test_roofline_analysis_mode(self):
        """Test roofline analysis mode"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_GRAPH_MAPPING,
            [
                "--model", "resnet18",
                "--hardware", "H100",
                "--analysis", "roofline",
                "--show-roofline"
            ]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "ROOFLINE ANALYSIS" in stdout
        assert "Arithmetic Intensity" in stdout
        assert "Bottleneck Distribution" in stdout

    def test_memory_analysis_mode(self):
        """Test memory analysis mode"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_GRAPH_MAPPING,
            [
                "--model", "resnet18",
                "--hardware", "H100",
                "--analysis", "memory"
            ]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "MEMORY ANALYSIS" in stdout
        assert "Peak Memory" in stdout
        assert "Hardware Fit" in stdout

    def test_full_analysis_mode(self):
        """Test full analysis mode (roofline + energy + memory)"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_GRAPH_MAPPING,
            [
                "--model", "resnet18",
                "--hardware", "H100",
                "--analysis", "full"
            ]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "PHASE 3 ADVANCED ANALYSIS" in stdout
        assert "ROOFLINE ANALYSIS" in stdout
        assert "ENERGY ANALYSIS" in stdout
        assert "MEMORY ANALYSIS" in stdout

    def test_all_visualizations(self):
        """Test all visualization flags together"""
        returncode, stdout, stderr = run_cli(
            ANALYZE_GRAPH_MAPPING,
            [
                "--model", "resnet18",
                "--hardware", "H100",
                "--analysis", "full",
                "--show-energy-breakdown",
                "--show-roofline",
                "--show-memory-timeline"
            ]
        )

        assert returncode == 0, f"Tool failed: {stderr}"
        assert "Energy Breakdown" in stdout
        assert "Roofline Plot" in stdout or "Operations breakdown" in stdout


# =============================================================================
# Cross-Tool Integration Tests
# =============================================================================

class TestCrossToolIntegration:
    """Tests that verify consistency across tools"""

    def test_consistent_results_comprehensive_vs_batch(self):
        """Test that analyze_comprehensive and analyze_batch give consistent results"""

        # Run analyze_comprehensive
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            comp_output = f.name

        try:
            run_cli(
                ANALYZE_COMPREHENSIVE,
                [
                    "--model", "resnet18",
                    "--hardware", "H100",
                    "--output-format", "json",
                    "--output-file", comp_output,
                    "--quiet"
                ]
            )

            with open(comp_output, 'r') as f:
                comp_data = json.load(f)

            # Run analyze_batch
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                batch_output = f.name

            try:
                run_cli(
                    ANALYZE_BATCH,
                    [
                        "--model", "resnet18",
                        "--hardware", "H100",
                        "--batch-size", "1",
                        "--output", batch_output,
                        "--format", "json",
                        "--quiet"
                    ],
                    timeout=180
                )

                with open(batch_output, 'r') as f:
                    batch_data = json.load(f)

                # Compare key metrics (allow small differences due to different partitioners)
                comp_energy = comp_data['energy']['total_energy_mj']
                batch_energy = batch_data[0]['total_energy_mj']

                # Energy should be similar (within 50% due to different partitioning)
                ratio = comp_energy / batch_energy if batch_energy > 0 else 1
                assert 0.5 < ratio < 2.0, \
                    f"Energy mismatch: comprehensive={comp_energy}, batch={batch_energy}"

            finally:
                if os.path.exists(batch_output):
                    os.unlink(batch_output)

        finally:
            if os.path.exists(comp_output):
                os.unlink(comp_output)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Tests to ensure tools run in reasonable time"""

    def test_comprehensive_performance(self):
        """Test that analyze_comprehensive completes in reasonable time"""
        import time

        start = time.time()
        returncode, stdout, stderr = run_cli(
            ANALYZE_COMPREHENSIVE,
            ["--model", "resnet18", "--hardware", "H100", "--quiet"],
            timeout=60
        )
        elapsed = time.time() - start

        assert returncode == 0, f"Tool failed: {stderr}"
        assert elapsed < 30, f"Tool too slow: {elapsed:.1f}s (should be < 30s)"

    def test_batch_sweep_performance(self):
        """Test that batch sweep completes in reasonable time"""
        import time

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            start = time.time()
            returncode, stdout, stderr = run_cli(
                ANALYZE_BATCH,
                [
                    "--model", "resnet18",
                    "--hardware", "H100",
                    "--batch-size", "1", "2", "4",
                    "--output", output_file,
                    "--quiet"
                ],
                timeout=300
            )
            elapsed = time.time() - start

            assert returncode == 0, f"Tool failed: {stderr}"
            assert elapsed < 180, f"Tool too slow: {elapsed:.1f}s (should be < 180s for 3 configs)"

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
