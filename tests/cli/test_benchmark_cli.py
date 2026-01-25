"""
Tests for Benchmark CLI Tool.

Tests cover:
- Listing benchmarks
- Running single benchmarks
- Running benchmark suites
- Device selection
- Output formats
- Filtering
- Error handling
"""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
cli_path = repo_root / "cli"
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(cli_path))

# Import from cli directory
import importlib.util
spec = importlib.util.spec_from_file_location("benchmark", cli_path / "benchmark.py")
benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)

list_benchmarks = benchmark_module.list_benchmarks
find_benchmark_by_name = benchmark_module.find_benchmark_by_name
get_gemm_specs = benchmark_module.get_gemm_specs
get_conv2d_specs = benchmark_module.get_conv2d_specs
get_memory_specs = benchmark_module.get_memory_specs
run_benchmark_suite = benchmark_module.run_benchmark_suite
format_results_table = benchmark_module.format_results_table
save_results_json = benchmark_module.save_results_json
save_results_csv = benchmark_module.save_results_csv

from graphs.benchmarks.schema import ExecutionConfig, Precision


class TestListBenchmarks:
    """Tests for listing available benchmarks."""

    def test_list_all_benchmarks(self):
        benchmarks = list_benchmarks()

        assert "gemm" in benchmarks
        assert "conv2d" in benchmarks
        assert "memory" in benchmarks

        assert len(benchmarks["gemm"]) > 0
        assert len(benchmarks["conv2d"]) > 0
        assert len(benchmarks["memory"]) > 0

    def test_list_gemm_suite(self):
        benchmarks = list_benchmarks(suite="gemm")

        assert "gemm" in benchmarks
        assert "conv2d" not in benchmarks
        assert "memory" not in benchmarks

    def test_list_conv2d_suite(self):
        benchmarks = list_benchmarks(suite="conv2d")

        assert "conv2d" in benchmarks
        assert "gemm" not in benchmarks

    def test_list_memory_suite(self):
        benchmarks = list_benchmarks(suite="memory")

        assert "memory" in benchmarks
        assert "gemm" not in benchmarks


class TestFindBenchmark:
    """Tests for finding benchmarks by name."""

    def test_find_existing_gemm(self):
        spec = find_benchmark_by_name("gemm_128x128_fp32")

        assert spec is not None
        assert spec.name == "gemm_128x128_fp32"
        assert spec.M == 128

    def test_find_existing_conv2d(self):
        spec = find_benchmark_by_name("conv2d_3x3_64_56x56")

        assert spec is not None
        assert spec.name == "conv2d_3x3_64_56x56"

    def test_find_existing_memory(self):
        spec = find_benchmark_by_name("memory_triad_10M")

        assert spec is not None
        assert spec.name == "memory_triad_10M"

    def test_find_nonexistent(self):
        spec = find_benchmark_by_name("nonexistent_benchmark_xyz")

        assert spec is None


class TestLoadSpecs:
    """Tests for loading benchmark specs from YAML."""

    def test_load_gemm_specs(self):
        specs = get_gemm_specs()

        assert len(specs) > 0
        assert all(hasattr(s, 'M') for s in specs)

    def test_load_conv2d_specs(self):
        specs = get_conv2d_specs()

        assert len(specs) > 0
        assert all(hasattr(s, 'in_channels') for s in specs)

    def test_load_memory_specs(self):
        specs = get_memory_specs()

        assert len(specs) > 0
        assert all(hasattr(s, 'pattern') for s in specs)

    def test_filter_gemm_by_tags(self):
        all_specs = get_gemm_specs()
        transformer_specs = get_gemm_specs(filter_tags=["transformer"])

        assert len(transformer_specs) <= len(all_specs)
        for spec in transformer_specs:
            assert "transformer" in spec.tags


class TestRunBenchmarkSuite:
    """Tests for running benchmark suites."""

    @pytest.fixture
    def fast_config(self):
        return ExecutionConfig(
            warmup_iterations=2,
            measurement_iterations=5,
        )

    def test_run_gemm_suite_cpu(self, fast_config):
        # Run just one benchmark to keep test fast
        results = run_benchmark_suite(
            suite="gemm",
            device="cpu",
            precision=Precision.FP32,
            config=fast_config,
            filter_tags=["tiny"],  # Only run smallest benchmark
            quiet=True,
        )

        # Should have at least one result
        assert len(results) >= 1
        # First result should be successful
        assert any(r.success for r in results)

    def test_run_memory_suite_cpu(self, fast_config):
        results = run_benchmark_suite(
            suite="memory",
            device="cpu",
            precision=Precision.FP32,
            config=fast_config,
            quiet=True,
        )

        assert len(results) > 0
        successful = [r for r in results if r.success]
        assert len(successful) > 0

        # Memory benchmarks should have bandwidth
        for r in successful:
            assert r.bandwidth_gbps > 0


class TestOutputFormats:
    """Tests for output format generation."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for formatting tests."""
        from datetime import datetime
        from graphs.benchmarks.schema import BenchmarkResult, TimingStats

        return [
            BenchmarkResult(
                spec_name="gemm_128x128",
                timestamp=datetime.now().isoformat(),
                device="cpu",
                device_name="CPU",
                precision="fp32",
                timing=TimingStats(
                    mean_ms=0.1,
                    std_ms=0.01,
                    min_ms=0.09,
                    max_ms=0.12,
                    median_ms=0.1,
                    p95_ms=0.11,
                    p99_ms=0.12,
                    num_iterations=10,
                ),
                gflops=100.0,
                success=True,
            ),
            BenchmarkResult(
                spec_name="gemm_256x256",
                timestamp=datetime.now().isoformat(),
                device="cpu",
                device_name="CPU",
                precision="fp32",
                timing=TimingStats(
                    mean_ms=0.5,
                    std_ms=0.05,
                    min_ms=0.45,
                    max_ms=0.55,
                    median_ms=0.5,
                    p95_ms=0.54,
                    p99_ms=0.55,
                    num_iterations=10,
                ),
                gflops=200.0,
                success=True,
            ),
        ]

    def test_format_table(self, sample_results):
        table = format_results_table(sample_results)

        assert "gemm_128x128" in table
        assert "gemm_256x256" in table
        assert "cpu" in table
        assert "OK" in table

    def test_save_json(self, sample_results):
        with NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            save_results_json(sample_results, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["spec_name"] == "gemm_128x128"
            assert data[1]["spec_name"] == "gemm_256x256"
        finally:
            output_path.unlink()

    def test_save_csv(self, sample_results):
        with NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            save_results_csv(sample_results, output_path)

            with open(output_path) as f:
                content = f.read()

            assert "spec_name" in content  # Header
            assert "gemm_128x128" in content
            assert "gemm_256x256" in content
        finally:
            output_path.unlink()


class TestCLIIntegration:
    """Integration tests for CLI execution."""

    def test_cli_list(self):
        result = subprocess.run(
            [sys.executable, str(repo_root / "cli" / "benchmark.py"), "--list"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "GEMM" in result.stdout
        assert "CONV2D" in result.stdout
        assert "MEMORY" in result.stdout

    def test_cli_list_suite(self):
        result = subprocess.run(
            [sys.executable, str(repo_root / "cli" / "benchmark.py"),
             "--list", "gemm"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "GEMM" in result.stdout
        assert "CONV2D" not in result.stdout

    def test_cli_single_benchmark(self):
        result = subprocess.run(
            [sys.executable, str(repo_root / "cli" / "benchmark.py"),
             "--benchmark", "gemm_128x128_fp32",
             "--device", "cpu",
             "--iterations", "5",
             "--warmup", "2"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0
        assert "gemm_128x128_fp32" in result.stdout
        assert "OK" in result.stdout

    def test_cli_nonexistent_benchmark(self):
        result = subprocess.run(
            [sys.executable, str(repo_root / "cli" / "benchmark.py"),
             "--benchmark", "nonexistent_xyz"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr

    def test_cli_json_output(self):
        with NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, str(repo_root / "cli" / "benchmark.py"),
                 "--benchmark", "gemm_128x128_fp32",
                 "--device", "cpu",
                 "--iterations", "5",
                 "--warmup", "2",
                 "--output", output_path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0

            with open(output_path) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["spec_name"] == "gemm_128x128_fp32"
            assert data[0]["success"] is True
        finally:
            Path(output_path).unlink()

    def test_cli_csv_output(self):
        with NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, str(repo_root / "cli" / "benchmark.py"),
                 "--benchmark", "gemm_128x128_fp32",
                 "--device", "cpu",
                 "--iterations", "5",
                 "--warmup", "2",
                 "--output", output_path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0

            with open(output_path) as f:
                content = f.read()

            assert "spec_name" in content
            assert "gemm_128x128_fp32" in content
        finally:
            Path(output_path).unlink()

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, str(repo_root / "cli" / "benchmark.py"), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "benchmark" in result.stdout.lower()
        assert "--suite" in result.stdout
        assert "--device" in result.stdout


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_suite(self):
        from graphs.benchmarks.schema import ExecutionConfig

        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)

        with pytest.raises(ValueError, match="Unknown suite"):
            run_benchmark_suite(
                suite="invalid_suite",
                device="cpu",
                precision=Precision.FP32,
                config=config,
                quiet=True,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
