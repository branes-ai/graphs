"""
Tests for Calibration Fitting CLI Tool.

Tests cover:
- Loading benchmark results from JSON
- Roofline fitting mode
- Energy coefficient fitting mode
- Utilization curve fitting mode
- Combined "all" mode
- Output file generation (JSON/YAML)
- Report generation (Markdown)
- Incremental updates
- Error handling
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
cli_path = repo_root / "cli"
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(cli_path))

from graphs.benchmarks.schema import BenchmarkResult, TimingStats


# =============================================================================
# FIXTURES
# =============================================================================


def _make_timing(time_ms: float) -> TimingStats:
    """Create a TimingStats object."""
    return TimingStats(
        mean_ms=time_ms,
        std_ms=time_ms * 0.05,
        min_ms=time_ms * 0.9,
        max_ms=time_ms * 1.1,
        median_ms=time_ms,
        p95_ms=time_ms * 1.05,
        p99_ms=time_ms * 1.08,
        num_iterations=100,
    )


def _make_gemm_result(
    M: int,
    N: int,
    K: int,
    time_ms: float,
    precision: str = "fp32",
    gflops: float = None,
    power_watts: float = None,
) -> dict:
    """Create a benchmark result dict for GEMM."""
    flops = 2 * M * N * K
    if gflops is None:
        gflops = (flops / 1e9) / (time_ms / 1000)

    result = BenchmarkResult(
        spec_name=f"gemm_{M}x{N}x{K}_{precision}",
        timestamp=datetime.now().isoformat(),
        device="cuda:0",
        device_name="Test GPU",
        precision=precision,
        timing=_make_timing(time_ms),
        gflops=gflops,
        avg_power_watts=power_watts,
        success=True,
        extra={
            "flops": flops,
            "bytes_transferred": 4 * (M * K + K * N + M * N),  # FP32
        },
    )
    return result.to_dict()


def _make_memory_result(
    array_size: int,
    time_ms: float,
    pattern: str = "triad",
    power_watts: float = None,
) -> dict:
    """Create a benchmark result dict for memory benchmark."""
    # Triad: 2 reads + 1 write = 3 * 8 bytes per element (FP64)
    bytes_transferred = 3 * 8 * array_size
    bandwidth_gbps = (bytes_transferred / 1e9) / (time_ms / 1000)

    result = BenchmarkResult(
        spec_name=f"memory_{pattern}_{array_size // 1_000_000}M",
        timestamp=datetime.now().isoformat(),
        device="cuda:0",
        device_name="Test GPU",
        precision="fp64",
        timing=_make_timing(time_ms),
        bandwidth_gbps=bandwidth_gbps,
        avg_power_watts=power_watts,
        success=True,
        extra={
            "bytes_transferred": bytes_transferred,
        },
    )
    return result.to_dict()


@pytest.fixture
def synthetic_results() -> list:
    """Create synthetic benchmark results for testing."""
    results = []

    # GEMM results at different sizes (for roofline & utilization)
    sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    for M, N, K in sizes:
        flops = 2 * M * N * K
        # Simulate increasing utilization with size
        peak_gflops = 50000
        utilization = 0.85 * (1 - 0.5 ** (flops / 1e10))
        actual_gflops = peak_gflops * utilization
        time_ms = (flops / 1e9) / actual_gflops * 1000

        # Add power measurement
        power = 100 + 250 * utilization  # 100W idle + 250W dynamic

        results.append(_make_gemm_result(
            M, N, K, time_ms,
            gflops=actual_gflops,
            power_watts=power,
        ))

    # Memory benchmark results (for roofline bandwidth)
    array_sizes = [1_000_000, 10_000_000, 100_000_000]
    for size in array_sizes:
        # Simulate ~80% of 2000 GB/s peak
        bandwidth = 1600
        bytes_transferred = 3 * 8 * size
        time_ms = (bytes_transferred / 1e9) / bandwidth * 1000

        power = 120 + (200 * 0.8)

        results.append(_make_memory_result(
            size, time_ms,
            power_watts=power,
        ))

    return results


@pytest.fixture
def results_file(synthetic_results, tmp_path) -> Path:
    """Create a temporary JSON file with synthetic results."""
    file_path = tmp_path / "benchmark_results.json"
    with open(file_path, 'w') as f:
        json.dump(synthetic_results, f, indent=2)
    return file_path


@pytest.fixture
def empty_results_file(tmp_path) -> Path:
    """Create a temporary JSON file with empty results."""
    file_path = tmp_path / "empty_results.json"
    with open(file_path, 'w') as f:
        json.dump([], f)
    return file_path


# =============================================================================
# UNIT TESTS - LOADING
# =============================================================================


class TestLoadBenchmarkResults:
    """Tests for loading benchmark results."""

    def test_load_results_list(self, results_file):
        """Test loading results from a list format."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = module.load_benchmark_results(results_file)

        assert len(results) > 0
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_load_results_dict_with_results_key(self, tmp_path, synthetic_results):
        """Test loading results from dict with 'results' key."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        file_path = tmp_path / "results_wrapped.json"
        with open(file_path, 'w') as f:
            json.dump({"results": synthetic_results}, f)

        results = module.load_benchmark_results(file_path)

        assert len(results) == len(synthetic_results)

    def test_load_empty_results(self, empty_results_file):
        """Test loading empty results file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = module.load_benchmark_results(empty_results_file)

        assert len(results) == 0


# =============================================================================
# UNIT TESTS - FITTING MODES
# =============================================================================


class TestRooflineMode:
    """Tests for roofline fitting mode."""

    def test_fit_roofline_basic(self, results_file):
        """Test basic roofline fitting."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = module.load_benchmark_results(results_file)
        roofline = module.fit_roofline_mode(results, quiet=True)

        # Should return valid roofline parameters
        assert 'error' not in roofline
        assert 'achieved_bandwidth_gbps' in roofline
        assert 'achieved_compute_gflops' in roofline
        assert 'ridge_point' in roofline

    def test_fit_roofline_with_peaks(self, results_file):
        """Test roofline fitting with theoretical peaks."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = module.load_benchmark_results(results_file)
        roofline = module.fit_roofline_mode(
            results,
            peak_bandwidth_gbps=2000.0,
            peak_gflops=50000.0,
            quiet=True,
        )

        assert 'error' not in roofline
        # Should have efficiency values when peaks are provided
        assert 'bandwidth_efficiency' in roofline
        assert 'compute_efficiency' in roofline


class TestEnergyMode:
    """Tests for energy coefficient fitting mode."""

    def test_fit_energy_basic(self, results_file):
        """Test basic energy fitting."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = module.load_benchmark_results(results_file)
        energy = module.fit_energy_mode(results, quiet=True)

        # Should return valid energy coefficients
        assert 'error' not in energy
        assert 'compute_pj_per_op' in energy
        assert 'memory_pj_per_byte' in energy
        assert 'static_power_watts' in energy

    def test_fit_energy_insufficient_data(self, tmp_path):
        """Test energy fitting with insufficient data."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Create file with results without power measurements
        file_path = tmp_path / "no_power.json"
        results = [_make_gemm_result(256, 256, 256, 1.0, power_watts=None)]
        with open(file_path, 'w') as f:
            json.dump(results, f)

        loaded = module.load_benchmark_results(file_path)
        energy = module.fit_energy_mode(loaded, quiet=True)

        assert 'error' in energy


class TestUtilizationMode:
    """Tests for utilization curve fitting mode."""

    def test_fit_utilization_basic(self, results_file):
        """Test basic utilization fitting."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = module.load_benchmark_results(results_file)
        util = module.fit_utilization_mode(
            results,
            peak_gflops=50000.0,
            peak_bandwidth_gbps=2000.0,
            quiet=True,
        )

        # Should return valid utilization profile
        assert 'error' not in util
        assert 'curves' in util

    def test_fit_utilization_no_peak(self, results_file):
        """Test utilization fitting without peak GFLOPS."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = module.load_benchmark_results(results_file)
        util = module.fit_utilization_mode(results, peak_gflops=0, quiet=True)

        assert 'error' in util


# =============================================================================
# UNIT TESTS - OUTPUT GENERATION
# =============================================================================


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_basic(self):
        """Test basic report generation."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        report = module.generate_report(
            roofline={
                'achieved_bandwidth_gbps': 1600.0,
                'achieved_compute_gflops': 42000.0,
                'ridge_point': 26.25,
                'bandwidth_efficiency': 0.8,
                'compute_efficiency': 0.84,
            },
            energy={
                'compute_pj_per_op': 0.5,
                'memory_pj_per_byte': 10.0,
                'static_power_watts': 100.0,
            },
            device_name="Test GPU",
        )

        assert "# Calibration Quality Report" in report
        assert "Test GPU" in report
        assert "Roofline Parameters" in report
        assert "Energy Coefficients" in report

    def test_generate_report_partial(self):
        """Test report generation with partial data."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Only roofline data
        report = module.generate_report(
            roofline={
                'achieved_bandwidth_gbps': 1600.0,
                'achieved_compute_gflops': 42000.0,
                'ridge_point': 26.25,
            },
        )

        assert "Roofline Parameters" in report
        assert "Energy Coefficients" not in report

    def test_generate_report_with_errors(self):
        """Test report generation with error results."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        report = module.generate_report(
            roofline={'error': 'Insufficient data'},
            energy={'error': 'No power measurements'},
        )

        # Errors should not appear in tables
        assert "Roofline Parameters" not in report
        assert "Energy Coefficients" not in report


class TestProfileSaveLoad:
    """Tests for profile saving and loading."""

    def test_save_load_json(self, tmp_path):
        """Test saving and loading JSON profile."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        profile = {
            'metadata': {'device_name': 'Test'},
            'roofline': {'ridge_point': 26.25},
        }

        file_path = tmp_path / "profile.json"
        module.save_profile(profile, file_path)

        loaded = module.load_profile(file_path)

        assert loaded['metadata']['device_name'] == 'Test'
        assert loaded['roofline']['ridge_point'] == 26.25

    def test_save_load_yaml(self, tmp_path):
        """Test saving and loading YAML profile."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fit_calibration", cli_path / "fit_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        profile = {
            'metadata': {'device_name': 'Test'},
            'roofline': {'ridge_point': 26.25},
        }

        file_path = tmp_path / "profile.yaml"
        module.save_profile(profile, file_path)

        loaded = module.load_profile(file_path)

        assert loaded['metadata']['device_name'] == 'Test'
        assert loaded['roofline']['ridge_point'] == 26.25


# =============================================================================
# CLI INTEGRATION TESTS
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI execution."""

    def test_cli_help(self):
        """Test CLI help output."""
        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "--input" in result.stdout
        assert "--output" in result.stdout
        assert "roofline" in result.stdout

    def test_cli_missing_input(self):
        """Test CLI with missing input file."""
        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "roofline",
             "--input", "/nonexistent/file.json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_cli_roofline_mode(self, results_file, tmp_path):
        """Test CLI roofline mode."""
        output_path = tmp_path / "profile.json"

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "roofline",
             "--input", str(results_file),
             "--output", str(output_path),
             "--quiet"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_path.exists()

        with open(output_path) as f:
            profile = json.load(f)

        assert 'roofline' in profile
        assert 'achieved_bandwidth_gbps' in profile['roofline']

    def test_cli_energy_mode(self, results_file, tmp_path):
        """Test CLI energy mode."""
        output_path = tmp_path / "profile.json"

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "energy",
             "--input", str(results_file),
             "--output", str(output_path),
             "--quiet"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_path.exists()

        with open(output_path) as f:
            profile = json.load(f)

        assert 'energy' in profile

    def test_cli_utilization_mode(self, results_file, tmp_path):
        """Test CLI utilization mode."""
        output_path = tmp_path / "profile.json"

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "utilization",
             "--input", str(results_file),
             "--output", str(output_path),
             "--peak-gflops", "50000",
             "--peak-bandwidth", "2000",
             "--quiet"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_path.exists()

        with open(output_path) as f:
            profile = json.load(f)

        assert 'utilization' in profile

    def test_cli_all_mode(self, results_file, tmp_path):
        """Test CLI all mode."""
        output_path = tmp_path / "profile.json"

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "all",
             "--input", str(results_file),
             "--output", str(output_path),
             "--peak-gflops", "50000",
             "--peak-bandwidth", "2000",
             "--device-name", "Test GPU",
             "--quiet"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_path.exists()

        with open(output_path) as f:
            profile = json.load(f)

        assert 'roofline' in profile
        assert 'energy' in profile
        assert 'utilization' in profile
        assert profile['metadata']['device_name'] == "Test GPU"

    def test_cli_report_generation(self, results_file, tmp_path):
        """Test CLI report generation."""
        report_path = tmp_path / "report.md"

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "all",
             "--input", str(results_file),
             "--report", str(report_path),
             "--peak-gflops", "50000",
             "--device-name", "Test GPU",
             "--quiet"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert report_path.exists()

        with open(report_path) as f:
            content = f.read()

        assert "# Calibration Quality Report" in content
        assert "Test GPU" in content

    def test_cli_yaml_output(self, results_file, tmp_path):
        """Test CLI YAML output."""
        output_path = tmp_path / "profile.yaml"

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "roofline",
             "--input", str(results_file),
             "--output", str(output_path),
             "--quiet"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_path.exists()

        import yaml
        with open(output_path) as f:
            profile = yaml.safe_load(f)

        assert 'roofline' in profile

    def test_cli_update_profile(self, results_file, tmp_path):
        """Test CLI incremental update."""
        # Create initial profile
        initial_path = tmp_path / "initial.json"
        initial_profile = {
            'metadata': {'device_name': 'Initial', 'created_at': '2026-01-01T00:00:00'},
            'roofline': {'ridge_point': 10.0},
        }
        with open(initial_path, 'w') as f:
            json.dump(initial_profile, f)

        output_path = tmp_path / "updated.json"

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "energy",
             "--input", str(results_file),
             "--update", str(initial_path),
             "--output", str(output_path),
             "--quiet"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(output_path) as f:
            profile = json.load(f)

        # Should preserve existing roofline and add energy
        assert 'roofline' in profile
        assert profile['roofline']['ridge_point'] == 10.0
        assert 'energy' in profile
        # Should preserve original created_at
        assert profile['metadata']['created_at'] == '2026-01-01T00:00:00'

    def test_cli_empty_results(self, empty_results_file):
        """Test CLI with empty results file."""
        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "all",
             "--input", str(empty_results_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "no valid" in result.stderr.lower()


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json_file(self, tmp_path):
        """Test handling of invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("not valid json{}")

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "roofline",
             "--input", str(invalid_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1

    def test_results_without_timing(self, tmp_path):
        """Test results that lack timing data."""
        # Create minimal results without timing
        results = [{
            'spec_name': 'test',
            'timestamp': '2026-01-01T00:00:00',
            'device': 'cpu',
            'success': True,
        }]
        file_path = tmp_path / "no_timing.json"
        with open(file_path, 'w') as f:
            json.dump(results, f)

        result = subprocess.run(
            [sys.executable, str(cli_path / "fit_calibration.py"),
             "--mode", "roofline",
             "--input", str(file_path),
             "--quiet"],
            capture_output=True,
            text=True,
        )

        # Should handle gracefully
        assert result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
