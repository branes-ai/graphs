"""Tests for verdict-first batch sweep CLI output.

Tests the --check-latency, --check-power, --check-memory, --check-energy
options and the --format verdict output in analyze_batch.py.
"""

import json
from pathlib import Path

import pytest

# Path to the CLI script
CLI_PATH = Path(__file__).parent.parent.parent / "cli" / "analyze_batch.py"


@pytest.fixture
def run_cli(cli_runner):
    """Wrap the global cli_runner with this file's CLI_PATH and JSON
    parsing. Returns parsed JSON output (skipping leading progress text)."""
    def _run(*args, timeout=180):
        rc, stdout, stderr = cli_runner(CLI_PATH, ["--quiet", *args])
        json_start = stdout.find("{")
        if json_start == -1:
            raise ValueError(
                f"No JSON output found. stdout: {stdout}, stderr: {stderr}"
            )
        return json.loads(stdout[json_start:])
    return _run


class TestBatchVerdictOutput:
    """Test verdict-first batch sweep output format."""

    def test_all_pass(self, run_cli):
        """Test PASS verdict when all batch sizes meet constraint."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4", "16",
            "--check-latency", "100.0",  # Very generous target
        )
        assert result["verdict"] == "PASS"
        assert result["passing_count"] == 3
        assert result["failing_count"] == 0
        assert "All" in result["summary"]

    def test_all_fail(self, run_cli):
        """Test FAIL verdict when no batch sizes meet constraint."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4",
            "--check-latency", "0.001",  # Impossible target
        )
        assert result["verdict"] == "FAIL"
        assert result["passing_count"] == 0
        assert result["failing_count"] == 2
        assert "No configurations" in result["summary"]
        assert len(result["suggestions"]) > 0

    def test_partial_pass(self, run_cli):
        """Test PARTIAL verdict when some batch sizes meet constraint."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4", "16",
            "--check-latency", "0.5",  # Tight target - batch=16 at ~0.83ms should fail
        )
        assert result["verdict"] == "PARTIAL"
        assert result["passing_count"] > 0
        assert result["failing_count"] > 0
        assert "of" in result["summary"]  # "X of Y configurations..."

    def test_memory_constraint(self, run_cli):
        """Test memory constraint checking."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4",
            "--check-memory", "1000.0",  # 1GB limit
        )
        assert result["verdict"] in ["PASS", "PARTIAL", "FAIL"]
        assert result["constraint"]["metric"] == "memory"
        assert result["constraint"]["threshold"] == 1000.0

    def test_power_constraint(self, run_cli):
        """Test power constraint checking."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4",
            "--check-power", "500.0",  # 500W budget
        )
        assert result["verdict"] in ["PASS", "PARTIAL", "FAIL"]
        assert result["constraint"]["metric"] == "power"

    def test_recommendations_in_passing(self, run_cli):
        """Test that passing configs include recommendations."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4", "16",
            "--check-latency", "100.0",
        )
        assert result["verdict"] == "PASS"
        group = result["group_summaries"][0]
        assert "recommendations" in group
        assert "for_latency" in group["recommendations"]
        assert "for_throughput" in group["recommendations"]
        assert "for_energy_efficiency" in group["recommendations"]

    def test_max_batch_suggestion_in_partial(self, run_cli):
        """Test that PARTIAL verdict includes max batch size suggestion."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4", "16",
            "--check-latency", "1.0",
        )
        if result["verdict"] == "PARTIAL":
            assert any("Maximum batch size" in s for s in result["suggestions"])


class TestBatchVerdictFormat:
    """Test explicit verdict format output."""

    def test_explicit_verdict_format(self, run_cli):
        """Test --format verdict without constraint."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4",
            "--format", "verdict",
        )
        assert result["verdict"] == "PASS"
        assert "total_configs" in result
        assert "configs" in result
        assert len(result["configs"]) == 2

    def test_verdict_has_all_metrics(self, run_cli):
        """Test that verdict output contains all metrics per config."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1", "4",
            "--check-latency", "100.0",
        )
        # Check first passing config has all metrics
        config = result["passing_configs"][0]
        assert "model" in config
        assert "hardware" in config
        assert "batch_size" in config
        assert "latency_ms" in config
        assert "throughput_fps" in config
        assert "energy_per_inference_mj" in config
        assert "peak_memory_mb" in config
        assert "margin_pct" in config


class TestBatchVerdictEdgeCases:
    """Test edge cases for batch verdict output."""

    def test_single_batch_size(self, run_cli):
        """Test with single batch size."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1",
            "--check-latency", "100.0",
        )
        assert result["total_configs"] == 1
        assert result["verdict"] == "PASS"

    def test_margin_calculation(self, run_cli):
        """Test that margin percentage is calculated correctly."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--batch-size", "1",
            "--check-latency", "100.0",
        )
        config = result["passing_configs"][0]
        # Margin should be (threshold - actual) / threshold * 100
        expected_margin = (100.0 - config["latency_ms"]) / 100.0 * 100
        assert abs(config["margin_pct"] - expected_margin) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
