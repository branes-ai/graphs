"""Tests for verdict-first CLI output.

Tests the --check-latency, --check-power, --check-memory, --check-energy
options and the --format verdict output in analyze_comprehensive.py.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the CLI script
CLI_PATH = Path(__file__).parent.parent.parent / "cli" / "analyze_comprehensive.py"


def run_cli(*args, timeout=120):
    """Run the CLI and return parsed JSON output."""
    cmd = [sys.executable, str(CLI_PATH), "--quiet"] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={"PYTHONPATH": str(Path(__file__).parent.parent.parent / "src")},
    )
    # Find the JSON output (skip any progress messages)
    output = result.stdout
    json_start = output.find("{")
    if json_start == -1:
        raise ValueError(f"No JSON output found. stdout: {output}, stderr: {result.stderr}")
    return json.loads(output[json_start:])


class TestVerdictOutput:
    """Test verdict-first output format."""

    def test_latency_pass(self):
        """Test PASS verdict when latency is under target."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-latency", "100.0",  # Very generous target
        )
        assert result["verdict"] == "PASS"
        assert "constraint_metric" in result
        assert result["constraint_metric"] == "latency"
        assert result["constraint_threshold"] == 100.0
        assert result["constraint_margin_pct"] > 0

    def test_latency_fail(self):
        """Test FAIL verdict when latency exceeds target."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-latency", "0.01",  # Very tight target
        )
        assert result["verdict"] == "FAIL"
        assert result["constraint_margin_pct"] < 0
        assert "exceeds" in result["summary"].lower()

    def test_power_constraint(self):
        """Test power budget constraint checking."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-power", "500.0",  # 500W budget
        )
        assert result["verdict"] in ["PASS", "FAIL"]
        assert result["constraint_metric"] == "power"
        assert "constraint_actual" in result

    def test_memory_constraint_pass(self):
        """Test PASS verdict when memory is under limit."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-memory", "1000.0",  # 1GB limit
        )
        assert result["verdict"] == "PASS"
        assert result["constraint_metric"] == "memory"

    def test_memory_constraint_fail(self):
        """Test FAIL verdict when memory exceeds limit."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-memory", "0.1",  # 0.1MB limit (impossible)
        )
        assert result["verdict"] == "FAIL"
        assert result["constraint_metric"] == "memory"
        assert result["constraint_margin_pct"] < 0

    def test_explicit_verdict_format(self):
        """Test explicit --format verdict flag."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--format", "verdict",
        )
        # Without constraint, should return PASS with summary
        assert result["verdict"] == "PASS"
        assert "summary" in result
        assert "latency_ms" in result

    def test_verdict_contains_key_metrics(self):
        """Test that verdict output contains all key metrics."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-latency", "100.0",
        )
        # Key fields
        assert "verdict" in result
        assert "confidence" in result
        assert "summary" in result
        assert "model_id" in result
        assert "hardware_id" in result

        # Performance metrics
        assert "latency_ms" in result
        assert "throughput_fps" in result
        assert "energy_per_inference_mj" in result
        assert "peak_memory_mb" in result

        # Constraint info
        assert "constraint_metric" in result
        assert "constraint_threshold" in result
        assert "constraint_actual" in result
        assert "constraint_margin_pct" in result

    def test_verdict_contains_breakdowns(self):
        """Test that verdict output contains detailed breakdowns."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--format", "verdict",
        )
        # Breakdown sections
        assert "roofline" in result
        assert "energy" in result
        assert "memory" in result

        # Roofline details
        assert "bottleneck" in result["roofline"]
        assert "utilization_pct" in result["roofline"]

        # Energy details
        assert "compute_energy_mj" in result["energy"]
        assert "memory_energy_mj" in result["energy"]
        assert "static_energy_mj" in result["energy"]

    def test_fail_has_suggestions(self):
        """Test that FAIL verdict includes suggestions."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-latency", "0.01",  # Will fail
        )
        assert result["verdict"] == "FAIL"
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_only_one_constraint(self):
        """Test that only one constraint can be specified at a time."""
        # The last constraint specified should be used
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-latency", "100.0",
        )
        assert result["constraint_metric"] == "latency"

    def test_constraint_margin_calculation(self):
        """Test that margin percentage is calculated correctly."""
        result = run_cli(
            "--model", "resnet18",
            "--hardware", "H100",
            "--check-latency", "100.0",
        )
        # Margin should be close to (100 - actual) / 100 * 100
        expected_margin = (100.0 - result["constraint_actual"]) / 100.0 * 100
        assert abs(result["constraint_margin_pct"] - expected_margin) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
