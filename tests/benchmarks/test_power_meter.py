"""
Tests for power measurement backends (power_meter.py).

All tests use mocked system interfaces so they run on any host,
including CI runners without RAPL, tegrastats, or NVML.
"""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

from graphs.benchmarks.collectors import PowerMeasurement
from graphs.benchmarks.power_meter import (
    RAPLPowerCollector,
    TegrastatsPowerCollector,
    NoOpPowerCollector,
    auto_select_power_collector,
)


class TestNoOpPowerCollector:
    """NoOp always succeeds start/stop but reports no energy."""

    def test_returns_unsuccessful_measurement(self):
        c = NoOpPowerCollector()
        c.start()
        time.sleep(0.01)
        c.stop()
        m = c.get_measurement()
        assert isinstance(m, PowerMeasurement)
        assert not m.success
        assert m.energy_joules == 0.0

    def test_duration_is_positive(self):
        c = NoOpPowerCollector()
        c.start()
        time.sleep(0.01)
        c.stop()
        assert c.duration_ms > 0


class TestRAPLPowerCollector:
    """RAPL collector with mocked sysfs reads."""

    def _make_collector(self):
        with patch("pathlib.Path.read_text", return_value="0"):
            return RAPLPowerCollector(domain="intel-rapl:0")

    def test_energy_delta(self):
        collector = self._make_collector()

        reads = iter(["1000000", "2000000"])

        with patch("pathlib.Path.read_text", side_effect=lambda: next(reads)):
            collector.start()
            time.sleep(0.01)
            collector.stop()

        m = collector.get_measurement()
        assert m.success
        assert m.energy_joules == pytest.approx(1.0, abs=0.01)
        assert m.avg_power_watts > 0

    def test_counter_wraparound(self):
        collector = self._make_collector()
        collector._max_energy_uj = 10_000_000

        reads = iter(["9000000", "1000000"])

        with patch("pathlib.Path.read_text", side_effect=lambda: next(reads)):
            collector.start()
            time.sleep(0.01)
            collector.stop()

        m = collector.get_measurement()
        assert m.success
        assert m.energy_joules == pytest.approx(2.0, abs=0.01)

    def test_negative_delta_without_max_reports_failure(self):
        collector = self._make_collector()
        collector._max_energy_uj = None

        reads = iter(["5000000", "2000000"])

        with patch("pathlib.Path.read_text", side_effect=lambda: next(reads)):
            collector.start()
            time.sleep(0.01)
            collector.stop()

        m = collector.get_measurement()
        assert not m.success
        assert m.energy_joules == 0.0

    def test_reset_clears_state(self):
        collector = self._make_collector()
        collector._energy_start_uj = 999
        collector._energy_end_uj = 999
        collector.reset()
        assert collector._energy_start_uj == 0
        assert collector._energy_end_uj == 0


class TestTegrastatsPowerCollector:
    """Tegrastats collector with mocked subprocess."""

    def test_parses_power_rails(self):
        sample_line = (
            b"RAM 3456/7620MB (lfb 1234x4MB) SWAP 0/3810MB "
            b"VDD_GPU_SOC 2596mW/2596mW VDD_CPU_CV 1534mW/1534mW\n"
        )

        mock_proc = MagicMock()
        mock_proc.stdout = iter([sample_line])

        with patch("subprocess.Popen", return_value=mock_proc):
            c = TegrastatsPowerCollector(interval_ms=100)
            c.start()
            if c._reader_thread:
                c._reader_thread.join(timeout=1.0)
            c.stop()

        m = c.get_measurement()
        assert m.success
        expected_watts = (2596 + 1534) / 1000.0
        assert len(m.samples) >= 1
        assert m.samples[0] == pytest.approx(expected_watts, abs=0.01)

    def test_no_samples_returns_failure(self):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])

        with patch("subprocess.Popen", return_value=mock_proc):
            c = TegrastatsPowerCollector(interval_ms=100)
            c.start()
            c.stop()

        m = c.get_measurement()
        assert not m.success


class TestAutoSelect:
    """auto_select_power_collector priority logic."""

    def test_cuda_prefers_nvml(self):
        with patch("graphs.benchmarks.power_meter._nvml_available", return_value=True):
            from graphs.benchmarks.collectors import PowerCollector
            c = auto_select_power_collector("cuda:0")
            assert isinstance(c, PowerCollector)

    def test_cpu_prefers_rapl_when_available(self):
        with (
            patch("graphs.benchmarks.power_meter._rapl_available", return_value=True),
            patch("pathlib.Path.read_text", return_value="0"),
        ):
            c = auto_select_power_collector("cpu")
            assert isinstance(c, RAPLPowerCollector)

    def test_falls_back_to_tegrastats(self):
        with (
            patch("graphs.benchmarks.power_meter._rapl_available", return_value=False),
            patch("graphs.benchmarks.power_meter._tegrastats_available", return_value=True),
        ):
            c = auto_select_power_collector("cpu")
            assert isinstance(c, TegrastatsPowerCollector)

    def test_falls_back_to_noop(self):
        with (
            patch("graphs.benchmarks.power_meter._nvml_available", return_value=False),
            patch("graphs.benchmarks.power_meter._rapl_available", return_value=False),
            patch("graphs.benchmarks.power_meter._tegrastats_available", return_value=False),
        ):
            c = auto_select_power_collector("cpu")
            assert isinstance(c, NoOpPowerCollector)

    def test_cuda_without_nvml_returns_noop(self):
        with (
            patch("graphs.benchmarks.power_meter._nvml_available", return_value=False),
            patch("graphs.benchmarks.power_meter._tegrastats_available", return_value=False),
        ):
            c = auto_select_power_collector("cuda:0")
            assert isinstance(c, NoOpPowerCollector)

    def test_cuda_without_nvml_prefers_tegrastats_on_jetson(self):
        with (
            patch("graphs.benchmarks.power_meter._nvml_available", return_value=False),
            patch("graphs.benchmarks.power_meter._tegrastats_available", return_value=True),
        ):
            c = auto_select_power_collector("cuda")
            assert isinstance(c, TegrastatsPowerCollector)


class TestRunnerPowerIntegration:
    """PyTorchRunner populates energy fields when power is available."""

    def test_runner_enables_power_by_default(self):
        from graphs.benchmarks.runner import PyTorchRunner
        runner = PyTorchRunner()
        assert runner._enable_power is True

    def test_runner_can_disable_power(self):
        from graphs.benchmarks.runner import PyTorchRunner
        runner = PyTorchRunner(enable_power_measurement=False)
        assert runner._enable_power is False

    def test_attach_power_populates_fields(self):
        from graphs.benchmarks.runner import PyTorchRunner
        from graphs.benchmarks.schema import BenchmarkResult

        result = BenchmarkResult(
            spec_name="test",
            timestamp="2026-04-16T00:00:00Z",
            device="cpu",
        )
        assert result.energy_joules is None

        mock_collector = MagicMock()
        mock_collector.get_measurement.return_value = PowerMeasurement(
            duration_ms=100.0,
            avg_power_watts=50.0,
            peak_power_watts=60.0,
            energy_joules=5.0,
            success=True,
        )

        result = PyTorchRunner._attach_power(result, mock_collector)
        assert result.energy_joules == 5.0
        assert result.avg_power_watts == 50.0
        assert result.peak_power_watts == 60.0

    def test_attach_power_noop_on_failure(self):
        from graphs.benchmarks.runner import PyTorchRunner
        from graphs.benchmarks.schema import BenchmarkResult

        result = BenchmarkResult(
            spec_name="test",
            timestamp="2026-04-16T00:00:00Z",
            device="cpu",
        )

        mock_collector = MagicMock()
        mock_collector.get_measurement.return_value = PowerMeasurement(
            duration_ms=100.0,
            success=False,
            error_message="no power",
        )

        result = PyTorchRunner._attach_power(result, mock_collector)
        assert result.energy_joules is None

    def test_attach_power_handles_none_collector(self):
        from graphs.benchmarks.runner import PyTorchRunner
        from graphs.benchmarks.schema import BenchmarkResult

        result = BenchmarkResult(
            spec_name="test",
            timestamp="2026-04-16T00:00:00Z",
            device="cpu",
        )
        result = PyTorchRunner._attach_power(result, None)
        assert result.energy_joules is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
