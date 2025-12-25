"""Tests for Pydantic output adapters.

Tests the conversion from UnifiedAnalysisResult to embodied-schemas
Pydantic models with verdict-first output.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

# Skip all tests if embodied-schemas is not installed
pytest.importorskip("embodied_schemas")

from embodied_schemas import (
    Verdict,
    Confidence,
    Bottleneck,
    RooflineResult,
    EnergyResult,
    MemoryResult,
    GraphAnalysisResult,
)

from graphs.adapters.pydantic_output import (
    make_verdict,
    convert_roofline_to_pydantic,
    convert_energy_to_pydantic,
    convert_memory_to_pydantic,
    convert_to_pydantic,
)
from graphs.analysis.roofline import RooflineReport, LatencyDescriptor
from graphs.analysis.energy import EnergyReport, EnergyDescriptor
from graphs.analysis.memory import MemoryReport


class TestMakeVerdict:
    """Tests for the make_verdict helper function."""

    def test_pass_with_headroom(self):
        """Test PASS verdict when actual is under threshold."""
        verdict, margin, summary = make_verdict(
            actual=8.0,
            threshold=10.0,
            metric="Latency",
            lower_is_better=True
        )
        assert verdict == Verdict.PASS
        assert margin == pytest.approx(20.0)
        assert "meets" in summary or "under" in summary

    def test_fail_over_threshold(self):
        """Test FAIL verdict when actual exceeds threshold."""
        verdict, margin, summary = make_verdict(
            actual=15.0,
            threshold=10.0,
            metric="Latency",
            lower_is_better=True
        )
        assert verdict == Verdict.FAIL
        assert margin == pytest.approx(-50.0)
        assert "exceeds" in summary

    def test_higher_is_better(self):
        """Test verdict when higher values are better (e.g., throughput)."""
        verdict, margin, summary = make_verdict(
            actual=120.0,
            threshold=100.0,
            metric="Throughput",
            lower_is_better=False
        )
        assert verdict == Verdict.PASS
        assert margin == pytest.approx(20.0)


class TestConvertRoofline:
    """Tests for RooflineReport to RooflineResult conversion."""

    @pytest.fixture
    def sample_roofline_report(self):
        """Create a sample RooflineReport."""
        latencies = [
            LatencyDescriptor(
                subgraph_id="sg_0",
                subgraph_name="conv_block_0",
                compute_time=0.005,
                memory_time=0.002,
                actual_latency=0.005,
                overhead=0.0,
                bottleneck="compute",
                bottleneck_ratio=2.5,
                arithmetic_intensity=100.0,
                arithmetic_intensity_breakpoint=156.0,
                attained_flops=200e12,
                peak_flops=312e12,
                flops_utilization=0.64,
            ),
            LatencyDescriptor(
                subgraph_id="sg_1",
                subgraph_name="conv_block_1",
                compute_time=0.002,
                memory_time=0.005,
                actual_latency=0.005,
                overhead=0.0,
                bottleneck="memory",
                bottleneck_ratio=2.5,
                arithmetic_intensity=10.0,
                arithmetic_intensity_breakpoint=156.0,
                attained_flops=50e12,
                peak_flops=312e12,
                flops_utilization=0.16,
            ),
        ]
        return RooflineReport(
            peak_flops=312e12,
            peak_bandwidth=2000e9,
            arithmetic_intensity_breakpoint=156.0,
            latencies=latencies,
            total_latency=0.010,
            total_compute_time=0.007,
            total_memory_time=0.007,
            total_overhead=0.0,
            num_compute_bound=1,
            num_memory_bound=1,
            num_balanced=0,
            average_flops_utilization=0.4,
            average_bandwidth_utilization=0.5,
        )

    def test_convert_roofline_basic(self, sample_roofline_report):
        """Test basic roofline conversion."""
        result = convert_roofline_to_pydantic(sample_roofline_report, 10.0)

        assert isinstance(result, RooflineResult)
        assert result.latency_ms == 10.0
        assert result.peak_flops == 312e12
        assert result.peak_bandwidth_gbps == 2000.0
        assert result.ridge_point == 156.0

    def test_convert_roofline_bottleneck(self, sample_roofline_report):
        """Test bottleneck classification."""
        # Equal compute and memory bound -> balanced
        result = convert_roofline_to_pydantic(sample_roofline_report, 10.0)
        assert result.bottleneck == Bottleneck.BALANCED

        # More compute bound
        sample_roofline_report.num_compute_bound = 3
        sample_roofline_report.num_memory_bound = 1
        result = convert_roofline_to_pydantic(sample_roofline_report, 10.0)
        assert result.bottleneck == Bottleneck.COMPUTE_BOUND

    def test_convert_roofline_utilization(self, sample_roofline_report):
        """Test utilization percentage conversion."""
        result = convert_roofline_to_pydantic(sample_roofline_report, 10.0)
        assert result.utilization_pct == pytest.approx(40.0)


class TestConvertEnergy:
    """Tests for EnergyReport to EnergyResult conversion."""

    @pytest.fixture
    def sample_energy_report(self):
        """Create a sample EnergyReport."""
        return EnergyReport(
            total_energy_j=0.150,
            total_energy_mj=150.0,
            energy_per_inference_j=0.150,
            compute_energy_j=0.080,
            memory_energy_j=0.050,
            static_energy_j=0.020,
            average_efficiency=0.75,
            average_utilization=0.80,
            wasted_energy_j=0.010,
            wasted_energy_percent=6.7,
            average_power_w=200.0,
            peak_power_w=350.0,
            total_latency_s=0.010,
            energy_descriptors=[],
            total_allocated_units_energy_j=0.015,
            total_unallocated_units_energy_j=0.005,
            total_power_gating_savings_j=0.005,
            power_gating_enabled=True,
            average_allocated_units=80.0,
        )

    def test_convert_energy_basic(self, sample_energy_report):
        """Test basic energy conversion."""
        result = convert_energy_to_pydantic(sample_energy_report)

        assert isinstance(result, EnergyResult)
        assert result.total_energy_mj == 150.0
        assert result.compute_energy_mj == pytest.approx(80.0)
        assert result.memory_energy_mj == pytest.approx(50.0)
        assert result.static_energy_mj == pytest.approx(20.0)

    def test_convert_energy_power(self, sample_energy_report):
        """Test power metrics conversion."""
        result = convert_energy_to_pydantic(sample_energy_report, hardware_tdp_w=450.0)

        assert result.average_power_w == 200.0
        assert result.peak_power_w == 350.0
        assert result.tdp_w == 450.0

    def test_convert_energy_power_gating(self, sample_energy_report):
        """Test power gating fields."""
        result = convert_energy_to_pydantic(sample_energy_report)

        assert result.power_gating_enabled is True
        assert result.power_gating_savings_mj == pytest.approx(5.0)


class TestConvertMemory:
    """Tests for MemoryReport to MemoryResult conversion."""

    @pytest.fixture
    def sample_memory_report(self):
        """Create a sample MemoryReport."""
        return MemoryReport(
            peak_memory_bytes=1024 * 1024 * 1024,  # 1 GB
            peak_memory_mb=1024.0,
            peak_memory_gb=1.0,
            activation_memory_bytes=600 * 1024 * 1024,
            weight_memory_bytes=400 * 1024 * 1024,
            workspace_memory_bytes=24 * 1024 * 1024,
            average_memory_bytes=800 * 1024 * 1024,
            memory_utilization=0.78,
            fragmentation_waste_bytes=0,
            fits_in_l2_cache=False,
            fits_in_shared_memory=False,
            fits_on_device=True,
            l2_cache_size_bytes=50 * 1024 * 1024,
            device_memory_bytes=80 * 1024 * 1024 * 1024,
        )

    def test_convert_memory_basic(self, sample_memory_report):
        """Test basic memory conversion."""
        result = convert_memory_to_pydantic(sample_memory_report)

        assert isinstance(result, MemoryResult)
        assert result.peak_memory_mb == 1024.0
        assert result.weights_mb == pytest.approx(400.0)
        assert result.activations_mb == pytest.approx(600.0)

    def test_convert_memory_fit(self, sample_memory_report):
        """Test hardware fit analysis."""
        result = convert_memory_to_pydantic(sample_memory_report)

        assert result.fits_in_l2 is False
        assert result.fits_in_device_memory is True
        assert result.l2_cache_mb == pytest.approx(50.0)
        assert result.device_memory_mb == pytest.approx(80 * 1024)

    def test_convert_memory_utilization(self, sample_memory_report):
        """Test memory utilization calculation."""
        result = convert_memory_to_pydantic(sample_memory_report)

        # 1 GB / 80 GB = 1.25%
        expected_util = (1024 * 1024 * 1024) / (80 * 1024 * 1024 * 1024) * 100
        assert result.memory_utilization_pct == pytest.approx(expected_util, rel=0.01)


def create_mock_unified_result():
    """Create a mock UnifiedAnalysisResult with all required fields."""
    from graphs.hardware.resource_model import Precision

    # Create mock hardware
    hardware = MagicMock()
    hardware.name = "H100-SXM5-80GB"
    hardware.tdp = 700.0

    # Create roofline report
    roofline = RooflineReport(
        peak_flops=312e12,
        peak_bandwidth=2000e9,
        arithmetic_intensity_breakpoint=156.0,
        latencies=[
            LatencyDescriptor(
                subgraph_id="sg_0",
                subgraph_name="conv_block_0",
                compute_time=0.005,
                memory_time=0.002,
                actual_latency=0.005,
                overhead=0.0,
                bottleneck="compute",
                bottleneck_ratio=2.5,
                arithmetic_intensity=100.0,
                arithmetic_intensity_breakpoint=156.0,
                attained_flops=250e12,
                peak_flops=312e12,
                flops_utilization=0.8,
            ),
        ],
        total_latency=0.010,
        num_compute_bound=1,
        num_memory_bound=0,
        num_balanced=0,
        average_flops_utilization=0.8,
        average_bandwidth_utilization=0.5,
    )

    # Create energy report
    energy = EnergyReport(
        total_energy_j=0.150,
        total_energy_mj=150.0,
        energy_per_inference_j=0.150,
        compute_energy_j=0.080,
        memory_energy_j=0.050,
        static_energy_j=0.020,
        average_power_w=200.0,
        peak_power_w=350.0,
        total_latency_s=0.010,
        energy_descriptors=[
            EnergyDescriptor(
                subgraph_id="sg_0",
                subgraph_name="conv_block_0",
                compute_ops=int(1e9),
                bytes_transferred=int(1e7),
                compute_energy_j=0.080,
                memory_energy_j=0.050,
                static_energy_j=0.020,
                total_energy_j=0.150,
                efficiency=0.8,
                latency_s=0.005,
            ),
        ],
    )

    # Create memory report
    memory = MemoryReport(
        peak_memory_bytes=1024 * 1024 * 1024,
        peak_memory_mb=1024.0,
        peak_memory_gb=1.0,
        activation_memory_bytes=600 * 1024 * 1024,
        weight_memory_bytes=400 * 1024 * 1024,
        workspace_memory_bytes=24 * 1024 * 1024,
        average_memory_bytes=800 * 1024 * 1024,
        memory_utilization=0.78,
        fits_in_l2_cache=False,
        fits_on_device=True,
        l2_cache_size_bytes=50 * 1024 * 1024,
        device_memory_bytes=80 * 1024 * 1024 * 1024,
    )

    # Create mock partition report using spec to avoid MagicMock auto-attributes
    partition = MagicMock(spec=['subgraphs'])
    sg = MagicMock(spec=['flops', 'memory_traffic', 'dominant_op'])
    sg.flops = 1e9
    sg.memory_traffic = 1e7
    sg.dominant_op = "conv2d"
    partition.subgraphs = [sg]

    # Create mock UnifiedAnalysisResult
    result = MagicMock()
    result.model_name = "resnet18"
    result.display_name = "ResNet-18"
    result.batch_size = 1
    result.precision = Precision.FP32
    result.hardware_name = "H100-SXM5-80GB"
    result.hardware_display_name = "H100-SXM5-80GB"
    result.hardware = hardware
    result.roofline_report = roofline
    result.energy_report = energy
    result.memory_report = memory
    result.partition_report = partition
    result.concurrency_report = None
    result.hardware_allocation = None
    result.total_latency_ms = 10.0
    result.throughput_fps = 100.0
    result.energy_per_inference_mj = 150.0
    result.peak_memory_mb = 1024.0
    result.validation_warnings = []
    result.analysis_timestamp = "2024-12-24T10:30:00"
    result._generate_recommendations = MagicMock(return_value=[])

    return result


class TestConvertToPydantic:
    """Integration tests for full UnifiedAnalysisResult conversion."""

    def test_convert_full_result(self):
        """Test full conversion without constraints."""
        mock_result = create_mock_unified_result()
        result = convert_to_pydantic(mock_result)

        assert isinstance(result, GraphAnalysisResult)
        assert result.model_id == "resnet18"
        assert result.hardware_id == "H100-SXM5-80GB"
        assert result.latency_ms == 10.0
        assert result.throughput_fps == 100.0
        assert result.verdict == Verdict.PASS

    def test_convert_with_passing_constraint(self):
        """Test conversion with passing latency constraint."""
        mock_result = create_mock_unified_result()
        result = convert_to_pydantic(
            mock_result,
            constraint_metric="latency",
            constraint_threshold=20.0
        )

        assert result.verdict == Verdict.PASS
        assert result.constraint_metric == "latency"
        assert result.constraint_threshold == 20.0
        assert result.constraint_actual == 10.0
        assert result.constraint_margin_pct == pytest.approx(50.0)

    def test_convert_with_failing_constraint(self):
        """Test conversion with failing latency constraint."""
        mock_result = create_mock_unified_result()
        result = convert_to_pydantic(
            mock_result,
            constraint_metric="latency",
            constraint_threshold=5.0
        )

        assert result.verdict == Verdict.FAIL
        assert result.constraint_margin_pct == pytest.approx(-100.0)
        assert len(result.suggestions) > 0

    def test_convert_has_all_breakdowns(self):
        """Test that all breakdown sections are present."""
        mock_result = create_mock_unified_result()
        result = convert_to_pydantic(mock_result)

        assert result.roofline is not None
        assert result.energy is not None
        assert result.memory is not None
        assert result.roofline.bottleneck == Bottleneck.COMPUTE_BOUND

    def test_convert_power_constraint(self):
        """Test conversion with power constraint."""
        mock_result = create_mock_unified_result()
        result = convert_to_pydantic(
            mock_result,
            constraint_metric="power",
            constraint_threshold=250.0
        )

        assert result.verdict == Verdict.PASS
        assert result.constraint_metric == "power"

    def test_convert_memory_constraint(self):
        """Test conversion with memory constraint."""
        mock_result = create_mock_unified_result()
        result = convert_to_pydantic(
            mock_result,
            constraint_metric="memory",
            constraint_threshold=500.0  # 500 MB limit
        )

        assert result.verdict == Verdict.FAIL  # 1024 MB > 500 MB
        assert result.constraint_metric == "memory"


class TestVerdictPatterns:
    """Test the verdict-first pattern for various scenarios."""

    def test_verdict_enables_llm_trust(self):
        """Verify the verdict-first pattern is correct for LLM consumption."""
        mock_result = create_mock_unified_result()

        pydantic_result = convert_to_pydantic(
            mock_result,
            constraint_metric="latency",
            constraint_threshold=15.0
        )

        # Verify verdict-first pattern
        assert pydantic_result.verdict in [Verdict.PASS, Verdict.FAIL, Verdict.PARTIAL, Verdict.UNKNOWN]
        assert pydantic_result.confidence in [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW]
        assert isinstance(pydantic_result.summary, str)
        assert len(pydantic_result.summary) > 10  # Meaningful summary

        # Verify LLM can use verdict directly
        if pydantic_result.verdict == Verdict.PASS:
            assert pydantic_result.constraint_margin_pct > 0
        elif pydantic_result.verdict == Verdict.FAIL:
            assert pydantic_result.constraint_margin_pct < 0
