"""
Tests for the confidence module.
"""

import pytest
from graphs.core.confidence import (
    ConfidenceLevel,
    EstimationConfidence,
    DEFAULT_CALIBRATED_SCORE,
    DEFAULT_INTERPOLATED_SCORE,
    DEFAULT_THEORETICAL_SCORE,
)


class TestConfidenceLevel:
    """Test ConfidenceLevel enum."""

    def test_values(self):
        """Test enum values."""
        assert ConfidenceLevel.CALIBRATED.value == "calibrated"
        assert ConfidenceLevel.INTERPOLATED.value == "interpolated"
        assert ConfidenceLevel.THEORETICAL.value == "theoretical"
        assert ConfidenceLevel.UNKNOWN.value == "unknown"

    def test_all_levels_defined(self):
        """Ensure all expected levels exist."""
        levels = list(ConfidenceLevel)
        assert len(levels) == 4


class TestEstimationConfidence:
    """Test EstimationConfidence dataclass."""

    def test_basic_creation(self):
        """Test basic confidence creation."""
        conf = EstimationConfidence(
            level=ConfidenceLevel.CALIBRATED,
            score=0.95,
            source="test calibration"
        )
        assert conf.level == ConfidenceLevel.CALIBRATED
        assert conf.score == 0.95
        assert conf.source == "test calibration"

    def test_score_validation(self):
        """Test that score must be in 0.0-1.0 range."""
        with pytest.raises(ValueError):
            EstimationConfidence(
                level=ConfidenceLevel.CALIBRATED,
                score=1.5,  # Invalid
                source="test"
            )

        with pytest.raises(ValueError):
            EstimationConfidence(
                level=ConfidenceLevel.CALIBRATED,
                score=-0.1,  # Invalid
                source="test"
            )

    def test_calibrated_factory(self):
        """Test calibrated() factory method."""
        conf = EstimationConfidence.calibrated(
            score=0.95,
            source="GPU benchmark",
            calibration_id="nvidia/h100/calibration.json"
        )
        assert conf.level == ConfidenceLevel.CALIBRATED
        assert conf.score == 0.95
        assert conf.calibration_id == "nvidia/h100/calibration.json"

    def test_interpolated_factory(self):
        """Test interpolated() factory method."""
        conf = EstimationConfidence.interpolated(
            score=0.75,
            source="interpolated from FP32 and FP16"
        )
        assert conf.level == ConfidenceLevel.INTERPOLATED
        assert conf.score == 0.75

    def test_theoretical_factory(self):
        """Test theoretical() factory method."""
        conf = EstimationConfidence.theoretical(
            score=0.50,
            source="vendor spec sheet"
        )
        assert conf.level == ConfidenceLevel.THEORETICAL
        assert conf.score == 0.50

    def test_unknown_factory(self):
        """Test unknown() factory method."""
        conf = EstimationConfidence.unknown()
        assert conf.level == ConfidenceLevel.UNKNOWN
        assert conf.score == 0.0

    def test_str_representation(self):
        """Test string representation."""
        conf = EstimationConfidence.calibrated(score=0.95)
        s = str(conf)
        assert "calibrated" in s
        assert "95%" in s

    def test_format_summary(self):
        """Test detailed format_summary()."""
        conf = EstimationConfidence(
            level=ConfidenceLevel.CALIBRATED,
            score=0.90,
            source="benchmark run",
            calibration_id="test/calibration.json",
            min_value=0.85,
            max_value=0.95
        )
        summary = conf.format_summary()
        assert "calibrated" in summary
        assert "90%" in summary
        assert "benchmark run" in summary
        assert "test/calibration.json" in summary
        assert "0.85" in summary
        assert "0.95" in summary

    def test_confidence_interval(self):
        """Test optional confidence interval."""
        conf = EstimationConfidence(
            level=ConfidenceLevel.THEORETICAL,
            score=0.50,
            source="spec sheet",
            min_value=0.001,
            max_value=0.002
        )
        assert conf.min_value == 0.001
        assert conf.max_value == 0.002


class TestDefaultScores:
    """Test default score constants."""

    def test_defaults_exist(self):
        """Test that default scores are defined."""
        assert 0.8 <= DEFAULT_CALIBRATED_SCORE <= 1.0
        assert 0.5 <= DEFAULT_INTERPOLATED_SCORE <= 0.9
        assert 0.3 <= DEFAULT_THEORETICAL_SCORE <= 0.7

    def test_ordering(self):
        """Test that defaults are ordered correctly."""
        assert DEFAULT_CALIBRATED_SCORE > DEFAULT_INTERPOLATED_SCORE
        assert DEFAULT_INTERPOLATED_SCORE > DEFAULT_THEORETICAL_SCORE


class TestDescriptorIntegration:
    """Test that descriptors have confidence fields."""

    def test_latency_descriptor_has_confidence(self):
        """Test LatencyDescriptor has confidence field."""
        from graphs.estimation.roofline import LatencyDescriptor
        from graphs.core.structures import BottleneckType

        lat = LatencyDescriptor(
            subgraph_id="test",
            subgraph_name="test_sg",
            compute_time=0.001,
            memory_time=0.002,
            actual_latency=0.002
        )
        assert hasattr(lat, 'confidence')
        assert lat.confidence.level == ConfidenceLevel.UNKNOWN

    def test_energy_descriptor_has_confidence(self):
        """Test EnergyDescriptor has confidence field."""
        from graphs.estimation.energy import EnergyDescriptor

        energy = EnergyDescriptor(
            subgraph_id="test",
            subgraph_name="test_sg",
            compute_energy_j=0.1,
            memory_energy_j=0.05,
            static_energy_j=0.01,
            total_energy_j=0.16,
            compute_ops=1000,
            bytes_transferred=4000,
            latency_s=0.001
        )
        assert hasattr(energy, 'confidence')
        assert energy.confidence.level == ConfidenceLevel.UNKNOWN

    def test_memory_descriptor_has_confidence(self):
        """Test MemoryDescriptor has confidence field."""
        from graphs.estimation.memory import MemoryDescriptor
        from graphs.core.structures import OperationType

        mem = MemoryDescriptor(
            subgraph_id="test",
            subgraph_name="test_sg",
            operation_type=OperationType.LINEAR,
            input_memory_bytes=1000,
            output_memory_bytes=1000,
            weight_memory_bytes=500,
            workspace_memory_bytes=0
        )
        assert hasattr(mem, 'confidence')
        assert mem.confidence.level == ConfidenceLevel.UNKNOWN
