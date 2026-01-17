"""
Confidence Levels for Estimation Results

This module defines confidence levels that indicate how reliable an estimate is,
based on whether it comes from calibrated measurements, interpolation, or
theoretical calculations.

Usage:
    from graphs.core.confidence import ConfidenceLevel, EstimationConfidence

    # When estimate is from calibrated hardware
    confidence = EstimationConfidence(
        level=ConfidenceLevel.CALIBRATED,
        score=0.95,
        source="nvidia/jetson_orin_nano/25W/calibration.json"
    )

    # When interpolating between calibration points
    confidence = EstimationConfidence(
        level=ConfidenceLevel.INTERPOLATED,
        score=0.75,
        source="interpolated from FP32 and FP16 calibration"
    )

    # When using theoretical peak specs
    confidence = EstimationConfidence(
        level=ConfidenceLevel.THEORETICAL,
        score=0.50,
        source="theoretical peak from hardware specs"
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConfidenceLevel(Enum):
    """
    Confidence level indicating the source of an estimate.

    CALIBRATED: Estimate from actual hardware benchmarks
        - Highest confidence (typically 0.85-1.0)
        - Based on measured performance on specific hardware
        - Example: GEMM throughput measured via calibration script

    INTERPOLATED: Estimate derived from nearby calibration points
        - Medium confidence (typically 0.60-0.85)
        - Interpolated between precisions, batch sizes, or operations
        - Example: INT8 estimated from FP16 and FP32 measurements

    THEORETICAL: Estimate from hardware specifications
        - Lower confidence (typically 0.30-0.60)
        - Based on vendor-published peak FLOPS/bandwidth
        - Example: Latency from peak TFLOPS without calibration

    UNKNOWN: Confidence level not determined
        - Used when analysis doesn't track confidence
        - Default for backward compatibility
    """
    CALIBRATED = "calibrated"
    INTERPOLATED = "interpolated"
    THEORETICAL = "theoretical"
    UNKNOWN = "unknown"


@dataclass
class EstimationConfidence:
    """
    Confidence metadata for an estimation result.

    Attributes:
        level: The confidence level (calibrated, interpolated, theoretical)
        score: Numeric confidence score from 0.0 (no confidence) to 1.0 (certain)
        source: Description of where the estimate came from
        calibration_id: Optional reference to calibration profile used
        min_value: Optional lower bound of confidence interval
        max_value: Optional upper bound of confidence interval
    """
    level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    score: float = 0.0  # 0.0 to 1.0
    source: str = ""

    # Optional calibration reference
    calibration_id: Optional[str] = None

    # Optional confidence interval
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        """Validate confidence score is in range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Confidence score must be 0.0-1.0, got {self.score}")

    @classmethod
    def calibrated(
        cls,
        score: float = 0.90,
        source: str = "calibration profile",
        calibration_id: Optional[str] = None
    ) -> 'EstimationConfidence':
        """Create a calibrated confidence with sensible defaults."""
        return cls(
            level=ConfidenceLevel.CALIBRATED,
            score=score,
            source=source,
            calibration_id=calibration_id
        )

    @classmethod
    def interpolated(
        cls,
        score: float = 0.70,
        source: str = "interpolated from calibration"
    ) -> 'EstimationConfidence':
        """Create an interpolated confidence with sensible defaults."""
        return cls(
            level=ConfidenceLevel.INTERPOLATED,
            score=score,
            source=source
        )

    @classmethod
    def theoretical(
        cls,
        score: float = 0.50,
        source: str = "theoretical peak from specs"
    ) -> 'EstimationConfidence':
        """Create a theoretical confidence with sensible defaults."""
        return cls(
            level=ConfidenceLevel.THEORETICAL,
            score=score,
            source=source
        )

    @classmethod
    def unknown(cls) -> 'EstimationConfidence':
        """Create an unknown confidence (for backward compatibility)."""
        return cls(
            level=ConfidenceLevel.UNKNOWN,
            score=0.0,
            source="confidence not tracked"
        )

    def __str__(self) -> str:
        """Short string representation."""
        return f"{self.level.value} ({self.score:.0%})"

    def format_summary(self) -> str:
        """Detailed summary for reports."""
        lines = [f"Confidence: {self.level.value} ({self.score:.0%})"]
        if self.source:
            lines.append(f"  Source: {self.source}")
        if self.calibration_id:
            lines.append(f"  Calibration: {self.calibration_id}")
        if self.min_value is not None and self.max_value is not None:
            lines.append(f"  Range: [{self.min_value:.4g}, {self.max_value:.4g}]")
        return "\n".join(lines)


# Default confidence values for different scenarios
DEFAULT_CALIBRATED_SCORE = 0.90
DEFAULT_INTERPOLATED_SCORE = 0.70
DEFAULT_THEORETICAL_SCORE = 0.50
