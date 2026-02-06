"""
GPU Calibration Data Loader

Loads efficiency curves from calibration_data/ and provides lookup interface
for the GPU mapper to get size-dependent efficiency factors.

Usage:
    from graphs.calibration.gpu_calibration import GPUCalibration

    # Load calibration for Jetson Orin AGX 50W
    cal = GPUCalibration.load("jetson_orin_agx_50w", "fp32")

    # Look up efficiency for matmul with 100M FLOPs
    eff = cal.get_efficiency("matmul", 1e8)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, List

from .efficiency_curves import (
    EfficiencyProfile,
    PiecewiseLinearCurve,
    AsymptoticCurve,
    ConstantCurve,
    CurveType,
)


# Default calibration data directory
CALIBRATION_DATA_DIR = Path(__file__).parent.parent.parent.parent / "calibration_data"


@dataclass
class GPUCalibration:
    """
    GPU calibration data loaded from efficiency_curves.json.

    Provides size-dependent efficiency lookup for GPU operations.
    """
    hardware_id: str
    precision: str
    profile: EfficiencyProfile
    device_name: str = ""
    calibration_date: str = ""
    measured_bandwidth_gbps: Optional[float] = None

    # Operation type aliases for mapping fusion patterns to calibration keys
    OP_TYPE_ALIASES: Dict[str, List[str]] = field(default_factory=lambda: {
        "matmul": ["matmul", "linear", "gemm", "mm"],
        "conv2d": ["conv2d", "conv"],
        "conv2d_batchnorm": ["conv2d_batchnorm", "conv_bn", "conv_bn_relu"],
        "activation": ["activation", "relu", "gelu", "softmax", "sigmoid", "tanh"],
        "unfused": ["unfused", "add", "mul", "elementwise"],
    })

    @classmethod
    def load(
        cls,
        hardware_id: str,
        precision: str = "fp32",
        calibration_dir: Optional[Path] = None
    ) -> Optional['GPUCalibration']:
        """
        Load calibration data from efficiency_curves.json.

        Args:
            hardware_id: Hardware identifier (e.g., "jetson_orin_agx_50w")
            precision: Precision to load (e.g., "fp32", "fp16")
            calibration_dir: Override calibration data directory

        Returns:
            GPUCalibration instance or None if not found
        """
        if calibration_dir is None:
            calibration_dir = CALIBRATION_DATA_DIR

        # Try precision subdirectory first, then legacy location
        curves_path = calibration_dir / hardware_id / precision / "efficiency_curves.json"
        if not curves_path.exists():
            # Try legacy structure
            curves_path = calibration_dir / hardware_id / "efficiency_curves.json"

        if not curves_path.exists():
            return None

        try:
            with open(curves_path) as f:
                data = json.load(f)

            return cls._from_json_data(data, hardware_id, precision)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to load calibration from {curves_path}: {e}")
            return None

    @classmethod
    def _from_json_data(
        cls,
        data: Dict[str, Any],
        hardware_id: str,
        precision: str
    ) -> 'GPUCalibration':
        """
        Parse calibration JSON data into GPUCalibration.

        The JSON structure is:
        {
            "curves": {
                "conv2d_batchnorm": {
                    "fitted_curve": {
                        "curve_type": "piecewise_linear",
                        "breakpoints": [[flops, eff], ...]
                    }
                },
                ...
            }
        }
        """
        profile = EfficiencyProfile(
            device_name=data.get("hardware_name", hardware_id),
            created_at=data.get("calibration_date", ""),
        )

        curves_data = data.get("curves", {})

        for op_type, op_data in curves_data.items():
            fitted_curve = op_data.get("fitted_curve")
            if not fitted_curve:
                continue

            curve_type = fitted_curve.get("curve_type", "piecewise_linear")

            if curve_type == "piecewise_linear":
                breakpoints = fitted_curve.get("breakpoints", [])
                if breakpoints:
                    # Convert to list of tuples
                    bp_tuples = [tuple(bp) for bp in breakpoints]
                    curve = PiecewiseLinearCurve(breakpoints=bp_tuples)
                    profile.add_curve(op_type, curve, precision)

            elif curve_type == "asymptotic":
                peak = fitted_curve.get("peak", 0.8)
                scale = fitted_curve.get("scale", 1e6)
                curve = AsymptoticCurve(peak=peak, scale=scale)
                profile.add_curve(op_type, curve, precision)

            elif curve_type == "constant":
                efficiency = fitted_curve.get("efficiency", 0.5)
                curve = ConstantCurve(efficiency=efficiency)
                profile.add_curve(op_type, curve, precision)

        return cls(
            hardware_id=hardware_id,
            precision=precision,
            profile=profile,
            device_name=data.get("hardware_name", ""),
            calibration_date=data.get("calibration_date", ""),
            measured_bandwidth_gbps=data.get("measured_bandwidth_gbps"),
        )

    def get_efficiency(
        self,
        op_type: str,
        flops: float,
        default: float = 0.5
    ) -> float:
        """
        Look up efficiency for operation type and FLOP count.

        Args:
            op_type: Operation type (e.g., "matmul", "conv2d")
            flops: Number of floating-point operations
            default: Default efficiency if no curve found

        Returns:
            Efficiency factor (0.0 to 1.0)
        """
        # Try direct lookup first
        eff = self.profile.predict_efficiency(op_type, flops, self.precision, default=-1.0)
        if eff >= 0:
            return eff

        # Try aliases
        for canonical, aliases in self.OP_TYPE_ALIASES.items():
            if op_type.lower() in aliases:
                eff = self.profile.predict_efficiency(canonical, flops, self.precision, default=-1.0)
                if eff >= 0:
                    return eff

        # Try partial match (e.g., "Conv2d_BatchNorm2d_ReLU" -> "conv2d_batchnorm")
        op_lower = op_type.lower()
        for curve_op in self.profile.curves.keys():
            if curve_op.lower() in op_lower or op_lower in curve_op.lower():
                eff = self.profile.predict_efficiency(curve_op, flops, self.precision, default=-1.0)
                if eff >= 0:
                    return eff

        # Fallback to "unfused" if available
        eff = self.profile.predict_efficiency("unfused", flops, self.precision, default=-1.0)
        if eff >= 0:
            return eff

        return default

    def list_operations(self) -> List[str]:
        """Return list of operation types with calibration data."""
        return list(self.profile.curves.keys())

    def __repr__(self) -> str:
        ops = self.list_operations()
        return (
            f"GPUCalibration(hardware_id='{self.hardware_id}', "
            f"precision='{self.precision}', operations={ops})"
        )
