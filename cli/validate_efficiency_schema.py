#!/usr/bin/env python3
"""
Validate efficiency calibration JSON files against the schema.

Usage:
    ./cli/validate_efficiency_schema.py hardware_registry/gpu/jetson_orin_agx/efficiency_curves.json
    ./cli/validate_efficiency_schema.py --all  # Validate all efficiency_curves.json files
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Schema definition
REQUIRED_TOP_LEVEL = [
    "schema_version",
    "hardware_id",
    "hardware_name",
    "device_type",
    "precision",
    "calibration_date",
    "calibration_tool_version",
    "curves",
]

VALID_DEVICE_TYPES = ["gpu", "cpu", "dsp", "tpu", "kpu", "dpu", "cgra"]
VALID_PRECISIONS = ["FP32", "FP16", "INT8", "TF32", "BF16"]
VALID_SOURCES = ["legacy_hardcoded", "measured", "interpolated", "extrapolated"]
VALID_CURVE_TYPES = ["piecewise_linear", "log_linear", "sigmoid", "polynomial"]

REQUIRED_DATA_POINT_FIELDS = [
    "flops",
    "efficiency_mean",
]

STATISTICAL_FIELDS = [
    "efficiency_std",
    "efficiency_min",
    "efficiency_max",
    "ci_lower",
    "ci_upper",
    "num_observations",
]

PROVENANCE_FIELDS = [
    "source",
    "source_subgraphs",
    "source_models",
]


class ValidationError:
    """Represents a validation error."""
    def __init__(self, path: str, message: str, severity: str = "error"):
        self.path = path
        self.message = message
        self.severity = severity  # "error", "warning", "info"

    def __str__(self):
        prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}[self.severity]
        return f"[{prefix}] {self.path}: {self.message}"


def validate_data_point(dp: dict, path: str) -> list:
    """Validate a single data point."""
    errors = []

    # Required fields
    for field in REQUIRED_DATA_POINT_FIELDS:
        if field not in dp:
            errors.append(ValidationError(path, f"Missing required field: {field}"))

    # Type checks
    if "flops" in dp:
        if not isinstance(dp["flops"], (int, float)):
            errors.append(ValidationError(path, f"flops must be numeric, got {type(dp['flops']).__name__}"))
        elif dp["flops"] <= 0:
            errors.append(ValidationError(path, f"flops must be positive, got {dp['flops']}"))

    if "efficiency_mean" in dp:
        if not isinstance(dp["efficiency_mean"], (int, float)):
            errors.append(ValidationError(path, f"efficiency_mean must be numeric"))
        elif dp["efficiency_mean"] < 0:
            errors.append(ValidationError(path, f"efficiency_mean cannot be negative: {dp['efficiency_mean']}"))
        elif dp["efficiency_mean"] > 10:
            errors.append(ValidationError(f"{path}", f"efficiency_mean unusually high: {dp['efficiency_mean']}", "warning"))

    # Statistical field checks
    for field in ["efficiency_std", "efficiency_min", "efficiency_max"]:
        if field in dp:
            if not isinstance(dp[field], (int, float)):
                errors.append(ValidationError(path, f"{field} must be numeric"))
            elif dp[field] < 0:
                errors.append(ValidationError(path, f"{field} cannot be negative"))

    # Consistency checks
    if all(f in dp for f in ["efficiency_mean", "efficiency_min", "efficiency_max"]):
        if dp["efficiency_min"] > dp["efficiency_mean"]:
            errors.append(ValidationError(path, f"efficiency_min ({dp['efficiency_min']}) > efficiency_mean ({dp['efficiency_mean']})"))
        if dp["efficiency_max"] < dp["efficiency_mean"]:
            errors.append(ValidationError(path, f"efficiency_max ({dp['efficiency_max']}) < efficiency_mean ({dp['efficiency_mean']})"))

    if all(f in dp for f in ["ci_lower", "ci_upper", "efficiency_mean"]):
        if dp["ci_lower"] > dp["efficiency_mean"]:
            errors.append(ValidationError(path, f"ci_lower ({dp['ci_lower']}) > efficiency_mean ({dp['efficiency_mean']})"))
        if dp["ci_upper"] < dp["efficiency_mean"]:
            errors.append(ValidationError(path, f"ci_upper ({dp['ci_upper']}) < efficiency_mean ({dp['efficiency_mean']})"))

    # Source validation
    if "source" in dp and dp["source"] not in VALID_SOURCES:
        errors.append(ValidationError(path, f"Invalid source: {dp['source']}. Valid: {VALID_SOURCES}", "warning"))

    # Check for missing statistical fields (warning)
    missing_stats = [f for f in STATISTICAL_FIELDS if f not in dp]
    if missing_stats:
        errors.append(ValidationError(path, f"Missing statistical fields: {missing_stats}", "info"))

    return errors


def validate_fitted_curve(curve: dict, path: str) -> list:
    """Validate a fitted curve definition."""
    errors = []

    if "curve_type" not in curve:
        errors.append(ValidationError(path, "Missing curve_type"))
    elif curve["curve_type"] not in VALID_CURVE_TYPES:
        errors.append(ValidationError(path, f"Invalid curve_type: {curve['curve_type']}. Valid: {VALID_CURVE_TYPES}"))

    if "breakpoints" not in curve:
        errors.append(ValidationError(path, "Missing breakpoints"))
    else:
        bps = curve["breakpoints"]
        if not isinstance(bps, list):
            errors.append(ValidationError(path, "breakpoints must be a list"))
        elif len(bps) < 2:
            errors.append(ValidationError(path, f"Need at least 2 breakpoints, got {len(bps)}"))
        else:
            # Check monotonicity
            prev_flops = 0
            for i, bp in enumerate(bps):
                if not isinstance(bp, list) or len(bp) != 2:
                    errors.append(ValidationError(f"{path}.breakpoints[{i}]", f"Breakpoint must be [flops, efficiency], got {bp}"))
                else:
                    flops, eff = bp
                    if flops <= prev_flops:
                        errors.append(ValidationError(f"{path}.breakpoints[{i}]", f"Breakpoints must have increasing FLOPs: {flops} <= {prev_flops}"))
                    prev_flops = flops

    return errors


def validate_efficiency_curve(curve: dict, op_type: str, path: str) -> list:
    """Validate a single efficiency curve."""
    errors = []

    # Operation type consistency
    if "operation_type" in curve:
        if curve["operation_type"] != op_type:
            errors.append(ValidationError(path, f"operation_type mismatch: key={op_type}, value={curve['operation_type']}"))

    # Data points
    if "data_points" not in curve:
        errors.append(ValidationError(path, "Missing data_points"))
    else:
        dps = curve["data_points"]
        if not isinstance(dps, list):
            errors.append(ValidationError(path, "data_points must be a list"))
        elif len(dps) == 0:
            errors.append(ValidationError(path, "data_points is empty"))
        else:
            for i, dp in enumerate(dps):
                errors.extend(validate_data_point(dp, f"{path}.data_points[{i}]"))

            # Check for monotonic FLOPs
            flops_list = [dp.get("flops", 0) for dp in dps]
            if flops_list != sorted(flops_list):
                errors.append(ValidationError(path, "data_points should be sorted by FLOPs", "warning"))

    # Fitted curve
    if "fitted_curve" in curve:
        errors.extend(validate_fitted_curve(curve["fitted_curve"], f"{path}.fitted_curve"))
    else:
        errors.append(ValidationError(path, "Missing fitted_curve", "warning"))

    # Range checks
    if "min_flops" in curve and "max_flops" in curve:
        if curve["min_flops"] >= curve["max_flops"]:
            errors.append(ValidationError(path, f"min_flops ({curve['min_flops']}) >= max_flops ({curve['max_flops']})"))

    return errors


def validate_efficiency_file(data: dict, filepath: str) -> list:
    """Validate an efficiency calibration file."""
    errors = []

    # Top-level required fields
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.append(ValidationError(filepath, f"Missing required field: {field}"))

    # Device type
    if "device_type" in data and data["device_type"] not in VALID_DEVICE_TYPES:
        errors.append(ValidationError(filepath, f"Invalid device_type: {data['device_type']}. Valid: {VALID_DEVICE_TYPES}"))

    # Precision
    if "precision" in data and data["precision"] not in VALID_PRECISIONS:
        errors.append(ValidationError(filepath, f"Invalid precision: {data['precision']}. Valid: {VALID_PRECISIONS}"))

    # Schema version
    if "schema_version" in data and data["schema_version"] != "1.0":
        errors.append(ValidationError(filepath, f"Unknown schema_version: {data['schema_version']}", "warning"))

    # Curves
    if "curves" in data:
        if not isinstance(data["curves"], dict):
            errors.append(ValidationError(filepath, "curves must be an object"))
        else:
            for op_type, curve in data["curves"].items():
                errors.extend(validate_efficiency_curve(curve, op_type, f"{filepath}#curves.{op_type}"))

    # Bandwidth curves (optional)
    if "bandwidth_curves" in data and data["bandwidth_curves"]:
        if not isinstance(data["bandwidth_curves"], dict):
            errors.append(ValidationError(filepath, "bandwidth_curves must be an object"))
        else:
            for op_type, curve in data["bandwidth_curves"].items():
                errors.extend(validate_efficiency_curve(curve, op_type, f"{filepath}#bandwidth_curves.{op_type}"))

    return errors


def validate_file(filepath: Path) -> tuple:
    """Validate a single file. Returns (errors, warnings, info)."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ([ValidationError(str(filepath), f"Invalid JSON: {e}")], [], [])
    except Exception as e:
        return ([ValidationError(str(filepath), f"Cannot read file: {e}")], [], [])

    all_errors = validate_efficiency_file(data, str(filepath))

    errors = [e for e in all_errors if e.severity == "error"]
    warnings = [e for e in all_errors if e.severity == "warning"]
    info = [e for e in all_errors if e.severity == "info"]

    return errors, warnings, info


def find_all_efficiency_files(base_path: Path) -> list:
    """Find all efficiency_curves.json files in the hardware registry."""
    return list(base_path.glob("**/efficiency_curves.json"))


def main():
    parser = argparse.ArgumentParser(description="Validate efficiency calibration JSON files")
    parser.add_argument("files", nargs="*", help="JSON files to validate")
    parser.add_argument("--all", "-a", action="store_true", help="Validate all efficiency_curves.json in hardware_registry/")
    parser.add_argument("--warnings", "-w", action="store_true", help="Show warnings")
    parser.add_argument("--info", "-i", action="store_true", help="Show info messages")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show summary")
    args = parser.parse_args()

    # Find files to validate
    files = []
    if args.all:
        # Find project root
        script_dir = Path(__file__).parent.parent
        registry_path = script_dir / "hardware_registry"
        if not registry_path.exists():
            print(f"Error: hardware_registry not found at {registry_path}")
            sys.exit(1)
        files = find_all_efficiency_files(registry_path)
        if not files:
            print("No efficiency_curves.json files found")
            sys.exit(0)
    elif args.files:
        files = [Path(f) for f in args.files]
    else:
        parser.print_help()
        sys.exit(1)

    # Validate
    total_errors = 0
    total_warnings = 0
    total_info = 0

    for filepath in files:
        errors, warnings, info = validate_file(filepath)

        if not args.quiet:
            if errors or (args.warnings and warnings) or (args.info and info):
                print(f"\n{filepath}:")
                for e in errors:
                    print(f"  {e}")
                if args.warnings:
                    for w in warnings:
                        print(f"  {w}")
                if args.info:
                    for i in info:
                        print(f"  {i}")

        total_errors += len(errors)
        total_warnings += len(warnings)
        total_info += len(info)

    # Summary
    print(f"\n{'='*60}")
    print(f"Validated {len(files)} file(s)")
    print(f"  Errors:   {total_errors}")
    print(f"  Warnings: {total_warnings}")
    print(f"  Info:     {total_info}")
    print(f"{'='*60}")

    if total_errors > 0:
        print("\nValidation FAILED")
        sys.exit(1)
    else:
        print("\nValidation PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
