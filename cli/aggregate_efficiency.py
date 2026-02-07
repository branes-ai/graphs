#!/usr/bin/env python3
"""
Efficiency Aggregation Tool

Aggregates per-subgraph efficiency measurements from multiple models into
efficiency curves for calibration. Groups data by operation type and FLOP
bins, computes pooled statistics, and fits parametric curves.

Usage:
    # Aggregate multiple model measurements
    ./cli/aggregate_efficiency.py \\
        --input measurements/resnet18_i7.json \\
        --input measurements/resnet50_i7.json \\
        --input measurements/vgg16_i7.json \\
        --output hardware_registry/cpu/i7-12700K/efficiency_curves.json

    # Aggregate all measurements in a directory
    ./cli/aggregate_efficiency.py \\
        --input-dir measurements/ \\
        --hardware i7-12700K \\
        --output hardware_registry/cpu/i7-12700K/efficiency_curves.json

    # Preview without writing
    ./cli/aggregate_efficiency.py --input measurements/*.json --dry-run
"""

import argparse
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.calibration.ground_truth import GroundTruthLoader


# ============================================================================
# FLOP Binning
# ============================================================================

# Logarithmic bin edges: 100K, 300K, 1M, 3M, 10M, 30M, 100M, 300M, 1G, 3G, 10G
FLOP_BIN_EDGES = [
    1e5,    # 100K
    3e5,    # 300K
    1e6,    # 1M
    3e6,    # 3M
    1e7,    # 10M
    3e7,    # 30M
    1e8,    # 100M
    3e8,    # 300M
    1e9,    # 1G
    3e9,    # 3G
    1e10,   # 10G
]

FLOP_BIN_NAMES = [
    "<100K", "100K-300K", "300K-1M", "1M-3M", "3M-10M",
    "10M-30M", "30M-100M", "100M-300M", "300M-1G", "1G-3G", "3G-10G", ">10G"
]


def get_flop_bin(flops: float) -> Tuple[int, str, float]:
    """Get the bin index, name, and center for a FLOP count.

    Returns:
        Tuple of (bin_index, bin_name, bin_center_flops)
    """
    for i, edge in enumerate(FLOP_BIN_EDGES):
        if flops < edge:
            center = FLOP_BIN_EDGES[i-1] * math.sqrt(10) if i > 0 else edge / 3
            return i, FLOP_BIN_NAMES[i], center
    # Above highest bin
    center = FLOP_BIN_EDGES[-1] * math.sqrt(10)
    return len(FLOP_BIN_EDGES), FLOP_BIN_NAMES[-1], center


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class EfficiencyObservation:
    """A single efficiency observation from a measurement."""
    flops: int
    efficiency_mean: float
    efficiency_std: float
    num_samples: int
    source_model: str
    fusion_pattern: str


@dataclass
class BinnedEfficiency:
    """Aggregated efficiency data for a FLOP bin."""
    bin_index: int
    bin_name: str
    bin_center_flops: float
    observations: List[EfficiencyObservation] = field(default_factory=list)

    @property
    def pooled_mean(self) -> float:
        """Compute pooled mean weighted by sample count."""
        if not self.observations:
            return 0.0
        total_weight = sum(o.num_samples for o in self.observations)
        if total_weight == 0:
            return statistics.mean(o.efficiency_mean for o in self.observations)
        weighted_sum = sum(o.efficiency_mean * o.num_samples for o in self.observations)
        return weighted_sum / total_weight

    @property
    def pooled_variance(self) -> float:
        """Compute pooled variance across observations.

        Uses the formula for pooled variance:
        s_p^2 = (sum((n_i-1)*s_i^2) + sum(n_i*(mean_i - mean_pooled)^2)) / (sum(n_i) - 1)
        """
        if len(self.observations) < 2:
            if self.observations:
                return self.observations[0].efficiency_std ** 2
            return 0.0

        pooled_mean = self.pooled_mean
        n_total = sum(o.num_samples for o in self.observations)

        if n_total <= 1:
            return statistics.variance([o.efficiency_mean for o in self.observations])

        # Within-group variance
        within_var = sum((o.num_samples - 1) * (o.efficiency_std ** 2)
                         for o in self.observations if o.num_samples > 1)

        # Between-group variance
        between_var = sum(o.num_samples * (o.efficiency_mean - pooled_mean) ** 2
                          for o in self.observations)

        return (within_var + between_var) / (n_total - 1)

    @property
    def pooled_std(self) -> float:
        return math.sqrt(max(0, self.pooled_variance))

    @property
    def ci_lower(self) -> float:
        """95% CI lower bound."""
        n = sum(o.num_samples for o in self.observations)
        se = self.pooled_std / math.sqrt(n) if n > 0 else self.pooled_std
        return max(0, self.pooled_mean - 1.96 * se)

    @property
    def ci_upper(self) -> float:
        """95% CI upper bound."""
        n = sum(o.num_samples for o in self.observations)
        se = self.pooled_std / math.sqrt(n) if n > 0 else self.pooled_std
        return self.pooled_mean + 1.96 * se

    @property
    def total_observations(self) -> int:
        return len(self.observations)

    @property
    def total_samples(self) -> int:
        return sum(o.num_samples for o in self.observations)

    @property
    def source_models(self) -> List[str]:
        return list(set(o.source_model for o in self.observations))


@dataclass
class OperationTypeCurve:
    """Efficiency curve for an operation type."""
    operation_type: str
    bins: Dict[int, BinnedEfficiency] = field(default_factory=dict)

    def add_observation(self, obs: EfficiencyObservation):
        bin_idx, bin_name, bin_center = get_flop_bin(obs.flops)
        if bin_idx not in self.bins:
            self.bins[bin_idx] = BinnedEfficiency(bin_idx, bin_name, bin_center)
        self.bins[bin_idx].observations.append(obs)


# ============================================================================
# Load measurements
# ============================================================================

def load_measurement_file(filepath: Path) -> Optional[dict]:
    """Load a measurement JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        if data.get("measurement_type") != "efficiency":
            print(f"  Skipping {filepath.name}: not an efficiency measurement")
            return None
        return data
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def extract_observations(data: dict) -> List[Tuple[str, EfficiencyObservation]]:
    """Extract observations from measurement data (dict or MeasurementRecord).

    Returns:
        List of (operation_type, EfficiencyObservation) tuples
    """
    observations = []

    # Handle both raw dicts and MeasurementRecord objects
    if hasattr(data, 'model'):
        # MeasurementRecord object
        model_name = data.model
        subgraphs = data.subgraphs
        for sg in subgraphs:
            if sg.flops == 0:
                continue
            if sg.efficiency.samples == 0:
                continue
            obs = EfficiencyObservation(
                flops=sg.flops,
                efficiency_mean=sg.efficiency.mean,
                efficiency_std=sg.efficiency.std,
                num_samples=sg.efficiency.samples,
                source_model=model_name,
                fusion_pattern=sg.fusion_pattern,
            )
            observations.append((sg.operation_type, obs))
    else:
        # Raw dict (legacy path)
        model_name = data.get("model", "unknown")
        for sg in data.get("subgraphs", []):
            op_type = sg.get("operation_type", "generic")
            flops = sg.get("flops", 0)
            eff = sg.get("efficiency", {})

            if flops == 0:
                continue
            if eff.get("samples", 0) == 0:
                continue

            obs = EfficiencyObservation(
                flops=flops,
                efficiency_mean=eff.get("mean", 0),
                efficiency_std=eff.get("std", 0),
                num_samples=eff.get("samples", 1),
                source_model=model_name,
                fusion_pattern=sg.get("fusion_pattern", "")
            )
            observations.append((op_type, obs))

    return observations


# ============================================================================
# Curve fitting
# ============================================================================

def fit_piecewise_linear_curve(bins: Dict[int, BinnedEfficiency]) -> dict:
    """Fit a piecewise linear curve to binned data.

    Returns breakpoints in (flops, efficiency) pairs.
    """
    if not bins:
        return {"curve_type": "piecewise_linear", "breakpoints": [], "r_squared": 0, "residual_std": 0}

    # Sort bins by index
    sorted_bins = sorted(bins.values(), key=lambda b: b.bin_index)

    # Create breakpoints
    breakpoints = []
    for b in sorted_bins:
        if b.pooled_mean > 0:  # Skip empty bins
            breakpoints.append([b.bin_center_flops, b.pooled_mean])

    if len(breakpoints) < 2:
        return {"curve_type": "piecewise_linear", "breakpoints": breakpoints, "r_squared": 0, "residual_std": 0}

    # Compute R^2 and residual std (simplified)
    # For piecewise linear, we assume perfect fit within bins
    residuals = []
    for b in sorted_bins:
        for obs in b.observations:
            predicted = b.pooled_mean  # Use bin mean as prediction
            residuals.append(obs.efficiency_mean - predicted)

    if residuals:
        residual_std = statistics.stdev(residuals) if len(residuals) > 1 else 0
        ss_res = sum(r**2 for r in residuals)
        mean_eff = statistics.mean(o.efficiency_mean for b in sorted_bins for o in b.observations)
        ss_tot = sum((o.efficiency_mean - mean_eff)**2 for b in sorted_bins for o in b.observations)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        residual_std = 0
        r_squared = 0

    return {
        "curve_type": "piecewise_linear",
        "breakpoints": breakpoints,
        "r_squared": max(0, r_squared),
        "residual_std": residual_std,
        "max_residual": max(abs(r) for r in residuals) if residuals else 0
    }


# ============================================================================
# Output generation
# ============================================================================

def generate_efficiency_curves_json(
    curves: Dict[str, OperationTypeCurve],
    hardware_id: str,
    hardware_name: str,
    device_type: str,
    precision: str,
    input_files: List[str]
) -> dict:
    """Generate efficiency_curves.json format output."""

    output = {
        "schema_version": "1.0",
        "hardware_id": hardware_id,
        "hardware_name": hardware_name,
        "device_type": device_type,
        "precision": precision.upper(),
        "calibration_date": datetime.now().isoformat(),
        "calibration_tool_version": "aggregate_efficiency.py v1.0",
        "curves": {},
        "bandwidth_curves": {},
        "validation": None,
        "notes": f"Aggregated from {len(input_files)} measurement files."
    }

    for op_type, curve in curves.items():
        if not curve.bins:
            continue

        sorted_bins = sorted(curve.bins.values(), key=lambda b: b.bin_index)

        # Build data points
        data_points = []
        for b in sorted_bins:
            if b.total_observations == 0:
                continue
            data_points.append({
                "flops": b.bin_center_flops,
                "efficiency_mean": b.pooled_mean,
                "efficiency_std": b.pooled_std,
                "efficiency_min": min(o.efficiency_mean for o in b.observations),
                "efficiency_max": max(o.efficiency_mean for o in b.observations),
                "ci_lower": b.ci_lower,
                "ci_upper": b.ci_upper,
                "num_observations": b.total_observations,
                "num_samples": b.total_samples,
                "source": "measured",
                "source_subgraphs": [o.fusion_pattern for o in b.observations[:5]],  # First 5
                "source_models": b.source_models
            })

        if not data_points:
            continue

        # Fit curve
        fitted_curve = fit_piecewise_linear_curve(curve.bins)

        # Collect all calibration models
        all_models = list(set(
            o.source_model for b in sorted_bins for o in b.observations
        ))

        # Compute FLOP range
        all_flops = [o.flops for b in sorted_bins for o in b.observations]
        min_flops = min(all_flops) if all_flops else 0
        max_flops = max(all_flops) if all_flops else 0

        output["curves"][op_type] = {
            "operation_type": op_type,
            "description": f"Measured efficiency for {op_type} operations",
            "data_points": data_points,
            "fitted_curve": fitted_curve,
            "min_flops": min_flops,
            "max_flops": max_flops,
            "calibration_models": all_models,
            "notes": f"Aggregated from {len(all_models)} models"
        }

    return output


# ============================================================================
# Display
# ============================================================================

def print_aggregation_summary(curves: Dict[str, OperationTypeCurve]):
    """Print summary of aggregated data."""
    print()
    print("Aggregation Summary:")
    print(f"  {'Operation Type':<20} {'Bins':>5} {'Obs':>6} {'Samples':>8} "
          f"{'Eff Range':>15} {'Models':>8}")
    print("  " + "-" * 75)

    for op_type, curve in sorted(curves.items()):
        if not curve.bins:
            continue
        sorted_bins = sorted(curve.bins.values(), key=lambda b: b.bin_index)
        total_obs = sum(b.total_observations for b in sorted_bins)
        total_samples = sum(b.total_samples for b in sorted_bins)
        all_means = [o.efficiency_mean for b in sorted_bins for o in b.observations]
        min_eff = min(all_means) if all_means else 0
        max_eff = max(all_means) if all_means else 0
        models = set(o.source_model for b in sorted_bins for o in b.observations)

        print(f"  {op_type:<20} {len(sorted_bins):>5} {total_obs:>6} {total_samples:>8} "
              f"{min_eff:.3f}-{max_eff:.3f}    {len(models):>8}")


def print_detailed_bins(curves: Dict[str, OperationTypeCurve]):
    """Print detailed bin information."""
    for op_type, curve in sorted(curves.items()):
        if not curve.bins:
            continue
        print(f"\n{op_type}:")
        print(f"  {'Bin':<15} {'Obs':>4} {'Samples':>7} {'Mean':>7} {'Std':>7} "
              f"{'CI':>15}")
        print("  " + "-" * 60)

        for b in sorted(curve.bins.values(), key=lambda b: b.bin_index):
            ci_str = f"[{b.ci_lower:.3f}, {b.ci_upper:.3f}]"
            print(f"  {b.bin_name:<15} {b.total_observations:>4} {b.total_samples:>7} "
                  f"{b.pooled_mean:>7.3f} {b.pooled_std:>7.3f} {ci_str:>15}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate efficiency measurements into calibration curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input measurements/*.json --output efficiency_curves.json
  %(prog)s --input-dir measurements/ --hardware i7-12700K --output curves.json
""")
    parser.add_argument('--input', '-i', action='append', dest='input_files',
                        help='Input measurement JSON file(s)')
    parser.add_argument('--input-dir', type=str,
                        help='Directory containing measurement JSON files')
    parser.add_argument('--hardware', type=str,
                        help='Filter by hardware ID (for input-dir mode)')
    parser.add_argument('--from-loader', action='store_true',
                        help='Load measurements via GroundTruthLoader instead of file globs')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output efficiency curves JSON path')
    parser.add_argument('--hardware-id', type=str,
                        help='Hardware ID for output (default: from first input)')
    parser.add_argument('--hardware-name', type=str,
                        help='Hardware name for output')
    parser.add_argument('--device-type', type=str, choices=['cpu', 'gpu'],
                        help='Device type (default: auto-detect)')
    parser.add_argument('--precision', type=str, default='FP32',
                        help='Precision (default: FP32)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without writing output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed bin information')

    args = parser.parse_args()

    # Load and aggregate
    curves: Dict[str, OperationTypeCurve] = defaultdict(OperationTypeCurve)
    loaded_files = []
    hardware_id = args.hardware_id
    device_type = args.device_type

    if args.from_loader and args.hardware_id:
        # Use GroundTruthLoader to find measurement files
        loader = GroundTruthLoader()
        prec = args.precision.lower() if args.precision else None
        records = loader.load_all(args.hardware_id, precision=prec)

        if not records:
            print(f"Error: No measurements found via loader for {args.hardware_id}"
                  f"{' / ' + prec if prec else ''}")
            return 1

        print(f"Loaded {len(records)} record(s) via GroundTruthLoader")

        if not hardware_id:
            hardware_id = args.hardware_id
        if not device_type and records:
            device_type = records[0].device

        for record in records:
            observations = extract_observations(record)
            print(f"  {record.model} (b{record.batch_size}): {len(observations)} observations")
            for op_type, obs in observations:
                if op_type not in curves:
                    curves[op_type] = OperationTypeCurve(operation_type=op_type)
                curves[op_type].add_observation(obs)
            loaded_files.append(f"{record.model}_b{record.batch_size}")

    else:
        # Collect input files via glob (original path)
        input_files = []
        if args.input_files:
            for pattern in args.input_files:
                input_files.extend(glob.glob(pattern))
        if args.input_dir:
            dir_files = glob.glob(str(Path(args.input_dir) / "*.json"))
            input_files.extend(dir_files)

        if not input_files:
            print("Error: No input files specified (use --input, --input-dir, or --from-loader)")
            parser.print_help()
            return 1

        input_files = list(set(input_files))  # Remove duplicates
        print(f"Found {len(input_files)} input file(s)")

        for filepath in input_files:
            fp = Path(filepath)
            print(f"  Loading {fp.name}...", end=" ")

            data = load_measurement_file(fp)
            if data is None:
                continue

            # Filter by hardware if specified
            if args.hardware and data.get("hardware_id") != args.hardware:
                print(f"skipped (hardware={data.get('hardware_id')})")
                continue

            # Extract metadata from first file
            if not hardware_id:
                hardware_id = data.get("hardware_id", "unknown")
            if not device_type:
                device_type = data.get("device", "cpu")

            # Extract observations
            observations = extract_observations(data)
            print(f"{len(observations)} observations")

            for op_type, obs in observations:
                if op_type not in curves:
                    curves[op_type] = OperationTypeCurve(operation_type=op_type)
                curves[op_type].add_observation(obs)

            loaded_files.append(filepath)

    if not loaded_files:
        print("Error: No valid measurement files loaded")
        return 1

    print(f"\nLoaded {len(loaded_files)} file(s)")

    # Print summary
    print_aggregation_summary(curves)

    if args.verbose:
        print_detailed_bins(curves)

    # Generate output
    hardware_name = args.hardware_name or f"Hardware {hardware_id}"

    output = generate_efficiency_curves_json(
        curves=curves,
        hardware_id=hardware_id,
        hardware_name=hardware_name,
        device_type=device_type,
        precision=args.precision,
        input_files=loaded_files
    )

    # Write output
    if args.dry_run:
        print("\n[DRY RUN] Would write to:", args.output)
        print("\nPreview (first 2000 chars):")
        preview = json.dumps(output, indent=2)[:2000]
        print(preview)
        if len(json.dumps(output, indent=2)) > 2000:
            print("...")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nWritten: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
