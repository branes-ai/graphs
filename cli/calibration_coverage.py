#!/usr/bin/env python3
"""
Calibration Coverage Analysis Tool

Analyzes the hardware registry to identify calibration gaps and priorities.

Features:
- Coverage by device type, category, vendor
- Staleness detection (calibrations older than threshold)
- Priority ranking for next calibrations
- JSON/CSV export for tracking over time

Usage:
    ./cli/calibration_coverage.py                    # Text report
    ./cli/calibration_coverage.py --stale-days 30    # Flag stale calibrations
    ./cli/calibration_coverage.py --output report.json
    ./cli/calibration_coverage.py --output report.csv
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.calibration.registry_sync import HardwareRegistry, HardwareEntry


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CalibrationStatus:
    """Calibration status for a single hardware entry."""
    hardware_id: str
    model: str
    device_type: str
    category: str
    vendor: str
    is_calibrated: bool
    calibration_count: int
    latest_calibration_date: Optional[datetime]
    days_since_calibration: Optional[int]
    is_stale: bool
    theoretical_peak_gflops: float
    calibrated_peak_gflops: Optional[float]
    efficiency: Optional[float]  # calibrated / theoretical
    priority_score: float  # Higher = should calibrate sooner
    is_suspicious: bool = False  # True if any calibration has efficiency > 100%
    suspicious_count: int = 0  # Number of suspicious calibrations


@dataclass
class CoverageReport:
    """Complete calibration coverage report."""
    timestamp: datetime
    total_hardware: int
    calibrated_count: int
    coverage_pct: float

    # Coverage by dimension
    by_device_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_vendor: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Detailed status
    hardware_status: List[CalibrationStatus] = field(default_factory=list)

    # Stale calibrations
    stale_calibrations: List[CalibrationStatus] = field(default_factory=list)

    # Suspicious calibrations (efficiency > 100%)
    suspicious_calibrations: List[CalibrationStatus] = field(default_factory=list)

    # Priority list
    priority_list: List[CalibrationStatus] = field(default_factory=list)


# =============================================================================
# PRIORITY SCORING
# =============================================================================

# Device type importance (higher = more important to calibrate)
DEVICE_TYPE_PRIORITY = {
    "gpu": 10,       # Most commonly used for DNN
    "tpu": 9,        # Important accelerator
    "cpu": 8,        # Common fallback
    "accelerator": 7,
    "kpu": 6,
    "dpu": 5,
    "dsp": 4,
    "cgra": 3,
}

# Category importance
CATEGORY_PRIORITY = {
    "datacenter": 10,  # Production workloads
    "edge": 8,         # Deployed systems
    "desktop": 6,      # Development
    "mobile": 5,
    "embedded": 4,
    "embodied": 7,     # Robotics
}

# Vendor importance (based on market presence)
VENDOR_PRIORITY = {
    "NVIDIA": 10,
    "Google": 9,
    "Intel": 8,
    "AMD": 8,
    "Ampere Computing": 7,
    "Qualcomm": 7,
    "ARM": 6,
    "Xilinx": 6,
    "Hailo": 5,
}


def calculate_priority_score(
    entry: HardwareEntry,
    is_calibrated: bool,
    days_since_calibration: Optional[int],
    stale_threshold_days: int
) -> float:
    """
    Calculate priority score for calibrating a hardware entry.

    Higher score = higher priority to calibrate.

    Factors:
    - Device type importance
    - Category importance
    - Vendor importance
    - Whether it's calibrated (uncalibrated = higher priority)
    - Staleness (stale calibration = higher priority)
    - Peak performance (higher peak = more important)
    """
    score = 0.0

    # Base priority from device type
    score += DEVICE_TYPE_PRIORITY.get(entry.device_type, 1) * 10

    # Category priority
    score += CATEGORY_PRIORITY.get(entry.product_category, 1) * 5

    # Vendor priority
    score += VENDOR_PRIORITY.get(entry.vendor, 1) * 3

    # Performance importance (log scale, normalized)
    peak = entry.theoretical_peaks.get("fp32", 0)
    if peak > 0:
        import math
        # Log scale: 1 TFLOPS = 10 points, 100 TFLOPS = 20 points
        score += math.log10(peak / 100 + 1) * 10

    # Calibration status penalty/bonus
    if not is_calibrated:
        # Uncalibrated: high priority
        score += 50
    elif days_since_calibration is not None:
        if days_since_calibration > stale_threshold_days:
            # Stale: medium priority
            score += 25
        elif days_since_calibration > stale_threshold_days // 2:
            # Getting old: low priority boost
            score += 10
        else:
            # Fresh: reduce priority
            score -= 20

    return score


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_coverage(
    registry: HardwareRegistry,
    stale_threshold_days: int = 30
) -> CoverageReport:
    """
    Analyze calibration coverage across the hardware registry.

    Args:
        registry: Hardware registry to analyze
        stale_threshold_days: Days after which calibration is considered stale

    Returns:
        CoverageReport with detailed analysis
    """
    now = datetime.now()
    hardware_status = []

    # Analyze each hardware entry
    for entry in registry.list_hardware():
        # Check calibration status
        is_calibrated = len(entry.calibrations) > 0
        calibration_count = len(entry.calibrations)

        # Find latest calibration date
        latest_date = None
        days_since = None
        if entry.calibrations:
            latest_date = max(c.timestamp for c in entry.calibrations)
            days_since = (now - latest_date).days

        # Check staleness
        is_stale = False
        if days_since is not None and days_since > stale_threshold_days:
            is_stale = True

        # Get peaks
        theoretical_peak = entry.theoretical_peaks.get("fp32", 0)
        calibrated_peak = None
        efficiency = None
        if is_calibrated:
            calibrated_peak = entry.get_calibrated_peak("fp32", skip_suspicious=False)
            if theoretical_peak > 0 and calibrated_peak > 0:
                efficiency = calibrated_peak / theoretical_peak

        # Check for suspicious calibrations (efficiency > 100%)
        suspicious_count = sum(1 for c in entry.calibrations if c.is_suspicious)
        is_suspicious = suspicious_count > 0

        # Calculate priority (suspicious calibrations need recalibration)
        priority = calculate_priority_score(
            entry, is_calibrated, days_since, stale_threshold_days
        )
        if is_suspicious:
            priority += 30  # Boost priority for suspicious calibrations

        status = CalibrationStatus(
            hardware_id=entry.id,
            model=entry.model,
            device_type=entry.device_type,
            category=entry.product_category,
            vendor=entry.vendor,
            is_calibrated=is_calibrated,
            calibration_count=calibration_count,
            latest_calibration_date=latest_date,
            days_since_calibration=days_since,
            is_stale=is_stale,
            theoretical_peak_gflops=theoretical_peak,
            calibrated_peak_gflops=calibrated_peak,
            efficiency=efficiency,
            priority_score=priority,
            is_suspicious=is_suspicious,
            suspicious_count=suspicious_count,
        )
        hardware_status.append(status)

    # Calculate coverage by dimension
    by_device_type = _calculate_dimension_coverage(hardware_status, "device_type")
    by_category = _calculate_dimension_coverage(hardware_status, "category")
    by_vendor = _calculate_dimension_coverage(hardware_status, "vendor")

    # Find stale calibrations
    stale = [s for s in hardware_status if s.is_stale]
    stale.sort(key=lambda s: s.days_since_calibration or 0, reverse=True)

    # Find suspicious calibrations (efficiency > 100%)
    suspicious = [s for s in hardware_status if s.is_suspicious]
    suspicious.sort(key=lambda s: (s.efficiency or 0), reverse=True)

    # Priority list (uncalibrated + stale + suspicious, sorted by priority)
    priority_list = [
        s for s in hardware_status
        if not s.is_calibrated or s.is_stale or s.is_suspicious
    ]
    priority_list.sort(key=lambda s: s.priority_score, reverse=True)

    # Overall stats
    total = len(hardware_status)
    calibrated = sum(1 for s in hardware_status if s.is_calibrated)
    coverage_pct = (calibrated / total * 100) if total > 0 else 0

    return CoverageReport(
        timestamp=now,
        total_hardware=total,
        calibrated_count=calibrated,
        coverage_pct=coverage_pct,
        by_device_type=by_device_type,
        by_category=by_category,
        by_vendor=by_vendor,
        hardware_status=hardware_status,
        stale_calibrations=stale,
        suspicious_calibrations=suspicious,
        priority_list=priority_list,
    )


def _calculate_dimension_coverage(
    status_list: List[CalibrationStatus],
    dimension: str
) -> Dict[str, Dict[str, Any]]:
    """Calculate coverage breakdown for a dimension."""
    from collections import defaultdict

    counts = defaultdict(lambda: {"total": 0, "calibrated": 0, "hardware": []})

    for s in status_list:
        key = getattr(s, dimension)
        counts[key]["total"] += 1
        if s.is_calibrated:
            counts[key]["calibrated"] += 1
        counts[key]["hardware"].append(s.hardware_id)

    result = {}
    for key, data in counts.items():
        total = data["total"]
        calibrated = data["calibrated"]
        result[key] = {
            "total": total,
            "calibrated": calibrated,
            "uncalibrated": total - calibrated,
            "coverage_pct": (calibrated / total * 100) if total > 0 else 0,
            "hardware": data["hardware"],
        }

    return result


# =============================================================================
# FORMATTING
# =============================================================================

def format_text_report(report: CoverageReport, show_all: bool = False) -> str:
    """Format coverage report as text."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("CALIBRATION COVERAGE REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overall stats
    lines.append("OVERALL COVERAGE")
    lines.append("-" * 40)
    lines.append(f"  Total hardware entries: {report.total_hardware}")
    lines.append(f"  Calibrated:             {report.calibrated_count}")
    lines.append(f"  Coverage:               {report.coverage_pct:.1f}%")
    lines.append("")

    # By device type
    lines.append("BY DEVICE TYPE")
    lines.append("-" * 40)
    for dt, data in sorted(report.by_device_type.items(),
                           key=lambda x: x[1]["coverage_pct"], reverse=True):
        bar = _progress_bar(data["coverage_pct"], 20)
        lines.append(f"  {dt:12s} {data['calibrated']:2d}/{data['total']:2d} {bar} {data['coverage_pct']:5.1f}%")
    lines.append("")

    # By category
    lines.append("BY CATEGORY")
    lines.append("-" * 40)
    for cat, data in sorted(report.by_category.items(),
                            key=lambda x: x[1]["coverage_pct"], reverse=True):
        bar = _progress_bar(data["coverage_pct"], 20)
        lines.append(f"  {cat:12s} {data['calibrated']:2d}/{data['total']:2d} {bar} {data['coverage_pct']:5.1f}%")
    lines.append("")

    # Suspicious calibrations (efficiency > 100%)
    if report.suspicious_calibrations:
        lines.append("SUSPICIOUS CALIBRATIONS (efficiency > 100%)")
        lines.append("-" * 40)
        lines.append("  These calibrations likely used iGPU or have measurement errors.")
        lines.append("  They are excluded from calibrated peak calculations.")
        lines.append("")
        for s in report.suspicious_calibrations[:10]:
            eff_pct = (s.efficiency or 0) * 100
            lines.append(f"  {s.hardware_id}: {eff_pct:.0f}% efficiency ({s.suspicious_count} bad runs)")
        if len(report.suspicious_calibrations) > 10:
            lines.append(f"  ... and {len(report.suspicious_calibrations) - 10} more")
        lines.append("")

    # Stale calibrations
    if report.stale_calibrations:
        lines.append("STALE CALIBRATIONS")
        lines.append("-" * 40)
        for s in report.stale_calibrations[:10]:
            days = s.days_since_calibration
            lines.append(f"  {s.hardware_id}: {days} days ago")
        if len(report.stale_calibrations) > 10:
            lines.append(f"  ... and {len(report.stale_calibrations) - 10} more")
        lines.append("")

    # Priority list
    lines.append("CALIBRATION PRIORITY (Top 10)")
    lines.append("-" * 40)
    for i, s in enumerate(report.priority_list[:10], 1):
        if s.is_suspicious:
            status = "SUSPICIOUS"
        elif s.is_stale:
            status = "STALE"
        else:
            status = "UNCALIBRATED"
        lines.append(f"  {i:2d}. {s.model[:40]:40s}")
        lines.append(f"      {s.device_type}/{s.category} [{status}] score={s.priority_score:.0f}")
    lines.append("")

    # Detailed list (if requested)
    if show_all:
        lines.append("ALL HARDWARE STATUS")
        lines.append("-" * 40)
        for s in sorted(report.hardware_status, key=lambda x: x.model):
            status = "OK" if s.is_calibrated and not s.is_stale else "STALE" if s.is_stale else "NONE"
            eff = f"{s.efficiency*100:.0f}%" if s.efficiency else "N/A"
            lines.append(f"  [{status:5s}] {s.model[:45]:45s} eff={eff:>5s}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def _progress_bar(pct: float, width: int) -> str:
    """Create ASCII progress bar."""
    filled = int(pct / 100 * width)
    empty = width - filled
    return "[" + "#" * filled + "-" * empty + "]"


def format_json_report(report: CoverageReport) -> str:
    """Format coverage report as JSON."""
    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, CalibrationStatus):
            return {
                "hardware_id": obj.hardware_id,
                "model": obj.model,
                "device_type": obj.device_type,
                "category": obj.category,
                "vendor": obj.vendor,
                "is_calibrated": obj.is_calibrated,
                "calibration_count": obj.calibration_count,
                "latest_calibration_date": obj.latest_calibration_date.isoformat() if obj.latest_calibration_date else None,
                "days_since_calibration": obj.days_since_calibration,
                "is_stale": obj.is_stale,
                "theoretical_peak_gflops": obj.theoretical_peak_gflops,
                "calibrated_peak_gflops": obj.calibrated_peak_gflops,
                "efficiency": obj.efficiency,
                "priority_score": obj.priority_score,
            }
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    data = {
        "timestamp": report.timestamp.isoformat(),
        "summary": {
            "total_hardware": report.total_hardware,
            "calibrated_count": report.calibrated_count,
            "coverage_pct": report.coverage_pct,
        },
        "by_device_type": report.by_device_type,
        "by_category": report.by_category,
        "by_vendor": report.by_vendor,
        "stale_calibrations": [serialize(s) for s in report.stale_calibrations],
        "priority_list": [serialize(s) for s in report.priority_list],
        "all_hardware": [serialize(s) for s in report.hardware_status],
    }

    return json.dumps(data, indent=2)


def format_csv_report(report: CoverageReport) -> str:
    """Format coverage report as CSV."""
    lines = []

    # Header
    lines.append(",".join([
        "hardware_id", "model", "device_type", "category", "vendor",
        "is_calibrated", "calibration_count", "days_since_calibration",
        "is_stale", "theoretical_peak_gflops", "calibrated_peak_gflops",
        "efficiency", "priority_score"
    ]))

    # Data rows
    for s in report.hardware_status:
        lines.append(",".join([
            s.hardware_id,
            f'"{s.model}"',
            s.device_type,
            s.category,
            s.vendor,
            str(s.is_calibrated),
            str(s.calibration_count),
            str(s.days_since_calibration or ""),
            str(s.is_stale),
            f"{s.theoretical_peak_gflops:.1f}",
            f"{s.calibrated_peak_gflops:.1f}" if s.calibrated_peak_gflops else "",
            f"{s.efficiency:.3f}" if s.efficiency else "",
            f"{s.priority_score:.1f}",
        ]))

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze calibration coverage across the hardware registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Text report to stdout
  %(prog)s --stale-days 14          # Flag calibrations older than 14 days
  %(prog)s --output report.json     # JSON output
  %(prog)s --output report.csv      # CSV output
  %(prog)s --all                    # Show all hardware details
        """
    )

    parser.add_argument(
        "--stale-days", type=int, default=30,
        help="Days after which calibration is considered stale (default: 30)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file (supports .json, .csv, .txt)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show all hardware details in text report"
    )
    parser.add_argument(
        "--device-type", type=str,
        help="Filter by device type (e.g., gpu, cpu, tpu)"
    )
    parser.add_argument(
        "--category", type=str,
        help="Filter by category (e.g., datacenter, edge)"
    )
    parser.add_argument(
        "--priority-only", action="store_true",
        help="Only show priority list (uncalibrated + stale)"
    )

    args = parser.parse_args()

    # Load registry
    registry = HardwareRegistry()

    # Analyze coverage
    report = analyze_coverage(registry, stale_threshold_days=args.stale_days)

    # Apply filters if specified
    if args.device_type:
        report.hardware_status = [
            s for s in report.hardware_status
            if s.device_type == args.device_type
        ]
        report.priority_list = [
            s for s in report.priority_list
            if s.device_type == args.device_type
        ]

    if args.category:
        report.hardware_status = [
            s for s in report.hardware_status
            if s.category == args.category
        ]
        report.priority_list = [
            s for s in report.priority_list
            if s.category == args.category
        ]

    # Format output
    if args.output:
        ext = Path(args.output).suffix.lower()
        if ext == ".json":
            content = format_json_report(report)
        elif ext == ".csv":
            content = format_csv_report(report)
        else:
            content = format_text_report(report, show_all=args.all)

        Path(args.output).write_text(content)
        print(f"Report written to {args.output}")
    else:
        if args.priority_only:
            print("CALIBRATION PRIORITY LIST")
            print("=" * 60)
            for i, s in enumerate(report.priority_list, 1):
                status = "STALE" if s.is_stale else "UNCALIBRATED"
                print(f"{i:2d}. {s.model}")
                print(f"    {s.device_type}/{s.category} [{status}] score={s.priority_score:.0f}")
                print()
        else:
            print(format_text_report(report, show_all=args.all))


if __name__ == "__main__":
    main()
