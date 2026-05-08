#!/usr/bin/env python
"""
Hardware Resources Spec-Sheet CLI

Produces the unified "what is this chip?" view across every registered
hardware mapper: identity, fabrication (die size, transistors, process node,
foundry, architecture), peak throughput at every precision, memory bandwidth,
TDP, and launch info.

Complementary to ``cli/list_hardware_mappers.py``, which answers "what does
the mapper see?" -- operational constraints used by roofline / energy /
scheduling estimators. Some output overlap is intentional; the audiences
differ.

Operational data comes from ``mapper.resource_model``. Physical / fab data
comes from ``mapper.physical_spec`` (a sibling dataclass introduced in PR
#133). For mappers without a populated ``physical_spec``, physical columns
render as ``N/A`` -- the table is honest about partial coverage.

Usage:
    python cli/list_hardware_resources.py
    python cli/list_hardware_resources.py --category gpu
    python cli/list_hardware_resources.py --sort die_size
    python cli/list_hardware_resources.py --output specs.csv
    python cli/list_hardware_resources.py --output specs.md
    python cli/list_hardware_resources.py --output specs.json --verbose
"""

import argparse
import contextlib
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from graphs.hardware.mappers import (
    get_mapper_by_name,
    get_mapper_info,
    list_all_mappers,
)
from graphs.hardware.physical_spec import PhysicalSpec
from graphs.hardware.resource_model import Precision


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class HardwareResourceInfo:
    """Spec-sheet view of one hardware product.

    Operational fields (peak throughput by precision, memory bandwidth, TDP)
    are always populated -- they come from the resource model. Physical /
    fab fields are Optional; ``None`` means the underlying ``physical_spec``
    is absent or doesn't list that field, and renderers should print
    ``N/A``.
    """

    # Identity
    name: str
    category: str
    vendor: str

    # Operational (from resource_model)
    compute_units: int
    peak_flops_fp64: float    # GFLOPS
    peak_flops_fp32: float    # GFLOPS
    peak_flops_fp16: float    # GFLOPS
    peak_flops_fp8: float     # GFLOPS
    peak_flops_int32: float   # GOPS
    peak_flops_int16: float   # GOPS
    peak_flops_int8: float    # GOPS
    memory_bandwidth: float   # GB/s
    power_tdp: float          # Watts

    # Physical / fab (from physical_spec; None when not populated)
    die_size_mm2: Optional[float]
    transistors_billion: Optional[float]
    transistor_density_mtx_mm2: Optional[float]
    process_node_nm: Optional[int]
    process_node_name: Optional[str]
    foundry: Optional[str]
    architecture: Optional[str]
    num_dies: int
    is_chiplet: bool
    package_type: Optional[str]
    launch_date: Optional[str]
    launch_msrp_usd: Optional[float]
    physical_spec_source: Optional[str]
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _safe_peak_gflops(mapper, precision: Precision) -> float:
    """Peak ops at ``precision`` in GFLOPS / GOPS, or 0.0 if unsupported.

    ``HardwareResourceModel.get_peak_ops`` raises on unsupported precisions
    (#53). For the spec-sheet view we render 0.0 = "not natively supported"
    so the table stays rectangular.
    """
    try:
        return mapper.resource_model.get_peak_ops(precision) / 1e9
    except ValueError:
        return 0.0


def _peak_fp8_gflops(mapper) -> float:
    """Max FP8 throughput across the generic and IEEE-variant enums.

    Most resource models register ``FP8_E4M3`` / ``FP8_E5M2`` but not the
    generic ``FP8`` enum. Take the max so the column reflects the chip's
    real FP8 capability (matches list_hardware_mappers.py).
    """
    return max(
        _safe_peak_gflops(mapper, Precision.FP8),
        _safe_peak_gflops(mapper, Precision.FP8_E4M3),
        _safe_peak_gflops(mapper, Precision.FP8_E5M2),
    )


def _tdp_watts(mapper) -> float:
    """TDP from the default thermal profile, or 0.0 if no profiles registered."""
    rm = mapper.resource_model
    profile_name = rm.default_thermal_profile
    if profile_name is None or profile_name not in rm.thermal_operating_points:
        if rm.thermal_operating_points:
            profile_name = next(iter(rm.thermal_operating_points.keys()))
        else:
            return 0.0
    return rm.thermal_operating_points[profile_name].tdp_watts


def _build_record(name: str, mapper, info: Dict[str, Any]) -> HardwareResourceInfo:
    """Project one mapper + its registry metadata into a HardwareResourceInfo."""
    spec: Optional[PhysicalSpec] = getattr(mapper, "physical_spec", None)

    return HardwareResourceInfo(
        name=name,
        category=info["category"],
        vendor=info["vendor"],
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp64=_safe_peak_gflops(mapper, Precision.FP64),
        peak_flops_fp32=_safe_peak_gflops(mapper, Precision.FP32),
        peak_flops_fp16=_safe_peak_gflops(mapper, Precision.FP16),
        peak_flops_fp8=_peak_fp8_gflops(mapper),
        peak_flops_int32=_safe_peak_gflops(mapper, Precision.INT32),
        peak_flops_int16=_safe_peak_gflops(mapper, Precision.INT16),
        peak_flops_int8=_safe_peak_gflops(mapper, Precision.INT8),
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=_tdp_watts(mapper),
        die_size_mm2=spec.die_size_mm2 if spec else None,
        transistors_billion=spec.transistors_billion if spec else None,
        transistor_density_mtx_mm2=spec.transistor_density_mtx_mm2 if spec else None,
        process_node_nm=spec.process_node_nm if spec else None,
        process_node_name=spec.process_node_name if spec else None,
        foundry=spec.foundry if spec else None,
        architecture=spec.architecture if spec else None,
        num_dies=spec.num_dies if spec else 1,
        is_chiplet=spec.is_chiplet if spec else False,
        package_type=spec.package_type if spec else None,
        launch_date=spec.launch_date if spec else None,
        launch_msrp_usd=spec.launch_msrp_usd if spec else None,
        physical_spec_source=spec.source if spec else None,
        extras=dict(spec.extras) if spec else {},
    )


def discover_all_resources(category_filter: Optional[str] = None) -> List[HardwareResourceInfo]:
    """Walk the mapper registry and build a HardwareResourceInfo per entry."""
    records: List[HardwareResourceInfo] = []
    for name in list_all_mappers():
        info = get_mapper_info(name)
        if info is None:
            continue
        if category_filter and info["category"].lower() != category_filter.lower():
            continue
        try:
            mapper = get_mapper_by_name(name)
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"warning: failed to instantiate {name!r}: {exc}",
                file=sys.stderr,
            )
            continue
        if mapper is None:
            continue
        records.append(_build_record(name, mapper, info))
    return records


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

# Sort keys are (missing_test, value_extractor) pairs. ``missing_test`` lets
# us partition records that have no value for the chosen key and append them
# at the end -- regardless of ``--reverse``. The naive single-key approach
# (key=(missing, value), reverse=True) would flip the missing partition to
# the FRONT under reverse, which is the opposite of what users want when
# they ask for "biggest die_size" and expect populated rows first.
_SORT_KEYS = {
    "name":          (lambda r: False,                                  lambda r: r.name.lower()),
    "die_size":      (lambda r: r.die_size_mm2 is None,                 lambda r: r.die_size_mm2 or 0.0),
    "transistors":   (lambda r: r.transistors_billion is None,          lambda r: r.transistors_billion or 0.0),
    "density":       (lambda r: r.transistor_density_mtx_mm2 is None,   lambda r: r.transistor_density_mtx_mm2 or 0.0),
    "launch_year":   (lambda r: r.launch_date is None,                  lambda r: r.launch_date or ""),
    "tdp":           (lambda r: False,                                  lambda r: r.power_tdp),
    "tops_per_watt": (
        lambda r: False,
        lambda r: (r.peak_flops_int8 / 1000.0) / r.power_tdp if r.power_tdp > 0 else 0.0,
    ),
}


def _sort_records(
    records: List[HardwareResourceInfo],
    sort_key: str,
    reverse: bool,
) -> List[HardwareResourceInfo]:
    missing_test, value_fn = _SORT_KEYS[sort_key]
    populated = [r for r in records if not missing_test(r)]
    missing = [r for r in records if missing_test(r)]
    populated.sort(key=value_fn, reverse=reverse)
    # Missing records keep their natural order (by name) and always trail.
    missing.sort(key=lambda r: r.name.lower())
    return populated + missing


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _na(value: Any) -> str:
    """Render Optional values as ``N/A`` when missing."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _format_msrp(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.0f}"


def generate_text_report(
    records: List[HardwareResourceInfo],
    verbose: bool = False,
) -> None:
    """Human-readable text report. Wide table, segmented by category."""
    table_width = 145
    print("=" * table_width)
    print("HARDWARE RESOURCES SPEC SHEET")
    print("=" * table_width)
    print()
    print(f"Total products: {len(records)}")
    populated = sum(1 for r in records if r.die_size_mm2 is not None)
    print(f"PhysicalSpec coverage: {populated}/{len(records)} populated, "
          f"{len(records) - populated} N/A")
    print()

    by_category: Dict[str, List[HardwareResourceInfo]] = {}
    for r in records:
        by_category.setdefault(r.category, []).append(r)

    header = (
        f"{'Hardware':<28}"
        f"{'Die mm2':>10}"
        f"{'Tx (B)':>9}"
        f"{'Mtx/mm2':>10}"
        f"{'Node':>10}"
        f"{'Foundry':>10}"
        f"{'Arch':>14}"
        f"{'TDP (W)':>10}"
        f"{'INT8 TOPS':>11}"
        f"{'Launched':>13}"
    )

    for category in sorted(by_category.keys()):
        cat_records = by_category[category]
        print("-" * table_width)
        print(f"CATEGORY: {category.upper()} ({len(cat_records)} products)")
        print("-" * table_width)
        print(header)
        print("-" * table_width)
        for r in cat_records:
            row = (
                f"{r.name[:27]:<28}"
                f"{_na(r.die_size_mm2):>10}"
                f"{_na(r.transistors_billion):>9}"
                f"{_na(r.transistor_density_mtx_mm2):>10}"
                # Prefer the named version; fall back to the raw nm value.
                # Don't pre-format with _na -- "N/A" is truthy and would mask
                # nm-only specs.
                f"{(r.process_node_name or _na(r.process_node_nm)):>10}"
                f"{_na(r.foundry):>10}"
                f"{(r.architecture or 'N/A')[:13]:>14}"
                f"{r.power_tdp:>10.1f}"
                f"{r.peak_flops_int8/1000:>11.1f}"
                f"{_na(r.launch_date):>13}"
            )
            print(row)

            if verbose:
                # Show full precision row + provenance under the main row
                precisions = "  ".join(
                    f"{label}={val:.1f}"
                    for label, val in [
                        ("FP64", r.peak_flops_fp64),
                        ("FP32", r.peak_flops_fp32),
                        ("FP16", r.peak_flops_fp16),
                        ("FP8", r.peak_flops_fp8),
                        ("INT32", r.peak_flops_int32),
                        ("INT16", r.peak_flops_int16),
                        ("INT8", r.peak_flops_int8),
                    ]
                    if val > 0
                )
                print(f"    Vendor: {r.vendor}   Compute units: {r.compute_units}   "
                      f"Mem BW: {r.memory_bandwidth:.1f} GB/s")
                print(f"    Peak (G[FL]OPS): {precisions or 'N/A'}")
                print(f"    Packaging: num_dies={r.num_dies} chiplet={r.is_chiplet} "
                      f"type={_na(r.package_type)}   MSRP: {_format_msrp(r.launch_msrp_usd)}")
                if r.physical_spec_source:
                    print(f"    PhysicalSpec source: {r.physical_spec_source}")
                if r.extras:
                    print(f"    Extras: {r.extras}")
        print()

    print("=" * table_width)
    print(f"Total: {len(records)} hardware products")
    print("=" * table_width)


def generate_json_report(
    records: List[HardwareResourceInfo],
    verbose: bool = False,
) -> None:
    """JSON report. Always round-trips through asdict; ``verbose`` is a no-op
    for JSON since the format is already complete."""
    payload = {
        "total_products": len(records),
        "populated_physical_spec": sum(1 for r in records if r.die_size_mm2 is not None),
        "products": [asdict(r) for r in records],
    }
    print(json.dumps(payload, indent=2))


def generate_csv_report(
    records: List[HardwareResourceInfo],
    verbose: bool = False,  # noqa: ARG001 - keep signature parity
) -> None:
    """One row per product, columns are HardwareResourceInfo fields.

    None values render as empty cells (CSV-idiomatic). dict ``extras`` is
    JSON-encoded so the cell stays single-valued.
    """
    field_names = [f.name for f in fields(HardwareResourceInfo)]
    writer = csv.writer(sys.stdout)
    writer.writerow(field_names)
    for r in records:
        row = []
        for name in field_names:
            value = getattr(r, name)
            if value is None:
                row.append("")
            elif isinstance(value, dict):
                row.append(json.dumps(value, sort_keys=True) if value else "")
            else:
                row.append(value)
        writer.writerow(row)


def generate_markdown_report(
    records: List[HardwareResourceInfo],
    verbose: bool = False,
) -> None:
    """Real Markdown table per category. Designed to render cleanly in
    GitHub / PR descriptions."""
    print("# Hardware Resources Spec Sheet")
    print()
    print(f"Total products: **{len(records)}**.   ", end="")
    populated = sum(1 for r in records if r.die_size_mm2 is not None)
    print(f"PhysicalSpec coverage: **{populated}/{len(records)}** populated.")
    print()

    by_category: Dict[str, List[HardwareResourceInfo]] = {}
    for r in records:
        by_category.setdefault(r.category, []).append(r)

    for category in sorted(by_category.keys()):
        cat_records = by_category[category]
        print(f"## {category.upper()} ({len(cat_records)} products)")
        print()
        print(
            "| Hardware | Vendor | Die mm² | Tx (B) | Mtx/mm² | Node | Foundry | "
            "Arch | TDP (W) | INT8 TOPS | Launched |"
        )
        print(
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |"
        )
        for r in cat_records:
            print(
                f"| {r.name} | {r.vendor} | {_na(r.die_size_mm2)} | "
                f"{_na(r.transistors_billion)} | {_na(r.transistor_density_mtx_mm2)} | "
                f"{(r.process_node_name or _na(r.process_node_nm))} | "
                f"{_na(r.foundry)} | {r.architecture or 'N/A'} | "
                f"{r.power_tdp:.1f} | {r.peak_flops_int8/1000:.1f} | "
                f"{_na(r.launch_date)} |"
            )
        print()

        if verbose:
            for r in cat_records:
                if r.physical_spec_source:
                    print(f"- *{r.name}* PhysicalSpec source: `{r.physical_spec_source}`")
            print()


# ---------------------------------------------------------------------------
# Output routing
# ---------------------------------------------------------------------------

_EXT_TO_FORMAT = {
    ".json": "json",
    ".csv": "csv",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}

# Tokens accepted on the --format flag. ``md`` is an alias for ``markdown``
# (to match the .md extension); both normalize to the same generator.
_FORMAT_ALIASES = {
    "md": "markdown",
}

_FORMAT_CHOICES = ("text", "json", "csv", "markdown", "md")

_GENERATORS = {
    "text": generate_text_report,
    "json": generate_json_report,
    "csv": generate_csv_report,
    "markdown": generate_markdown_report,
}


def _resolve_format(output_path: Optional[str], explicit_format: Optional[str]) -> str:
    """Pick the output format with explicit-flag precedence.

    Precedence: explicit ``--format`` > file extension > "text" default.
    The flag wins so that ``--output spec.csv --format json`` writes JSON
    (matches the user's stated intent rather than silently re-detecting).
    """
    if explicit_format:
        return _FORMAT_ALIASES.get(explicit_format, explicit_format)
    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        if ext in _EXT_TO_FORMAT:
            return _EXT_TO_FORMAT[ext]
    return "text"


def _emit(fmt: str, records: List[HardwareResourceInfo], verbose: bool) -> None:
    _GENERATORS[fmt](records, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spec-sheet view of every registered hardware product",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default text report, all products
  python cli/list_hardware_resources.py

  # Filter to one category
  python cli/list_hardware_resources.py --category gpu

  # Sort by die area (descending), with full per-product detail
  python cli/list_hardware_resources.py --sort die_size --reverse --verbose

  # Write Markdown table for a PR description
  python cli/list_hardware_resources.py --output specs.md

  # CSV for analysis in a spreadsheet
  python cli/list_hardware_resources.py --output specs.csv

  # JSON for downstream tools
  python cli/list_hardware_resources.py --output specs.json
        """,
    )

    parser.add_argument(
        "--category",
        help="Filter to a single category (gpu, cpu, dsp, tpu, kpu, dpu, "
             "cgra, accelerator, dfm). Case-insensitive.",
    )
    parser.add_argument(
        "--format",
        choices=list(_FORMAT_CHOICES),
        default=None,
        help="Output format. Always wins over --output extension when "
             "explicitly provided. ``md`` is accepted as an alias for "
             "``markdown``. If omitted, format is auto-detected from "
             "--output extension; falls back to ``text`` for stdout.",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension "
             "(.json, .csv, .md/.markdown, .txt) unless --format is "
             "explicitly given (which always wins).",
    )
    parser.add_argument(
        "--sort",
        choices=sorted(_SORT_KEYS.keys()),
        default="name",
        help="Sort key (default: name). 'density' uses transistor_density_mtx_mm2; "
             "rows with missing values for the chosen key sort last.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the sort order (descending).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Per-product detail block: full precision row, packaging, "
             "MSRP, PhysicalSpec source citation, extras dict.",
    )

    args = parser.parse_args()

    records = discover_all_resources(category_filter=args.category)
    records = _sort_records(records, args.sort, reverse=args.reverse)

    fmt = _resolve_format(args.output, args.format)

    if args.output:
        with open(args.output, "w", newline="") as fh:
            with contextlib.redirect_stdout(fh):
                _emit(fmt, records, args.verbose)
        print(f"Wrote {fmt} report to {args.output}", file=sys.stderr)
    else:
        _emit(fmt, records, args.verbose)


if __name__ == "__main__":
    main()
