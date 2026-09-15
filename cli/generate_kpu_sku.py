#!/usr/bin/env python
"""
KPU SKU Generator CLI

Reads a KPUSKUInputSpec from a YAML file, generates a fully-populated
KPUEntry by computing roll-up fields (die size, transistor count,
performance, idle power) from the architecture + silicon_bin + the
referenced ProcessNode, and writes the result as YAML or JSON.

Optionally runs the validator framework against the generated SKU and
returns non-zero exit code if any ERROR finding is produced -- the
generator and validator together let an architect author a new SKU
with confidence that the roll-up numbers are self-consistent.

Two modes:

1. Generate from a hand-authored input spec:

    python cli/generate_kpu_sku.py --input my_sku_spec.yaml \\
        --output stillwater_kpu_new.yaml --validate

2. Extract a spec from an existing SKU and regenerate (template / round-trip):

    python cli/generate_kpu_sku.py --from-sku kpu_t256_32x32_lp5x16_16nm_tsmc_ffp \\
        --output regenerated_t256.yaml --validate

3. Roadmap sweep -- override PE-array size while keeping everything else:

    for pe in 16x16 24x24 32x32 40x40; do
      python cli/generate_kpu_sku.py --from-sku kpu_t256_32x32_lp5x16_16nm_tsmc_ffp \\
          --pe-array $pe --output t256_$pe.yaml
    done

4. Heterogeneous KPUs (graphs#268 C3): tiles in an input spec may reference
   the tile-class library ({use: <id>, num_tiles: N, overrides: {...}});
   --pe-array CLASS=RxC resizes one pe_fabric class (a bare RxC resizes every
   pe_fabric class); --tile-mix CLASS=N,... sets tile counts:

    python cli/generate_kpu_sku.py --input hetero_spec.yaml \\
        --pe-array pe_int8_mac_i32=16x16 --tile-mix pe_int8_mac_i32=32,systolic_int8_ws=2

Exit codes:
    0 = generation succeeded (and --validate, if used, found no ERROR)
    1 = validator produced ERROR finding(s) on the generated SKU
    2 = generator error (bad spec, missing process node, etc.)
"""

import argparse
import json
import os
import sys
from typing import Optional

import yaml

from embodied_schemas import load_cooling_solutions, load_process_nodes

from graphs.hardware.compute_product_loader import load_compute_products_unified
from graphs.hardware.kpu_access import kpu_die_of
from graphs.hardware.kpu_sku_generator import (
    GeneratorError,
    apply_pe_array_override,
    apply_tile_mix,
    generate_kpu_sku,
    input_spec_from_compute_product,
)
from graphs.hardware.kpu_sku_input import KPUSKUInputSpec
from graphs.hardware.sku_validators import (
    Severity,
    ValidatorContext,
    default_registry,
    has_errors,
    load_validators,
)


def _load_spec(path: str) -> KPUSKUInputSpec:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return KPUSKUInputSpec.model_validate(data)


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "yaml"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {"yaml": "yaml", "yml": "yaml", "json": "json"}.get(ext, "yaml")


def _serialize(entry, fmt: str) -> str:
    payload = entry.model_dump(mode="json", exclude_none=True)
    if fmt == "json":
        return json.dumps(payload, indent=2) + "\n"
    # YAML; use safe_dump for round-trippable output.
    return yaml.safe_dump(
        payload, sort_keys=False, default_flow_style=False, indent=2
    )


def _parse_pe_array(arg: str) -> tuple[Optional[str], int, int]:
    """'32x32' -> (None, 32, 32); 'cls=16x8' -> ('cls', 16, 8). ValueError if malformed."""
    tile_class, _, dims = arg.rpartition("=")
    rows_str, cols_str = dims.lower().split("x", 1)
    return (tile_class or None), int(rows_str), int(cols_str)


def _parse_tile_mix(arg: str) -> dict[str, int]:
    """'a=24,b=4' -> {'a': 24, 'b': 4}. ValueError if malformed."""
    mix: dict[str, int] = {}
    for part in arg.split(","):
        ref, sep, n = part.strip().partition("=")
        if not sep or not ref:
            raise ValueError(f"expected CLASS=N, got {part!r}")
        try:
            mix[ref] = int(n)
        except ValueError:
            raise ValueError(f"expected an integer count, got {part!r}") from None
    return mix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a KPU SKU YAML from an input spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input",
        help="Path to a KPUSKUInputSpec YAML.",
    )
    src.add_argument(
        "--from-sku",
        help="Extract a spec from an existing KPU SKU id and regenerate "
        "(template / round-trip mode).",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension "
        "(.yaml / .yml / .json). Default: stdout YAML.",
    )
    parser.add_argument(
        "--pe-array",
        metavar="[CLASS=]ROWSxCOLS",
        action="append",
        help="Override PE-array dimensions. A bare ROWSxCOLS (e.g., '16x16') "
        "resizes every pe_fabric tile class; CLASS=ROWSxCOLS resizes one "
        "(CLASS = tile_class_id or tile_type). Repeatable. ops_per_tile_per_clock "
        "and pipeline fill/drain cycles are scaled to match. Use for roadmap "
        "sweeps: `--from-sku kpu_t256_32x32_lp5x16_16nm_tsmc_ffp --pe-array 16x16` "
        "regenerates T256 with smaller PEs to compare cost / capability.",
    )
    parser.add_argument(
        "--tile-mix",
        metavar="CLASS=N[,CLASS=N...]",
        action="append",
        help="Set num_tiles per tile class (CLASS = tile_class_id or tile_type). "
        "Repeatable. total_tiles and an auto checkerboard's spare sites follow.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the validator registry against the generated SKU. "
        "Exit non-zero on any ERROR finding.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="With --validate, treat WARNING findings as ERROR for "
        "exit-code purposes.",
    )
    args = parser.parse_args()

    # Load catalogs once.
    try:
        process_nodes = load_process_nodes()
        cooling_solutions = load_cooling_solutions()
    except Exception as exc:
        print(f"error: failed to load process-node / cooling catalogs: {exc}",
              file=sys.stderr)
        return 2

    # Build the spec.
    if args.input:
        try:
            spec = _load_spec(args.input)
        except Exception as exc:
            print(f"error: failed to parse spec {args.input!r}: {exc}",
                  file=sys.stderr)
            return 2
    else:
        try:
            kpus = load_compute_products_unified()
        except Exception as exc:
            print(f"error: failed to load KPU catalog: {exc}", file=sys.stderr)
            return 2
        existing = kpus.get(args.from_sku)
        if existing is None:
            print(
                f"error: no KPU with id={args.from_sku!r}. "
                f"Available: {', '.join(sorted(kpus))}",
                file=sys.stderr,
            )
            return 2
        spec = input_spec_from_compute_product(existing)

    # Apply the tile mix, then PE-array overrides (after spec load, before
    # generation).
    for mix_arg in args.tile_mix or []:
        try:
            spec = apply_tile_mix(spec, _parse_tile_mix(mix_arg))
        except ValueError as exc:
            print(f"error: --tile-mix: {exc}", file=sys.stderr)
            return 2
    for pe_arg in args.pe_array or []:
        try:
            tile_class, rows, cols = _parse_pe_array(pe_arg)
        except ValueError:
            print(
                f"error: --pe-array must be ROWSxCOLS or CLASS=ROWSxCOLS "
                f"(e.g., '32x32', 'pe_int8_mac_i32=16x16'); got {pe_arg!r}",
                file=sys.stderr,
            )
            return 2
        try:
            spec = apply_pe_array_override(spec, rows, cols, tile_class=tile_class)
        except ValueError as exc:
            print(f"error: --pe-array: {exc}", file=sys.stderr)
            return 2

    # Generate.
    try:
        entry = generate_kpu_sku(spec, process_nodes=process_nodes)
    except GeneratorError as exc:
        print(f"error: generator failed: {exc}", file=sys.stderr)
        return 2

    # Serialize.
    fmt = _detect_format(args.output)
    rendered = _serialize(entry, fmt)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(f"info: wrote {fmt} to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)

    # Validate.
    if args.validate:
        load_validators()
        ctx = ValidatorContext(
            sku=entry,
            process_node=process_nodes[kpu_die_of(entry).process_node_id],
            cooling_solutions=cooling_solutions,
        )
        findings = default_registry.run_all(ctx)
        if findings:
            print(
                f"info: validator produced {len(findings)} finding(s). "
                f"by severity: " + ", ".join(
                    f"{s.value}={sum(1 for f in findings if f.severity == s)}"
                    for s in Severity
                ),
                file=sys.stderr,
            )
            for f in findings:
                if f.severity in (Severity.ERROR, Severity.WARNING):
                    print("  " + f.render_one_line(), file=sys.stderr)
        if has_errors(findings):
            return 1
        if args.strict and any(f.severity == Severity.WARNING for f in findings):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
