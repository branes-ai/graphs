#!/usr/bin/env python
"""
PDK Ingestion Tool (Phase 7)

Converts a structured "PDK summary" file (YAML or JSON, shaped like a
``ProcessNodeEntry``) into a validated catalog YAML, ready to drop
into ``${PROCESS_NODE_DATA_DIR}/<id>.yaml`` or
``embodied-schemas/data/process-nodes/<foundry>/<id>.yaml``.

Vendor PDK file formats vary widely (TSMC, Samsung, GlobalFoundries
each ship different LEF/LIB/CAD characterization data); this tool
does NOT parse vendor-native PDK files directly. The expected
workflow is:

  1. Vendor-specific extraction script -> structured PDK summary
     (YAML or JSON; shape matches ``ProcessNodeEntry``).
  2. ``cli/import_pdk.py --input <summary>`` -> validated catalog
     YAML with ``confidence: calibrated`` and a citation.
  3. Drop the result into ``${PROCESS_NODE_DATA_DIR}`` (a private
     directory outside this repo), and the
     ``embodied_schemas.load_process_nodes()`` overlay path picks
     it up automatically. CALIBRATED entries supersede the public
     THEORETICAL estimates.

What this tool guarantees:

- The input is a valid ``ProcessNodeEntry`` (Pydantic validation).
- The output's ``confidence`` is ``calibrated`` by default (PDK data
  shouldn't masquerade as THEORETICAL after ingestion). Override
  with ``--confidence`` if interpolating from PDK data.
- The output's ``source`` is set to the PDK rev / citation passed
  in via ``--source``, ensuring downstream provenance is traceable.

Usage:

    # Basic: validate + emit catalog YAML
    python cli/import_pdk.py \\
        --input pdk_summary.yaml \\
        --source "TSMC N16FF+ PDK rev 2024Q1" \\
        --output /private/process-nodes/tsmc_n16_calibrated.yaml

    # Auto-place under PROCESS_NODE_DATA_DIR if set
    export PROCESS_NODE_DATA_DIR=/private/process-nodes
    python cli/import_pdk.py --input pdk_summary.yaml \\
        --source "TSMC N16FF+ PDK rev 2024Q1"
    # -> writes to ${PROCESS_NODE_DATA_DIR}/<id>.yaml

    # JSON input also accepted (auto-detected from extension)
    python cli/import_pdk.py --input pdk_summary.json \\
        --source "..." --output node.yaml

Exit codes:
    0 = wrote the catalog YAML successfully
    1 = validation failed (input doesn't conform to ProcessNodeEntry)
    2 = I/O / argument error
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from embodied_schemas import ProcessNodeEntry
from embodied_schemas.process_node import DataConfidence


def _load_input(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON input file. Format is detected by extension."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    raise ValueError(
        f"Unrecognized input extension {suffix!r}. Expected .yaml, .yml, "
        f"or .json."
    )


def _resolve_output_path(
    args: argparse.Namespace, entry: ProcessNodeEntry
) -> Path:
    """Pick an output path.

    Priority: explicit ``--output`` -> ``$PROCESS_NODE_DATA_DIR/<id>.yaml``
    -> ``./<id>.yaml`` in cwd.
    """
    if args.output:
        return Path(args.output)
    overlay_dir = os.environ.get("PROCESS_NODE_DATA_DIR")
    if overlay_dir:
        d = Path(overlay_dir)
        if not d.is_dir():
            print(
                f"warning: PROCESS_NODE_DATA_DIR={overlay_dir!r} is not "
                f"an existing directory. Writing to cwd.",
                file=sys.stderr,
            )
            return Path(f"{entry.id}.yaml")
        return d / f"{entry.id}.yaml"
    return Path(f"{entry.id}.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a structured PDK summary (YAML/JSON shaped like "
            "ProcessNodeEntry) into a validated catalog YAML with "
            "confidence: calibrated."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the PDK summary input (YAML or JSON).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Source citation -- PDK rev, characterization run id, or "
            "internal reference. Written to ``source:`` in the output "
            "and propagated as ProcessNodeEntry provenance."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Output catalog YAML path. Defaults to "
            "${PROCESS_NODE_DATA_DIR}/<id>.yaml when the env var is set, "
            "else <id>.yaml in cwd."
        ),
    )
    parser.add_argument(
        "--confidence",
        choices=[c.value for c in DataConfidence],
        default=DataConfidence.CALIBRATED.value,
        help=(
            "Override the output's ``confidence:``. Default 'calibrated' "
            "for PDK-derived data. Use 'interpolated' if the input is "
            "extrapolated between PDK characterization points."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: input not found: {in_path}", file=sys.stderr)
        return 2

    try:
        raw = _load_input(in_path)
    except Exception as exc:
        print(f"error: failed to parse input: {exc}", file=sys.stderr)
        return 2

    if not isinstance(raw, dict):
        print(
            f"error: input must be a mapping/object at the top level, "
            f"got {type(raw).__name__}",
            file=sys.stderr,
        )
        return 2

    # Force confidence + source before validation so the resulting
    # ProcessNodeEntry has the right provenance.
    raw["confidence"] = args.confidence
    raw["source"] = args.source

    try:
        entry = ProcessNodeEntry.model_validate(raw)
    except Exception as exc:
        print(
            f"error: input does not validate as ProcessNodeEntry:\n{exc}",
            file=sys.stderr,
        )
        return 1

    out_path = _resolve_output_path(args, entry)
    if out_path.exists() and not args.force:
        print(
            f"error: output already exists: {out_path}. Pass --force to "
            f"overwrite.",
            file=sys.stderr,
        )
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = entry.model_dump(mode="json", exclude_none=True)
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    print(f"info: wrote {out_path}", file=sys.stderr)
    print(f"  id:         {entry.id}")
    print(f"  foundry:    {entry.foundry.value}")
    print(f"  node:       {entry.node_name} ({entry.node_nm} nm)")
    print(f"  topology:   {entry.transistor_topology.value}")
    print(f"  libraries:  {len(entry.densities)}")
    print(f"  confidence: {entry.confidence.value}")
    print(f"  source:     {entry.source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
