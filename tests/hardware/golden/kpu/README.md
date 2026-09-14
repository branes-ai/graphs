# KPU golden snapshots

This directory holds one JSON file per KPU SKU in the embodied-schemas
catalog. Each file pins what every KPU modeling layer produces for that SKU:

- the generator round-trip, including the TDP breakdown
- silicon area and power per block
- both floorplans
- the PhysicalSpec
- the resource model
- the mapper results on synthetic subgraphs
- the validator findings

These files are the zero-diff gate for the KPU heterogeneous-tile refactor
(`docs/plans/kpu-heterogeneous-tile-refactor-plan.md`, Phase A0).
`tests/hardware/test_kpu_golden.py` enforces it.

- Code: `src/graphs/hardware/kpu_golden.py`
- Check: `python cli/kpu_golden_snapshot.py`
- Regenerate, only when a change is intended:
  `python cli/kpu_golden_snapshot.py --update [--sku ID]`

Regenerate when:

- a PR is a declared model change, or
- the embodied-schemas catalog data changed.

Before regenerating, review the diff the check prints. Commit the updated
JSON in the same PR as the change that caused it.

The goldens are generated against the embodied-schemas commit pinned in
`.github/workflows/ci.yml`. The `_meta` block records the package version
for provenance only; it is excluded from comparison.

Do not hand-edit these files. `test_golden_file_is_canonical` rejects any
file that is not byte-identical to what `dumps_snapshot` writes.
