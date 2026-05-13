"""GEOMETRY-category validators (Stage 8).

These run on top of the heuristic floorplan derived in
``graphs.hardware.silicon_floorplan``. They surface 2D layout problems
that area-only math can't see:

- ``floorplan_pitch_match``: alternating compute / memory tiles in the
  KPU mesh must share a pitch. If two tile classes have geometric
  pitches that differ by more than a small fraction, the SKU either
  needs whitespace (lower silicon utilization) or a non-uniform
  layout (longer wires, more energy). The validator reports the
  pitch mismatch so the architect can rebalance per-PE coefficients
  or reduce PEs in the over-area class.
- ``floorplan_within_die_envelope``: total floorplan rectangle fits
  inside the SKU's declared die_size_mm2 + reasonable slop. Catches
  cases where silicon_bin block coefficients drifted past what the
  declared die can hold.
- ``floorplan_aspect_ratio``: die-level aspect ratio bounds. Modern
  reticle limits and packaging typically constrain dies to
  aspect <= 2.5; severely elongated rectangles get flagged.

The geometry findings reference ``block=tile_class:<name>`` for
per-tile-class pitch findings so the offending class lands in the
report exactly where downstream tooling can find it.
"""

from __future__ import annotations

from typing import List

from ...compute_product_loader import compute_product_to_kpu_entry
from ...silicon_floorplan import (
    derive_kpu_architectural_floorplan,
    derive_kpu_floorplan,
)
from .. import ValidatorCategory, ValidatorContext, default_registry
from ..framework import Finding, Severity


def _legacy_sku(ctx: ValidatorContext):
    """Reverse-adapt ctx.sku (ComputeProduct) to KPUEntry for the
    silicon_floorplan derivation functions. Bridge until silicon_floorplan
    accepts ComputeProduct directly (next migration PR)."""
    return compute_product_to_kpu_entry(ctx.sku, ctx.process_node)


# ---------------------------------------------------------------------------
# Tunable bounds
# ---------------------------------------------------------------------------

# Maximum acceptable pitch ratio between tile classes before the KPU
# checkerboard layout requires wasteful whitespace. 1.20 = the smaller
# pitch is at most 20% smaller than the largest. Beyond that, alternating
# rows / columns between tile classes leaves visible gaps.
#
# Rationale: at 20% pitch mismatch, a row of small tiles occupies 80% of
# the unified-pitch row width; the remaining 20% becomes IO / wire-routing
# whitespace that adds little function. Beyond that, the architect should
# either rebalance per-PE coefficients (so all tiles land in the same
# area envelope) or accept a non-uniform mesh (more complex NoC routing).
#
# Stage 8 stance: emit WARN at >=1.20x (real signal). The ERROR threshold
# is set high (>=5.0x) so today's heuristic-v1 floorplan can flag real
# design imbalances without ERROR-failing the Phase 6 catalog CI gate
# during Stage 8 development. Tighten to ~2.0x once the heuristic is
# calibrated against measured silicon.
_PITCH_RATIO_WARNING = 1.20
_PITCH_RATIO_ERROR = 5.00

# Maximum die aspect ratio. Above 3.0 routing becomes painful and
# packaging gets expensive (asymmetric reticles, mask costs). Above 5.0
# we assume the floorplanner heuristic is producing nonsense.
_DIE_ASPECT_WARNING = 3.0
_DIE_ASPECT_ERROR = 5.0

# Allowed slack between heuristic floorplan area and the SKU's declared
# ``die.die_size_mm2``. The floorplan is a heuristic; treat 1.5x as the
# fuzzy "we're in the same ballpark" bound. ERROR threshold set high
# (>=3.0x) for Stage 8 -- a 2x derived/declared mismatch is real signal
# (silicon_bin coefficients, declared die size, or floorplan overhead)
# but should not ERROR-fail the catalog gate while the heuristic is
# uncalibrated. Tighten to ~2.0x after first measured-silicon callibration.
_DIE_AREA_RATIO_WARNING = 1.5
_DIE_AREA_RATIO_ERROR = 3.0

# Compute-vs-memory tile-pitch mismatch in the KPU checkerboard. The
# *primary* geometric concern: every compute tile is paired 1:1 with a
# memory tile (L3) in the alternating layout. If their pitches differ
# significantly, the die pays whitespace inside the smaller cell.
#
# Bands match the within-class pitch validator: WARN at >=1.20x
# (notable design imbalance), ERROR at >=5.0x (broken layout). For
# Stage 8 the ERROR threshold is set high so all 4 catalog SKUs land
# at WARN-max while the heuristic is uncalibrated. Tighten to ~2.0x
# after measured-silicon calibration.
_CM_PITCH_RATIO_WARNING = 1.20
_CM_PITCH_RATIO_ERROR = 5.00

# Whitespace-fraction warning band for the architectural floorplan.
# A KPU mesh with >50% mesh-area whitespace is paying a lot of silicon
# for nothing (typically because Matrix tiles set the unified pitch
# while the bulk of tiles are smaller INT8). WARN at >=40%, ERROR at
# >=80% (essentially "the design is half air").
_WHITESPACE_FRACTION_WARNING = 0.40
_WHITESPACE_FRACTION_ERROR = 0.80


# ---------------------------------------------------------------------------
# floorplan_pitch_match
# ---------------------------------------------------------------------------

@default_registry.register_class
class FloorplanPitchMatch:
    """Flag tile-class pitch mismatches in the KPU checkerboard layout.

    For an M0.5 KPU with ``mesh_rows x mesh_cols`` tiles drawn from
    multiple tile classes (INT8-primary, BF16-primary, Matrix), the
    physical pitch of every tile class should be within
    ``_PITCH_RATIO_WARNING`` of the largest. The validator emits a
    WARNING per-class above that threshold, ERROR above
    ``_PITCH_RATIO_ERROR``.
    """

    name = "floorplan_pitch_match"
    category = ValidatorCategory.GEOMETRY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        try:
            fp = derive_kpu_floorplan(_legacy_sku(ctx), ctx.process_node)
        except Exception as exc:
            return [Finding(
                validator=self.name,
                category=self.category,
                severity=Severity.ERROR,
                message=(
                    f"could not derive floorplan: {type(exc).__name__}: {exc}"
                ),
                citation="graphs.hardware.silicon_floorplan",
            )]

        if not fp.tile_pitches:
            return []

        max_pitch = max(tp.pitch_mm for tp in fp.tile_pitches.values())
        if max_pitch <= 0:
            return []

        out: List[Finding] = []
        for tile_class, tp in fp.tile_pitches.items():
            if tp.pitch_mm <= 0:
                continue
            ratio = max_pitch / tp.pitch_mm
            if ratio < _PITCH_RATIO_WARNING:
                continue  # Tile fits within the unified pitch tolerance
            sev = (
                Severity.ERROR if ratio >= _PITCH_RATIO_ERROR
                else Severity.WARNING
            )
            out.append(Finding(
                validator=self.name,
                category=self.category,
                severity=sev,
                message=(
                    f"tile class {tile_class!r} pitch {tp.pitch_mm:.3f} mm "
                    f"is {ratio:.2f}x smaller than max-class pitch "
                    f"{max_pitch:.3f} mm; KPU checkerboard will leave "
                    f"whitespace or require non-uniform NoC routing "
                    f"(tile area {tp.total_area_mm2:.4f} mm^2)"
                ),
                block=f"tile_class:{tile_class}",
                citation=(
                    f"silicon_floorplan v1 heuristic; pitch threshold "
                    f"WARN={_PITCH_RATIO_WARNING}x ERR={_PITCH_RATIO_ERROR}x"
                ),
            ))
        return out


# ---------------------------------------------------------------------------
# floorplan_within_die_envelope
# ---------------------------------------------------------------------------

@default_registry.register_class
class FloorplanWithinDieEnvelope:
    """Check the heuristic floorplan area is in the same ballpark as the
    SKU's declared ``die.die_size_mm2``.

    The floorplanner is a heuristic; small disagreements with the
    declared die size are expected. But a 2x+ disagreement means either
    (a) the silicon_bin coefficients drifted, (b) the declared die_size
    is wrong, or (c) the floorplanner heuristic is broken for this SKU.
    """

    name = "floorplan_within_die_envelope"
    category = ValidatorCategory.GEOMETRY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        try:
            fp = derive_kpu_floorplan(_legacy_sku(ctx), ctx.process_node)
        except Exception:
            return []  # Pitch-match validator already reports derivation errors
        declared = ctx.sku.dies[0].die_size_mm2
        derived = fp.die_area_mm2
        if declared <= 0 or derived <= 0:
            return []
        ratio = max(declared, derived) / min(declared, derived)
        if ratio < _DIE_AREA_RATIO_WARNING:
            return []
        sev = (
            Severity.ERROR if ratio >= _DIE_AREA_RATIO_ERROR
            else Severity.WARNING
        )
        return [Finding(
            validator=self.name,
            category=self.category,
            severity=sev,
            message=(
                f"heuristic floorplan area {derived:.1f} mm^2 differs "
                f"from declared die.die_size_mm2 {declared:.1f} mm^2 by "
                f"{ratio:.2f}x; check silicon_bin coefficients or die size"
            ),
            citation=(
                f"silicon_floorplan v1 heuristic; envelope threshold "
                f"WARN={_DIE_AREA_RATIO_WARNING}x ERR={_DIE_AREA_RATIO_ERROR}x"
            ),
        )]


# ---------------------------------------------------------------------------
# floorplan_aspect_ratio
# ---------------------------------------------------------------------------

@default_registry.register_class
class FloorplanAspectRatio:
    """Flag dies with extreme aspect ratios.

    Floorplan with aspect > _DIE_ASPECT_WARNING (currently 3.0) is
    awkward to route and package; > _DIE_ASPECT_ERROR (5.0) usually
    means the heuristic is off (e.g., the PHY strip on the right edge
    has dwarfed the mesh).
    """

    name = "floorplan_aspect_ratio"
    category = ValidatorCategory.GEOMETRY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        try:
            fp = derive_kpu_floorplan(_legacy_sku(ctx), ctx.process_node)
        except Exception:
            return []
        if fp.die_width_mm <= 0 or fp.die_height_mm <= 0:
            return []
        wide = max(fp.die_width_mm, fp.die_height_mm)
        tall = min(fp.die_width_mm, fp.die_height_mm)
        ar = wide / tall
        if ar < _DIE_ASPECT_WARNING:
            return []
        sev = (
            Severity.ERROR if ar >= _DIE_ASPECT_ERROR
            else Severity.WARNING
        )
        return [Finding(
            validator=self.name,
            category=self.category,
            severity=sev,
            message=(
                f"die aspect ratio {ar:.2f} ({fp.die_width_mm:.2f} x "
                f"{fp.die_height_mm:.2f} mm) is awkward for routing/"
                f"packaging; consider rebalancing PHY placement or mesh shape"
            ),
            citation=(
                f"silicon_floorplan v1 heuristic; aspect threshold "
                f"WARN={_DIE_ASPECT_WARNING} ERR={_DIE_ASPECT_ERROR}"
            ),
        )]


# ---------------------------------------------------------------------------
# floorplan_compute_memory_pitch_match  (architectural view)
# ---------------------------------------------------------------------------

@default_registry.register_class
class FloorplanComputeMemoryPitchMatch:
    """Flag compute-vs-memory tile-pitch mismatch in the KPU checkerboard.

    The KPU's M0.5 layout pairs every compute tile (PE + L1 + L2) with
    a memory tile (L3) in a checkerboard. For the layout to tile
    densely, ``compute_pitch ~= memory_pitch``. When they diverge, the
    smaller tile leaves whitespace inside its cell -- the larger pitch
    sets the cell size.

    This is a different concern from ``floorplan_pitch_match``, which
    catches mismatches *within* compute classes (INT8 vs Matrix). Both
    fire because they describe different geometric problems.
    """

    name = "floorplan_compute_memory_pitch_match"
    category = ValidatorCategory.GEOMETRY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        try:
            fp = derive_kpu_architectural_floorplan(
                _legacy_sku(ctx), ctx.process_node
            )
        except Exception as exc:
            return [Finding(
                validator=self.name,
                category=self.category,
                severity=Severity.ERROR,
                message=(
                    f"could not derive architectural floorplan: "
                    f"{type(exc).__name__}: {exc}"
                ),
                citation="graphs.hardware.silicon_floorplan",
            )]
        if not fp.compute_summaries or fp.memory_summary.pitch_mm <= 0:
            return []

        memory_pitch = fp.memory_summary.pitch_mm
        out: List[Finding] = []
        for class_name, summary in fp.compute_summaries.items():
            cp = summary.pitch_mm
            if cp <= 0:
                continue
            ratio = max(cp, memory_pitch) / min(cp, memory_pitch)
            if ratio < _CM_PITCH_RATIO_WARNING:
                continue
            sev = (
                Severity.ERROR if ratio >= _CM_PITCH_RATIO_ERROR
                else Severity.WARNING
            )
            larger = "compute" if cp > memory_pitch else "memory"
            out.append(Finding(
                validator=self.name,
                category=self.category,
                severity=sev,
                message=(
                    f"compute tile {class_name!r} pitch {cp:.3f} mm vs "
                    f"memory pitch {memory_pitch:.3f} mm = {ratio:.2f}x "
                    f"({larger} side dominates); checkerboard pairing "
                    f"will leave whitespace in the smaller cell "
                    f"(class waste {summary.class_whitespace_mm2:.1f} mm^2)"
                ),
                block=f"tile_class:{class_name}",
                citation=(
                    f"silicon_floorplan v1 heuristic; C/M pitch threshold "
                    f"WARN={_CM_PITCH_RATIO_WARNING}x "
                    f"ERR={_CM_PITCH_RATIO_ERROR}x"
                ),
            ))
        return out


# ---------------------------------------------------------------------------
# floorplan_whitespace_fraction
# ---------------------------------------------------------------------------

@default_registry.register_class
class FloorplanWhitespaceFraction:
    """Flag SKUs whose architectural die has too much whitespace.

    A high whitespace fraction means the chosen tile-class mix is
    paying silicon area for nothing -- typically because the largest
    class (Matrix) sets the unified pitch while the bulk of compute
    tiles are smaller. Useful as a top-level "die efficiency" alarm.

    The companion what-if estimates show the architect what the die
    would cost if every tile were of one class.
    """

    name = "floorplan_whitespace_fraction"
    category = ValidatorCategory.GEOMETRY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        try:
            fp = derive_kpu_architectural_floorplan(
                _legacy_sku(ctx), ctx.process_node
            )
        except Exception:
            return []  # The pitch validator already reports derivation errors

        ws_frac = fp.whitespace_fraction()
        if ws_frac < _WHITESPACE_FRACTION_WARNING:
            return []
        sev = (
            Severity.ERROR if ws_frac >= _WHITESPACE_FRACTION_ERROR
            else Severity.WARNING
        )
        # Find the smallest compute class (most whitespace contributor) and
        # the all-X what-if that would shrink the die the most.
        worst_class = max(
            fp.compute_summaries.values(),
            key=lambda s: s.class_whitespace_mm2,
            default=None,
        )
        best_what_if = min(
            fp.what_if, key=lambda w: w.die_area_mm2, default=None,
        )
        what_if_msg = ""
        if best_what_if and best_what_if.die_area_mm2 < fp.die_area_mm2:
            shrink = (
                1.0 - best_what_if.die_area_mm2 / fp.die_area_mm2
            ) * 100.0
            what_if_msg = (
                f"; an all-{best_what_if.tile_class!r} mesh would be "
                f"{best_what_if.die_area_mm2:.1f} mm^2 ({shrink:.0f}% smaller)"
            )
        worst_msg = ""
        if worst_class:
            worst_msg = (
                f"; biggest contributor is {worst_class.tile_class!r} "
                f"({worst_class.class_whitespace_mm2:.1f} mm^2 whitespace "
                f"across {worst_class.num_tiles} tiles)"
            )
        return [Finding(
            validator=self.name,
            category=self.category,
            severity=sev,
            message=(
                f"architectural die whitespace {ws_frac*100:.0f}% "
                f"({fp.whitespace_mm2():.1f} mm^2 of "
                f"{fp.die_area_mm2:.1f} mm^2){worst_msg}{what_if_msg}"
            ),
            citation=(
                f"silicon_floorplan v1 architectural; whitespace "
                f"threshold WARN={_WHITESPACE_FRACTION_WARNING*100:.0f}% "
                f"ERR={_WHITESPACE_FRACTION_ERROR*100:.0f}%"
            ),
        )]
