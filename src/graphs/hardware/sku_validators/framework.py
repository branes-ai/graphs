"""SKU validator framework -- core types.

The validator framework is a registry of pluggable, categorized risk-checks
that operate on a ComputeSolution (= ProcessNode + CoolingSolution + SKU).
Each product-risk class -- electrical, area, energy, thermal, reliability,
geometry -- is its own validator class. Adding a new risk check is one
new class plus one ``@register`` line; the rest of the framework stays
the same.

This module defines the framework primitives only: Severity,
ValidatorCategory, Finding, ValidatorContext, the SKUValidator Protocol,
and ValidatorRegistry. Individual validators live in
``graphs.hardware.sku_validators.validators.*`` and self-register on
import via ``register(..)`` calls.

Design notes:

- ``Severity`` separates advisory (INFO), correctable (WARNING), and
  fatal (ERROR) findings. The CLI returns non-zero on any ERROR.
- ``ValidatorCategory`` lets users focus on one risk class at a time
  when triaging a SKU; new categories are added here as the framework
  grows (GEOMETRY arrives with the floorplanner in Stage 8).
- ``Finding`` carries a ``block`` field for per-silicon_bin findings so
  the offending block name lands in the report.
- ``ValidatorContext`` carries the full cooling-solution catalog rather
  than a single solution: thermal/EM validators iterate over thermal
  profiles and each profile names its own cooling.
- ``SKUValidator`` is a Protocol so users can supply validators via duck
  typing, no inheritance required. The registry holds a list of
  registered validators with their metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, List, Optional, Protocol, runtime_checkable

from embodied_schemas.cooling_solution import CoolingSolutionEntry
from embodied_schemas.kpu import KPUEntry
from embodied_schemas.process_node import ProcessNodeEntry


# ---------------------------------------------------------------------------
# Severity / Category enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Finding severity.

    INFO:    advisory; describes a property of the SKU, not a defect.
    WARNING: correctable; the SKU likely works but a property looks off.
    ERROR:   fatal; the SKU is implausible, inconsistent, or unsafe.

    The CLI returns non-zero on any ERROR finding. Tests can assert no
    ERRORs across the catalog as a CI gate.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        """Ordinal for sorting / threshold comparisons (ERROR > WARNING > INFO)."""
        return {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2}[self]


class ValidatorCategory(str, Enum):
    """Coarse risk-class for grouping findings.

    INTERNAL:    SKU self-consistency (sums match, refs resolve).
    ELECTRICAL:  bandwidth math, power-profile monotonicity, voltage rails.
    AREA:        per-block area, composite density, library validity.
    ENERGY:      TOPS/W envelope per-block.
    THERMAL:     power density per block vs cooling-solution ceiling.
    RELIABILITY: electromigration, NBTI/PBTI, yield risk.
    GEOMETRY:    floorplan pitch-match, aspect ratios, NoC routability
                 (added in Stage 8 with the floorplanner).
    """

    INTERNAL = "internal"
    ELECTRICAL = "electrical"
    AREA = "area"
    ENERGY = "energy"
    THERMAL = "thermal"
    RELIABILITY = "reliability"
    GEOMETRY = "geometry"


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One validator finding.

    Immutable so a registry run can be cached / re-rendered without risk
    of mutation. Equality is structural for easy assertion in tests.
    """

    validator: str
    """Name of the validator that produced the finding."""

    category: ValidatorCategory
    """Risk class this finding belongs to."""

    severity: Severity
    """INFO / WARNING / ERROR."""

    message: str
    """Human-readable explanation. Should be self-contained (no need to
    cross-reference other findings)."""

    block: Optional[str] = None
    """Name of the silicon_bin block this finding applies to, or None for
    chip-level findings."""

    profile: Optional[str] = None
    """Thermal-profile name this finding applies to, or None when the
    finding is profile-independent."""

    citation: Optional[str] = None
    """Citation supporting the bound the validator enforced (PDK rev,
    datasheet, public estimate). Helps reviewers judge whether to trust
    the finding."""

    def render_one_line(self) -> str:
        """Compact one-line rendering for table-style output."""
        loc_parts = []
        if self.block:
            loc_parts.append(f"block={self.block}")
        if self.profile:
            loc_parts.append(f"profile={self.profile}")
        loc = f"[{','.join(loc_parts)}] " if loc_parts else ""
        return (
            f"{self.severity.value.upper():7s} "
            f"{self.category.value:11s} "
            f"{self.validator:30s} "
            f"{loc}{self.message}"
        )


# ---------------------------------------------------------------------------
# Validator context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatorContext:
    """Everything a validator needs about one SKU.

    A ComputeSolution is the blend of ProcessNode + CoolingSolution + SKU.
    The context carries the SKU and the resolved ProcessNode plus the
    full cooling-solution catalog -- thermal/EM validators iterate over
    the SKU's thermal_profiles and each profile names its own cooling
    solution by id.

    Carrying the full cooling catalog (not just one solution) lets the
    framework handle per-profile cooling cleanly without forcing a
    'context per profile' iteration on every validator.
    """

    sku: KPUEntry
    """The SKU being validated. Today only KPUEntry; future GPU/CPU SKUs
    will be a Union here."""

    process_node: ProcessNodeEntry
    """The ProcessNodeEntry referenced by ``sku.process_node_id``."""

    cooling_solutions: dict[str, CoolingSolutionEntry]
    """Map of cooling_solution_id -> CoolingSolutionEntry. Includes every
    cooling solution in the catalog so validators can resolve any
    thermal_profile's reference."""

    extras: dict = field(default_factory=dict)
    """Free-form dict for validators or callers to stash extra data
    (e.g., a measured-vs-claimed comparison set). Not consumed by the
    framework itself."""

    def cooling_for(self, profile_name: str) -> Optional[CoolingSolutionEntry]:
        """Look up the CoolingSolution for a thermal profile by name.

        Returns None if the profile or its cooling_solution_id can't be
        resolved -- validators report that as a Finding rather than
        crashing.
        """
        for tp in self.sku.power.thermal_profiles:
            if tp.name == profile_name:
                return self.cooling_solutions.get(tp.cooling_solution_id)
        return None


# ---------------------------------------------------------------------------
# Validator Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SKUValidator(Protocol):
    """Duck-typed validator interface.

    Any object with ``name``, ``category``, and a ``check(ctx)`` method
    that returns a list of Findings can be registered. A class is
    typical, but a module-level function wrapped in a tiny adapter works
    too -- see ``register_callable`` for the function-style form.
    """

    name: str
    category: ValidatorCategory

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ValidatorRegistry:
    """Mutable registry of SKU validators.

    Exposed as the module-level singleton ``default_registry`` in
    ``graphs.hardware.sku_validators.__init__``. Validators self-register
    by importing that module and calling ``register(my_validator)`` (or
    via the ``@register`` decorator on a class). Tests construct a fresh
    ``ValidatorRegistry()`` instance to avoid bleed.
    """

    def __init__(self) -> None:
        self._validators: dict[str, SKUValidator] = {}

    # -- registration ------------------------------------------------------

    def register(self, validator: SKUValidator) -> SKUValidator:
        """Register a validator. Returns the validator unchanged so this
        can be used as a decorator on a class:

            @default_registry.register_class
            class MyValidator:
                name = "my_check"
                category = ValidatorCategory.AREA
                def check(self, ctx): ...
        """
        if not isinstance(validator, SKUValidator):
            raise TypeError(
                f"validator {validator!r} does not satisfy the SKUValidator "
                f"protocol (must have name, category, check(ctx))"
            )
        if validator.name in self._validators:
            raise ValueError(
                f"validator name {validator.name!r} already registered"
            )
        self._validators[validator.name] = validator
        return validator

    def register_class(self, cls: type) -> type:
        """Decorator for class-form validators. Instantiates with no
        args, registers the instance, returns the class so further
        decorators can chain.
        """
        instance = cls()
        self.register(instance)
        return cls

    def unregister(self, name: str) -> None:
        """Remove a validator (test convenience)."""
        self._validators.pop(name, None)

    def clear(self) -> None:
        """Drop all validators (test convenience)."""
        self._validators.clear()

    # -- introspection -----------------------------------------------------

    def names(self) -> List[str]:
        """Sorted list of registered validator names."""
        return sorted(self._validators)

    def categories(self) -> List[ValidatorCategory]:
        """Sorted list of categories that have at least one registered
        validator."""
        cats = {v.category for v in self._validators.values()}
        return sorted(cats, key=lambda c: c.value)

    def get(self, name: str) -> Optional[SKUValidator]:
        return self._validators.get(name)

    def __len__(self) -> int:
        return len(self._validators)

    def __contains__(self, name: str) -> bool:
        return name in self._validators

    # -- run ---------------------------------------------------------------

    def run_all(self, ctx: ValidatorContext) -> List[Finding]:
        """Run every registered validator against ``ctx``. Returns
        Findings sorted by (-severity, category, validator, block).
        Validator exceptions are caught and converted to ERROR Findings
        so a single broken validator doesn't crash the whole run.
        """
        return self._run(ctx, self._validators.values())

    def run_category(
        self, ctx: ValidatorContext, category: ValidatorCategory
    ) -> List[Finding]:
        """Run only validators in ``category``."""
        selected = [v for v in self._validators.values() if v.category == category]
        return self._run(ctx, selected)

    def run_one(self, name: str, ctx: ValidatorContext) -> List[Finding]:
        """Run a single named validator. Raises KeyError if not registered."""
        v = self._validators.get(name)
        if v is None:
            raise KeyError(f"validator {name!r} is not registered")
        return self._run(ctx, [v])

    def _run(
        self, ctx: ValidatorContext, validators: Iterable[SKUValidator]
    ) -> List[Finding]:
        findings: List[Finding] = []
        for v in validators:
            try:
                result = v.check(ctx)
            except Exception as exc:
                # Convert validator crashes into ERROR findings so the
                # CLI / tests see them rather than aborting the whole run.
                findings.append(
                    Finding(
                        validator=v.name,
                        category=v.category,
                        severity=Severity.ERROR,
                        message=f"validator crashed: {type(exc).__name__}: {exc}",
                        citation="framework: caught exception",
                    )
                )
                continue
            findings.extend(result)
        # Sort: highest severity first, then category / validator / block
        # for stable rendering.
        findings.sort(
            key=lambda f: (
                -f.severity.rank,
                f.category.value,
                f.validator,
                f.block or "",
                f.profile or "",
            )
        )
        return findings


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def make_callable_validator(
    name: str,
    category: ValidatorCategory,
    fn: Callable[[ValidatorContext], List[Finding]],
) -> SKUValidator:
    """Wrap a function as an SKUValidator-compliant object.

    Useful for one-off validators in tests; production validators should
    be class-based for discoverability.
    """

    class _Wrapped:
        pass

    obj = _Wrapped()
    obj.name = name
    obj.category = category
    obj.check = fn
    return obj  # type: ignore[return-value]


def filter_findings(
    findings: Iterable[Finding],
    *,
    min_severity: Optional[Severity] = None,
    category: Optional[ValidatorCategory] = None,
) -> List[Finding]:
    """Filter a finding list by severity floor and / or category.

    Used by the CLI's --severity and --category flags.
    """
    out: List[Finding] = []
    for f in findings:
        if min_severity is not None and f.severity.rank < min_severity.rank:
            continue
        if category is not None and f.category != category:
            continue
        out.append(f)
    return out


def has_errors(findings: Iterable[Finding]) -> bool:
    """True if any finding has severity == ERROR. Used by CLI for exit code."""
    return any(f.severity == Severity.ERROR for f in findings)
