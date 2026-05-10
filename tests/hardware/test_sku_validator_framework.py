"""Smoke tests for the SKU validator framework (Phase 2a).

These tests exercise the framework primitives -- registry, context
builder, finding sorting, severity filtering -- without depending on
any specific validator. Phase 2b validators will get their own tests.

The tests construct fresh ``ValidatorRegistry()`` instances rather
than mutating the module-level ``default_registry`` so they don't
bleed across tests or with future Phase 2b validator registrations.
"""

from __future__ import annotations

import pytest

from graphs.hardware.sku_validators import (
    ContextError,
    Finding,
    Severity,
    ValidatorCategory,
    ValidatorRegistry,
    build_context_for_kpu,
    filter_findings,
    has_errors,
    make_callable_validator,
)


# ---------------------------------------------------------------------------
# Severity / Category enums
# ---------------------------------------------------------------------------

def test_severity_rank_orders_correctly():
    assert Severity.INFO.rank < Severity.WARNING.rank < Severity.ERROR.rank


def test_validator_category_has_all_planned_categories():
    # Phase 2a should expose every planned category, including GEOMETRY
    # for the Stage-8 floorplan validators (no validators registered
    # against it yet, but the enum value must exist).
    expected = {
        "internal", "electrical", "area",
        "energy", "thermal", "reliability", "geometry",
    }
    actual = {c.value for c in ValidatorCategory}
    assert expected <= actual


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

def test_finding_is_immutable():
    f = Finding(
        validator="x", category=ValidatorCategory.AREA,
        severity=Severity.ERROR, message="m",
    )
    with pytest.raises(Exception):  # FrozenInstanceError is a dataclass-specific subclass
        f.severity = Severity.INFO  # type: ignore[misc]


def test_finding_equality_is_structural():
    a = Finding(validator="x", category=ValidatorCategory.AREA,
                severity=Severity.ERROR, message="m")
    b = Finding(validator="x", category=ValidatorCategory.AREA,
                severity=Severity.ERROR, message="m")
    assert a == b


def test_finding_render_one_line_includes_block_and_profile():
    f = Finding(
        validator="thermal_hotspot", category=ValidatorCategory.THERMAL,
        severity=Severity.ERROR, message="too hot",
        block="pe_int8", profile="50W",
    )
    line = f.render_one_line()
    assert "pe_int8" in line
    assert "50W" in line
    assert "ERROR" in line
    assert "thermal" in line


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

def _make_validator(name="test", category=ValidatorCategory.AREA,
                    findings=None, raises=None):
    findings = findings or []

    def check(ctx):
        if raises is not None:
            raise raises
        return list(findings)

    return make_callable_validator(name, category, check)


def test_registry_registers_validator_via_explicit_call():
    r = ValidatorRegistry()
    v = _make_validator()
    r.register(v)
    assert "test" in r
    assert len(r) == 1
    assert r.get("test") is v


def test_registry_register_class_decorator_instantiates_and_registers():
    r = ValidatorRegistry()

    @r.register_class
    class MyValidator:
        name = "decorated"
        category = ValidatorCategory.ENERGY
        def check(self, ctx):
            return []

    assert "decorated" in r
    assert r.get("decorated").category == ValidatorCategory.ENERGY


def test_registry_rejects_duplicate_name():
    r = ValidatorRegistry()
    r.register(_make_validator("dup"))
    with pytest.raises(ValueError, match="already registered"):
        r.register(_make_validator("dup"))


def test_registry_rejects_object_missing_protocol_fields():
    r = ValidatorRegistry()
    class NotAValidator:
        pass
    with pytest.raises(TypeError, match="SKUValidator protocol"):
        r.register(NotAValidator())


def test_registry_clear_and_unregister():
    r = ValidatorRegistry()
    r.register(_make_validator("a"))
    r.register(_make_validator("b"))
    assert len(r) == 2
    r.unregister("a")
    assert "a" not in r and "b" in r
    r.clear()
    assert len(r) == 0


# ---------------------------------------------------------------------------
# Registry execution
# ---------------------------------------------------------------------------

@pytest.fixture
def real_ctx():
    """Build a context against the real catalog. Skipped if catalog not
    reachable (e.g., embodied-schemas not installed in this environment)."""
    try:
        return build_context_for_kpu("stillwater_kpu_t256")
    except Exception as exc:
        pytest.skip(f"catalog unreachable: {exc}")
        # pytest.skip raises Skipped; the explicit raise below makes the
        # control-flow contract clear to static analyzers (no implicit
        # None fall-through after the try/except).
        raise AssertionError("unreachable")  # pragma: no cover


def test_run_all_returns_empty_when_registry_empty(real_ctx):
    r = ValidatorRegistry()
    assert r.run_all(real_ctx) == []


def test_run_all_collects_findings_from_every_validator(real_ctx):
    r = ValidatorRegistry()
    f1 = Finding(validator="a", category=ValidatorCategory.AREA,
                 severity=Severity.WARNING, message="warn")
    f2 = Finding(validator="b", category=ValidatorCategory.ENERGY,
                 severity=Severity.ERROR, message="err")
    r.register(_make_validator("a", ValidatorCategory.AREA, [f1]))
    r.register(_make_validator("b", ValidatorCategory.ENERGY, [f2]))
    findings = r.run_all(real_ctx)
    assert {f.validator for f in findings} == {"a", "b"}


def test_run_all_sorts_by_severity_descending(real_ctx):
    r = ValidatorRegistry()
    info = Finding(validator="i", category=ValidatorCategory.AREA,
                   severity=Severity.INFO, message="info")
    err = Finding(validator="e", category=ValidatorCategory.AREA,
                  severity=Severity.ERROR, message="err")
    r.register(_make_validator("i", ValidatorCategory.AREA, [info]))
    r.register(_make_validator("e", ValidatorCategory.AREA, [err]))
    findings = r.run_all(real_ctx)
    assert findings[0].severity == Severity.ERROR
    assert findings[-1].severity == Severity.INFO


def test_run_category_filters_to_one_category(real_ctx):
    r = ValidatorRegistry()
    f1 = Finding(validator="a", category=ValidatorCategory.AREA,
                 severity=Severity.INFO, message="m")
    f2 = Finding(validator="b", category=ValidatorCategory.ENERGY,
                 severity=Severity.INFO, message="m")
    r.register(_make_validator("a", ValidatorCategory.AREA, [f1]))
    r.register(_make_validator("b", ValidatorCategory.ENERGY, [f2]))
    out = r.run_category(real_ctx, ValidatorCategory.AREA)
    assert len(out) == 1 and out[0].validator == "a"


def test_run_one_executes_named_validator(real_ctx):
    r = ValidatorRegistry()
    f = Finding(validator="solo", category=ValidatorCategory.INTERNAL,
                severity=Severity.WARNING, message="m")
    r.register(_make_validator("solo", ValidatorCategory.INTERNAL, [f]))
    out = r.run_one("solo", real_ctx)
    assert len(out) == 1


def test_run_one_raises_for_unknown_validator(real_ctx):
    r = ValidatorRegistry()
    with pytest.raises(KeyError, match="not registered"):
        r.run_one("nonexistent", real_ctx)


def test_validator_exception_becomes_error_finding(real_ctx):
    """A crashing validator must not abort the whole run -- the framework
    converts the exception into an ERROR Finding so the rest of the
    validators still execute."""
    r = ValidatorRegistry()
    r.register(_make_validator("crashes", raises=RuntimeError("kaboom")))
    r.register(_make_validator("ok"))
    findings = r.run_all(real_ctx)
    crash_findings = [f for f in findings if f.validator == "crashes"]
    assert len(crash_findings) == 1
    assert crash_findings[0].severity == Severity.ERROR
    assert "kaboom" in crash_findings[0].message


def test_categories_returns_only_categories_with_validators():
    r = ValidatorRegistry()
    r.register(_make_validator("a", ValidatorCategory.AREA))
    r.register(_make_validator("b", ValidatorCategory.THERMAL))
    cats = set(r.categories())
    assert cats == {ValidatorCategory.AREA, ValidatorCategory.THERMAL}


# ---------------------------------------------------------------------------
# Filters / convenience
# ---------------------------------------------------------------------------

def test_filter_findings_by_severity_floor():
    findings = [
        Finding(validator="a", category=ValidatorCategory.AREA,
                severity=Severity.INFO, message="m"),
        Finding(validator="b", category=ValidatorCategory.AREA,
                severity=Severity.WARNING, message="m"),
        Finding(validator="c", category=ValidatorCategory.AREA,
                severity=Severity.ERROR, message="m"),
    ]
    assert len(filter_findings(findings, min_severity=Severity.INFO)) == 3
    assert len(filter_findings(findings, min_severity=Severity.WARNING)) == 2
    assert len(filter_findings(findings, min_severity=Severity.ERROR)) == 1


def test_filter_findings_by_category():
    findings = [
        Finding(validator="a", category=ValidatorCategory.AREA,
                severity=Severity.INFO, message="m"),
        Finding(validator="b", category=ValidatorCategory.ENERGY,
                severity=Severity.INFO, message="m"),
    ]
    out = filter_findings(findings, category=ValidatorCategory.AREA)
    assert len(out) == 1 and out[0].category == ValidatorCategory.AREA


def test_has_errors_detects_errors():
    findings = [
        Finding(validator="a", category=ValidatorCategory.AREA,
                severity=Severity.WARNING, message="m"),
    ]
    assert not has_errors(findings)
    findings.append(
        Finding(validator="b", category=ValidatorCategory.AREA,
                severity=Severity.ERROR, message="m")
    )
    assert has_errors(findings)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def test_build_context_for_existing_kpu():
    ctx = build_context_for_kpu("stillwater_kpu_t256")
    assert ctx.sku.id == "stillwater_kpu_t256"
    assert ctx.process_node.id == "tsmc_n16"
    assert "active_fan" in ctx.cooling_solutions


def test_build_context_unknown_sku_raises():
    with pytest.raises(ContextError, match="no KPU SKU with id"):
        build_context_for_kpu("not_a_real_sku")


def test_build_context_passes_in_memory_catalogs():
    """Tests can short-circuit disk IO by passing pre-built dicts."""
    real = build_context_for_kpu("stillwater_kpu_t256")
    ctx = build_context_for_kpu(
        "stillwater_kpu_t256",
        kpus={real.sku.id: real.sku},
        process_nodes={real.process_node.id: real.process_node},
        cooling_solutions=real.cooling_solutions,
    )
    assert ctx.sku is real.sku
    assert ctx.process_node is real.process_node


def test_context_cooling_for_resolves_existing_profile():
    ctx = build_context_for_kpu("stillwater_kpu_t256")
    cs = ctx.cooling_for("15W")
    assert cs is not None and cs.id == "passive_heatsink_large"


def test_context_cooling_for_returns_none_for_unknown_profile():
    ctx = build_context_for_kpu("stillwater_kpu_t256")
    assert ctx.cooling_for("not_a_profile") is None


def test_context_unresolved_process_node_raises():
    """Hand-build a SKU that points at a non-existent process node and
    confirm build_context surfaces it as ContextError. Uses an
    in-memory catalog override so we don't need to author a broken YAML."""
    real = build_context_for_kpu("stillwater_kpu_t256")
    broken_sku = real.sku.model_copy(update={"process_node_id": "nonexistent_node"})
    with pytest.raises(ContextError, match="does not resolve"):
        build_context_for_kpu(
            broken_sku.id,
            kpus={broken_sku.id: broken_sku},
            process_nodes={},  # empty -> any reference fails
            cooling_solutions={},
        )
