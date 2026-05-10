"""SKU validator framework -- public API.

Exposes the framework primitives plus a module-level ``default_registry``
that individual validators self-register against. Phase 2a ships the
framework with no validators; Phase 2b populates ``validators/`` and
each module's import triggers registration.

Typical use:

    from graphs.hardware.sku_validators import (
        default_registry,
        build_context_for_kpu,
    )

    ctx = build_context_for_kpu("stillwater_kpu_t256")
    findings = default_registry.run_all(ctx)
    for f in findings:
        print(f.render_one_line())
"""

from .context import ContextError, build_context_for_kpu
from .framework import (
    Finding,
    Severity,
    SKUValidator,
    ValidatorCategory,
    ValidatorContext,
    ValidatorRegistry,
    filter_findings,
    has_errors,
    make_callable_validator,
)


# Module-level singleton. Validators in
# ``graphs.hardware.sku_validators.validators.*`` self-register against
# this on import. Tests should construct a fresh ``ValidatorRegistry()``
# rather than mutating this.
default_registry = ValidatorRegistry()


def register(validator: SKUValidator) -> SKUValidator:
    """Convenience shortcut for ``default_registry.register(validator)``."""
    return default_registry.register(validator)


def register_class(cls: type) -> type:
    """Convenience shortcut for ``default_registry.register_class(cls)``."""
    return default_registry.register_class(cls)


def load_validators() -> int:
    """Import the validators package so all validators self-register.

    Phase 2a: the validators package is empty, so this is a no-op that
    returns 0 registered validators. Phase 2b adds modules under
    ``sku_validators/validators/`` and this becomes the discovery hook.

    Returns the count of validators registered against the
    ``default_registry`` after the import.
    """
    try:
        # Importing the package triggers its __init__.py, which in turn
        # imports each individual validator module so they self-register.
        from . import validators  # noqa: F401
    except ImportError:
        # The validators sub-package is optional during Phase 2a.
        pass
    return len(default_registry)


__all__ = [
    # Core types
    "Finding",
    "Severity",
    "ValidatorCategory",
    "ValidatorContext",
    "SKUValidator",
    "ValidatorRegistry",
    # Helpers
    "filter_findings",
    "has_errors",
    "make_callable_validator",
    "build_context_for_kpu",
    "ContextError",
    # Default registry + shortcuts
    "default_registry",
    "register",
    "register_class",
    "load_validators",
]
