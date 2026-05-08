"""
PhysicalSpec YAML loader (issue #136 Phase 3.5).

Reads PhysicalSpec data from the embodied-schemas hardware database
(``embodied-schemas/data/gpus/<vendor>/<id>.yaml``) instead of carrying
copies of the values as Python literals in factory functions. Replaces
the Phase 3 inline backfill (#139) with a single source of truth.

Design:

- ``load_physical_spec(base_id)`` -- read YAML, return PhysicalSpec.
- ``validate_physical_spec(spec, peak_bandwidth_gbps, dram_rate_gtps)``
  -- bandwidth-math sanity check that catches the bug class we just hit
  in #139 (where the YAMLs themselves had inconsistent values).
- ``KNOWN_OVERRIDES`` -- field-level corrections for documented YAML
  bugs. Each entry MUST cite the upstream issue and the math/source
  that justifies the override. Removed when upstream fix lands.

Path resolution (in priority order):

1. ``EMBODIED_SCHEMAS_DATA_DIR`` env var -- explicit override, highest
   priority. Useful for CI / testing with a frozen snapshot.
2. Installed ``embodied_schemas`` package -- ``embodied_schemas.loaders
   .get_data_dir()`` if importable. The recommended deployment path.
3. Sibling clone -- heuristic walk up from this file looking for
   ``../embodied-schemas/src/embodied_schemas/data/``. Works for local
   development with both repos checked out alongside each other.

If none resolve, ``load_physical_spec`` raises ``FileNotFoundError``
with guidance on how to make the data reachable.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from graphs.hardware.physical_spec import PhysicalSpec


# ---------------------------------------------------------------------------
# Known overrides for upstream YAML bugs
# ---------------------------------------------------------------------------

# Field-level corrections applied AFTER the YAML load. Each entry MUST
# cite the upstream issue and the verification source. Remove the entry
# when the upstream fix lands.
KNOWN_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # branes-ai/embodied-schemas#8: YAML lists 512-bit, but bandwidth math
    # (273 GB/s / 8.533 GT/s LPDDR5X = 256 bits) and NVIDIA's Jetson Thor
    # announcement blog ("256-bit LPDDR5X, 273 GB/s") both confirm 256.
    "nvidia_thor_gpu_128gb_lpddr5x": {
        "memory_bus_width_bits": 256,
    },
    # branes-ai/embodied-schemas#8: YAML lists 64-bit, but bandwidth math
    # (68 GB/s / 4.267 GT/s LPDDR5-4267 = 128 bits) and NVIDIA's Orin
    # Nano datasheet both confirm 128.
    "nvidia_orin_nano_gpu_8gb_lpddr5": {
        "memory_bus_width_bits": 128,
    },
}


# ---------------------------------------------------------------------------
# Foundry display map (for process_node_name composition)
# ---------------------------------------------------------------------------

# YAML stores ``process_name`` as just the node label (e.g., "N4", "8LPP").
# We compose ``"{Foundry} {process_name}"`` so the rendered value reads
# naturally on the spec sheet ("TSMC N4", "Samsung 8LPP"). Maps the
# lowercase YAML foundry value to display-cased name. Unknown foundries
# fall through to .title() casing.
_FOUNDRY_DISPLAY = {
    "tsmc": "TSMC",
    "samsung": "Samsung",
    "intel": "Intel",
    "globalfoundries": "GlobalFoundries",
    "smic": "SMIC",
}


def _format_process_node_name(foundry: Optional[str], process_name: Optional[str]) -> Optional[str]:
    """Compose a human-readable process node name from foundry + node label.

    Returns None if both inputs are absent. Otherwise composes
    ``"{Foundry} {process_name}"`` using a display-case map for known
    foundries (e.g., ``tsmc`` -> ``TSMC``, ``samsung`` -> ``Samsung``).
    Unknown foundries fall through to title case.
    """
    if not foundry and not process_name:
        return None
    foundry_display = _FOUNDRY_DISPLAY.get((foundry or "").lower(), (foundry or "").title())
    if not process_name:
        return foundry_display or None
    if not foundry_display:
        return str(process_name)
    return f"{foundry_display} {process_name}"


# ---------------------------------------------------------------------------
# Data directory resolution
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> Path:
    """Find the embodied-schemas data directory.

    Resolution order documented in the module docstring. Raises
    ``FileNotFoundError`` if no candidate path exists.
    """
    # 1. Env var override.
    env_path = os.environ.get("EMBODIED_SCHEMAS_DATA_DIR")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"EMBODIED_SCHEMAS_DATA_DIR points to {env_path!r} but the path "
            f"does not exist."
        )

    # 2. Installed embodied_schemas package.
    try:
        from embodied_schemas.loaders import get_data_dir as _es_get_data_dir
        candidate = _es_get_data_dir()
        if candidate.exists():
            return candidate
    except ImportError:
        pass

    # 3. Sibling-clone heuristic. Walk up from this file looking for a
    # peer ``embodied-schemas`` repo checkout.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "embodied-schemas" / "src" / "embodied_schemas" / "data"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the embodied-schemas data directory. Set the "
        "EMBODIED_SCHEMAS_DATA_DIR environment variable, install the "
        "embodied-schemas package (e.g., `pip install -e .[schemas]`), "
        "or check out the embodied-schemas repo as a sibling of graphs/."
    )


# ---------------------------------------------------------------------------
# YAML lookup
# ---------------------------------------------------------------------------

def _yaml_path_for(base_id: str, *, vendor: Optional[str] = None) -> Path:
    """Locate the YAML file for ``base_id`` within the data directory.

    Searches by vendor sub-directory if ``vendor`` is given, else scans
    every gpu/cpu/npu/chip vendor directory until it finds a file whose
    ``id:`` field matches ``base_id``. Caches nothing -- this is meant
    to be called O(N) times per process at factory-import time.
    """
    data_dir = _resolve_data_dir()
    # The known top-level categories that hold individual chip YAMLs.
    categories = ("gpus", "cpus", "npus", "chips")
    if vendor:
        # Direct lookup -- caller knows category + vendor.
        for category in categories:
            candidate_dir = data_dir / category / vendor.lower()
            if not candidate_dir.is_dir():
                continue
            for yaml_path in candidate_dir.glob("*.yaml"):
                doc = yaml.safe_load(yaml_path.read_text())
                if doc and doc.get("id") == base_id:
                    return yaml_path
    # Broad scan.
    for category in categories:
        cat_dir = data_dir / category
        if not cat_dir.is_dir():
            continue
        for yaml_path in cat_dir.rglob("*.yaml"):
            doc = yaml.safe_load(yaml_path.read_text())
            if doc and doc.get("id") == base_id:
                return yaml_path
    raise FileNotFoundError(
        f"No YAML found for base_id={base_id!r} under {data_dir}."
    )


# ---------------------------------------------------------------------------
# YAML -> PhysicalSpec mapping
# ---------------------------------------------------------------------------

def _physical_spec_from_yaml_dict(doc: Dict[str, Any], *, source: str) -> PhysicalSpec:
    """Project a parsed YAML dict onto a PhysicalSpec instance.

    Only fields that PhysicalSpec defines are surfaced. Other YAML
    blocks (compute, clocks, performance, features) are ignored here --
    they belong to other consumers (resource_model, etc.).
    """
    die = doc.get("die") or {}
    memory = doc.get("memory") or {}
    market = doc.get("market") or {}

    # Process node: prefer the typed ``process_nm`` (int) for nm, and
    # compose the human-readable name from foundry + ``process_name``.
    process_node_nm = die.get("process_nm")
    foundry = die.get("foundry")
    process_node_name = _format_process_node_name(foundry, die.get("process_name"))

    # Packaging: derive package_type from ``is_chiplet`` / ``num_dies``
    # since the YAML doesn't have an explicit field.
    is_chiplet = bool(die.get("is_chiplet", False))
    num_dies = int(die.get("num_dies", 1) or 1)
    package_type = "chiplet" if is_chiplet else "monolithic"

    return PhysicalSpec(
        die_size_mm2=die.get("die_size_mm2"),
        transistors_billion=die.get("transistors_billion"),
        process_node_nm=process_node_nm,
        process_node_name=process_node_name,
        foundry=foundry,
        architecture=die.get("architecture"),
        num_dies=num_dies,
        is_chiplet=is_chiplet,
        package_type=package_type,
        memory_type=memory.get("memory_type"),
        memory_bus_width_bits=memory.get("memory_bus_bits"),
        launch_date=market.get("launch_date"),
        launch_msrp_usd=market.get("launch_msrp_usd"),
        source=source,
    )


def _apply_overrides(spec: PhysicalSpec, base_id: str) -> PhysicalSpec:
    """Apply documented field-level overrides for known YAML bugs.

    Mutates ``spec`` in place and returns it for chaining.
    """
    overrides = KNOWN_OVERRIDES.get(base_id, {})
    for field_name, value in overrides.items():
        setattr(spec, field_name, value)
    return spec


def load_physical_spec(base_id: str, *, vendor: Optional[str] = None) -> PhysicalSpec:
    """Load a PhysicalSpec for ``base_id`` from the embodied-schemas data.

    Args:
        base_id: The chip's ``id:`` field value as it appears in the
            YAML (e.g., ``"nvidia_h100_sxm5_80gb_hbm3"``).
        vendor: Optional vendor sub-directory hint (e.g., ``"nvidia"``)
            to short-circuit the broad directory scan.

    Returns:
        A PhysicalSpec instance with KNOWN_OVERRIDES applied. The
        ``source`` field cites the YAML path so the provenance is
        traceable.

    Raises:
        FileNotFoundError: If the embodied-schemas data directory or
            the requested YAML cannot be located.
    """
    yaml_path = _yaml_path_for(base_id, vendor=vendor)
    doc = yaml.safe_load(yaml_path.read_text())
    # Make the source citation point to the in-repo path under data/
    # so it's stable across host machines.
    try:
        rel = yaml_path.relative_to(_resolve_data_dir())
        source = f"embodied-schemas:{rel.as_posix()}"
    except ValueError:
        source = f"embodied-schemas:{yaml_path.name}"
    spec = _physical_spec_from_yaml_dict(doc, source=source)
    return _apply_overrides(spec, base_id)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

# Tolerance for bandwidth math in percent. LPDDR5 / HBM3 figures often
# round in vendor specs (e.g., LPDDR5-4267 is really 4266.66... MT/s),
# and bandwidth values in datasheets typically round to 1-3 significant
# figures, so a tight 1% tolerance would generate noise. 5% catches
# real bugs (the embodied-schemas#8 errors are 100% / 200% off) without
# false-positives on rounding artifacts.
_BANDWIDTH_TOLERANCE = 0.05


def validate_physical_spec(
    spec: PhysicalSpec,
    *,
    peak_bandwidth_gbps: Optional[float] = None,
    dram_rate_gtps: Optional[float] = None,
    base_id: Optional[str] = None,
) -> List[str]:
    """Run sanity checks on a loaded PhysicalSpec.

    Returns a list of human-readable warning strings (empty list = ok).
    Designed for "warn loudly, never crash" semantics: callers decide
    whether to raise or log.

    Checks performed:

    1. **Bandwidth math**: when both ``peak_bandwidth_gbps`` and
       ``dram_rate_gtps`` are supplied alongside the spec's
       ``memory_bus_width_bits``, asserts
       ``bandwidth = (bus_width / 8) * dram_rate`` within a 5%
       tolerance. This catches the bug class hit by embodied-schemas#8
       (Thor's 512-bit / 273 GB/s claim implied an impossible DRAM
       rate; Nano's 64-bit / 68 GB/s likewise).
    2. **Density math**: confirms transistor density is in a sane range
       given the process node, when both die_size and transistors are
       set. Doesn't fail on small deviations -- vendors don't always
       publish exact die sizes.
    """
    warnings: List[str] = []
    label = base_id or "<unknown id>"

    # 1. Bandwidth math.
    if (
        spec.memory_bus_width_bits is not None
        and peak_bandwidth_gbps is not None
        and dram_rate_gtps is not None
        and spec.memory_bus_width_bits > 0
        and dram_rate_gtps > 0
    ):
        expected_bw = (spec.memory_bus_width_bits / 8.0) * dram_rate_gtps
        if peak_bandwidth_gbps > 0:
            rel_err = abs(expected_bw - peak_bandwidth_gbps) / peak_bandwidth_gbps
            if rel_err > _BANDWIDTH_TOLERANCE:
                warnings.append(
                    f"{label}: memory bandwidth math inconsistent. "
                    f"bus_width_bits={spec.memory_bus_width_bits}, "
                    f"dram_rate_gtps={dram_rate_gtps:.3f} -> implies "
                    f"{expected_bw:.1f} GB/s, but peak_bandwidth_gbps="
                    f"{peak_bandwidth_gbps:.1f}. Relative error "
                    f"{rel_err:.1%} exceeds {_BANDWIDTH_TOLERANCE:.0%} "
                    f"tolerance. One of bus_width / dram_rate / "
                    f"bandwidth is wrong."
                )

    # 2. Density math (very loose plausibility).
    density = spec.transistor_density_mtx_mm2
    if density is not None and spec.process_node_nm is not None:
        # Modern process nodes hit roughly:
        #   16/14 nm: 25-40 Mtx/mm^2
        #    7 nm:    60-100 Mtx/mm^2
        #    5 nm:    130-180 Mtx/mm^2
        #    4/3 nm:  150-220 Mtx/mm^2
        # These are rough envelopes for Logic+Cache; numbers vary
        # widely with cache fraction and IP type. We only flag really
        # implausible values (>5x out of range).
        if density < 5.0 or density > 500.0:
            warnings.append(
                f"{label}: transistor density {density:.1f} Mtx/mm^2 is "
                f"outside the plausible range [5, 500] for any modern "
                f"process node. Check die_size_mm2 ({spec.die_size_mm2}) "
                f"and transistors_billion ({spec.transistors_billion})."
            )

    return warnings
