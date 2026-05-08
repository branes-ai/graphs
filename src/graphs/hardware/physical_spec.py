"""
PhysicalSpec: chip-level physical / fabrication attributes.

Sibling to HardwareResourceModel. Holds the spec-sheet view of a compute
product -- die size, transistor budget, process node, foundry, launch info --
that is NOT consumed by the mapping path (roofline / energy / scheduling
estimators read HardwareResourceModel, not this).

This separation keeps the mapper hot path uncluttered while still letting
spec-sheet tools (e.g., cli/list_hardware_resources.py) report the full
chip-level view.

Source-of-truth roadmap: today these values are populated inline in factory
functions, sourced from datasheets. When the embodied-schemas ComputeProduct
unification (RFC 0001) lands, PhysicalSpec will become a thin loader that
reads from a ComputeProduct YAML, and consumers won't change.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


@dataclass
class PhysicalSpec:
    """Physical / fabrication attributes of a compute product.

    All fields are optional. Many will be unknown for research-stage chips
    (e.g., proprietary KPUs, hypothetical CGRAs); leave those as None and
    let downstream tools render "N/A".

    Multi-die products (chiplets, MCMs) report SUM values for ``die_size_mm2``
    and ``transistors_billion`` across all dies in the package, with
    ``num_dies`` and ``is_chiplet`` describing the packaging.
    """

    # Fabrication
    die_size_mm2: Optional[float] = None
    """Total silicon area in mm^2 (sum across all dies in the package)."""

    transistors_billion: Optional[float] = None
    """Total transistor count in billions (sum across all dies)."""

    process_node_nm: Optional[int] = None
    """Process node in nanometers. May be a marketing-rounded value
    (e.g., 4 for TSMC N4 or Intel 4); see ``process_node_name`` for the
    exact node string."""

    process_node_name: Optional[str] = None
    """Vendor name for the process node, e.g. ``TSMC N4``, ``Intel 7``,
    ``Samsung 5LPE``."""

    foundry: Optional[str] = None
    """Foundry that fabricated the silicon, e.g. ``tsmc``, ``samsung``,
    ``intel``, ``globalfoundries``."""

    # Architecture / generation
    architecture: Optional[str] = None
    """Microarchitecture or codename, e.g. ``Hopper``, ``Blackwell``,
    ``Alder Lake``, ``Zen 4``."""

    # Packaging
    num_dies: int = 1
    """Number of silicon dies in the package. >1 for chiplet / MCM /
    multi-die products (B100 = 2, MI300X = 8 + 4, etc.)."""

    is_chiplet: bool = False
    """True if the product uses chiplet packaging (multiple dies bridged
    by an interposer or organic substrate)."""

    package_type: Optional[str] = None
    """Packaging classification: ``monolithic``, ``mcm``, ``chiplet``,
    ``board``, ``system``."""

    # Market
    launch_date: Optional[str] = None
    """ISO date string of public launch, e.g. ``2022-09-20``."""

    launch_msrp_usd: Optional[float] = None
    """Launch MSRP in USD. None for products without public list pricing."""

    # Provenance
    source: Optional[str] = None
    """Where the values came from, e.g. ``embodied-schemas:nvidia/h100_sxm5_80gb_hbm3.yaml``,
    ``vendor datasheet 2023-04``, ``Wikichip``. Use a short citation that
    survives schema changes."""

    extras: Dict[str, Any] = field(default_factory=dict)
    """Vendor-specific or category-specific fields that don't fit the spine.
    Keep typed values when possible. Examples: ``{"chiplet_count": 4,
    "interposer_type": "CoWoS"}``."""

    @property
    def transistor_density_mtx_mm2(self) -> Optional[float]:
        """Transistor density in millions of transistors per mm^2.

        Returns None if either die size or transistor count is unknown.
        Useful for cross-process comparisons (e.g., 5nm vs 3nm density).
        """
        if self.die_size_mm2 is None or self.transistors_billion is None:
            return None
        # Treat non-positive values as invalid: a real chip has die_size > 0
        # and transistors > 0. Guard so a malformed PhysicalSpec doesn't
        # produce a nonsensical negative density.
        if self.die_size_mm2 <= 0 or self.transistors_billion <= 0:
            return None
        return (self.transistors_billion * 1000.0) / self.die_size_mm2

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (e.g., for JSON output)."""
        return asdict(self)
