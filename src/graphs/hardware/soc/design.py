"""SoC designs: a composition of IP blocks (graphs#269 PR 2.1).

A design names IP templates and how many of each, plus the layout overheads
that are not any one block's silicon. It is node-independent: the node is
applied by ``compose_soc`` (PR 2.2), so one design can be priced at 8 nm, 7 nm
and 5 nm and the three compared.

Shared blocks -- the system cache, the NoC, memory controllers and PHYs, IO --
are IP blocks like any other, with counts. That keeps one area path for
everything, and makes "how many LPDDR5 channels" a line in the design rather
than a special field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .ip_block import IPBlockTemplate


class DesignBlock(BaseModel):
    """One placement of an IP template in a design."""

    instance: str = Field(..., pattern=r"^[a-z0-9_]+$")
    ip: str
    count: int = Field(1, gt=0)
    #: An explicit clock always wins over the template's retargeted fmax.
    clock_ghz: Optional[float] = Field(None, gt=0)
    notes: str = ""

    model_config = {"extra": "forbid"}


class Layout(BaseModel):
    """Area that belongs to no block.

    ``whitespace_fraction`` covers placement and routing slack between blocks;
    ``io_ring_mm`` is the pad ring's depth around the die.
    """

    whitespace_fraction: float = Field(0.15, ge=0, lt=1)
    io_ring_mm: float = Field(0.30, ge=0)
    source: str = ""

    model_config = {"extra": "forbid"}


class Reference(BaseModel):
    """Published figures a design reconstructs, for acceptance checks."""

    die_area_mm2: Optional[float] = Field(None, gt=0)
    transistors_billion: Optional[float] = Field(None, gt=0)
    process_node: Optional[str] = None
    peak_tops: Dict[str, float] = Field(default_factory=dict)
    source: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}


class SoCDesign(BaseModel):
    """A node-independent SoC composition."""

    id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    name: str
    process_node: str = Field(..., description="Default node; compose_soc can override")
    blocks: List[DesignBlock] = Field(..., min_length=1)
    layout: Layout = Field(default_factory=Layout)
    reference: Optional[Reference] = None
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _unique_instances(self) -> "SoCDesign":
        names = [b.instance for b in self.blocks]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"design {self.id!r}: duplicate block instances {dup}")
        return self

    def check_against(self, library: Mapping[str, IPBlockTemplate]) -> None:
        """Every block must name an IP in the library."""
        missing = sorted({b.ip for b in self.blocks if b.ip not in library})
        if missing:
            raise KeyError(f"design {self.id!r}: unknown IP {missing}")


DEFAULT_DESIGN_DIR = Path(__file__).resolve().parents[4] / "soc_designs" / "designs"


def load_design(path: Path) -> SoCDesign:
    return SoCDesign.model_validate(yaml.safe_load(Path(path).read_text()))


def load_designs(path: Optional[Path] = None) -> Dict[str, SoCDesign]:
    """Every ``*.yaml`` design under ``path``, keyed by id."""
    root = Path(path) if path else DEFAULT_DESIGN_DIR
    designs: Dict[str, SoCDesign] = {}
    for file in sorted(root.glob("*.yaml")):
        design = load_design(file)
        if design.id != file.stem:
            raise ValueError(f"{file.name}: id {design.id!r} must match the file name")
        designs[design.id] = design
    return designs
