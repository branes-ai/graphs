"""SoC composition: node-independent IP blocks assembled into a die (graphs#269).

``ip_block`` defines the templates, ``design`` the compositions. The node is
applied later, by ``compose_soc``, so one design prices at any node.
"""

from .compose import BlockInstance, LineArea, SoCInstance, compose_soc
from .design import DesignBlock, Layout, Reference, SoCDesign, load_design, load_designs
from .ip_block import (
    Confidence,
    EngineKind,
    IPBlockTemplate,
    IPClock,
    IPCompute,
    IPShoreline,
    IPSilicon,
    load_ip_library,
)

__all__ = [
    "BlockInstance",
    "LineArea",
    "SoCInstance",
    "compose_soc",
    "Confidence",
    "DesignBlock",
    "EngineKind",
    "IPBlockTemplate",
    "IPClock",
    "IPCompute",
    "IPShoreline",
    "IPSilicon",
    "Layout",
    "Reference",
    "SoCDesign",
    "load_design",
    "load_designs",
    "load_ip_library",
]
