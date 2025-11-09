"""
IP Core Hardware Models

This module contains resource models for licensable IP cores that can be
integrated into custom SoCs. These are distinct from standalone accelerators
in that they are soft IP that gets synthesized into the customer's chip design.

Categories:
- Vision/DSP IP Cores: Cadence Vision Q8, Synopsys ARC EV7x
- Neural Processing IP: CEVA NeuPro NPM11

These IP cores are licensed to SoC vendors and integrated into their designs,
rather than being sold as discrete chips.
"""

from .cadence_vision_q8 import cadence_vision_q8_resource_model
from .ceva_neupro_npm11 import ceva_neupro_npm11_resource_model
from .synopsys_arc_ev7x import synopsys_arc_ev7x_resource_model

__all__ = [
    'cadence_vision_q8_resource_model',
    'ceva_neupro_npm11_resource_model',
    'synopsys_arc_ev7x_resource_model',
]
