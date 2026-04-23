# TDP Constraints for Embedded Compute

Why is the Jetson running at 650MHz? At 8nm it should easily be able to run at 1.5GHz. 
Is this to satisfy the TDP constraints of embedded computing use cases?


— 0.65 GHz is a TDP cap, not a silicon cap. At 8 nm, Samsung's 8LPP/8LPH node (what GA10B uses) can sustain 1.3–1.5 GHz comfortably in
  products that aren't thermally boxed:

  - RTX 3050 Mobile (GA107, same 8 nm process family): 1.35 GHz boost sustained
  - RTX 3060 Mobile (GA106): 1.28–1.70 GHz depending on TGP
  - Desktop GA106: 1.78 GHz boost

Orin AGX sits at 0.65–0.92 GHz under sustained load only because 16 SMs × 1.5 GHz worth of dynamic power would draw ~24 W 
of compute alone — far past the 30 W SoC envelope once they add L2, memory controller, CPU cluster, video engines, etc. 
DVFS clamps the clock to fit the TDP.

This means my current building-block comparison is mixing two different axes:

|        Clock basis         |          SM          | KPU tile |
|----------------------------|----------------------|----------|
| Silicon capability at 8 nm | 1.5 GHz              | 1.5 GHz  |
| TDP-constrained deployment | 0.65 GHz (Orin 30 W) | ?        |

We gave the SM its deployed/TDP-constrained clock (0.65 GHz) while giving the KPU tile its silicon-capability clock (1.5 GHz). 
That's not apples-to-apples for a pure architectural comparison.

Two clean options:

  1. Silicon-capability view (both at 1.5 GHz). The SM's per-SM power goes from 666 mW → 1.54 W, and 16 SMs draws ~25 W just for compute. The SM's pJ/MAC is
  unchanged (0.25 pJ/MAC — dynamic energy per MAC is clock-independent at matched activity), but absolute TOPS triples. This is the fair "what can the
  silicon do" comparison.
  2. Deployed-product view. Keep Orin at 0.65 GHz because that's what the chip actually sustains in the 15–30 W envelope; show the KPU tile at whatever clock
   its SKU envelope dictates (which for a T128 in a 12–18 W package is also probably 1.0–1.3 GHz, not the 1.5 GHz nominal).

The silicon-capability comparison is what makes the architectural-advantage story clean; the deployed-product comparison is
what customers actually experience. I'd bump the canonical SM block to 1.5 GHz to match the KPU tile (architectural comparison) and then add a
SoC-composition row that reflects the Orin-like TDP-constrained deployment (SM @ 0.65 GHz, 16 instances) separately.

