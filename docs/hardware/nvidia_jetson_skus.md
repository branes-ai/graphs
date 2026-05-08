● PhysicalSpec for all 4 Jetson SKUs landed
  PR: #135 (draft) (https://github.com/branes-ai/graphs/pull/135) — feat(hardware): populate PhysicalSpec for 4 Jetson SKUs (relates #130)
  Branch: feat/issue-130-jetson-physical-specs (off main)
  Status: 18 PhysicalSpec tests pass, 205 hardware tests pass, no regressions

  Data captured
  Orin AGX 64GB / NX 16GB / Nano 8GB — all three share the GA10B (T234) silicon; module bins differ but die does not. 
  Single helper _orin_soc_physical_spec() 
  builds the shared chip-level fields:
  ┌─────────────────────────────────────┬──────────────────────────────┐
  │ Chip-level (identical across all 3) │            Value             │
  ├─────────────────────────────────────┼──────────────────────────────┤
  │ die_size_mm²                        │ 455.0                        │
  ├─────────────────────────────────────┼──────────────────────────────┤
  │ transistors (B)                     │ 17.0                         │
  ├─────────────────────────────────────┼──────────────────────────────┤
  │ process                             │ Samsung 8N (8LPP derivative) │
  ├─────────────────────────────────────┼──────────────────────────────┤
  │ foundry                             │ samsung                      │
  ├─────────────────────────────────────┼──────────────────────────────┤
  │ architecture                        │ Ampere                       │
  ├─────────────────────────────────────┼──────────────────────────────┤
  │ density                             │ 37.4 Mtx/mm² (derived)       │
  └─────────────────────────────────────┴──────────────────────────────┘                                                                                      
  
  ┌────────────────────────┬────────────────────────────┬────────────────┐
  │ Module-level (per SKU) │           Launch           │      MSRP      │
  ├────────────────────────┼────────────────────────────┼────────────────┤
  │ AGX 64GB               │ 2022-11-08 (GTC Fall 2022) │ $1,999         │
  ├────────────────────────┼────────────────────────────┼────────────────┤
  │ NX 16GB                │ 2023-01-04                 │ $899           │
  ├────────────────────────┼────────────────────────────┼────────────────┤
  │ Nano 8GB               │ 2023-03-22 (GTC 2023)      │ $499 (dev kit) │
  └────────────────────────┴────────────────────────────┴────────────────┘
                                                            
  Thor 128GB — different generation (Blackwell on TSMC 4N), launched 2025-08-25 at $2,999. 
  die_size_mm² and transistors_billion are None — NVIDIA hasn't publicly disclosed those for Thor; 
  the spec-sheet view will render them as N/A (graceful degradation by design).

  Caveats noted in commit message and PR body
 
  - The 455 mm² figure is NVIDIA's commonly-cited number; an X die-shot photo measurement reported 448 mm². 
    NVIDIA has not officially published an exact die area. The discrepancy is small (~1.5%) and the 
    17B transistor count is solid (NVIDIA-disclosed at Hot Chips).
  - The Orin Nano 8GB MSRP of $499 reflects the dev kit launch price; the production module sold for ~$199 later. Source field documents this.

  Open PRs in flight
  - #133 (https://github.com/branes-ai/graphs/pull/133) — merged earlier (PhysicalSpec foundation)
  - #134 (https://github.com/branes-ai/graphs/pull/134) — cli/list_hardware_resources.py (the spec-sheet CLI; in draft)
  - #135 (https://github.com/branes-ai/graphs/pull/135) — this one (Jetson backfill; in draft)

  Once #135 and #134 both land, running python cli/list_hardware_resources.py --category gpu --sort die_size --reverse will show H100 first, 
  then Orin AGX/NX/Nano with their populated GA10B specs, then Thor (process node populated, die size N/A).

  Sources
  - NVIDIA Jetson AGX Orin Series Technical Brief
    (https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)
  - Wikipedia: NVIDIA Tegra (Orin / Thor) (https://en.wikipedia.org/wiki/Tegra)
  - NVIDIA's Next-Gen GPU Architecture Powers The DRIVE AGX Orin (WCCFTech)
    (https://wccftech.com/nvidia-next-gen-gpu-architecture-powered-orin-soc-announced/) — 17B transistor announcement
  - GA10B die size measurement (X / @lafaiel) (https://x.com/lafaiel/status/1869311677000458418) — die-shot reporting 448 mm²
  - Introducing NVIDIA Jetson Thor (NVIDIA Technical Blog)
    (https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/) — Thor T5000 specs
  - NVIDIA Blackwell-Powered Jetson Thor Now Available (NVIDIA Newsroom)
    (https://nvidianews.nvidia.com/news/nvidia-blackwell-powered-jetson-thor-now-available-accelerating-the-age-of-general-robotics) 
    — 2025-08-25 launch + $2,999 module pricing                                     
  - $3499 NVIDIA Jetson AGX Thor Developer Kit (CNX Software)
    (https://www.cnx-software.com/2025/08/19/3499-nvidia-jetson-agx-thor-developer-kit-2070-tops-jetson-t5000-som-for-robotics-and-edge-ai/) — pricing detail 
