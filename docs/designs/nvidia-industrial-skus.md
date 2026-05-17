# NVIDIA Industrial Modules

● Verified table
     
  SKU: Jetson TX2i (900-83489-0000-000)
  GPU architecture: Pascal — 256-core, Tegra X2 / Parker
  GA date: Apr 2018 (announced Mar 2018, distributor support Apr 24 2018)
  EOL date: Last Time Buy: Jul 15 2026; Last Time Ship: Jul 15 2027 (pulled in from 2028 due to LPDDR4 EOL)                                                    
  Reference URL: NVIDIA Dev Forum PCN (https://forums.developer.nvidia.com/t/jetson-product-eol-updates/370052)
  ────────────────────────────────────────
  SKU: Jetson AGX Xavier Industrial (900-82888-0070-000)
  GPU architecture: Volta — 512-core, 64 Tensor Cores
  GA date: Jul 2021 ($1,449; announced Jun 15 2021)
  EOL date: Last Time Buy: Jul 15 2026; Last Time Ship: Jul 15 2027 (pulled in from 2031 due to LPDDR4 supply)                                                 
  Reference URL: NVIDIA Blog launch announcement (https://blogs.nvidia.com/blog/jetson-agx-xavier-industrial-use-ai/)
  ────────────────────────────────────────
  SKU: Jetson AGX Orin Industrial (900-13701-0080-000)
  GPU architecture: Ampere — 2048-core, 64 Tensor Cores
  GA date: Mid-2023 (announced COMPUTEX 2023, May 29 2023)
  EOL date: Active; lifecycle published through Jul 2033 (no LTB announced)                                                                                    
  Reference URL: NVIDIA Technical Blog (https://developer.nvidia.com/blog/step-into-the-future-of-industrial-grade-edge-ai-with-nvidia-jetson-agx-orin-industrial)
  ────────────────────────────────────────
  SKU: (Thor industrial)
  GPU architecture: Blackwell — 2,560 CUDA + 96 Gen-5 Tensor Cores
  GA date: No "Jetson AGX Thor Industrial" SKU exists yet. What ships: (a) Jetson T5000 commercial production module GA Aug 25, 2025 ($2,999); (b) IGX T5000 SoM + IGX T7000
  board
     kit (industrial / medical-grade) announced GTC DC Oct 2025, GA Dec 2025. DRIVE AGX Thor (automotive) is dev-kit-only — no automotive production module shipping.
  EOL date: Active (no EOL); Jetson T5000 lifecycle through Aug 2035
  Reference URL: NVIDIA Blog: IGX Thor announcement (https://blogs.nvidia.com/blog/igx-thor-processor-physical-ai-industrial-medical-edge/)

  Caveats worth carrying into the slide

  1. Thor industrial branding is split. NVIDIA does not yet ship a SKU literally named "Jetson AGX Thor Industrial." The industrial-grade Blackwell-Thor platform is branded IGX 
  Thor — separate enterprise/medical/industrial line, distinct from the Jetson robotics line. The commercial Jetson T5000 (Aug 2025) is the Thor production module but is not
  industrial-grade (no extended-temp / ECC / functional-safety guarantees of IGX). For the industrial roadmap, the closest equivalent is IGX T5000 (Dec 2025).
  2. TX2i / Xavier Industrial EOLs were both accelerated in an April 2026 PCN due to LPDDR4 manufacturer EOL. The dates above reflect the revised schedule; NVIDIA's lifecycle
  page may still show stale "available through 2031" entries — treat the forum PCN as authoritative.
  3. Architecture lineage is what we want for "intelligence per Watt vs process+architecture":
    - TX2i: Pascal, TSMC 16nm FinFET
    - AGX Xavier Industrial: Volta, TSMC 12nm FFN
    - AGX Orin Industrial: Ampere, Samsung 8nm
    - IGX T5000 / Jetson T5000: Blackwell, TSMC 4NP
