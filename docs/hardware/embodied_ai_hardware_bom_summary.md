# Embodied AI Hardware BOM Cost Summary

**Date:** 2025-11-09
**Purpose:** Complete BOM cost analysis for market positioning and competitive comparison

---

## Overview

This document provides a comprehensive Bill of Materials (BOM) cost breakdown for all Embodied AI hardware accelerators in our comparison framework. All costs are estimated at 10K+ unit volumes for 2025 deployment.

---

## Entry-Level Edge AI ($25-$300 BOM, 1-15W)

### Stillwater KPU-T64
**Target**: Battery-powered drones, robots, edge AI devices
| Component | Cost |
|-----------|------|
| Silicon die (16nm TSMC, 64 tiles) | $75 |
| Package (flip-chip BGA) | $15 |
| Memory (2GB LPDDR4X on-package) | $20 |
| PCB assembly | $8 |
| Thermal solution (small heatsink) | $2 |
| Other (testing, connectors) | $5 |
| **Total BOM** | **$125** |
| **Retail Price** (2.4× margin) | **$300** |

**Key Metrics**:
- 16 TOPS INT8 peak, ~10 TOPS effective (60%)
- 3-10W TDP range
- Cost per TOPS: $12.50 BOM / $30 retail
- Positioning: Premium vs Coral ($25 BOM), competitive vs Qualcomm ($85 BOM), superior to Jetson Nano ($205 BOM)

---

### Hailo-8
**Target**: Edge AI cameras, drones, industrial vision, ADAS
| Component | Cost |
|-----------|------|
| Silicon die (16nm, dataflow architecture) | $25 |
| Package (BGA) | $8 |
| Memory (all on-chip SRAM, no external DRAM) | $0 |
| PCB assembly | $4 |
| Thermal solution (tiny heatsink, 2.5W) | $1 |
| Other | $2 |
| **Total BOM** | **$40** |
| **Retail Price** (4.0× margin) | **$160** |

**Key Metrics**:
- 26 TOPS INT8 peak, ~22 TOPS effective (85%)
- 2.5W TDP (best power efficiency in class: 10.4 TOPS/W)
- Cost per TOPS: $1.54 BOM / $6.15 retail
- Positioning: Best $/TOPS and TOPS/W in entry segment, dataflow architecture

---

### Google Coral Edge TPU
**Target**: IoT cameras, embedded vision, ultra-low-power edge
| Component | Cost |
|-----------|------|
| Silicon die (14nm, small TPU) | $12 |
| Package | $5 |
| Memory (uses host) | $0 |
| PCB assembly | $3 |
| Thermal solution | $1 |
| Other | $4 |
| **Total BOM** | **$25** |
| **Retail Price** (3.0× margin) | **$75** |

**Key Metrics**:
- 4 TOPS INT8, ~3.4 TOPS effective (85%)
- 0.5-2W TDP
- Cost per TOPS: $6.25 BOM / $18.75 retail
- Positioning: Lowest BOM cost, INT8-only, best for cost-sensitive IoT

---

### Qualcomm QCS6490 (Hexagon V79)
**Target**: Mobile robots, drones, edge AI cameras
| Component | Cost |
|-----------|------|
| Silicon die (6nm, integrated SoC) | $45 |
| Package (advanced flip-chip) | $12 |
| Memory (2GB LPDDR4X on-package) | $15 |
| PCB assembly | $6 |
| Thermal solution | $2 |
| Other (testing, certification) | $5 |
| **Total BOM** | **$85** |
| **Retail Price** (2.8× margin) | **$238** |

**Key Metrics**:
- 12 TOPS mixed precision peak, ~6-8 TOPS INT8 effective (50-70%)
- 5-15W TDP range
- Cost per TOPS: $10.63 BOM / $29.75 retail
- Positioning: Integrated SoC (CPU+GPU+NPU), competitive with KPU-T64

---

### NVIDIA Jetson Orin Nano
**Target**: Entry-level edge AI, robotics prototyping
| Component | Cost |
|-----------|------|
| Silicon die (12nm Ampere, 16 SMs) | $120 |
| Package | $25 |
| Memory (8GB LPDDR5) | $40 |
| PCB assembly | $10 |
| Thermal solution | $3 |
| Other | $7 |
| **Total BOM** | **$205** |
| **Retail Price** (1.46× margin) | **$299** |

**Key Metrics**:
- 67 TOPS INT8 (advertised sparse), ~2-4 TOPS effective (3-6% due to throttling!)
- 7-15W TDP range
- Cost per EFFECTIVE TOPS: $51.25 BOM / $74.75 retail (terrible efficiency!)
- Positioning: Poor value (highest BOM, lowest effective TOPS), NVIDIA market penetration pricing

---

## Mid-Range Edge AI ($350-$700 BOM, 15-60W)

### Stillwater KPU-T256
**Target**: Autonomous mobile robots, delivery robots, inspection drones
| Component | Cost |
|-----------|------|
| Silicon die (16nm TSMC, 256 tiles) | $280 |
| Package (advanced flip-chip) | $45 |
| Memory (8GB LPDDR5) | $90 |
| PCB assembly | $20 |
| Thermal solution (larger heatsink) | $8 |
| Other | $12 |
| **Total BOM** | **$455** |
| **Retail Price** (2.4× margin) | **$1,092** |

**Key Metrics**:
- 64 TOPS INT8 peak, ~45 TOPS effective (70%)
- 15-50W TDP range
- Cost per TOPS: $10.11 BOM / $24.27 retail
- Positioning: 3-9× effective TOPS vs Jetson AGX at same power, superior efficiency

---

### Hailo-10H
**Target**: Edge Gen AI, on-device LLMs, vision-language models
| Component | Cost |
|-----------|------|
| Silicon die (16nm, transformer-optimized dataflow) | $30 |
| Package (with LPDDR4X interface) | $10 |
| Memory (4GB LPDDR4X on-module for KV cache) | $20 |
| PCB assembly | $5 |
| Thermal solution (tiny heatsink, 2.5W) | $1 |
| Other | $4 |
| **Total BOM** | **$70** |
| **Retail Price** (3.5× margin) | **$240** |

**Key Metrics**:
- 40 TOPS INT4, 20 TOPS INT8 peak
- ~32 TOPS INT4, ~17 TOPS INT8 effective (80-85%)
- 2.5W TDP (16 TOPS/W INT4, best for edge LLMs)
- Cost per TOPS: $1.75 BOM / $6.00 retail (INT4 basis)
- Positioning: First edge Gen AI accelerator, KV cache support, 2B parameter LLMs @ 10 tokens/sec

---

### Qualcomm SA8775P (Snapdragon Ride)
**Target**: ADAS L2+/L3, cockpit compute, automotive vision
| Component | Cost |
|-----------|------|
| Silicon die (5nm automotive, ASIL D) | $180 |
| Package (automotive-grade) | $35 |
| Memory (16GB LPDDR5 automotive) | $80 |
| PCB assembly (automotive) | $25 |
| Thermal solution (automotive) | $15 |
| Other (safety certification) | $15 |
| **Total BOM** | **$350** |
| **Retail Price** (2.2× margin) | **$770** |

**Key Metrics**:
- 32 TOPS INT8 (estimated), ~16-20 TOPS effective (50-60%)
- 20-45W TDP range
- Cost per TOPS: $17.50 BOM / $38.50 retail
- Positioning: Automotive-certified (ASIL D), integrated cockpit+ADAS

---

### NVIDIA Jetson Orin AGX
**Target**: High-end robotics, prototyping, research
| Component | Cost |
|-----------|------|
| Silicon die (8nm Ampere, 32 SMs) | $380 |
| Package | $60 |
| Memory (64GB LPDDR5) | $160 |
| PCB assembly | $35 |
| Thermal solution | $15 |
| Other | $20 |
| **Total BOM** | **$670** |
| **Retail Price** (1.34× margin) | **$899** |

**Key Metrics**:
- 275 TOPS INT8 (advertised sparse), ~5-17 TOPS effective (2-6% due to throttling!)
- 15-60W TDP range
- Cost per EFFECTIVE TOPS: $39.41 BOM / $52.88 retail (very poor efficiency)
- Positioning: Expensive, poor thermal design, fails in real deployments

---

## High-End Automotive/Robotics ($800-$1600 BOM, 60-130W)

### Stillwater KPU-T768
**Target**: Autonomous vehicles (L3-L4), humanoid robots, industrial AGVs
| Component | Cost |
|-----------|------|
| Silicon die (7nm TSMC, 768 tiles) | $680 |
| Package (multi-chip or interposer) | $120 |
| Memory (32GB LPDDR5X or HBM2e) | $280 |
| PCB assembly | $65 |
| Thermal solution (liquid cooling) | $45 |
| Other | $35 |
| **Total BOM** | **$1,225** |
| **Retail Price** (2.4× margin) | **$2,940** |

**Key Metrics**:
- 192 TOPS INT8 peak, ~135 TOPS effective (70%)
- 60-100W TDP range
- Cost per TOPS: $9.07 BOM / $21.78 retail
- Positioning: 1.3-4.5× effective TOPS vs Jetson Thor, $275 cheaper BOM

---

### Qualcomm Snapdragon Ride (700 TOPS config)
**Target**: L3/L4/L5 autonomous driving, robotaxis
| Component | Cost |
|-----------|------|
| Silicon die (4nm automotive, large) | $380 |
| Package (multi-chip automotive) | $85 |
| Memory (32GB LPDDR5X automotive) | $180 |
| PCB assembly (automotive) | $50 |
| Thermal solution (liquid cooling) | $60 |
| Other (certification, safety) | $45 |
| **Total BOM** | **$800** |
| **Retail Price** (2.0× margin) | **$1,600** |

**Key Metrics**:
- 700 TOPS mixed precision peak, ~350-420 TOPS INT8 effective (50-60%)
- 65-130W TDP range
- Cost per TOPS: $1.14 BOM / $2.29 retail
- Positioning: Scalable platform (30-2000 TOPS), automotive focus, competitive with KPU-T768

---

### NVIDIA Jetson Thor
**Target**: Next-gen autonomous vehicles, humanoid robots (2025+)
| Component | Cost |
|-----------|------|
| Silicon die (4nm Blackwell, 64 SMs) | $850 |
| Package (advanced CoWoS-like) | $180 |
| Memory (128GB HBM3) | $350 |
| PCB assembly | $90 |
| Thermal solution (liquid cooling) | $80 |
| Other | $50 |
| **Total BOM** | **$1,600** |
| **Retail Price** (1.56× margin) | **$2,500** |

**Key Metrics**:
- 2000 TOPS INT8 (advertised sparse), ~30-100 TOPS effective (1.5-5% due to throttling!)
- 30-130W TDP range
- Cost per EFFECTIVE TOPS: $16.00-$53.33 BOM / $25-$83.33 retail (terrible efficiency)
- Positioning: Extremely expensive, unproven thermal design, likely fails deployment

---

## Competitive Analysis Summary

### BOM Cost vs Effective Performance (@ typical deployment power)

| Product | BOM Cost | Effective TOPS | $/TOPS (BOM) | Power Budget | Efficiency |
|---------|----------|----------------|--------------|--------------|------------|
| **Entry-Level** |
| Coral Edge TPU | $25 | 3.4 | $7.35 | 2W | 85% |
| Hailo-8 | $40 | 22 | $1.82 | 2.5W | 85% |
| QCS6490 | $85 | 7 | $12.14 | 10W | 58% |
| KPU-T64 | $125 | 10 | $12.50 | 6W | 60% |
| Jetson Orin Nano | $205 | 2.5 | $82.00 | 7W | 4% |
| **Mid-Range** |
| Hailo-10H | $70 | 17 (INT8) | $4.12 | 2.5W | 85% |
| KPU-T256 | $455 | 45 | $10.11 | 30W | 70% |
| SA8775P | $350 | 18 | $19.44 | 30W | 56% |
| Jetson Orin AGX | $670 | 10 | $67.00 | 30W | 4% |
| **High-End** |
| Snapdragon Ride | $800 | 320 | $2.50 | 100W | 46% |
| KPU-T768 | $1,225 | 135 | $9.07 | 80W | 70% |
| Jetson Thor | $1,600 | 60 | $26.67 | 100W | 6% |

### Key Insights

1. **Hailo Leads $/TOPS**: Hailo-8 achieves **$1.82/TOPS** (BOM), the best cost efficiency in the entry-level segment. Hailo-10H delivers **$4.12/TOPS** (INT8) with Gen AI capabilities.

2. **NVIDIA's Thermal Disaster**: Jetson products achieve only 2-6% of advertised TOPS due to severe DVFS throttling, making their $/effective-TOPS **10-20× worse** than competitors.

3. **KPU Advantage**: Stillwater KPU achieves 60-70% efficiency across all SKUs, delivering **3-9× effective TOPS per dollar** vs NVIDIA Jetson.

4. **Hailo Power Efficiency**: Both Hailo-8 (10.4 TOPS/W) and Hailo-10H (16 TOPS/W INT4) lead their segments in power efficiency, ideal for battery-powered applications.

5. **Qualcomm Competitiveness**: Qualcomm products achieve 50-60% efficiency, competitive with KPU but at lower absolute performance.

6. **Coral TPU Niche**: Google Coral has lowest BOM at \$25 and good \$/TOPS for ultra-low-power, but limited to INT8 and 4 TOPS peak.

7. **BOM vs Retail Margin**:
   - **NVIDIA**: 1.3-1.6× margin (market penetration pricing, subsidizes poor efficiency)
   - **Qualcomm**: 2.0-2.8× margin (typical B2B automotive/mobile)
   - **Stillwater KPU**: 2.4× margin (competitive positioning)
   - **Hailo**: 3.5-4.0× margin (high-performance specialized accelerators)
   - **Google Coral**: 3.0× margin (high-volume consumer IoT)

8. **Process Node Economics**:
   - **7nm/6nm/5nm**: Sweet spot for automotive (SA8775P, KPU-T768)
   - **4nm**: Expensive but efficient (Snapdragon Ride, Jetson Thor)
   - **16nm**: Cost-effective for mid-volume edge (KPU-T64, KPU-T256, Hailo-8, Hailo-10H)
   - **14nm**: Mature, low-cost (Coral Edge TPU)
   - **12nm**: Obsolete (Jetson Orin family thermal problems)

9. **Memory Cost Impact**:
   - HBM3 adds $350 BOM (Jetson Thor)
   - LPDDR5X automotive adds $80-180 (Qualcomm, KPU-T768)
   - LPDDR4X adds $15-40 (entry-level products, Hailo-10H for KV cache)
   - All-on-chip SRAM saves $20-40 (Hailo-8 advantage)
   - Host memory (Coral TPU) saves $40-80

---

## Market Positioning Recommendations

### KPU-T64 ($125 BOM, $300 retail)
**Target Customers**: IoT OEMs, drone manufacturers, robotics startups
- **Competitive against**: Jetson Orin Nano ($205 BOM, poor performance)
- **Value proposition**: 4× effective TOPS at 40% lower BOM
- **Pricing strategy**: $299 retail (matches Jetson Nano pricing, superior value)

### KPU-T256 ($455 BOM, $1,092 retail)
**Target Customers**: AMR manufacturers, delivery robot companies, inspection robots
- **Competitive against**: Jetson Orin AGX ($670 BOM, poor performance)
- **Value proposition**: 4.5× effective TOPS at 32% lower BOM
- **Pricing strategy**: $999 retail (undercut Jetson AGX @ $899, position as superior alternative)

### KPU-T768 ($1,225 BOM, $2,940 retail)
**Target Customers**: Automotive OEMs, autonomous vehicle platforms, humanoid robotics
- **Competitive against**: Jetson Thor ($1,600 BOM, unproven), Snapdragon Ride ($800 BOM, competitive)
- **Value proposition**: 2.3× effective TOPS vs Thor, proven vs unproven platform
- **Pricing strategy**: $2,499 retail (20% below Thor, competitive with Snapdragon Ride ecosystem)

---

## Next Steps

1. **Validation**: Cross-check BOM estimates with semiconductor industry sources
2. **Benchmarking**: Run YOLO/segmentation workloads to validate effective TOPS estimates
3. **TCO Analysis**: Add power consumption costs for 3-year deployment lifecycle
4. **Volume Pricing**: Model BOM cost reduction at 100K+ and 1M+ volumes
5. **Competitive Intelligence**: Monitor NVIDIA/Qualcomm pricing changes and product announcements

---

## Document Metadata

- **Created**: 2025-11-09
- **Version**: 1.0
- **Calibration Status**: ⚠️ ESTIMATED (needs validation against published benchmarks)
- **Next Review**: 2025-12-01
- **Owner**: Product Management / Hardware Engineering
