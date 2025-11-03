# Command-Line Tools

This directory contains command-line utilities for graph characterization, profiling, and model analysis.

---

## 📚 Detailed Documentation

Comprehensive how-to guides for each tool:


### Discovery Tools: Profiling & Partitioning
- **[discover_models.py](docs/discover_models.md)** - Find FX-traceable models (140+ models)
- **[profile_graph.py](docs/profile_graph.md)** - Hardware-independent graph profiling
- **[profile_graph_with_fvcore](docs/profile_graph.md)** - PyTorch Engineering fvcore-based graph profiling
- **[graph_explorer.py](docs/graph_explorer.md)** - Explore FX graphs interactively (discovery → summary → visualization)
- **[partition_analyzer.py](docs/partition_analyzer.md)** - Analyze and compare partitioning strategies
- **[list_hardware_mappers.py](docs/list_hardware_mappers.md)** - Discover available hardware (35+ models)

### Core Analysis Tools
- **[compare_models.py](docs/compare_models.md)** - Compare models across hardware targets
- **[analyze_graph_mapping.py](docs/analyze_graph_mapping.md)** - Complete guide to hardware mapping analysis

### Advanced Analysis Tools (Phase 4.1)
- **[analyze_comprehensive.py](docs/analyze_comprehensive.md)** - Deep-dive analysis with roofline, energy, and memory profiling
- **[analyze_batch.py](docs/analyze_batch.md)** - Batch size sweeps and configuration comparison
- **Enhanced analyze_graph_mapping.py** - Now includes Phase 3 analysis modes (--analysis flag)

### Specialized Comparisons
- **[Comparison Tools](docs/comparison_tools.md)** - Automotive, Edge, IP Cores, Datacenter

*💡 Tip: Start with the detailed guides above for step-by-step instructions, examples, and troubleshooting.*

---

## Common Conventions

### Unified Range Selection

CLI tools that visualize graphs (`graph_explorer.py`, `partition_analyzer.py`) use unified node addressing:

**Node Numbering:**
- Node numbers are **1-based** (matching the display output)
- Ranges are **inclusive** on both ends
- Example: `--start 5 --end 10` shows nodes 5, 6, 7, 8, 9, and 10

**Range Selection Methods:**
1. **Explicit Range**: `--start N --end M` (show nodes N through M)
2. **Context View**: `--around N --context K` (show K nodes before/after N)
3. **Max Nodes**: `--max-nodes N` (show first N nodes from start)

**Examples:**
```bash
# Show nodes 5-10 (inclusive, 6 nodes total)
./cli/graph_explorer.py --model resnet18 --start 5 --end 10

# Show 10 nodes around node 35 (nodes 25-45)
./cli/graph_explorer.py --model resnet18 --around 35 --context 10

# Show first 20 nodes (nodes 1-20)
./cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --max-nodes 20
```

---

## Tools

### `partition_analyzer.py`
Analyze and compare different partitioning strategies to quantify fusion benefits.

**Usage:**
```bash
# Compare all strategies
./cli/partition_analyzer.py --model resnet18 --strategy all --compare

# Visualize with specific range
./cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --start 5 --end 20

# Investigate around specific node
./cli/partition_analyzer.py --model mobilenet_v2 --strategy fusion --visualize --around 15 --context 5
```

**Features:**
- Compare partitioning strategies (unfused vs fusion)
- Visualize partitioned graphs with range selection
- Unified range selection (--start/--end, --around/--context, --max-nodes)
- **Node addressing**: 1-based, inclusive ranges matching display output
- Quantify fusion benefits (subgraph reduction, memory savings)
- Analyze fusion patterns and bottlenecks

---

### `graph_explorer.py`
Explore FX computational graphs interactively with three progressive modes.

**Three Modes:**
```bash
# 1. Discover models (no arguments)
./cli/graph_explorer.py

# 2. Get model summary (model only)
./cli/graph_explorer.py --model resnet18

# 3. Visualize sections (model + range)
./cli/graph_explorer.py --model resnet18 --max-nodes 20
./cli/graph_explorer.py --model resnet18 --start 5 --end 20
./cli/graph_explorer.py --model resnet18 --around 35 --context 10
```

**Features:**
- Progressive disclosure: models → summary → visualization
- Prevents accidental output floods (large models have 300+ nodes)
- Comprehensive summary statistics (FLOPs, bottlenecks, operation distribution)
- Side-by-side visualization of FX graph and partitions
- Unified range selection (--start/--end, --around/--context, --max-nodes)
- **Node addressing**: 1-based, inclusive ranges matching display output
- Export to file (--output)
- Shows operation details, arithmetic intensity, partition reasoning

---

### `profile_graph.py`
Profile PyTorch models to understand computational characteristics.

**Usage:**
```bash
# Profile ResNet-18
./cli/profile_graph.py --model resnet18

# Profile with custom input shape
./cli/profile_graph.py --model efficientnet_b0 --input-shape 1,3,240,240

# Output profiling data
./cli/profile_graph.py --model mobilenet_v2 --output profile.json
```

**Outputs:**
- FLOPs per layer
- Memory per layer
- Arithmetic intensity
- Bottleneck analysis (compute vs memory bound)
- Critical path identification

---

### `profile_graph_with_fvcore.py`
Compare our FLOP estimates against fvcore library.

**Usage:**
```bash
# Compare ResNet-18
./cli/profile_graph_with_fvcore.py --model resnet18

# Compare multiple models
./cli/profile_graph_with_fvcore.py --models resnet18,mobilenet_v2,efficientnet_b0
```

**Outputs:**
- Side-by-side FLOP comparison
- Accuracy percentages
- Discrepancy analysis

---

### `discover_models.py`
Discover and list available models from torchvision and custom sources.

**Usage:**
```bash
# List all torchvision models
./cli/discover_models.py

# Filter by pattern
./cli/discover_models.py --filter resnet

# Show model details
./cli/discover_models.py --model resnet18 --details
```

**Outputs:**
- Model names
- Parameter counts
- Input shapes
- Model families (ResNet, MobileNet, etc.)

---

### `model_registry_tv2dot7.py`
Model registry for torchvision 2.7 compatibility.

**Usage:**
```python
from cli.model_registry_tv2dot7 import get_model

model = get_model('resnet18')
```

---

## Hardware Comparison Tools

### `compare_automotive_adas.py`
Compare AI accelerators for automotive Advanced Driver Assistance Systems (ADAS Level 2-3).

**Usage:**
```bash
# Run full automotive comparison
python cli/compare_automotive_adas.py
```

**Features:**
- **Category 1**: Front Camera ADAS (10-15W) - Lane Keep, ACC, TSR
- **Category 2**: Multi-Camera ADAS (15-25W) - Surround View, Parking
- **Hardware**: TI TDA4VM, Jetson Orin Nano/AGX, KPU-T256
- **Models**: ResNet-50, FCN lane segmentation, YOLOv5 automotive
- **Metrics**: 30 FPS requirement, <100ms latency, ASIL-D certification

**Output:**
```
CATEGORY 1 RESULTS: Front Camera ADAS (10-15W)
--------------------------------------------------
Hardware              Power  TDP   Latency  FPS    FPS/W   30FPS?  <100ms?  Util%
TI-TDA4VM-C7x         10W    10.0  110.76   9.0    0.90    ✗       ✗        47.7
Jetson-Orin-Nano      15W    15.0  5.45     183.5  12.23   ✓       ✓        97.9
```

---

### `compare_edge_ai_platforms.py`
Compare edge AI accelerators for embodied AI and robotics platforms.

**Usage:**
```bash
# Run edge AI comparison
python cli/compare_edge_ai_platforms.py
```

**Features:**
- **Category 1**: Computer Vision / Low Power (≤10W) - Drones, robots, cameras
- **Category 2**: Transformers / Higher Power (≤50W) - Autonomous vehicles, edge servers
- **Hardware**: Hailo-8/10H, Jetson Orin, KPU-T64/T256, QRB5165, TI TDA4VM
- **Models**: ResNet-50, DeepLabV3+, ViT-Base
- **Metrics**: Latency, throughput, power efficiency (FPS/W), TOPS/W

**Output:**
```
CATEGORY 1: Computer Vision / Low Power (≤10W)
--------------------------------------------------
Hardware              Peak TOPS  FPS/W   Best for
Hailo-8 @ 2.5W        26         10.4    Edge cameras
Jetson-Orin-Nano      40         12.2    Robots
QRB5165-Hexagon698    15         2.1     Mobile robots
```

---

### `compare_i7_12700k_mappers.py`
Compare CPU mapper performance for Intel i7-12700K (standard vs large L3 cache).

**Usage:**
```bash
# Run CPU mapper comparison
python cli/compare_i7_12700k_mappers.py
```

**Features:**
- Standard i7-12700K (25 MB L3)
- Large cache variant (30 MB L3)
- Models: ResNet-50, DeepLabV3+, ViT-Base
- Metrics: Latency, throughput, cache efficiency

---

### `compare_ip_cores.py`
Compare licensable AI/compute IP cores for custom SoC integration.

**Usage:**
```bash
# Run IP core comparison
python cli/compare_ip_cores.py
```

**Features:**
- **Traditional Architectures** (Stored-Program Extensions):
  * CEVA NeuPro-M NPM11: 20 TOPS INT8 @ 2W (DSP + NPU)
  * Cadence Tensilica Vision Q8: 3.8 TOPS INT8 @ 1W (Vision DSP)
  * Synopsys ARC EV7x: 35 TOPS INT8 @ 5W (CPU + VPU + DNN)
  * ARM Mali-G78 MP20: 1.94 TFLOPS FP32 @ 5W (GPU)

- **Dataflow Architectures** (AI-Native):
  * KPU-T64: 6.9 TOPS INT8 @ 6W (64-tile dataflow)
  * KPU-T256: 33.8 TOPS INT8 @ 30W (256-tile dataflow)

- **Models**: ResNet-50, DeepLabV3+, ViT-Base
- **Metrics**: Peak TOPS, latency, FPS/W, architecture comparison

**Output:**
```
ALL IP CORES - COMPREHENSIVE RESULTS
-------------------------------------------------------
IP Core                Vendor      Type              Power  Latency    FPS     FPS/W   Util%
CEVA NeuPro-M NPM11    CEVA        DSP+NPU IP        2.0    150.57     6.6     3.32    29.3
Cadence Vision Q8      Cadence     Vision DSP IP     1.0    225.30     4.4     4.44    47.7
Synopsys ARC EV7x      Synopsys    CPU+VPU+DNN IP    5.0    364.06     2.7     0.55    14.7
ARM Mali-G78 MP20      ARM         GPU IP            5.0    1221.83    0.8     0.16    99.2
KPU-T64                KPU         Dataflow NPU IP   6.0    4.19       238.8   39.79   98.8
KPU-T256               KPU         Dataflow NPU IP   30.0   1.12       893.2   29.77   90.9
```

**Key Insight:**
KPU dataflow architecture achieves superior efficiency through AI-native design,
not just higher power. Traditional IPs extend stored-program machines, while KPU
is purpose-built for AI workloads from the ground up.

**Typical Use Cases:**
- Mobile flagship: CEVA NeuPro, ARM Mali-G78
- Automotive ADAS: Synopsys ARC EV7x (traditional), KPU-T64/T256 (dataflow)
- Edge AI / Embodied AI: KPU-T64/T256
- Edge servers: KPU-T256
- Base station servers: KPU-T768 (larger variant)

---

### `compare_datacenter_cpus.py`
Compare ARM and x86 datacenter server processors for AI inference workloads.

**Usage:**
```bash
# Run datacenter CPU comparison
python cli/compare_datacenter_cpus.py
```

**Features:**
- **Ampere AmpereOne 192-core**: ARM v8.6+ (5nm TSMC)
  * 192 cores, 22.1 TOPS INT8, 332.8 GB/s memory
  * Best for cloud-native microservices

- **Intel Xeon Platinum 8490H**: x86 Sapphire Rapids (10nm Intel 7)
  * 60 cores, 88.7 TOPS INT8 (AMX), 307 GB/s memory
  * Best for CNN inference (4-10× faster with AMX)

- **AMD EPYC 9654**: x86 Genoa (5nm TSMC)
  * 96 cores, 7.4 TOPS INT8, 460.8 GB/s memory
  * Best for Transformer inference (highest bandwidth)

**Models Tested:**
- ResNet-50 (CNN): Intel Xeon wins (1144 FPS vs 236 FPS Ampere, 217 FPS AMD)
- DeepLabV3+ (Segmentation): Intel Xeon wins (118 FPS vs 13.5 FPS Ampere, 11.7 FPS AMD)
- ViT-Base (Transformer): AMD EPYC wins (878 FPS vs 654 FPS Ampere, 606 FPS Intel)

**Key Insights:**
- **Intel AMX** dominates CNN workloads (4-10× faster)
- **AMD's high bandwidth** (460 GB/s) excels at Transformers
- **Ampere's 192 cores** best for general-purpose compute, not AI

**Output:**
```
DATACENTER CPU COMPARISON RESULTS
============================================================================
ResNet-50
----------------------------------------------------------------------------
CPU                            Cores    TDP      Latency      FPS        FPS/W
Ampere AmpereOne 192-core      192      283      4.24         235.8      0.83
Intel Xeon Platinum 8490H      60       350      0.87         1143.6     3.27  ← Winner
AMD EPYC 9654                  96       360      4.61         216.8      0.60
```

**Documentation**: See `docs/DATACENTER_CPU_COMPARISON.md` for comprehensive analysis

---

## Advanced Analysis Tools

### Phase 4.2: Unified Framework (Recommended)

The refactored v2 tools use the unified analysis framework for simplified code and consistent results. These are **production-ready drop-in replacements** for the Phase 4.1 tools.

**Key Benefits:**
- **Simpler API**: Single UnifiedAnalyzer orchestrates all analysis
- **Consistent Output**: All tools use same ReportGenerator
- **Less Code**: 61.5% code reduction while maintaining all functionality
- **Better Maintenance**: Fix bugs once, benefit everywhere
- **More Formats**: Text, JSON, CSV, Markdown all supported

#### `analyze_comprehensive.py` (Recommended)
Deep-dive comprehensive analysis using the unified framework.

**Usage:**
```bash
# Basic analysis (text output)
./cli/analyze_comprehensive.py --model resnet18 --hardware H100

# JSON output with all details
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output results.json

# CSV output for spreadsheet analysis
./cli/analyze_comprehensive.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
  --output results.csv

# Markdown report
./cli/analyze_comprehensive.py --model efficientnet_b0 --hardware KPU-T256 \
  --output report.md

# FP16 precision analysis
./cli/analyze_comprehensive.py --model resnet50 --hardware H100 \
  --precision fp16 --batch-size 32

# Custom output format
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \
  --format json --quiet
```

**Features:**
- **Roofline Analysis**: Latency, bottlenecks, utilization
- **Energy Analysis**: Three-component model (compute, memory, static)
- **Memory Analysis**: Peak memory, activation/weight breakdown, hardware fit
- **Executive Summary**: Quick overview with recommendations
- **Multiple Formats**: text, JSON, CSV, markdown (auto-detected from extension)
- **Selective Sections**: Choose which sections to include
- **Simplified Code**: 73% less code than v1 (262 lines vs 962 lines)

**Output Example:**
```
═══════════════════════════════════════════════════════════════
                 COMPREHENSIVE ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────
Model:                   ResNet-18
Hardware:                H100 SXM5 80GB
Precision:               FP32
Batch Size:              1

Performance:             0.43 ms latency, 2318 fps
Energy:                  48.9 mJ total (48.9 mJ/inference)
Energy per Inference:    48.9 mJ (93% static overhead)
Efficiency:              10.2% hardware utilization

Memory:                  Peak 55.0 MB
                         (activations: 10.8 MB, weights: 46.8 MB)
                         ✗ Does not fit in L2 cache (52.4 MB)

RECOMMENDATIONS
─────────────────────────────────────────────────────────────────
  1. Increase batch size to amortize static energy (93% overhead)
  2. Consider FP16 for 2× speedup with minimal accuracy loss
  3. Consider tiling or model partitioning to improve cache locality
```

---

#### `analyze_batch.py` (Recommended)
Batch size impact analysis using the unified framework.

**Usage:**
```bash
# Batch size sweep (single model/hardware)
./cli/analyze_batch.py --model resnet18 --hardware H100 \
  --batch-size 1 2 4 8 16 32 --output results.csv

# Model comparison (same hardware, same batch sizes)
./cli/analyze_batch.py --models resnet18 mobilenet_v2 efficientnet_b0 \
  --hardware H100 --batch-size 1 16 32

# Hardware comparison (same model, same batch sizes)
./cli/analyze_batch.py --model resnet50 \
  --hardware H100 Jetson-Orin-AGX KPU-T256 \
  --batch-size 1 8 16

# JSON output with insights
./cli/analyze_batch.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
  --batch-size 1 2 4 8 --output results.json --format json

# Quiet mode (no progress output)
./cli/analyze_batch.py --model resnet18 --hardware H100 \
  --batch-size 1 4 16 32 --output results.csv --quiet
```

**Features:**
- **Batch Size Sweeps**: Understand batching impact on latency, throughput, energy
- **Model Comparison**: Compare different models with same hardware/batch sizes
- **Hardware Comparison**: Compare different hardware with same model/batch sizes
- **Intelligent Insights**: Automatic analysis and recommendations
- **Multiple Formats**: CSV, JSON, text, markdown
- **Simplified Code**: 42% less code than v1 (329 lines vs 572 lines)

**Key Insights Provided:**
- Throughput improvement (e.g., "4.0× throughput increase from batch 1 to 16")
- Energy per inference improvement (e.g., "3.4× better energy efficiency")
- Latency vs throughput trade-offs
- Memory growth analysis
- Recommended batch sizes for different scenarios

**Output Example:**
```
═══════════════════════════════════════════════════════════════
                      BATCH SIZE INSIGHTS
═══════════════════════════════════════════════════════════════

ResNet-18 on H100 SXM5 80GB:
─────────────────────────────────────────────────────────────────
  • Throughput improvement: 4.0× (batch 1: 2318 fps → batch 16: 9260 fps)
  • Energy/inference improvement: 3.4× (batch 1: 48.9 mJ → batch 16: 14.3 mJ)
  • Latency increase: 4.0× (0.43 ms → 1.73 ms)
  • Memory growth: 3.8× (55.0 MB → 210.0 MB)

  Recommendations:
    - For energy efficiency: Use batch 16
    - For throughput: Use batch 16
    - For low latency: Use batch 1
```

**Migration from v1:**
The v2 tools are **drop-in replacements** with identical command-line arguments:
```bash
# Old (Phase 4.1)
./cli/analyze_comprehensive.py --model resnet18 --hardware H100

# New (Phase 4.2) - same command!
./cli/analyze_comprehensive.py --model resnet18 --hardware H100
```

---

### Phase 4.1: Original Tools (Legacy)

The original Phase 4.1 tools are still available but **v2 tools are recommended** for new work.

#### `analyze_comprehensive.py` (Legacy)
Deep-dive comprehensive analysis combining roofline modeling, energy profiling, and memory analysis.

**Usage:**
```bash
# Basic analysis (text output)
./cli/analyze_comprehensive.py --model resnet18 --hardware H100

# JSON output with all details
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output results.json

# CSV output for spreadsheet analysis
./cli/analyze_comprehensive.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
  --output results.csv --format csv

# Markdown report
./cli/analyze_comprehensive.py --model efficientnet_b0 --hardware KPU-T256 \
  --output report.md --format markdown

# FP16 precision analysis
./cli/analyze_comprehensive.py --model resnet50 --hardware H100 \
  --precision fp16 --batch-size 32
```

**Features:**
- **Roofline Analysis**: Latency, bottlenecks (compute vs memory-bound), utilization
- **Energy Analysis**: Three-component model (compute, memory, static/leakage)
- **Memory Analysis**: Peak memory, activation/weight breakdown, hardware fit
- **Multiple Output Formats**: text, JSON, CSV, markdown
- **Executive Summary**: Quick overview with key metrics and recommendations
- **Top Energy Consumers**: Identify optimization opportunities

**Output Example:**
```
═══════════════════════════════════════════════════════════════
                 COMPREHENSIVE ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────
Model:                   ResNet-18
Hardware:                H100 SXM5 80GB
Precision:               FP32
Batch Size:              1

Performance:             5.98 ms latency, 167.2 fps
Energy:                  29.4 mJ total (10.1 mJ compute, 2.4 mJ memory, 16.9 mJ static)
Energy per Inference:    29.4 mJ (57% static overhead)
Efficiency:              5.5% hardware utilization

Memory:                  Peak 44.7 MB (activations: 22.4 MB, weights: 44.7 MB)
                         ✓ Fits in L2 cache (50 MB)

Recommendations:
  • Increase batch size to amortize static energy (57% overhead)
  • Consider FP16 for 2× speedup with minimal accuracy loss
  • Current bottleneck: Memory-bound (optimize data layout)
```

**Documentation**: See `cli/docs/analyze_comprehensive.md` for detailed guide

---

### `analyze_batch.py`
Analyze the impact of batching on performance, energy, and efficiency.

**Usage:**
```bash
# Batch size sweep (single model/hardware)
./cli/analyze_batch.py --model resnet18 --hardware H100 \
  --batch-size 1 2 4 8 16 32 --output results.csv

# Model comparison (same hardware, same batch sizes)
./cli/analyze_batch.py --models resnet18 mobilenet_v2 efficientnet_b0 \
  --hardware H100 --batch-size 1 16 32

# Hardware comparison (same model, same batch sizes)
./cli/analyze_batch.py --model resnet50 \
  --hardware H100 Jetson-Orin-AGX KPU-T256 \
  --batch-size 1 8 16

# JSON output with insights
./cli/analyze_batch.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
  --batch-size 1 2 4 8 --output results.json --format json

# Quiet mode (no progress output)
./cli/analyze_batch.py --model resnet18 --hardware H100 \
  --batch-size 1 4 16 32 --output results.csv --quiet
```

**Features:**
- **Batch Size Sweeps**: Understand batching impact on latency, throughput, energy
- **Model Comparison**: Compare different models with same hardware/batch sizes
- **Hardware Comparison**: Compare different hardware with same model/batch sizes
- **Intelligent Insights**: Automatic analysis and recommendations
- **Multiple Output Formats**: CSV, JSON, text

**Key Insights Provided:**
- Throughput improvement (e.g., "3.2× throughput increase from batch 1 to 32")
- Energy per inference improvement (e.g., "3.7× better energy efficiency with batching")
- Latency vs throughput trade-offs
- Memory growth analysis
- Recommended batch sizes for different scenarios

**Output Example:**
```
═══════════════════════════════════════════════════════════════
         BATCH SIZE ANALYSIS: resnet18 on H100 SXM5 80GB
═══════════════════════════════════════════════════════════════

Batch  Latency    Throughput  Energy/Inf  Peak Mem  Efficiency
  1    5.98 ms    167.2 fps   29.4 mJ     44.7 MB   5.5%
  2    6.45 ms    310.1 fps   18.9 mJ     89.4 MB   9.6%
  4    7.40 ms    540.5 fps   13.7 mJ     178.8 MB  14.6%
  8    9.29 ms    861.1 fps   10.8 mJ     357.6 MB  17.3%
 16    13.08 ms   1223.5 fps  10.7 mJ     715.2 MB  24.4%
 32    20.65 ms   1549.5 fps  13.3 mJ     1430.4 MB 31.0%

KEY INSIGHTS:
─────────────────────────────────────────────────────────────────
Throughput Improvement:
  • 9.3× throughput increase (batch 1: 167 fps → batch 32: 1550 fps)
  • Best throughput: batch 32 at 1549.5 fps

Energy Per Inference:
  • 2.7× energy efficiency improvement with batching
  • Best efficiency: batch 16 at 10.7 mJ/inference
  • Static energy dominates at small batches (57% at batch 1)

Memory Growth:
  • 32× memory increase (44.7 MB → 1430.4 MB)
  • Sub-linear growth: weights reused across batch

Recommendations:
  • For latency-critical: Use batch 1-2 (<7ms latency)
  • For throughput-critical: Use batch 16-32 (>1200 fps)
  • For energy efficiency: Use batch 16 (best energy/inference)
```

**Documentation**: See `cli/docs/analyze_batch.md` for detailed guide

---

### Enhanced `analyze_graph_mapping.py`
Now includes Phase 3 analysis modes via `--analysis` flag.

**New Analysis Modes:**
```bash
# Basic mode (backward compatible - allocation analysis only)
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100

# Energy analysis mode
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --analysis energy

# Roofline analysis mode
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --analysis roofline

# Memory analysis mode
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --analysis memory

# Full analysis (roofline + energy + memory)
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --analysis full

# All analysis modes
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 --analysis all
```

**New Visualization Flags:**
```bash
# Show energy breakdown chart
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 \
  --analysis energy --show-energy-breakdown

# Show roofline plot
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 \
  --analysis roofline --show-roofline

# Show memory timeline
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 \
  --analysis memory --show-memory-timeline

# All visualizations
./cli/analyze_graph_mapping.py --model resnet18 --hardware H100 \
  --analysis all --show-energy-breakdown --show-roofline --show-memory-timeline
```

**Analysis Modes:**
- **basic**: Original allocation analysis (backward compatible)
- **energy**: Three-component energy model (compute, memory, static)
- **roofline**: Bottleneck analysis (compute vs memory-bound)
- **memory**: Peak memory, activation/weight breakdown, hardware fit
- **full**: Combines roofline + energy + memory
- **all**: Everything including concurrency analysis

**Backward Compatibility:**
- Default mode is `--analysis basic` (original behavior)
- All existing scripts and workflows continue to work unchanged
- Phase 3 analysis only runs when explicitly requested

**Documentation**: See `cli/docs/analyze_graph_mapping.md` for comprehensive guide

---

## Power Management Analysis

**NEW (2025-11-03)**: Enhanced energy analysis with hardware mapper integration and power gating support.

### Overview

The unified framework now provides accurate power management modeling by:
- **Hardware Mapper Integration**: Uses actual compute unit allocations (e.g., 24/132 SMs on H100) instead of thread-based estimates
- **Power Gating**: Models the ability to turn off unused compute units (unallocated units consume 0W idle power)
- **Per-Unit Energy Accounting**: Tracks energy for allocated vs unallocated units separately

**Impact**: Up to 61.7% idle energy savings on low-utilization workloads (e.g., ResNet-18 batch size 1).

### Basic Usage

```bash
# Enable power gating for accurate energy estimates
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --power-gating

# Compare with and without power gating
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output no_pg.json
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output with_pg.json --power-gating

# Disable hardware mapping (fallback to thread-based estimation)
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --no-hardware-mapping
```

### Understanding the Output

**Power Management Section** (appears in energy analysis when `--power-gating` is enabled):

```
ENERGY ANALYSIS
-------------------------------------------------------------------------------
Total Energy:            20.9 mJ
  Compute Energy:        1.8 mJ
  Memory Energy:         1.8 mJ
  Static Energy:         17.3 mJ

Energy per Inference:    20.9 mJ
Average Power:           48.5 W
Peak Power:              117.0 W
Energy Efficiency:       16.8%

Power Management:
  Average Units Allocated: 48.1
  Allocated Units Idle:    17.3 mJ
  Unallocated Units Idle:  0.0 mJ
  Power Gating:            ENABLED
  Power Gating Savings:    28.0 mJ (61.7%)
```

**Without power gating** (conservative estimate):
```
Power Management:
  Average Units Allocated: 48.1
  Allocated Units Idle:    17.3 mJ
  Unallocated Units Idle:  28.0 mJ
  Power Gating:            DISABLED (conservative estimate)
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Average Units Allocated** | Average compute units (SMs/tiles/cores) allocated across all operations |
| **Allocated Units Idle** | Idle energy consumed by units actively allocated to workload |
| **Unallocated Units Idle** | Idle energy consumed by unused units (0 with power gating) |
| **Power Gating Savings** | Energy saved by turning off unused units |

### Use Cases

**1. Low-Utilization Workloads** (batch size 1, small models)
```bash
# Power gating has maximum impact on low-utilization workloads
./cli/analyze_comprehensive.py --model mobilenet_v2 --hardware H100 \
    --batch-size 1 --power-gating --output mobile_pg.json
```
Expected savings: 50-70% idle energy reduction

**2. Edge Device Power Budgeting**
```bash
# Accurate power modeling for battery-powered devices
./cli/analyze_comprehensive.py --model efficientnet_b0 --hardware Jetson-Orin-Nano \
    --power-gating --precision fp16 --output edge_power.json
```
Use the "Energy per Inference" metric for battery life estimation.

**3. Datacenter TCO Analysis**
```bash
# Compare power gating impact across different batch sizes
./cli/analyze_batch.py --model resnet50 --hardware H100 \
    --batch-size 1 2 4 8 16 32 64 128 \
    --power-gating --output datacenter_tco.csv
```
Higher batch sizes reduce power gating benefit (better utilization).

**4. Hardware Comparison with Accurate Power**
```bash
# Compare energy efficiency across hardware with realistic idle power
for hw in H100 A100 Jetson-Orin-AGX; do
    ./cli/analyze_comprehensive.py --model resnet18 --hardware $hw \
        --power-gating --output ${hw}_power.json
done
```

**5. EDP (Energy-Delay Product) Comparison**

Compare hardware efficiency using EDP (Energy × Latency), which balances energy and performance trade-offs:

```bash
# Compare edge accelerators: Jetson-Orin-AGX vs KPU-T256
# Lower EDP = better efficiency

# Jetson-Orin-AGX (GPU-based edge device)
./cli/analyze_comprehensive.py --model efficientnet_b0 --hardware Jetson-Orin-AGX \
    --power-gating --precision fp16 --output jetson_edp.json

# KPU-T256 (dataflow NPU)
./cli/analyze_comprehensive.py --model efficientnet_b0 --hardware KPU-T256 \
    --power-gating --precision int8 --output kpu_edp.json

# Extract EDP from results
python -c "
import json
for hw in ['jetson', 'kpu']:
    with open(f'{hw}_edp.json') as f:
        data = json.load(f)
        energy_mj = data['derived_metrics']['energy_per_inference_mj']
        latency_ms = data['derived_metrics']['latency_ms']
        edp_ujs = energy_mj * latency_ms  # mJ × ms = µJ·s
        throughput = data['derived_metrics']['throughput_fps']
        print(f'{hw.upper()}: EDP={edp_ujs:.2f} µJ·s, E={energy_mj:.2f} mJ, L={latency_ms:.2f} ms, T={throughput:.0f} fps')
"
```

**Example Output:**
```
JETSON: EDP=27.47 µJ·s, E=12.39 mJ, L=2.22 ms, T=451 fps
KPU: EDP=6.95 µJ·s, E=8.49 mJ, L=0.82 ms, T=1222 fps

→ KPU has 75% better EDP (4.0× more efficient overall)
→ KPU has 63% better latency (2.7× faster inference)
→ KPU has 31% better energy efficiency
→ KPU has 2.7× better throughput
```

**Analysis:** For EfficientNet-B0, KPU-T256 dominates Jetson-Orin-AGX across all metrics due to its specialized dataflow architecture optimized for depthwise separable convolutions.

**Interpretation:**
- **EDP < 1**: Excellent efficiency (datacenter GPUs at high batch size)
- **EDP 1-5**: Good efficiency (edge accelerators, optimized workloads)
- **EDP 5-20**: Moderate efficiency (CPUs, low-batch GPU)
- **EDP > 20**: Poor efficiency (unoptimized workloads)

Lower EDP is better - it means you get the work done with less energy and in less time.

**Advanced EDP Analysis:**

For more detailed EDP breakdown and subgraph-level analysis, use the specialized architecture comparison tool:
```bash
# Comprehensive EDP comparison with subgraph breakdown
./cli/compare_architectures.py --model efficientnet_b0 --architectures GPU KPU \
    --level subgraph --output edp_detailed.html

# See which specific operations drive EDP differences
./cli/compare_architectures.py --model efficientnet_b0 \
    --explain-difference GPU KPU --metric energy
```

This provides EDP breakdown by architecture component (compute, memory, control overhead) and per-subgraph EDP analysis.

### When to Use Power Gating

**Enable `--power-gating` when:**
- ✅ Analyzing low-utilization workloads (batch size 1-4, small models)
- ✅ Estimating battery life for edge devices
- ✅ Comparing energy efficiency across hardware
- ✅ You have control over hardware power management policies

**Use default (no power gating) when:**
- ⚠️ You want conservative (worst-case) energy estimates
- ⚠️ Hardware doesn't support power gating (older GPUs, some FPGAs)
- ⚠️ Workload keeps all units busy (high batch size, large models)

### Python API

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.resource_model import Precision

# Enable power gating in analysis
config = AnalysisConfig(
    run_hardware_mapping=True,      # Get actual unit allocations
    power_gating_enabled=True,       # Model turning off unused units
    run_roofline=True,
    run_energy=True,
    run_memory=True
)

analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model('resnet18', 'H100', batch_size=1, config=config)

# Access power management metrics
print(f"Total Energy: {result.total_energy_mj:.1f} mJ")
print(f"Power Gating Savings: {result.energy_report.total_power_gating_savings_j * 1000:.1f} mJ")
print(f"Average Allocated Units: {result.energy_report.average_allocated_units:.1f}")
```

### Technical Details

**Hardware Mapper Integration:**
- Maps each subgraph to specific compute units (e.g., SMs on GPU, tiles on TPU)
- Provides `compute_units_allocated` for each operation
- Accounts for wave quantization and occupancy limits

**Per-Unit Idle Power:**
```
idle_power_per_unit = total_idle_power / total_compute_units

# Without power gating:
static_energy = idle_power_per_unit × (allocated + unallocated) × latency

# With power gating:
static_energy = idle_power_per_unit × allocated × latency
```

**Accuracy Improvements:**
- **Utilization**: 48× more accurate (36.5% actual vs 0.76% from thread count)
- **Idle Energy**: 61.7% savings for ResNet-18 batch size 1 on H100
- **Functional Composition**: Energy composes correctly from unit → subgraph → model

### Related Documentation

- **Design Document**: `docs/designs/functional_energy_composition.md`
- **Validation Tests**: `validation/analysis/test_phase1_mapper_integration.py`
- **Enhanced Reporting**: `validation/analysis/test_power_management_reporting.py`

---

## Common Usage Patterns

### Quick Model Analysis
```bash
# 1. Discover available models
./cli/discover_models.py --filter resnet

# 2. Profile the model
./cli/profile_graph.py --model resnet18

# 3. Partition into subgraphs
./cli/partition_analyzer.py --model resnet18 --output results.json

# 4. Compare against fvcore
./cli/show_fvcore_table.py --model resnet18
```

### Custom Model Workflow
```bash
# 1. Profile your model
./cli/profile_graph.py --model path/to/model.py --input-shape 1,3,224,224

# 2. Partition and analyze
./cli/partition_analyzer.py --model path/to/model.py --verbose

# 3. Export for further analysis
./cli/partition_analyzer.py --model path/to/model.py --output analysis.json
```

### Advanced Analysis Workflows

**Recommended: Use Phase 4.2 v2 tools for simplified workflows**

**1. Deep-Dive Model Analysis**
```bash
# Comprehensive analysis with all Phase 3 components (v2 recommended)
./cli/analyze_comprehensive.py --model resnet50 --hardware H100 \
  --output comprehensive_analysis.json

# Generate markdown report for documentation
./cli/analyze_comprehensive.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
  --output edge_deployment_report.md

# CSV format with subgraph details
./cli/analyze_comprehensive.py --model efficientnet_b0 --hardware KPU-T256 \
  --output detailed_analysis.csv --subgraph-details
```

**2. Batch Size Optimization**
```bash
# Find optimal batch size for throughput (v2 recommended)
./cli/analyze_batch.py --model resnet18 --hardware H100 \
  --batch-size 1 2 4 8 16 32 --output batch_sweep.csv

# Compare batching behavior across models
./cli/analyze_batch.py --models resnet18 mobilenet_v2 efficientnet_b0 \
  --hardware H100 --batch-size 1 16 32 --output model_comparison.csv

# Quiet mode for scripting
./cli/analyze_batch.py --model resnet18 --hardware H100 \
  --batch-size 1 4 16 32 --output batch_sweep.csv --quiet --no-insights
```

**3. Hardware Selection for Deployment**
```bash
# Compare hardware options (v2 recommended)
./cli/analyze_batch.py --model resnet50 \
  --hardware H100 Jetson-Orin-AGX KPU-T256 \
  --batch-size 1 8 16 --output hardware_comparison.csv

# Deep-dive into specific hardware
./cli/analyze_comprehensive.py --model resnet50 --hardware KPU-T256 \
  --precision fp16 --batch-size 8 --output kpu_deployment.json
```

**4. Energy Efficiency Analysis**
```bash
# Comprehensive energy analysis (v2 recommended)
./cli/analyze_comprehensive.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
  --output energy_analysis.json

# Find energy-optimal batch size
./cli/analyze_batch.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
  --batch-size 1 2 4 8 --output energy_optimization.csv
# Look for "Best efficiency" in the insights

# Alternative: Use enhanced graph mapping tool
./cli/analyze_graph_mapping.py --model mobilenet_v2 --hardware Jetson-Orin-Nano \
  --analysis energy --show-energy-breakdown
```

**5. Complete Deployment Analysis**
```bash
# Step 1: Comprehensive analysis (v2 recommended)
./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-AGX \
  --output analysis_report.json

# Step 2: Batch size sweep
./cli/analyze_batch.py --model resnet18 --hardware Jetson-Orin-AGX \
  --batch-size 1 2 4 8 --output batch_analysis.csv

# Step 3: Full Phase 3 analysis with visualizations (use original tool)
./cli/analyze_graph_mapping.py --model resnet18 --hardware Jetson-Orin-AGX \
  --analysis full --show-energy-breakdown --show-roofline --show-memory-timeline
```

**Legacy workflows:** The original Phase 4.1 tools (`analyze_comprehensive.py`, `analyze_batch.py`) still work but v2 is recommended for new work.

## Output Formats

### JSON Format
```json
{
  "model": "resnet18",
  "total_flops": 3.6e9,
  "total_memory": 44.6e6,
  "subgraphs": [
    {
      "id": 0,
      "operations": ["conv2d", "batchnorm2d", "relu"],
      "flops": 1.2e8,
      "memory": 2.1e6
    },
    ...
  ]
}
```

### CSV Format
```csv
subgraph_id,operations,flops,memory,bottleneck
0,"conv2d+bn+relu",1.2e8,2.1e6,compute
1,"conv2d+bn+relu",3.7e7,1.5e6,memory
...
```

### Text Format
```
Model: resnet18
Total FLOPs: 3.60 G
Total Memory: 44.6 MB
Subgraphs: 32

Subgraph 0: conv2d+bn+relu
  FLOPs: 120 M
  Memory: 2.1 MB
  Bottleneck: compute-bound (AI=57)
```

## Requirements

All tools require:
- Python 3.8+
- PyTorch
- torchvision

Optional:
- fvcore (for `show_fvcore_table.py`)

Install:
```bash
pip install torch torchvision fvcore
```

## Environment Setup

Tools use the repo root as the working directory:
```bash
# Run from repo root
./cli/partition_analyzer.py --model resnet18

# Or set PYTHONPATH
export PYTHONPATH=/path/to/graphs/repo
python cli/partition_analyzer.py --model resnet18
```

## Troubleshooting

**Import errors:**
- Run from repo root directory
- Check PYTHONPATH includes repo root
- Verify `src/graphs/` package structure

**Model not found:**
- Use `./cli/discover_models.py` to list available models
- Check torchvision version (some models require v0.13+)
- For custom models, provide absolute path

**FLOP mismatch with fvcore:**
- Different counting methodologies
- Our counts include operations fvcore may skip
- ±10% variance is expected and acceptable

## Adding New Tools

1. Create `tool_name.py` in `cli/`
2. Add shebang and make executable: `chmod +x cli/tool_name.py`
3. Include argparse for CLI arguments
4. Add tool to this README
5. Add usage examples

Template:
```python
#!/usr/bin/env python
"""Tool description"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.graphs.characterize.<module> import <Component>


def main():
    parser = argparse.ArgumentParser(description="Tool description")
    parser.add_argument('--model', required=True, help="Model name")
    args = parser.parse_args()

    # Tool logic here
    ...


if __name__ == '__main__':
    main()
```

---

## Quick Reference

### Common Workflows

**1. Discover and Profile a Model**
```bash
# Find available models
python3 cli/discover_models.py

# Profile the model
python3 cli/profile_graph.py --model resnet50

# Comprehensive analysis (Phase 4.1)
python3 cli/analyze_comprehensive.py --model resnet50 --hardware H100
```

**2. Compare Hardware Options**
```bash
# List available hardware
python3 cli/list_hardware_mappers.py

# Compare multiple hardware targets
python3 cli/analyze_graph_mapping.py --model resnet50 \
  --compare "H100,Jetson-Orin-AGX,KPU-T256"
```

**3. Evaluate Edge Deployment**
```bash
# Quick edge platform comparison
python3 cli/compare_edge_ai_platforms.py

# Detailed edge hardware analysis with energy profiling (Phase 4.1)
python3 cli/analyze_comprehensive.py --model mobilenet_v2 \
  --hardware Jetson-Orin-Nano --output edge_analysis.json
```

**4. Specialized Comparisons**
```bash
# Automotive ADAS platforms
python3 cli/compare_automotive_adas.py

# Datacenter CPUs
python3 cli/compare_datacenter_cpus.py

# IP cores for SoC integration
python3 cli/compare_ip_cores.py
```

**5. Advanced Analysis (Phase 4.1)**
```bash
# Comprehensive roofline/energy/memory analysis
python3 cli/analyze_comprehensive.py --model resnet18 --hardware H100 \
  --output comprehensive_report.json

# Batch size optimization
python3 cli/analyze_batch.py --model resnet18 --hardware H100 \
  --batch-size 1 2 4 8 16 32 --output batch_analysis.csv

# Full Phase 3 analysis with enhanced tool
python3 cli/analyze_graph_mapping.py --model resnet18 --hardware H100 \
  --analysis full --show-energy-breakdown --show-roofline
```

### Tool Selection Guide

| Goal | Tool | Notes |
|------|------|-------|
| Find available models | `discover_models.py` | |
| Profile model (HW-independent) | `profile_graph.py` | |
| Find available hardware | `list_hardware_mappers.py` | |
| Analyze single HW target | `analyze_graph_mapping.py --hardware` | |
| Compare multiple HW targets | `analyze_graph_mapping.py --compare` | |
| Compare models on same HW | `compare_models.py` | |
| **Deep-dive analysis** | **`analyze_comprehensive.py`** | ⭐ Recommended (Phase 4.2) |
| **Batch size impact analysis** | **`analyze_batch.py`** | ⭐ Recommended (Phase 4.2) |
| **Roofline/energy/memory analysis** | **`analyze_graph_mapping.py --analysis full`** | |
| Automotive deployment | `compare_automotive_adas.py` | |
| Edge deployment | `compare_edge_ai_platforms.py` | |
| Datacenter CPUs | `compare_datacenter_cpus.py` | |

**Note:** Phase 4.2 v2 tools (`*_v2.py`) are recommended for new work. They use the unified framework and are drop-in replacements for Phase 4.1 tools.

---

## Documentation

See also:
- `cli/docs/` - **Detailed how-to guides for each tool**
- `../examples/README.md` - Usage demonstrations
- `../validation/README.md` - Validation tests
- `../docs/` - Architecture documentation
