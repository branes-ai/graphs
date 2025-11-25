# Compare hardware on DNN models

Here are the tools for loading models, partitioning, and comparing hardware mappings:

## Loading and Partitioning Models

  CLI Tool - Partition Analyzer:
```bash
Partition Analyzer CLI
======================

Command-line tool to analyze and compare different partitioning strategies for FX graphs.
Compares unfused (baseline) vs fusion strategies, quantifying benefits of operator fusion.

Usage:
    # Compare all strategies on ResNet-18
    python cli/partition_analyzer.py --model resnet18 --strategy all

    # Test fusion strategy with visualization
    python cli/partition_analyzer.py --model mobilenet_v2 --strategy fusion --visualize

    # Compare unfused vs fusion
    python cli/partition_analyzer.py --model efficientnet_b0 --strategy all --compare

    # Visualize specific node range
    python cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --start 5 --end 20

    # Investigate around specific node
    python cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --around 10 --context 5

Command-line Options:
    --model:       Choose model (resnet18, mobilenet_v2, etc.)
    --strategy:    Select strategy (unfused, fusion, all)
    --compare:     Show side-by-side comparison
    --quantify:    Show detailed metrics
    --visualize:   Show graph visualization

    Range Selection (for visualization):
    --start:       Start node (1-based, inclusive)
    --end:         End node (1-based, inclusive)
    --around:      Center node for context view
    --context:     Nodes before/after center (default: 10)
    --max-nodes:   Maximum nodes from start (default: 20)

    --input-shape: Customize input tensor dimensions
```

Models supported: FX-traceable torchvision and transformers
 - resnet18,resnet34,resnet50,resnet101,resnet152
 - mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
 - efficientnet_{b0,b1,b2,b3,b4,b5,b6,b7}
 - vit_b_16,vit_b_32,vit_l_16
 - swin_t,swin_s,swin_b

  Python API:
```python
  from graphs.analysis.unified_analyzer import UnifiedAnalyzer

  analyzer = UnifiedAnalyzer()
  result = analyzer.analyze_model('resnet18', 'Jetson-Orin-Nano')
  # result contains partition_report with subgraphs
```

## Comparing Hardware Mappings

CLI Tool - Hardware Mapping Analysis:
```bash
./cli/analyze_graph_mapping.py --model resnet18 --hardware Jetson-Orin-Nano --precision fp16
```


CLI Tool - Comprehensive Analysis (compare across hardware):
```bash
# Compare same model on different hardware
./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-Nano
./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-AGX
./cli/analyze_comprehensive.py --model resnet18 --hardware Coral-Edge-TPU
```

Python API for Custom Comparisons

```python
  from graphs.analysis.unified_analyzer import UnifiedAnalyzer
  from graphs.reporting import ReportGenerator

  analyzer = UnifiedAnalyzer()
  generator = ReportGenerator()

  # Analyze same model on different hardware
  hardware_targets = ['KPU-T64', 'Jetson-Orin-Nano', 'Jetson-Orin-AGX', 'Coral-Edge-TPU']
  results = {}
  for hw in hardware_targets:
      results[hw] = analyzer.analyze_model('resnet18', hw)

  # Generate comparison reports
  for hw, result in results.items():
      print(f"\n=== {hw} ===")
      print(generator.generate_text_report(result))
```

## Analysis of Graph Mappings

CLI Tool - Graph Mapper
```bash
Graph Mapping Analysis Tool

Analyzes how computational graphs are partitioned and mapped onto hardware resources.

Provides detailed insight into:
- Graph partitioning into subgraphs
- Memory and compute requirements per subgraph
- Hardware resource allocation per subgraph
- Power and latency estimates per subgraph
- Sequential execution modeling
- Total power and latency for complete execution

This tool helps compiler and hardware designers understand:
- How computational graphs use hardware
- Where performance is lost (low utilization, bottlenecks)
- Optimization opportunities (fusion, data layout, etc.)

Usage:
    ./cli/analyze_graph_mapping.py --model resnet18 --hardware KPU-T64
    ./cli/analyze_graph_mapping.py --model mobilenet_v2 --compare Jetson-Orin-Nano,KPU-T64 --batch-size 4
    ./cli/analyze_graph_mapping.py --model resnet50 --hardware Coral-Edge-TPU --precision int8

Command-line Options:
  -h, --help            show this help message and exit
  --model MODEL         Model name (resnet18, resnet50, mobilenet_v2, etc.)
  --hardware HARDWARE   Exact hardware name (run with invalid name to see full list)
  --compare COMPARE     Compare multiple hardware targets (comma-separated, e.g., "KPU-T64,Jetson-Orin-AGX,Coral-Edge-TPU")
  
  Workload and Data path Configuration:
  --batch-size BATCH_SIZE
                        Batch size (default: 1)
  --precision {fp32,fp16,int8,int4}
                        Precision (default: fp16)
  --thermal-profile THERMAL_PROFILE
                        Thermal/power profile (10W, 350W, etc.) - uses hardware default if not specified
  --analysis {basic,full,energy,roofline,memory,all}
                        Analysis mode (default: basic - allocation only, full: roofline+energy+memory, all: everything)

  Reporting Options:
  --show-energy-breakdown
                        Show detailed energy breakdown visualization
  --show-roofline       Show ASCII roofline plot
  --show-memory-timeline
                        Show memory timeline
  --show-mapping-visualization
                        Show three-column mapping visualization (FX Graph → Subgraphs → Hardware)
  
  Range Selection (for visualization):
  --mapping-viz-start MAPPING_VIZ_START
                        Starting subgraph index for mapping visualization
  --mapping-viz-end MAPPING_VIZ_END
                        Ending subgraph index for mapping visualization 
```

## Key Files

  | Purpose          | Location                                                    |
  |------------------|-------------------------------------------------------------|
  | Unified analyzer | src/graphs/analysis/unified_analyzer.py                     |
  | Hardware mappers | src/graphs/hardware/mappers/ (cpu.py, gpu.py, tpu.py, etc.) |
  | Partitioner      | src/graphs/transform/partitioning/                          |
  | CLI tools        | cli/analyze_comprehensive.py, cli/analyze_graph_mapping.py  |
  | Validation tests | validation/hardware/                                        |

## Appendix

## Validation Tests for Hardware Comparison

10-way hardware comparison:
```bash
  python validation/hardware/test_all_hardware.py
```

Specific comparisons:
```bash
  python validation/hardware/test_cpu_vs_gpu_mapping.py
  python validation/hardware/test_gpu_cpu_kpu_comparison.py
```


