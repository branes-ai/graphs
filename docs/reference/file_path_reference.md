KEY FILE PATHS FOR REUSE
========================

GRAPH PARTITIONING
==================
/home/stillwater/dev/branes/clones/graphs/src/graphs/ir/structures.py
  - SubgraphDescriptor (core unit of computation)
  - PartitionReport (complete partitioning results)
  - ParallelismDescriptor, TensorDescriptor, BottleneckType, PartitionReason

/home/stillwater/dev/branes/clones/graphs/src/graphs/transform/partitioning/fusion_partitioner.py
  - FusionBasedPartitioner (main partitioning algorithm)
  - Unified visualization: visualize_partitioning(), visualize_partitioning_colored()
  - Analysis: analyze_balance()

/home/stillwater/dev/branes/clones/graphs/src/graphs/transform/partitioning/graph_partitioner.py
  - GraphPartitioner (single-op baseline, less commonly used)


HARDWARE MAPPING & RESOURCE MODELS
===================================
/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/resource_model.py
  - HardwareResourceModel (capabilities, memory, bandwidth)
  - HardwareAllocation (single subgraph mapping result)
  - GraphHardwareAllocation (complete graph allocation)
  - ComputeFabric (specific compute unit type)
  - Physics-based energy model: PROCESS_NODE_ENERGY, CIRCUIT_TYPE_MULTIPLIER, get_base_alu_energy()

/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/base.py
  - HardwareMapper (abstract base class)

GPU MAPPERS
-----------
/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/gpu.py
  - GPUMapper (SM allocation, wave quantization)
  - Factory functions: create_h100_sxm5_80gb_mapper(), create_jetson_orin_agx_64gb_mapper(), etc.

CPU MAPPERS
-----------
/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/cpu.py
  - CPUMapper (core allocation, SIMD units)
  - Factory functions: create_amd_epyc_9754_mapper(), create_intel_xeon_platinum_8490h_mapper()

ACCELERATOR MAPPERS
-------------------
/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/accelerators/kpu.py
  - KPUMapper (tile allocation, scratchpad constraints)
  - Factory: create_kpu_t64_mapper(), create_kpu_t256_mapper(), create_kpu_t768_mapper()

/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/accelerators/tpu.py
  - TPUMapper (systolic array, weight-stationary)
  - Factory: create_tpu_v4_mapper(), create_coral_edge_tpu_mapper()

/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/accelerators/dpu.py
  - DPUMapper (Xilinx Vitis AI, FPGA)
  - Factory: create_dpu_vitis_ai_mapper()

/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/accelerators/cgra.py
  - CGRAMapper (spatial dataflow, Plasticine-style)
  - Factory: create_plasticine_v2_mapper()

/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/dsp.py
  - DSPMapper (vector/tensor units)
  - Factory: create_qrb5165_mapper(), create_ti_tda4vm_mapper()

/home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/mappers/accelerators/hailo.py
  - HailoMapper (Hailo inference accelerators)
  - Factory: create_hailo10h_mapper(), create_hailo8_mapper()


VISUALIZATION
=============
/home/stillwater/dev/branes/clones/graphs/src/graphs/transform/visualization.py
  - Terminal capability detection
  - ANSIColor (ANSI color codes)
  - BoxChars (UTF-8 and ASCII box drawing)
  - export_to_dot() (Graphviz DOT export)
  - Utilities: get_bottleneck_color(), colorize(), create_legend()


ANALYSIS
========
/home/stillwater/dev/branes/clones/graphs/src/graphs/analysis/unified_analyzer.py
  - UnifiedAnalyzer (main orchestrator)
  - AnalysisConfig (configurable options)
  - UnifiedAnalysisResult (complete analysis results)

/home/stillwater/dev/branes/clones/graphs/src/graphs/analysis/roofline.py
  - RooflineAnalyzer (latency estimation, bottleneck analysis)
  - RooflineReport, LatencyDescriptor

/home/stillwater/dev/branes/clones/graphs/src/graphs/analysis/energy.py
  - EnergyAnalyzer (3-component energy model)
  - EnergyReport

/home/stillwater/dev/branes/clones/graphs/src/graphs/analysis/memory.py
  - MemoryEstimator (peak memory, memory timeline)
  - MemoryReport

/home/stillwater/dev/branes/clones/graphs/src/graphs/analysis/concurrency.py
  - ConcurrencyAnalyzer (graph-level parallelism)
  - ConcurrencyDescriptor


REPORTING
=========
/home/stillwater/dev/branes/clones/graphs/src/graphs/reporting/report_generator.py
  - ReportGenerator (multi-format output)
  - Supports: text, JSON, CSV, markdown, HTML


CLI EXAMPLES
============
/home/stillwater/dev/branes/clones/graphs/cli/analyze_comprehensive.py
  - Comprehensive single-model analysis
  - Uses UnifiedAnalyzer with configurable options
  - Multi-format output support

/home/stillwater/dev/branes/clones/graphs/cli/analyze_batch.py
  - Batch size impact analysis
  - Sweeps batch sizes, generates comparison reports

/home/stillwater/dev/branes/clones/graphs/cli/partition_analyzer.py
  - Partitioning strategy comparison
  - Graph visualization with optional range selection

/home/stillwater/dev/branes/clones/graphs/cli/analyze_graph_mapping.py
  - Hardware mapping analysis
  - Detailed allocation visualization


VALIDATION / TESTS
==================
/home/stillwater/dev/branes/clones/graphs/validation/hardware/test_all_hardware.py
  - 10-way hardware comparison example
  - Shows end-to-end usage pattern
  - DeepLabV3-ResNet101 benchmark across all hardware types

/home/stillwater/dev/branes/clones/graphs/validation/estimators/test_conv2d.py
/home/stillwater/dev/branes/clones/graphs/validation/estimators/test_resnet18.py
  - FLOP computation validation
  - Estimator accuracy testing


MODELS
======
/home/stillwater/dev/branes/clones/graphs/src/graphs/models/__init__.py
  - Model factory functions
  - Integration with torchvision

Used in all CLI tools via:
  from graphs.models import make_resnet18, make_mobilenet_v2, etc.

