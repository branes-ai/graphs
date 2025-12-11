# Tensor Shape Analysis

Summary: DNN Tensor Shape Analysis and Systolic Array Utilization Research Facility

## Package Structure

  src/graphs/research/
      shape_collection/
          extractor.py       - TensorShapeRecord, ShapeExtractor
          categorizer.py     - DNNClassifier (CNN/Encoder/Decoder/FullTransformer)
          database.py        - ShapeDatabase (Parquet/CSV/JSON I/O)
      visualization/
          distributions.py   - Histograms, box plots, CDFs
          heatmaps.py        - 2D heatmaps for (M,N), (M,K), (K,N)
          publication.py     - Publication-ready styling, LaTeX tables
      systolic/
          utilization.py     - SystolicArrayConfig, UtilizationCalculator
          sweep.py           - ArraySizeSweeper for 13 array sizes
          visualization.py   - Utilization plots and heatmaps
      dataflow/
          tiling.py          - TileSchedule, TileScheduler
          loop_nests.py      - LoopNest, LoopLevel representation
          data_movement.py   - DataMovementBreakdown, energy analysis
          dataflows.py       - Weight/Output/Row stationary implementations

  cli/research/
      collect_shapes.py      - Collect shapes from TorchVision/HF models
      analyze_shapes.py      - Generate distribution plots
      sweep_array_sizes.py   - Utilization sweep analysis
      analyze_dataflow.py    - Tiling and dataflow analysis

## Key Features

  1. Shape Collection: Extracts tensor shapes from 140+ TorchVision models, categorizes by DNN class, derives (M,K,N) matmul dimensions
  2. Systolic Utilization: Calculates utilization using TPU-style formula:
    - Spatial: min(M/rows, 1) * min(N/cols, 1)
    - Pipeline: K / (K + rows)
    - Sweeps 13 array sizes: 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128
  3. Dataflow Analysis: Full tiling schedules with:
    - Weight-stationary (TPU-style)
    - Output-stationary
    - Row-stationary (Eyeriss-style)
    - Energy breakdown at RF/L1/L2/DRAM levels
  4. Visualization: Publication-ready plots (PDF/SVG, 300 DPI, Times font)

## CLI Usage

```bash
  # Collect shapes
  python cli/research/collect_shapes.py --output shapes.parquet --verbose

  # Analyze distributions
  python cli/research/analyze_shapes.py --input shapes.parquet --output plots/ --by-class

  # Utilization sweep
  python cli/research/sweep_array_sizes.py --input shapes.parquet --output utilization/

  # Dataflow analysis
  python cli/research/analyze_dataflow.py --M 1024 --K 512 --N 1024 --compare-dataflows
```

## Research Facility Summary

### Generated Artifacts

  Shape Data: shapes.csv - 6,604 records from 30 models (2,262 matmul ops)

  Visualization Plots (outputs/research/plots/):
  - dimension_histograms_overall.pdf - M, K, N distribution histograms
  - dimension_by_family.pdf - Box plots by model family
  - cumulative_distribution.pdf - CDFs of dimensions
  - mn_heatmap.pdf, mk_heatmap.pdf, kn_heatmap.pdf - 2D pair frequencies
  - m_vs_array_size.pdf, n_vs_array_size.pdf - Dimension vs array size analysis
  - op_type_breakdown.pdf - Operation type distribution

  Utilization Results (outputs/research/utilization/):
  - utilization_vs_size.pdf - Utilization curves
  - utilization_heatmap.pdf - Array size x model heatmap
  - utilization_histograms.pdf - Distributions per array size
  - optimal_size_analysis.pdf - Optimal size recommendations

## Key Findings

  | Array Size | Mean Util | Weighted Util | Ops >75% |
  |------------|-----------|---------------|----------|
  | 16x16      | 75.9%     | 97.5%         | 74.3%    |
  | 32x32      | 70.4%     | 95.5%         | 65.6%    |
  | 64x64      | 59.0%     | 90.0%         | 44.8%    |
  | 128x128    | 44.5%     | 80.1%         | 29.0%    |

  Optimal size for BF16: 16x16 (97.52% weighted utilization)

## CLI Commands

```bash
  # Collect shapes from models
  python cli/research/collect_shapes.py --output shapes.csv --verbose

  # Analyze shape distributions
  python cli/research/analyze_shapes.py --input shapes.csv --output plots/ --all

  # Sweep array utilization
  python cli/research/sweep_array_sizes.py --input shapes.csv --output utilization/

  # Analyze dataflow for specific model
  python cli/research/analyze_dataflow.py --model resnet18 --array-size 128 --verbose
```
