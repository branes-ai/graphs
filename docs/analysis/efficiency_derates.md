# Efficiency Derate Design

● Write(docs/design/efficiency_derate_analysis.md)
  ⎿  Wrote 345 lines to docs/design/efficiency_derate_analysis.md
     # Analysis: Hard-Coded Efficiency Derates in Latency Estimation
     ## Executive Summary
     Your concern is well-founded. The current implementation contains **extensive hard-coded efficiency derates** that:
     1. Are calibrated primarily on **Jetson Orin AGX 50W** for GPU
     2. Have limited calibration data for **CPU** (only i7-12700K with sparse data points)
     3. Embed assumptions that may not transfer across hardware
     4. Are difficult to validate, maintain, and improve systematically
     … +335 lines (ctrl+o to expand)

I've created a detailed analysis at docs/design/efficiency_derate_analysis.md. Here's the summary:

  Your Concern is Valid

  The current implementation has ~60+ hard-coded efficiency parameters with these problems:

  1. GPU model: Calibrated on Jetson Orin AGX 50W, applied everywhere
  2. CPU model: Has operation-size-dependent scaling (lines 788-823), but only 3 sparse calibration points
  3. No feedback loop: Values can't self-correct when wrong
  4. Tech debt risk: Fixes to one model may break others

  Analysis of Your Validation Results

  Looking at the i7-12700K results, there's a clear pattern:
  ┌─────────────────────────┬──────────────┬────────────────────────────────────────────────────────────────────────────┐
  │        Category         │    Error     │                              What's Happening                              │
  ├─────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ VGG, deep ResNets       │ +25% to +50% │ Overestimated - large convs achieve better efficiency than modeled         │
  ├─────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ MobileNet, EfficientNet │ -12% to -37% │ Underestimated - small/depthwise ops achieve worse efficiency than modeled │
  ├─────────────────────────┼──────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ ViT models              │ +1% to +21%  │ Accurate - these were the calibration targets                              │
  └─────────────────────────┴──────────────┴────────────────────────────────────────────────────────────────────────────┘
  The CPU efficiency curve in roofline.py (lines 799-823) says:
  - Small ops (10-100M): 0.25-0.70 scale
  - Large ops (>100M): 0.70-1.00 scale

  But measured data suggests:
  - Small ops actually achieve: ~0.15-0.40 scale (worse)
  - Large ops actually achieve: ~1.0-1.5 scale (better)

  Recommended Path Forward

  Don't remove efficiency scaling - it's conceptually correct that small ops are less efficient than large ops. The problem is the parameters are wrong
  and hard-coded.

  Proposed solution: Externalize parameters to calibration files:

  hardware_registry/cpu/i7-12700K/efficiency_curves.json
  hardware_registry/gpu/jetson_orin_agx/efficiency_curves.json

  This enables:
  1. Auditability: See exactly what values are used
  2. Per-hardware calibration: Different curves for different CPUs
  3. Systematic updates: Run calibration tool, update JSON, track improvement
  4. No code changes: Adjust accuracy without touching roofline.py

  Immediate Low-Risk Actions

  1. Document current values - Add explicit calibration source comments
  2. Visualize efficiency curves - Tool to show assumed vs measured
  3. Separate data from code - Load efficiency curves from config files

  The full analysis with implementation details is in the design doc.


  7-Phase Roadmap: Efficiency Factor Externalization
  ┌───────┬──────┬──────────────────────────────────────────────────────────────────────────┐
  │ Phase │ Week │                             Key Deliverables                             │
  ├───────┼──────┼──────────────────────────────────────────────────────────────────────────┤
  │ 1     │ 1    │ JSON schema with statistical fields, extract current hard-coded values   │
  ├───────┼──────┼──────────────────────────────────────────────────────────────────────────┤
  │ 2     │ 2    │ measure_efficiency.py, aggregate_efficiency.py                           │
  ├───────┼──────┼──────────────────────────────────────────────────────────────────────────┤
  │ 3     │ 3    │ analyze_efficiency_distribution.py, uncertainty propagation in estimates │
  ├───────┼──────┼──────────────────────────────────────────────────────────────────────────┤
  │ 4     │ 4    │ calibrate_hardware.py, validate_calibration.py                           │
  ├───────┼──────┼──────────────────────────────────────────────────────────────────────────┤
  │ 5     │ 5    │ visualize_efficiency.py, audit_calibration.py                            │
  ├───────┼──────┼──────────────────────────────────────────────────────────────────────────┤
  │ 6     │ 6    │ Refactor roofline.py to load from data files                             │
  ├───────┼──────┼──────────────────────────────────────────────────────────────────────────┤
  │ 7     │ 7    │ Documentation                                                            │
  └───────┴──────┴──────────────────────────────────────────────────────────────────────────┘
  Statistical Features Throughout

  Per-measurement:
  - Mean, variance, std dev, min, max
  - 95% confidence intervals
  - Sample size tracking

  Aggregation across models:
  - Pooled mean and variance
  - Weighted least squares curve fitting
  - R-squared, residual std dev

  Distribution analysis:
  - Normality tests (Shapiro-Wilk)
  - Heteroscedasticity tests (does variance change with FLOP size?)
  - Outlier detection (>3 sigma)

  Uncertainty propagation:
  - Efficiency curves carry uncertainty bounds
  - Latency estimates include confidence intervals
  - Total uncertainty from summing subgraph variances

  The full roadmap with detailed tasks is at docs/roadmaps/efficiency_externalization_roadmap.md.

