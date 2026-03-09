---
paths:
  - "src/graphs/estimation/**/*.py"
---

# Estimation Module Rules

- Every estimator output descriptor MUST include an `EstimationConfidence` field
- Use `ConfidenceLevel.THEORETICAL` for analytical models without calibration data
- Use `ConfidenceLevel.CALIBRATED` only when backed by measured ground truth
- Energy = compute_energy + memory_energy + static_energy (three-component model)
- Latency = max(compute_time, memory_time) for roofline (not sum)
- Power = energy / latency must never exceed TDP
- All units must be documented: seconds, joules, bytes, FLOPS
- Follow the Analyzer class pattern: `analyze(partition_report, ...) -> Report`
