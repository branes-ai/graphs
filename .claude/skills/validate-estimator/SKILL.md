---
name: validate-estimator
description: Run estimator validation and analyze accuracy. Use when checking estimator correctness, comparing estimates against ground truth, or investigating estimation accuracy issues.
argument-hint: "[estimator-or-model] [hardware]"
context: fork
allowed-tools: Read, Grep, Glob, Bash
---

# Estimator Validation

Validate estimator accuracy for: $ARGUMENTS

## Procedure

1. **Identify scope**: Which estimator(s) and model(s) to validate
2. **Run existing validation tests**:
   ```bash
   # Full validation suite
   python -m pytest validation/ -v --tb=short

   # Specific estimator
   python validation/estimators/test_resnet18.py

   # Hardware mappers
   python validation/hardware/test_all_hardware.py
   ```

3. **Analyze results**:
   - Check confidence levels are appropriate (CALIBRATED vs THEORETICAL)
   - Compare roofline predictions against published benchmarks
   - Verify energy estimates are physically plausible (check watts * seconds = joules)
   - Check memory estimates against `torch.cuda.memory_allocated()` where possible

4. **Cross-consistency checks**:
   - `energy_total = compute_energy + memory_energy + static_energy`
   - `latency = max(compute_time, memory_time)` (roofline)
   - `throughput = batch_size / latency`
   - `power = energy / latency` should be <= TDP

5. **Report findings** with:
   - Pass/fail summary
   - Accuracy metrics (% error vs ground truth where available)
   - Confidence level appropriateness
   - Recommendations for improvement
