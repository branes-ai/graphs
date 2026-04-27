# Validation framework

The bottom-up validation framework docs live in three places:

Plans & assessments:

  - docs/plans/bottom-up-microbenchmark-plan.md — the master plan
  - docs/assessments/bottom-up-microbenchmark-coverage.md — coverage assessment
  - docs/plans/model-based-validation-plan.md — model-based validation plan
  - docs/assessments/empirical-vs-model-based-validation.md — the "why bottom-up" rationale

Active tasks (TASK-2026-012 through 020): docs/tasks/active/TASK-2026-01[2-9].yaml, TASK-2026-020.yaml

Code landing spots:

  - src/graphs/benchmarks/layer1_alu/, layer2_register_simd/, power_meter.py, schema.py
  - validation/composition/test_layer_composition.py

Related architecture:

  - docs/architecture/functional_energy_composition.md
  - docs/hardware/architectural_energy.md
