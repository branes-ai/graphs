---
paths:
  - "validation/**/*.py"
  - "tests/**/*.py"
---

# Validation and Test Rules

- Validation scripts in `validation/` are standalone (runnable without pytest)
- Unit tests in `tests/` use pytest
- Every new estimator needs a validation test against at least resnet18
- Every new mapper needs a validation test with basic mapping
- Cross-check estimates for physical plausibility:
  - Energy per inference should be in range [0.001 mJ, 10000 mJ]
  - Latency should be in range [0.001 ms, 10000 ms]
  - Memory should not exceed hardware total memory
  - Utilization should be in [0, 1]
- Use `test_all_hardware.py` as the integration gate for mapper changes
