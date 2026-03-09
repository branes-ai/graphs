---
paths:
  - "cli/**/*.py"
---

# CLI Tool Rules

- Use argparse with descriptive help text
- Support `--output` flag with auto-detected format (JSON, CSV, MD, text)
- Use `UnifiedAnalyzer` for analysis (not raw estimator calls)
- Use `ReportGenerator` for output formatting
- Exit code 0 on success, 1 on error
- Print human-readable text to stdout by default
- Print errors/warnings to stderr
- Support `--verbose` flag for detailed output
- Support `--hardware` accepting mapper names from the registry
- Support `--model` accepting model names from the model factory
