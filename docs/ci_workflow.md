# Continuous Integration (CI) Workflow

**Location**: `.github/workflows/ci.yml`

**Purpose**: Automated testing and validation to catch regressions before they reach the main branch.

---

## Overview

The CI workflow runs automatically on:
- Every push to `main` branch
- Every pull request targeting `main`
- Manual trigger from GitHub Actions tab

It consists of **7 parallel jobs** that validate different aspects of the codebase:

1. **Package Installation** - Verifies clean installation
2. **Unit Tests** - Runs test suite across Python 3.8-3.11
3. **CLI Tools** - Validates all command-line tools
4. **Examples** - Ensures example scripts work
5. **Code Quality** - Linting and formatting checks
6. **Hardware Tests** - Hardware mapper validation
7. **CI Success** - Summary job (only runs if all pass)

**Total Runtime**: ~8-12 minutes (with caching)

---

## Job Details

### 1. Package Installation Check

**What it does**:
- Installs the package cleanly with `pip install -e .`
- Tests imports from all major modules

**Catches**:
- Package structure issues (like the `pyproject.toml` bug)
- Missing dependencies
- Broken imports after reorganization

**Validated imports**:
```python
from graphs.ir.structures import TensorDescriptor
from graphs.transform.partitioning import GraphPartitioner
from graphs.transform.partitioning import FusionBasedPartitioner
from graphs.analysis.concurrency import analyze_parallelism
from graphs.hardware.resource_model import HardwareResourceModel
from graphs.hardware.mappers.cpu import CPUMapper
from graphs.hardware.mappers.gpu import GPUMapper
```

### 2. Unit Tests (Matrix Build)

**What it does**:
- Runs entire test suite with `pytest`
- Tests across Python 3.8, 3.9, 3.10, 3.11 in parallel
- Generates code coverage reports
- Runs smoke tests from validation suite

**Catches**:
- Algorithm bugs
- Python version incompatibilities
- Breaking API changes
- Reduced test coverage

**Test locations**:
- `tests/` - Core unit tests
- `validation/hardware/` - Hardware mapper tests (smoke tests only)
- `validation/estimators/` - Estimator accuracy tests (smoke tests only)

**Matrix strategy**: Tests run in parallel across 4 Python versions using GitHub's matrix feature.

### 3. CLI Tools Smoke Tests

**What it does**:
- Executes every CLI tool with basic arguments
- Tests range selection features explicitly
- Validates output generation

**Catches**:
- CLI regressions (like range selection off-by-one bugs)
- Argument parsing issues
- Tool crashes
- Import errors in CLI scripts

**Tools tested**:
```bash
# Discovery and exploration
python cli/discover_models.py
python cli/graph_explorer.py --model resnet18
python cli/graph_explorer.py --model resnet18 --max-nodes 10
python cli/graph_explorer.py --model resnet18 --start 5 --end 10
python cli/graph_explorer.py --model resnet18 --around 10 --context 2

# Partitioning and analysis
python cli/partition_analyzer.py --model resnet18 --strategy fusion --quantify
python cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --max-nodes 10
python cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --start 5 --end 10

# Hardware analysis
python cli/profile_graph.py --model resnet18
python cli/list_hardware_mappers.py
python cli/analyze_graph_mapping.py --model resnet18 --hardware H100
```

**Special focus**: Range selection testing explicitly verifies the unified behavior fixed in session 2025-10-28.

### 4. Examples Validation

**What it does**:
- Runs all example scripts to completion
- Ensures examples stay synchronized with API changes

**Catches**:
- Broken examples after API updates
- Import errors in examples
- Examples using deprecated features

**Examples tested**:
- `quick_start_partitioner.py` - 30-second intro
- `visualize_partitioning.py` - API demonstration
- `demo_fusion_comparison.py` - Fusion benefits

### 5. Code Quality (Linting)

**What it does**:
- Runs fast linting with `ruff`
- Checks code formatting with `black`
- Validates import order with `isort`
- Type checking with `mypy`

**Current status**: Non-blocking (warnings only)

**Rationale**: These checks currently emit warnings instead of failing the build, allowing gradual code cleanup without blocking development.

**Tools used**:
- **ruff**: Fast Python linter (replaces flake8, pylint, etc.)
- **black**: Opinionated code formatter
- **isort**: Import statement organizer
- **mypy**: Static type checker

**Future**: Will become blocking once codebase reaches target quality level.

### 6. Hardware Tests

**What it does**:
- Runs hardware validation test suite
- Tests mapper implementations
- Validates resource models

**Catches**:
- Hardware mapper bugs
- Resource model inconsistencies
- TDP modeling errors

**Test location**: `tests/hardware/`

### 7. CI Success (Summary)

**What it does**:
- Final job that depends on all others
- Only runs if all previous jobs succeed
- Provides summary of passed checks

**Purpose**:
- Clean indication that all checks passed
- Single job to require in branch protection rules
- Nice summary for PR reviewers

---

## Interpreting Results

### ✅ All Checks Passed

Green checkmark on your commit/PR means:
- Package installs cleanly
- All tests pass across Python 3.8-3.11
- All CLI tools work correctly
- All examples run successfully
- Code quality checks completed
- Hardware tests passed

**Ready to merge!**

### ❌ Check Failed

If a job fails, click on it to see details:

**Common failures and fixes**:

| Failure | Likely Cause | Fix |
|---------|--------------|-----|
| Package Installation | Import error, missing package | Check `pyproject.toml` |
| Unit Tests | Test failure, assertion error | Fix the test or the code |
| CLI Tools | Argument error, crash | Test CLI locally, check arguments |
| Examples | Import error, API change | Update examples to match API |
| Code Quality | Style violation | Run `ruff`, `black`, `isort` locally |
| Hardware Tests | Mapper logic error | Check hardware mapper implementation |

### 🟡 Code Quality Warnings

Code quality checks currently show warnings but don't fail the build. These should be addressed over time but won't block merging.

---

## Running CI Tests Locally

### Run All Tests

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install pytest pytest-cov ruff black isort mypy

# Run tests
pytest tests/ -v --cov=src/graphs

# Run validation smoke tests
pytest validation/hardware/ -v -k "not slow"

# Run CLI smoke tests
python cli/graph_explorer.py --model resnet18
python cli/partition_analyzer.py --model resnet18 --strategy fusion --quantify

# Run linting
ruff check src/ tests/ cli/
black --check src/ tests/ cli/
isort --check-only src/ tests/ cli/
```

### Test Specific Python Version

```bash
# Use pyenv or conda to switch Python version
pyenv local 3.8  # or 3.9, 3.10, 3.11

# Run tests
pytest tests/ -v
```

### Quick Pre-commit Check

```bash
# Minimal validation before pushing
pip install -e .
python -c "from graphs.transform.partitioning import GraphPartitioner"
pytest tests/ -v
python cli/graph_explorer.py --model resnet18
```

---

## Caching Strategy

The workflow uses GitHub Actions caching to speed up builds:

```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
```

**What's cached**:
- PyTorch and torchvision packages (~2GB)
- Other pip packages

**Cache invalidation**:
- Automatic when `pyproject.toml` changes
- Manual: Re-run workflow with cache disabled

**Speed improvement**:
- First run: ~8-12 minutes
- Cached run: ~3-5 minutes

---

## Maintenance

### Adding New Tests

1. **Unit tests**: Add to `tests/`
2. **Hardware tests**: Add to `tests/hardware/`
3. **CLI tests**: Add command to `cli-tools` job
4. **Examples**: Add script to `examples/` and `examples` job

### Adding New CLI Tools

When adding a new CLI tool:

1. Create the tool in `cli/`
2. Add smoke test to `.github/workflows/ci.yml`:
   ```yaml
   - name: Test my_new_tool
     run: |
       python cli/my_new_tool.py --basic-test
       echo "✅ my_new_tool.py works"
   ```

### Updating Python Versions

To test a new Python version (e.g., 3.12):

```yaml
matrix:
  python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

### Making Linting Blocking

When ready to enforce code quality:

```yaml
- name: Run ruff
  run: ruff check src/ tests/ cli/  # Remove "|| echo ..." part
```

---

## Troubleshooting

### Job Timeouts

**Problem**: Job exceeds 6-hour GitHub limit
**Solution**: Add timeout:
```yaml
jobs:
  test:
    timeout-minutes: 30
```

### Out of Memory

**Problem**: Test runs out of memory
**Solution**:
- Use CPU-only PyTorch: `torch.device('cpu')`
- Reduce batch sizes in tests
- Skip memory-intensive tests in CI

### Flaky Tests

**Problem**: Tests occasionally fail for non-deterministic reasons
**Solution**:
- Mark with `@pytest.mark.flaky`
- Use fixed random seeds
- Add retries for network-dependent tests

### Cache Issues

**Problem**: Cached dependencies causing issues
**Solution**: Clear cache from GitHub Actions UI or change cache key

---

## Branch Protection Rules

To enforce CI before merging:

1. Go to GitHub repo → Settings → Branches
2. Add rule for `main` branch:
   - ✅ Require status checks to pass
   - ✅ Select: "CI Success"
   - ✅ Require branches to be up to date

This prevents merging PRs with failing tests.

---

## CI Badge

The README includes a CI status badge:

```markdown
[![CI](https://github.com/USERNAME/graphs/workflows/CI/badge.svg)](https://github.com/USERNAME/graphs/actions/workflows/ci.yml)
```

**Note**: Replace `USERNAME` with your actual GitHub username/organization.

Badge colors:
- 🟢 Green: All checks passed
- 🔴 Red: One or more checks failed
- 🟡 Yellow: Checks in progress
- ⚪ Gray: No checks run yet

---

## Historical Context

This CI workflow was created on **2025-10-28** after fixing critical bugs:

1. **Package import structure bug** - Would have been caught by package installation check
2. **Range selection off-by-one bugs** - Would have been caught by CLI tools tests

**Lessons learned**:
- Import structure issues need automated verification
- CLI argument behavior needs explicit testing
- Range selection is critical for user experience

See: `docs/sessions/2025-10-28_unified_range_selection.md`

---

## Future Enhancements

### Potential Additions

1. **Performance Benchmarks**
   - Track performance regression over time
   - Store benchmark results as artifacts

2. **GPU Tests**
   - Use GitHub GPU runners (when available)
   - Test actual hardware mapping execution

3. **Documentation Generation**
   - Auto-generate API docs
   - Deploy to GitHub Pages

4. **Security Scanning**
   - Dependency vulnerability scanning
   - SAST (Static Application Security Testing)

5. **Release Automation**
   - Automatic versioning
   - PyPI package publishing

### Monitoring

Consider adding:
- Slack/Discord notifications for failures
- Weekly summary reports
- Trend analysis for test coverage

---

## Related Documentation

- [CLI Tools Documentation](../cli/README.md)
- [Testing Guide](../tests/README.md)
- [Session Log: Unified Range Selection](sessions/2025-10-28_unified_range_selection.md)
- [Contributing Guide](../CONTRIBUTING.md) (if exists)

---

## Support

**Questions about CI failures?**
1. Check job logs in GitHub Actions
2. Run tests locally to reproduce
3. See troubleshooting section above
4. Open an issue with CI failure details

**Need to skip CI?**
Add `[skip ci]` to commit message (use sparingly!)
