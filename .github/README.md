# GitHub Configuration

This directory contains GitHub-specific configuration files.

## CI/CD Workflows

### `workflows/ci.yml`

Continuous Integration workflow that runs on every push and pull request.

**See documentation**: [docs/ci_workflow.md](../docs/ci_workflow.md)

## Setup Instructions

### 1. CI Badge in README

The README.md contains a CI status badge configured for this repository:

```markdown
[![CI](https://github.com/branes-ai/graphs/workflows/CI/badge.svg)](https://github.com/branes-ai/graphs/actions/workflows/ci.yml)
```

This badge will automatically update to show:
- 🟢 Green: All checks passed
- 🔴 Red: One or more checks failed
- 🟡 Yellow: Checks in progress

### 2. First CI Run

After pushing this workflow to GitHub:

1. Go to your repository on GitHub
2. Click the "Actions" tab
3. You should see the "CI" workflow
4. The first run will take ~10-12 minutes (downloading PyTorch)
5. Subsequent runs will be faster (~3-5 minutes) due to caching

### 3. Enable Branch Protection (Optional but Recommended)

To require CI to pass before merging:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Check "Require status checks to pass before merging"
4. Select "CI Success" in the list
5. Check "Require branches to be up to date before merging"

This ensures no code is merged without passing all tests.

### 4. Notifications (Optional)

To get notified of CI failures:

1. Go to your GitHub notification settings
2. Enable Actions notifications
3. Or integrate with Slack/Discord for team notifications

## Workflow Structure

The CI workflow consists of 7 jobs:

1. **Package Installation Check** - Verifies clean installation
2. **Unit Tests** - Python 3.8, 3.9, 3.10, 3.11 matrix
3. **CLI Tools** - Smoke tests for all CLI scripts
4. **Examples** - Validates example scripts
5. **Code Quality** - Linting (currently non-blocking)
6. **Hardware Tests** - Hardware mapper validation
7. **CI Success** - Summary (only runs if all pass)

See [../docs/ci_workflow.md](../docs/ci_workflow.md) for complete documentation.

## Troubleshooting

### CI Badge Shows "Unknown"

- Workflow hasn't run yet - push a commit to trigger it
- Badge URL has wrong username - update the README.md
- Workflow file has syntax errors - check GitHub Actions tab

### CI Failing Immediately

- Check job logs in GitHub Actions tab
- Run tests locally: `pytest tests/ -v`
- Verify package installs: `pip install -e .`

### All Jobs Skipped

- Check if commit message contains `[skip ci]`
- Check branch protection rules
- Verify workflow file is on the branch being pushed

## Future Enhancements

Potential additions to this directory:

- **ISSUE_TEMPLATE/** - GitHub issue templates
- **PULL_REQUEST_TEMPLATE.md** - PR template
- **workflows/release.yml** - Automated releases
- **workflows/docs.yml** - Documentation deployment
- **dependabot.yml** - Automated dependency updates

## Related Files

- [../docs/ci_workflow.md](../docs/ci_workflow.md) - Complete CI documentation
- [../README.md](../README.md) - Main project README with badge
- [../CONTRIBUTING.md](../CONTRIBUTING.md) - Contributing guidelines (if exists)
