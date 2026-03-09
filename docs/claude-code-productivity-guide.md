# Claude Code Productivity Guide for the Graphs Repository

This document describes the Claude Code (CC) configuration for the `graphs`
repository -- the quantitative/analytical tool-call layer of the Embodied AI
Architect agentic system.

## Table of Contents

- [Why: The Productivity Case](#why-the-productivity-case)
- [What: Capability Overview](#what-capability-overview)
- [How: Configuration Reference](#how-configuration-reference)
- [Practical Examples](#practical-examples)

---

## Why: The Productivity Case

The `graphs` repo provides repeatable, confidence-tracked estimates that the
Embodied-AI-Architect LLM orchestrator consumes via tool calls. The scope is
expanding from DNN-only to full embodied AI pipelines (perception, planning,
control, sensor fusion, Kalman filters, compiler transforms, runtime
concurrency).

Working effectively requires:

1. **Navigating 15+ packages** -- finding the right mapper, estimator, or data
   structure across `core/`, `estimation/`, `hardware/mappers/`, `calibration/`,
   `frontends/`, `transform/`, `reporting/`, and `models/`.

2. **Maintaining estimation integrity** -- confidence tracking, three-component
   energy models, roofline semantics, and physical plausibility constraints
   must be correct in every estimator.

3. **Cross-cutting changes** -- adding a hardware mapper touches `resource_model.py`,
   `hardware/mappers/__init__.py`, the mapper file itself, validation tests, and
   potentially CLI tools and the agentic tool definitions in the Architect repo.

4. **Multi-repo consistency** -- changes here must stay compatible with
   `embodied-schemas` (shared data) and `Embodied-AI-Architect` (consumer).

5. **50+ hardware targets** -- the mapper registry spans CPUs, GPUs, DSPs, TPUs,
   KPUs, DPUs, CGRAs, Hailo, and research architectures.

The CC configuration below encodes domain knowledge so that Claude enforces
these constraints automatically.

---

## What: Capability Overview

### Project-Specific Skills (`.claude/skills/`)

| Skill | Invocation | Purpose |
|-------|-----------|---------|
| `/estimator-dev` | Auto + manual | Guided 5-phase estimator development with confidence tracking |
| `/hardware-mapper` | Auto + manual | Guided hardware mapper development with registry integration |
| `/validate-estimator` | Manual | Run validation suites and analyze accuracy |
| `/hardware-sweep` | Manual | Compare a model across hardware targets, find Pareto frontier |
| `/wrapup` | Manual | Generate changelog, session log, contribution record |

### User-Level Skills (`~/.claude/skills/`)

| Skill | Invocation | Purpose |
|-------|-----------|---------|
| `/cpp-model-invariant` | Auto: C++ class generation | Enforces model-invariant-first design |

### Built-in Skills

| Skill | Purpose |
|-------|---------|
| `/simplify` | Review changed code for quality and efficiency |
| `/loop` | Run a prompt on a recurring interval |
| `/claude-api` | Load Claude API reference material |
| `/cpp-modernize` | Analyze and modernize C++ code |
| `/cpp-review` | Comprehensive modern C++ code review |
| `/cpp-optimize` | Analyze performance characteristics |

### Custom Subagents (`.claude/agents/`)

| Agent | Tools | Purpose |
|-------|-------|---------|
| **estimator-researcher** | Read-only + web | Research hardware specs, benchmarks, modeling techniques |
| **validation-runner** | Read-only + Bash | Run test/validation suites and report pass/fail |
| **cross-repo-checker** | Read-only + Bash | Check consistency across graphs/schemas/architect repos |

### Built-in Subagents

| Agent | Speed | Purpose |
|-------|-------|---------|
| **Explore** | Fast | Quick codebase search and understanding |
| **Plan** | Normal | Design implementation strategies |
| **general-purpose** | Normal | Multi-step tasks with full tool access |

### Active Hooks (`.claude/settings.json`)

| Hook | Event | What It Does |
|------|-------|-------------|
| `syntax-check-python.sh` | PostToolUse (Write/Edit) | `py_compile` every Python file after modification |
| `check-estimator-confidence.sh` | PostToolUse (Write/Edit) | Warn if estimation/ files lack confidence tracking |
| `protect-calibration-data.sh` | PreToolUse (Write/Edit/Bash) | Block overwrites/deletes of calibration profiles |

### Path-Specific Rules (`.claude/rules/`)

| Rule File | Applies To | Key Constraints |
|-----------|-----------|-----------------|
| `estimation.md` | `src/graphs/estimation/**` | Confidence required, energy = 3 components, power <= TDP |
| `hardware-mappers.md` | `src/graphs/hardware/**` | Naming conventions, registry integration, utilization bounds |
| `cli-tools.md` | `cli/**` | Use UnifiedAnalyzer, support --output format auto-detection |
| `validation.md` | `validation/**`, `tests/**` | Physical plausibility ranges, integration gate requirements |

### MCP Server Infrastructure (`.mcp.json`)

The project `.mcp.json` is scaffolded and ready for adding servers:

```bash
# GitHub integration (issues, PRs from conversation)
claude mcp add --transport http github https://api.githubcopilot.com/mcp/

# Hardware spec database (if you build one)
claude mcp add --transport stdio hw-specs \
  --env DB_PATH=./hardware_registry/specs.db \
  -- python -m graphs.tools.mcp_hw_server
```

### Memory and Context

| System | What | Loaded |
|--------|------|--------|
| `CLAUDE.md` | Mission, interfaces, conventions, taxonomy | Every session (full) |
| `.claude/rules/*.md` | Path-specific constraints | When files in matching paths are touched |
| Auto Memory | Build commands, debugging insights, patterns | First 200 lines at startup |
| Plugins: `clangd-lsp` | C++ language server (go-to-def, diagnostics) | Active for C++ files |

---

## How: Configuration Reference

### File Layout

```
.claude/
  settings.json              # Hooks configuration
  rules/
    estimation.md            # Estimator constraints (path-scoped)
    hardware-mappers.md      # Mapper conventions (path-scoped)
    cli-tools.md             # CLI tool standards (path-scoped)
    validation.md            # Test requirements (path-scoped)
  skills/
    estimator-dev/SKILL.md   # New estimator development guide
    hardware-mapper/SKILL.md # New mapper development guide
    validate-estimator/SKILL.md  # Run validation + analyze
    hardware-sweep/SKILL.md  # Multi-hardware comparison
    wrapup/SKILL.md          # Session documentation
  agents/
    estimator-researcher/README.md  # Research agent (read-only + web)
    validation-runner/README.md     # Test runner agent (read-only + bash)
    cross-repo-checker/README.md    # Multi-repo consistency check
  hooks/
    syntax-check-python.sh          # Python syntax verification
    check-estimator-confidence.sh   # Confidence tracking enforcement
    protect-calibration-data.sh     # Calibration data protection
  contributions/                    # Monthly contribution logs (JSONL)
  decisions/                        # Architectural decision records (YAML)
.mcp.json                           # MCP server configuration (project-shared)
CLAUDE.md                           # Primary project instructions
```

### Adding Your Own

**New skill**: Create `.claude/skills/<name>/SKILL.md` with YAML frontmatter.

**New agent**: Create `.claude/agents/<name>/README.md` with YAML frontmatter.

**New hook**: Add script to `.claude/hooks/`, register in `.claude/settings.json`.

**New rule**: Create `.claude/rules/<topic>.md` with `paths:` frontmatter.

**New MCP server**: Run `claude mcp add ...` or edit `.mcp.json` directly.

---

## Practical Examples

### Example 1: Building a Pipeline Stage Estimator

**Scenario**: You need an estimator for sensor fusion pipeline latency that
models the data flow from camera + LiDAR through detection, tracking, and
planning stages.

```
You: I need a pipeline stage estimator that models end-to-end latency for a
     perception-to-planning pipeline with 3 stages.
```

**What CC does:**

1. **`/estimator-dev` skill auto-triggers** (description matches "creating
   estimators"): Walks you through the 5-phase workflow:
   - Phase 1: Scope -- PipelineAnalyzer consuming per-stage PartitionReports
   - Phase 2: Descriptors -- PipelineStageDescriptor with stage_latency,
     inter_stage_transfer, pipeline_depth, confidence
   - Phase 3: Implementation in `estimation/pipeline.py`
   - Phase 4: UnifiedAnalyzer integration
   - Phase 5: Validation test

2. **`estimation.md` rule activates** when editing `estimation/pipeline.py`:
   Reminds that confidence fields are required, energy must use 3-component
   model, power cannot exceed TDP.

3. **`syntax-check-python.sh` hook fires** after every Edit/Write to
   `estimation/pipeline.py`: Catches syntax errors immediately.

4. **`check-estimator-confidence.sh` hook fires**: Warns if the new Analyzer
   class doesn't reference `ConfidenceLevel` or `EstimationConfidence`.

5. **validation-runner agent** (you invoke): Runs the full test suite to
   verify nothing broke:
   ```
   You: Run the validation-runner agent to check everything passes
   ```

6. **cross-repo-checker agent** (you invoke): Verifies the new estimator's
   interface is compatible with how the Architect repo would call it.

**Capabilities used**: skill (estimator-dev), rules (estimation.md), hooks
(syntax-check, confidence-check), agents (validation-runner, cross-repo-checker).

### Example 2: Adding an Automotive DSP and Comparing It

**Scenario**: You need to add the TI TDA4VE DSP mapper and compare it against
existing automotive hardware.

```
You: Add a TI TDA4VE mapper and compare it against existing automotive DSPs
```

**What CC does:**

1. **`/hardware-mapper` skill auto-triggers**: Guides you through:
   - Architecture classification (VLIW + vector/tensor, closest to existing DSP mappers)
   - Resource model parameters from TI datasheet
   - Factory function: `create_ti_tda4ve_mapper()`
   - Registry integration in `__init__.py`
   - Validation test in `validation/hardware/test_tda4ve_mapper.py`

2. **`hardware-mappers.md` rule activates** when editing files under
   `src/graphs/hardware/`: Enforces naming conventions, registry pattern,
   utilization bounds.

3. **`protect-calibration-data.sh` hook** silently permits writes to mapper
   files but blocks any accidental touch of `calibration/profiles/`.

4. **`/hardware-sweep` skill** (you invoke after mapper is done):
   ```
   You: /hardware-sweep resnet18 automotive
   ```
   Runs resnet18 across all automotive-category mappers (TDA4VM, TDA4VL,
   TDA4AL, TDA4VH, SA8775P, and the new TDA4VE), generates comparison table
   with latency, energy, EDP, and Pareto frontier.

5. **estimator-researcher agent** (auto-dispatched for datasheet lookup):
   Searches the web for TI TDA4VE specifications if you don't provide them.

**Capabilities used**: skill (hardware-mapper, hardware-sweep), rules
(hardware-mappers.md), hooks (protect-calibration-data), agent
(estimator-researcher).

### Example 3: Debugging an Energy Estimation Discrepancy

**Scenario**: The agentic system reports that energy estimates for MobileNetV2
on Jetson Orin seem too low. You need to investigate.

```
You: Energy estimates for mobilenet_v2 on Jetson Orin AGX look too low.
     Can you investigate?
```

**What CC does:**

1. **Explore subagent** (auto-dispatched): Searches `estimation/energy.py`,
   `hardware/mappers/gpu.py`, and `calibration/profiles/` for Jetson Orin
   parameters. Returns the three-component energy formula, the Orin resource
   model, and any calibration data.

2. **`/validate-estimator` skill** (you invoke):
   ```
   You: /validate-estimator energy mobilenet_v2 jetson_orin_agx_64gb
   ```
   Runs in a forked context: executes the analysis, cross-checks that
   `power = energy / latency <= TDP(60W)`, verifies the three-component
   breakdown sums correctly, and compares against published Jetson benchmarks.

3. **estimator-researcher agent** (you invoke):
   ```
   You: Research published MobileNetV2 inference energy on Jetson Orin AGX
   ```
   Searches the web for NVIDIA's published power measurements, MLPerf edge
   results, and community benchmarks. Returns structured findings with sources.

4. **Fix + hooks enforce quality**: When you edit `energy.py` or `gpu.py` to
   fix the issue, the `syntax-check-python.sh` and `check-estimator-confidence.sh`
   hooks fire automatically.

5. **validation-runner agent** (post-fix verification):
   ```
   You: Run validation-runner to make sure the fix doesn't regress other hardware
   ```
   Runs the full suite: unit tests, estimator validation, hardware validation.

6. **`/wrapup`** (end of session):
   ```
   You: /wrapup
   ```
   Generates changelog entry documenting the energy model fix, session log
   with the investigation details, and contribution record.

**Capabilities used**: subagent (Explore), skill (validate-estimator, wrapup),
agent (estimator-researcher, validation-runner), hooks (syntax-check,
confidence-check).

### Example 4: Continuous Development with Loop and Background Agents

**Scenario**: You are extending the roofline model to support multi-stream
concurrency and want continuous feedback.

```
You: /loop 3m /validate-estimator roofline resnet18
```

This runs roofline validation every 3 minutes while you work on
`estimation/roofline.py`. Each iteration reports accuracy metrics.

Meanwhile, in the same session:

```
You: In the background, research how NVIDIA multi-stream scheduling affects
     roofline predictions for H100
```

The **estimator-researcher agent** runs in the background, searching for
multi-stream scheduling papers and NVIDIA documentation, while you continue
coding. You get notified when results are ready.

**Capabilities used**: skill (loop, validate-estimator), agent
(estimator-researcher in background).

---

## Summary

| Capability | Configuration | Key Benefit |
|------------|--------------|-------------|
| **Skills** (5 project + 1 user + 6 built-in) | `.claude/skills/` | Domain-guided workflows for estimators and mappers |
| **Agents** (3 custom + 3 built-in) | `.claude/agents/` | Isolated research, validation, cross-repo checking |
| **Hooks** (3 active) | `.claude/settings.json` | Syntax checking, confidence enforcement, data protection |
| **Rules** (4 path-scoped) | `.claude/rules/` | Context-sensitive constraints per package |
| **MCP** (scaffolded) | `.mcp.json` | Ready for GitHub, database, monitoring integration |
| **Memory** (CLAUDE.md + auto) | Root + `~/.claude/` | Mission context + accumulated learnings |
| **Plugin** (clangd-lsp) | `~/.claude/settings.json` | C++ language intelligence |

The system is designed so that **domain knowledge is encoded once** and
**enforced automatically** -- whether you are building a new Kalman filter
estimator, adding a RISC-V mapper, or debugging energy estimates for the
agentic system. The hooks catch mechanical errors, the rules enforce
conventions, and the skills guide you through proven workflows.
