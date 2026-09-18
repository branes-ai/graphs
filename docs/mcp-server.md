# Graphs MCP Server

The graphs repository includes a Model Context Protocol (MCP) server that
exposes the estimation framework as tool calls. This is the primary interface
used by the Embodied-AI-Architect LLM orchestrator to get quantitative
answers during hardware selection and optimization sessions.

## Location

```
src/graphs/mcp/
  __init__.py
  __main__.py      # Entry point: python -m graphs.mcp
  server.py        # 7 tools (554 lines)
  transport.py     # stdio + SSE/HTTP transport (205 lines)
  auth.py          # Optional Bearer token authentication
```

## Tools

| Tool | Purpose |
|------|---------|
| `analyze_model` | Full unified roofline + energy + memory analysis |
| `estimate_latency` | Roofline-based latency with compute/memory breakdown |
| `estimate_energy` | Three-component energy (compute, memory, static/leakage) |
| `estimate_memory` | Peak memory, activation timeline, device fit analysis |
| `compare_hardware` | Multi-target ranking by latency, energy, or memory |
| `list_hardware` | Hardware catalog with type filter and fuzzy search |
| `get_hardware_specs` | Detailed hardware profile (FLOPS, bandwidth, TDP) |
| `list_soc_designs` | SoC designs, nodes, efficiency tables, studies, mission profiles |
| `analyze_soc` | One SoC design on one pipeline mission profile |
| `sweep_soc_study` | SoC design sweeps with bound-aware Pareto and union-of-regimes reports |

### analyze_model

Full unified analysis returning latency, energy, peak memory, bottleneck
classification, utilization, and confidence level. Supports thermal profiles
and power gating.

### estimate_latency

Roofline-based prediction with compute vs memory time breakdown.
Per-subgraph latency analysis with confidence levels
(CALIBRATED/INTERPOLATED/THEORETICAL).

### estimate_energy

Component-wise energy breakdown: compute energy (from FLOPs), memory energy
(from data transfers), and static/leakage energy (from latency). Supports
power-gating savings and thermal-aware TDP.

### estimate_memory

Peak memory usage and activation timeline. Memory reuse patterns and
device fit analysis (does the model fit in on-chip memory?).

### compare_hardware

Compares a model across multiple hardware targets. Sortable by latency,
energy, or memory. Returns ranked comparison with utilization metrics.

### list_hardware

Discover available hardware targets. Filter by device type (cpu, gpu, dsp,
tpu, kpu, accelerator). Supports fuzzy search (e.g., "jetson", "orin").

### get_hardware_specs

Detailed hardware profile: peak FLOPS by precision, memory bandwidth, total
memory, TDP, architecture, compute units, calibration status, power profiles,
and thermal data.

### SoC tools (graphs#269)

These tools run the SoC study framework: IP composition, pipeline workload
scheduling and power.

- **Every figure carries its bound.** An input with gaps (unanchored
  silicon, an unmeasured efficiency, a missing node energy) makes area,
  power and oversubscription lower bounds and TOPS/W an upper bound, and
  the result flags each one.
- **Feasibility has three states.** `feasible` is `true` only when proven,
  `false` on a proven violation, and `null` when gaps leave it open.
- **Read `confidence_summary.limited_by` before using a number.**

The tools:

- **`list_soc_designs`** lists what can be analyzed: designs (and whether
  their silicon is fully priced), nodes, efficiency tables (`annex_v1`
  pooled, `default_v1` per engine), shipped studies, and the workload's
  mission profiles and regimes.
- **`analyze_soc`** runs one design on one profile at a node. By default it
  returns `detail: "summary"`:
  - the verdicts and bounds;
  - memory;
  - the power totals;
  - the stages that are over or unpriced.

  `detail: "full"` returns the whole `SoCAnalysisResult`.
- **`sweep_soc_study`** runs a shipped study or an ad-hoc sweep, and
  optionally adds:
  - `pareto` metrics, which classify points as front, dominated or
    undecided. A lower bound never lands on a front.
  - `union: true`, which reports the smallest design proven feasible in
    every profile.

The Embodied-AI-Architect lists these tools dynamically (`branes mcp tools`).
Its LLM tool layer defines its own schemas, so the orchestrator's agent
reaches them only once they are added there.

## Transport Modes

### stdio (default)

No extra dependencies. Used for local Claude Code integration.

```bash
python -m graphs.mcp
```

### SSE/HTTP

For remote or team access. Requires additional packages: `mcp`, `starlette`,
`uvicorn`.

```bash
python -m graphs.mcp --sse --port 8100
```

## Client Configuration

### Claude Code (this repo)

Add to `.mcp.json` at the repo root or personal `~/.claude.json`:

```json
{
  "mcpServers": {
    "graphs": {
      "command": "python",
      "args": ["-m", "graphs.mcp"],
      "env": {
        "PYTHONPATH": "/home/stillwater/dev/branes/clones/graphs/src"
      }
    }
  }
}
```

### Embodied-AI-Architect

Already configured in `.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "graphs": {
      "command": "/home/stillwater/dev/branes/clones/embodied-ai-architect/.venv/bin/python",
      "args": ["-m", "graphs.mcp"],
      "env": {
        "PYTHONPATH": "/home/stillwater/dev/branes/clones/graphs/src"
      }
    }
  }
}
```

### CLI (via Architect)

The Embodied-AI-Architect wraps the MCP server with a human-friendly CLI:

```bash
branes mcp tools                                  # List available tools
branes mcp hardware                               # List all hardware targets
branes mcp analyze resnet18 jetson_orin_nano      # Full analysis
branes mcp latency resnet50 jetson_orin_nano      # Latency only
branes mcp energy resnet18 h100_sxm5              # Energy analysis
branes mcp memory yolov8n jetson_orin_nano        # Memory analysis
branes mcp compare resnet18 jetson_orin_nano h100_sxm5  # Hardware comparison
branes mcp specs jetson_orin_nano                 # Hardware specifications
branes mcp server --sse --port 8100               # Start MCP server
```

## Authentication

Optional Bearer token authentication for SSE/HTTP mode:

- Set environment variable: `GRAPHS_MCP_TOKEN`
- Uses HMAC constant-time comparison
- Auth is disabled if the environment variable is not set (local dev mode)
- Generate a token: `python -c "import secrets; print(secrets.token_hex(32))"`

## Design Documentation

- Server design: `../Embodied-AI-Architect/docs/graphs-mcp-server-design.md`
- MCP architecture patterns: `../Embodied-AI-Architect/docs/mcp-architectures.md`
- MCP tools reference: `../Embodied-AI-Architect/docs-site/src/content/docs/reference/mcp-tools.md`
