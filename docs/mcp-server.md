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
