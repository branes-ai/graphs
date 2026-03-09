---
name: hardware-sweep
description: Run a model across multiple hardware targets and compare results. Use when the user wants to compare hardware options, find the best platform for a workload, or generate comparison tables.
argument-hint: "[model] [hardware-list-or-category]"
context: fork
allowed-tools: Read, Grep, Glob, Bash
---

# Hardware Sweep Comparison

Compare hardware targets for: $ARGUMENTS

## Procedure

1. **Determine scope**:
   - If specific hardware listed, use those
   - If category given (e.g., "edge", "datacenter", "automotive"), use:
     ```python
     from graphs.hardware.mappers import list_mappers_by_category
     ```
   - Default: representative set (H100, Jetson Orin AGX, Coral Edge TPU, Qualcomm SA8775P, KPU T256)

2. **Run analysis** for each hardware target:
   ```bash
   ./cli/analyze_comprehensive.py --model <model> --hardware <hw> --output /tmp/<hw>.json
   ```

3. **Generate comparison table** with columns:
   - Hardware name, category, TDP
   - Latency (ms), Throughput (inferences/sec)
   - Energy per inference (mJ)
   - Peak memory (MB)
   - Bottleneck (compute/memory-bound)
   - Confidence level
   - Energy-Delay Product (EDP)

4. **Identify Pareto-optimal** platforms:
   - Latency vs Energy frontier
   - Throughput vs Power frontier
   - Note which platforms are dominated

5. **Generate verdict** for agentic consumption:
   ```json
   {
     "best_latency": "<hardware>",
     "best_efficiency": "<hardware>",
     "best_edge": "<hardware>",
     "pareto_frontier": ["<hw1>", "<hw2>"],
     "confidence": "THEORETICAL"
   }
   ```
