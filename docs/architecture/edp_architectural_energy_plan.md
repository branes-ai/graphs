# EDP-Based Architectural Energy Comparison Framework

**Date:** 2025-11-03
**Status:** Planning → Implementation
**Goal:** Quantify, compare, and contrast architectural energy events using Energy-Delay Product (EDP)

---

## Executive Summary

This plan extends the existing architectural energy modeling (./src/graphs/hardware/architectural_energy.py) to support EDP-based comparisons across hardware architectures. The framework will preserve the existing simple energy model (compute + memory + static) while adding detailed EDP analysis that explains WHY one architecture is more energy-efficient than another.

**Key Innovation:** Two-tier energy modeling approach
- **Tier 1 (Simple):** Fast level-setting with compute + memory + static energy
- **Tier 2 (Architectural):** Detailed energy event modeling with EDP-based comparisons

**Validation Strategy:** Start with parameterized MLP (linear + bias + ReLU) as trusted test case

---

## Current State Inventory

### Two-Tier Energy Modeling Architecture

#### 1. Simple Energy Model (src/graphs/analysis/energy.py)
- **Formula:** `energy_per_flop × FLOPs + energy_per_byte × bytes + idle_power × latency`
- **Components:** Compute, memory, static (leakage)
- **Usage:** UnifiedAnalyzer, Phase 4.2 framework
- **Purpose:** Quick level-setting and efficiency metrics

#### 2. Architectural Energy Model (src/graphs/hardware/architectural_energy.py)
- **Architecture Classes (7):**
  - `STORED_PROGRAM` (CPU, DSP) - 1.0× baseline
  - `DATA_PARALLEL` (GPU) - 2.5-3.0× overhead (coherence machinery)
  - `SYSTOLIC_ARRAY` (TPU) - 0.10-0.20× (85% reduction)
  - `DOMAIN_FLOW` (KPU) - 0.25-0.40× (programmable spatial)
  - `DATA_FLOW_MACHINE` (Reference) - 0.30-0.50× (CAM-limited)
  - `SPATIAL_PARTITION` (Cerebras, Hailo) - 0.15-0.30×
  - `ADAPTIVE_DATAPATH` (FPGA, CGRA, DPU) - 0.15-0.30× + reconfig overhead

- **Energy Events Modeled:**
  - Instruction fetch, operand fetch, pipeline control (CPU)
  - Coherence machinery, thread scheduling, warp divergence (GPU)
  - Schedule setup, data injection/extraction (TPU)
  - Domain tracking, wavefront management (KPU)
  - CAM lookups, token matching (DFM)
  - Graph partitioning, mesh communication (Cerebras)
  - Reconfiguration, routing overhead (FPGA)

#### 3. Current Integration Points

**HardwareResourceModel** (resource_model.py:377):
```python
architecture_energy_model: Optional['ArchitecturalEnergyModel'] = None
```

**HardwareMapper.estimate_energy()** (resource_model.py:771-777):
- Uses architectural model if available
- Returns baseline + architectural breakdown

**ArchitectureComparator** (architecture_comparator.py:209-230):
- Extracts architectural breakdowns
- Compares energy across architectures
- Shows energy winners

**Hardware Mappers** (Configured with energy models):
- CPU: StoredProgramEnergyModel
- GPU: DataParallelEnergyModel
- TPU: SystolicArrayEnergyModel
- KPU: DomainFlowEnergyModel
- DFM: DataFlowMachineEnergyModel

**Demo Example** (examples/demo_architectural_energy.py):
- Comprehensive architectural comparison
- Shows energy progression: CPU → GPU → TPU → KPU
- Explains WHY each architecture differs

### What's Missing for EDP Analysis

❌ No EDP (Energy-Delay Product) calculation
❌ No EDP-based comparisons or visualizations
❌ No MLP-specific validation harness
❌ Old estimateEDP.py experiment uses deprecated API
❌ No EDP-aware explanations in architectural models
❌ No EDP breakdown (compute_edp, memory_edp, architectural_edp)

---

## Proposed Plan: EDP-Based Architectural Comparison Framework

### Phase 1: Add EDP Metrics (Minimal Changes, Preserve Existing)

**Goal:** Add EDP calculations without breaking existing functionality

#### 1.1 Add EDP to Core Data Structures

Location: `src/graphs/analysis/architecture_comparator.py`

**Changes to `ArchitectureMetrics`:**
```python
@dataclass
class ArchitectureMetrics:
    # ... existing fields ...

    # NEW: EDP metrics
    edp: float = 0.0  # Energy-Delay Product (J·s)
    edp_normalized: float = 0.0  # Normalized to baseline

    # Energy breakdown for EDP analysis
    compute_edp: float = 0.0  # Compute energy × latency
    memory_edp: float = 0.0   # Memory energy × latency
    architectural_edp: float = 0.0  # Architectural overhead × latency
```

**Rationale:** EDP captures the complete efficiency story. A system can have low energy but high latency (slow), or high energy but low latency (wasteful). EDP = Energy × Delay identifies the optimal balance.

#### 1.2 Add EDP to ComparisonSummary

```python
@dataclass
class ComparisonSummary:
    # ... existing fields ...

    # NEW: EDP winner
    edp_winner: str  # Best EDP (lowest)
    edp_ratios: Dict[str, float]  # EDP ratios vs baseline
```

#### 1.3 Update `_extract_metrics()` to Calculate EDP

```python
def _extract_metrics(self, name: str, result: UnifiedAnalysisResult) -> ArchitectureMetrics:
    # ... existing calculations ...

    # Calculate EDP
    energy_j = result.energy_per_inference_mj / 1000.0
    latency_s = result.total_latency_ms / 1000.0
    edp = energy_j * latency_s

    # Component EDPs
    compute_edp = compute_energy * latency_s
    memory_edp = memory_energy * latency_s
    architectural_edp = architectural_overhead * latency_s

    return ArchitectureMetrics(
        # ... existing fields ...
        edp=edp,
        compute_edp=compute_edp,
        memory_edp=memory_edp,
        architectural_edp=architectural_edp,
    )
```

#### 1.4 Update `_generate_summary()` to Find EDP Winner

```python
def _generate_summary(self) -> ComparisonSummary:
    # ... existing winner calculations ...

    # EDP winner: minimize energy × latency product
    edp_winner = min(self.metrics.items(), key=lambda x: x[1].edp)[0]

    # Calculate EDP ratios
    baseline_edp = self.metrics[baseline].edp
    edp_ratios = {
        name: metrics.edp / baseline_edp
        for name, metrics in self.metrics.items()
    }

    return ComparisonSummary(
        # ... existing fields ...
        edp_winner=edp_winner,
        edp_ratios=edp_ratios,
    )
```

**Deliverables:**
- ✅ EDP calculated for all architectures
- ✅ EDP winner identified
- ✅ EDP ratios computed vs baseline
- ✅ No breaking changes to existing API

---

### Phase 2: EDP-Focused Reporting

**Goal:** Create EDP-centric views and explanations

#### 2.1 Add EDP to Summary Report

Update `generate_summary()`:
```python
# Recommendations section
lines.append(f"  Best for Energy:      {self.summary.energy_winner}")
lines.append(f"  Best for Latency:     {self.summary.latency_winner}")
lines.append(f"  Best EDP (E×D):       {self.summary.edp_winner}")  # NEW
lines.append(f"  Best Balance:         {self.summary.balance_winner}")

# Add EDP comparison table
lines.append("")
lines.append("Energy-Delay Product (EDP) Comparison:")
lines.append(f"{'Architecture':<12} {'EDP (nJ·s)':<15} {'vs Baseline':<12} {'Breakdown'}")
lines.append("-" * 80)

for name in sorted(self.metrics.keys()):
    metrics = self.metrics[name]
    edp_nj_s = metrics.edp * 1e9  # Convert J·s to nJ·s
    edp_ratio = self.summary.edp_ratios[name]

    # Breakdown percentages
    total = metrics.compute_edp + metrics.memory_edp + metrics.architectural_edp
    if total > 0:
        compute_pct = metrics.compute_edp / total * 100
        memory_pct = metrics.memory_edp / total * 100
        arch_pct = metrics.architectural_edp / total * 100
        breakdown = f"C:{compute_pct:.0f}% M:{memory_pct:.0f}% A:{arch_pct:.0f}%"
    else:
        breakdown = "—"

    lines.append(f"{name:<12} {edp_nj_s:<15.3f} {edp_ratio:.2f}× {breakdown}")
```

**Output Example:**
```
Energy-Delay Product (EDP) Comparison:
Architecture     EDP (nJ·s)       vs Baseline    Breakdown
--------------------------------------------------------------------------------
CPU              45.200           1.00×          C:40% M:50% A:10%
GPU              67.800           1.50×          C:35% M:45% A:20%
TPU               8.940           0.20×          C:60% M:35% A:5%
KPU              13.560           0.30×          C:55% M:38% A:7%
```

#### 2.2 Create EDP Explanation Method

```python
def explain_edp_difference(self, arch1: str, arch2: str) -> str:
    """
    Explain EDP difference through energy AND latency breakdown.

    EDP captures the full efficiency story:
    - Low energy + high latency → mediocre EDP (slow)
    - High energy + low latency → mediocre EDP (wasteful)
    - Low energy + low latency → excellent EDP (efficient)

    Args:
        arch1: First architecture name
        arch2: Second architecture name

    Returns:
        Human-readable explanation of EDP difference
    """
    if arch1 not in self.metrics or arch2 not in self.metrics:
        raise ValueError(f"Architecture not found: {arch1} or {arch2}")

    m1 = self.metrics[arch1]
    m2 = self.metrics[arch2]

    lines = []
    lines.append("=" * 80)
    lines.append(f"EDP Analysis: {arch1} vs {arch2}")
    lines.append("=" * 80)
    lines.append("")

    # EDP comparison
    edp_ratio = m1.edp / m2.edp
    if edp_ratio > 1.0:
        lines.append(f"{arch1} has {edp_ratio:.1f}× WORSE EDP than {arch2}")
        winner = arch2
        loser = arch1
    else:
        lines.append(f"{arch1} has {1.0/edp_ratio:.1f}× BETTER EDP than {arch2}")
        winner = arch1
        loser = arch2

    lines.append("")
    lines.append("Breakdown:")
    lines.append(f"  {arch1}:")
    lines.append(f"    Energy:  {m1.total_energy_j*1e6:.2f} µJ")
    lines.append(f"    Latency: {m1.total_latency_s*1e3:.2f} ms")
    lines.append(f"    EDP:     {m1.edp*1e9:.2f} nJ·s")
    lines.append("")
    lines.append(f"  {arch2}:")
    lines.append(f"    Energy:  {m2.total_energy_j*1e6:.2f} µJ")
    lines.append(f"    Latency: {m2.total_latency_s*1e3:.2f} ms")
    lines.append(f"    EDP:     {m2.edp*1e9:.2f} nJ·s")
    lines.append("")

    # EDP breakdown analysis
    lines.append("EDP Component Breakdown:")
    lines.append(f"  {arch1}:")
    lines.append(f"    Compute EDP:        {m1.compute_edp*1e9:.2f} nJ·s")
    lines.append(f"    Memory EDP:         {m1.memory_edp*1e9:.2f} nJ·s")
    lines.append(f"    Architectural EDP:  {m1.architectural_edp*1e9:.2f} nJ·s")
    lines.append("")
    lines.append(f"  {arch2}:")
    lines.append(f"    Compute EDP:        {m2.compute_edp*1e9:.2f} nJ·s")
    lines.append(f"    Memory EDP:         {m2.memory_edp*1e9:.2f} nJ·s")
    lines.append(f"    Architectural EDP:  {m2.architectural_edp*1e9:.2f} nJ·s")
    lines.append("")

    # Explanation of why EDP differs
    lines.append(f"Why {winner} Wins EDP:")

    m_winner = m1 if winner == arch1 else m2
    m_loser = m1 if loser == arch1 else m2

    energy_better = m_winner.total_energy_j < m_loser.total_energy_j
    latency_better = m_winner.total_latency_s < m_loser.total_latency_s

    if energy_better and latency_better:
        lines.append(f"  {winner} is BOTH more energy efficient AND faster!")
        lines.append(f"  This is the ideal case - no trade-off needed.")
    elif energy_better:
        lines.append(f"  {winner} is more energy efficient, offsetting slower latency.")
        energy_ratio = m_loser.total_energy_j / m_winner.total_energy_j
        latency_ratio = m_winner.total_latency_s / m_loser.total_latency_s
        lines.append(f"  Energy advantage: {energy_ratio:.2f}×")
        lines.append(f"  Latency penalty:  {latency_ratio:.2f}×")
        lines.append(f"  Net EDP gain:     {1.0/edp_ratio:.2f}×")
    else:
        lines.append(f"  {winner} is faster, offsetting higher energy consumption.")
        energy_ratio = m_winner.total_energy_j / m_loser.total_energy_j
        latency_ratio = m_loser.total_latency_s / m_winner.total_latency_s
        lines.append(f"  Latency advantage: {latency_ratio:.2f}×")
        lines.append(f"  Energy penalty:    {energy_ratio:.2f}×")
        lines.append(f"  Net EDP gain:      {1.0/edp_ratio:.2f}×")

    lines.append("")
    lines.append("Architectural Insights:")

    # Use architectural breakdowns if available
    if m_winner.architectural_breakdown and m_loser.architectural_breakdown:
        b_winner = m_winner.architectural_breakdown
        b_loser = m_loser.architectural_breakdown

        control_diff = b_loser.control_overhead - b_winner.control_overhead
        if control_diff > 0:
            lines.append(f"  • {winner} saves {control_diff*1e9:.2f} nJ in control overhead")
            lines.append(f"    (eliminates instruction fetch, scheduling machinery)")

        memory_diff = b_loser.data_movement_overhead - b_winner.data_movement_overhead
        if memory_diff > 0:
            lines.append(f"  • {winner} saves {memory_diff*1e9:.2f} nJ in memory overhead")
            lines.append(f"    (more efficient memory access patterns)")

    lines.append("")
    return "\n".join(lines)
```

#### 2.3 Add EDP Visualization to HTML Export

Update `export_html()`:

**Add EDP Chart:**
```javascript
// EDP Chart (new)
new Chart(document.getElementById('edpChart'), {
    type: 'bar',
    data: {
        labels: arch_names,
        datasets: [{
            label: 'EDP (nJ·s)',
            data: edps,  // Array of EDP values
            backgroundColor: colors,
            borderColor: borderColors,
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: false },
            title: {
                display: true,
                text: 'Energy-Delay Product (Lower is Better)'
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: 'EDP (nJ·s)' }
            }
        }
    }
});
```

**Add EDP Breakdown Stacked Bar:**
```javascript
// EDP Breakdown (stacked bar)
new Chart(document.getElementById('edpBreakdownChart'), {
    type: 'bar',
    data: {
        labels: arch_names,
        datasets: [
            {
                label: 'Compute EDP',
                data: compute_edps,
                backgroundColor: 'rgba(255, 99, 132, 0.8)'
            },
            {
                label: 'Memory EDP',
                data: memory_edps,
                backgroundColor: 'rgba(54, 162, 235, 0.8)'
            },
            {
                label: 'Architectural EDP',
                data: architectural_edps,
                backgroundColor: 'rgba(255, 206, 86, 0.8)'
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            x: { stacked: true },
            y: { stacked: true, beginAtZero: true }
        }
    }
});
```

**Add Energy vs Latency Scatter with EDP Iso-lines:**
```javascript
// Energy vs Latency Scatter (with EDP iso-lines)
const scatterData = arch_names.map((name, i) => ({
    x: latencies[i],  // ms
    y: energies[i],   // mJ
    label: name
}));

new Chart(document.getElementById('energyLatencyScatter'), {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Architectures',
            data: scatterData,
            backgroundColor: colors,
            pointRadius: 8
        }]
    },
    options: {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: 'Energy vs Latency Trade-off (Diagonal lines = constant EDP)'
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const point = context.raw;
                        const edp = point.y * point.x;  // mJ * ms = nJ·s
                        return `${point.label}: ${point.y.toFixed(2)} mJ, ${point.x.toFixed(2)} ms, EDP=${edp.toFixed(2)} nJ·s`;
                    }
                }
            }
        },
        scales: {
            x: {
                title: { display: true, text: 'Latency (ms)' },
                type: 'logarithmic'
            },
            y: {
                title: { display: true, text: 'Energy (mJ)' },
                type: 'logarithmic'
            }
        }
    }
});
```

**Deliverables:**
- ✅ EDP in summary report with breakdown
- ✅ `explain_edp_difference()` method
- ✅ HTML visualizations: EDP bar, EDP breakdown, energy-latency scatter

---

### Phase 3: MLP Validation Harness

**Goal:** Create trusted test case for validation

#### 3.1 Create Parameterized MLP Test Suite

Location: `validation/energy/test_mlp_edp.py`

```python
"""
MLP EDP Validation Suite

Tests EDP calculations using parameterized MLP as trusted baseline.

MLP Structure:
- Linear layer: matmul (input @ weight) + bias
- ReLU activation: max(0, x)
- Optional: Softmax (separate analysis)

Validation Approach:
1. Analytical FLOP/byte calculation
2. Compare EDP ordering across architectures
3. Validate expected ratios (TPU < KPU < CPU < GPU)
4. Check architectural energy event explanations
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, List

from graphs.models.mlp import make_mlp
from graphs.analysis.architecture_comparator import ArchitectureComparator
from graphs.hardware.mappers.cpu import intel_xeon_platinum_8490h_mapper
from graphs.hardware.mappers.gpu import h100_pcie_mapper
from graphs.hardware.mappers.accelerators.tpu import tpu_v4_mapper
from graphs.hardware.mappers.accelerators.kpu import kpu_t256_mapper
from graphs.hardware.resource_model import Precision


class MLPEDPValidator(unittest.TestCase):
    """Validates EDP calculations using parameterized MLP"""

    def setUp(self):
        """Setup architecture mappers"""
        self.architectures = {
            'CPU': intel_xeon_platinum_8490h_mapper(),
            'GPU': h100_pcie_mapper(),
            'TPU': tpu_v4_mapper(),
            'KPU': kpu_t256_mapper(),
        }

    def calculate_expected_flops(self, input_dim: int, hidden_dim: int, output_dim: int) -> int:
        """
        Calculate expected FLOPs for MLP.

        Linear1: matmul (input_dim × hidden_dim × 2) + bias (hidden_dim)
        ReLU: hidden_dim comparisons (negligible FLOPs)
        Linear2: matmul (hidden_dim × output_dim × 2) + bias (output_dim)
        """
        # Linear1: input @ weight^T + bias
        linear1_matmul = input_dim * hidden_dim * 2  # 2 ops per MAC
        linear1_bias = hidden_dim

        # ReLU: max(0, x) - negligible
        relu_ops = 0

        # Linear2: hidden @ weight^T + bias
        linear2_matmul = hidden_dim * output_dim * 2
        linear2_bias = output_dim

        total_flops = linear1_matmul + linear1_bias + relu_ops + linear2_matmul + linear2_bias
        return total_flops

    def calculate_expected_params(self, input_dim: int, hidden_dim: int, output_dim: int) -> int:
        """Calculate expected parameters"""
        linear1_params = input_dim * hidden_dim + hidden_dim  # weight + bias
        linear2_params = hidden_dim * output_dim + output_dim
        return linear1_params + linear2_params

    def test_mlp_128_256_64(self):
        """Test MLP(128, 256, 64) - small network"""
        input_dim = 128
        hidden_dim = 256
        output_dim = 64

        # Expected values
        expected_flops = self.calculate_expected_flops(input_dim, hidden_dim, output_dim)
        expected_params = self.calculate_expected_params(input_dim, hidden_dim, output_dim)

        print(f"\nMLP({input_dim}, {hidden_dim}, {output_dim})")
        print(f"  Expected FLOPs: {expected_flops:,}")
        print(f"  Expected Params: {expected_params:,}")

        # Create model
        model = make_mlp(in_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim)
        input_tensor = torch.randn(1, input_dim)  # Batch size 1

        # Run comparison
        comparator = ArchitectureComparator(
            model_name=f"MLP_{input_dim}_{hidden_dim}_{output_dim}",
            architectures=self.architectures,
            batch_size=1,
            precision=Precision.FP32
        )

        comparator.analyze_all()

        # Validate EDP ordering: TPU < KPU < CPU < GPU (expected for small batch)
        edps = {name: metrics.edp for name, metrics in comparator.metrics.items()}

        print(f"\nEDP Results:")
        for name in ['CPU', 'GPU', 'TPU', 'KPU']:
            edp_nj_s = edps[name] * 1e9
            print(f"  {name}: {edp_nj_s:.2f} nJ·s")

        # Assertions
        self.assertLess(edps['TPU'], edps['KPU'], "TPU should have lower EDP than KPU")
        self.assertLess(edps['KPU'], edps['CPU'], "KPU should have lower EDP than CPU")

        # GPU may be worse than CPU at batch=1 due to coherence overhead
        # This is expected behavior!

        # Check ratios
        tpu_vs_gpu = edps['TPU'] / edps['GPU']
        kpu_vs_gpu = edps['KPU'] / edps['GPU']

        print(f"\nRatios:")
        print(f"  TPU vs GPU: {tpu_vs_gpu:.2f}× (expect 0.10-0.20×)")
        print(f"  KPU vs GPU: {kpu_vs_gpu:.2f}× (expect 0.25-0.40×)")

        # Validate ratios are in expected range
        self.assertGreater(tpu_vs_gpu, 0.05, "TPU should be at least 20× better than GPU")
        self.assertLess(tpu_vs_gpu, 0.30, "TPU advantage should be realistic")

        self.assertGreater(kpu_vs_gpu, 0.15, "KPU should be at least 2.5× better than GPU")
        self.assertLess(kpu_vs_gpu, 0.50, "KPU advantage should be realistic")

    def test_mlp_edp_breakdown(self):
        """Test EDP breakdown into compute, memory, architectural components"""
        model = make_mlp(in_dim=128, hidden_dim=256, out_dim=64)
        input_tensor = torch.randn(1, 128)

        comparator = ArchitectureComparator(
            model_name="MLP_test",
            architectures=self.architectures,
            batch_size=1,
            precision=Precision.FP32
        )

        comparator.analyze_all()

        print(f"\nEDP Breakdown:")
        for name, metrics in comparator.metrics.items():
            total_edp = metrics.edp * 1e9  # nJ·s
            compute_edp = metrics.compute_edp * 1e9
            memory_edp = metrics.memory_edp * 1e9
            arch_edp = metrics.architectural_edp * 1e9

            print(f"\n{name}:")
            print(f"  Total EDP:         {total_edp:.2f} nJ·s")
            print(f"  Compute EDP:       {compute_edp:.2f} nJ·s ({compute_edp/total_edp*100:.1f}%)")
            print(f"  Memory EDP:        {memory_edp:.2f} nJ·s ({memory_edp/total_edp*100:.1f}%)")
            print(f"  Architectural EDP: {arch_edp:.2f} nJ·s ({arch_edp/total_edp*100:.1f}%)")

            # Validate breakdown sums to total
            computed_total = compute_edp + memory_edp + arch_edp
            self.assertAlmostEqual(computed_total, total_edp, delta=0.01,
                                   msg=f"{name} EDP breakdown doesn't sum correctly")

    def test_batch_size_scaling(self):
        """Test that EDP scales appropriately with batch size"""
        model = make_mlp(in_dim=128, hidden_dim=256, out_dim=64)

        batch_sizes = [1, 8, 32]
        gpu_edps = []

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 128)

            comparator = ArchitectureComparator(
                model_name="MLP_batch_test",
                architectures={'GPU': self.architectures['GPU']},
                batch_size=batch_size,
                precision=Precision.FP32
            )

            comparator.analyze_all()
            gpu_edps.append(comparator.metrics['GPU'].edp)

        print(f"\nBatch Size Scaling (GPU):")
        for i, batch_size in enumerate(batch_sizes):
            edp = gpu_edps[i] * 1e9
            print(f"  Batch {batch_size}: {edp:.2f} nJ·s")

        # GPU should show better efficiency at larger batch sizes
        # EDP per sample should decrease
        edp_per_sample = [gpu_edps[i] / batch_sizes[i] for i in range(len(batch_sizes))]

        print(f"\nEDP per Sample:")
        for i, batch_size in enumerate(batch_sizes):
            edp_ps = edp_per_sample[i] * 1e9
            print(f"  Batch {batch_size}: {edp_ps:.2f} nJ·s per sample")

        # GPU should have lower EDP per sample at batch=32 vs batch=1
        self.assertLess(edp_per_sample[2], edp_per_sample[0],
                        "GPU EDP per sample should improve at larger batch sizes")


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

#### 3.2 Baseline Validation Dataset

Location: `validation/energy/mlp_baselines.json`

```json
{
  "mlp_128_256_64_b1": {
    "config": {
      "input_dim": 128,
      "hidden_dim": 256,
      "output_dim": 64,
      "batch_size": 1
    },
    "expected_flops": 98304,
    "expected_params": 98560,
    "expected_edp_order": ["TPU", "KPU", "CPU", "GPU"],
    "expected_ratios": {
      "TPU_vs_GPU": {"min": 0.05, "max": 0.30},
      "KPU_vs_GPU": {"min": 0.15, "max": 0.50},
      "CPU_vs_GPU": {"min": 0.60, "max": 1.20}
    },
    "notes": "Small batch (b=1) favors fixed-function accelerators. GPU coherence overhead dominates."
  },
  "mlp_128_256_64_b32": {
    "config": {
      "input_dim": 128,
      "hidden_dim": 256,
      "output_dim": 64,
      "batch_size": 32
    },
    "expected_flops": 3145728,
    "expected_params": 98560,
    "expected_edp_order": ["TPU", "GPU", "KPU", "CPU"],
    "expected_ratios": {
      "TPU_vs_GPU": {"min": 0.10, "max": 0.30},
      "GPU_vs_CPU": {"min": 0.30, "max": 0.70},
      "KPU_vs_GPU": {"min": 0.80, "max": 1.50}
    },
    "notes": "Large batch (b=32) amortizes GPU coherence overhead. GPU becomes competitive."
  },
  "mlp_512_1024_256_b1": {
    "config": {
      "input_dim": 512,
      "hidden_dim": 1024,
      "output_dim": 256,
      "batch_size": 1
    },
    "expected_flops": 1573888,
    "expected_params": 1049856,
    "expected_edp_order": ["TPU", "KPU", "CPU", "GPU"],
    "expected_ratios": {
      "TPU_vs_GPU": {"min": 0.05, "max": 0.25},
      "KPU_vs_GPU": {"min": 0.15, "max": 0.45}
    },
    "notes": "Larger MLP, small batch. Same ordering as 128-256-64."
  }
}
```

**Deliverables:**
- ✅ MLP EDP validation test suite
- ✅ Baseline dataset with expected orderings and ratios
- ✅ Batch size scaling tests
- ✅ EDP breakdown validation

---

### Phase 4: EDP-Focused CLI Tool

**Goal:** Make EDP comparison easily accessible

#### 4.1 Create `cli/compare_edp.py`

```python
#!/usr/bin/env python3
"""
EDP-Focused Multi-Architecture Comparison

Analyzes Energy-Delay Product to identify optimal architecture
for a given model and workload characteristics.

Usage:
    # MLP comparison (for validation)
    ./cli/compare_edp.py --model mlp --input-dim 128 --hidden-dim 256 --output-dim 64

    # Standard models
    ./cli/compare_edp.py --model resnet18 --batch-size 1
    ./cli/compare_edp.py --model mobilenet_v2 --batch-size 8

    # Custom architecture selection
    ./cli/compare_edp.py --model resnet18 --architectures CPU GPU TPU KPU

    # Output formats
    ./cli/compare_edp.py --model resnet18 --output edp_report.html
    ./cli/compare_edp.py --model resnet18 --output edp_report.json
    ./cli/compare_edp.py --model resnet18 --output edp_report.csv

    # EDP explanation
    ./cli/compare_edp.py --model resnet18 --explain GPU TPU
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.models as models

from graphs.models.mlp import make_mlp
from graphs.analysis.architecture_comparator import ArchitectureComparator
from graphs.hardware.mappers.cpu import intel_xeon_platinum_8490h_mapper
from graphs.hardware.mappers.gpu import h100_pcie_mapper
from graphs.hardware.mappers.accelerators.tpu import tpu_v4_mapper
from graphs.hardware.mappers.accelerators.kpu import kpu_t256_mapper
from graphs.hardware.resource_model import Precision


def get_architecture_mappers(arch_names=None):
    """Get architecture mappers"""
    all_mappers = {
        'CPU': intel_xeon_platinum_8490h_mapper(),
        'GPU': h100_pcie_mapper(),
        'TPU': tpu_v4_mapper(),
        'KPU': kpu_t256_mapper(),
    }

    if arch_names:
        return {name: all_mappers[name] for name in arch_names if name in all_mappers}

    return all_mappers


def create_mlp_model(input_dim, hidden_dim, output_dim, batch_size):
    """Create MLP model and input tensor"""
    model = make_mlp(in_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim)
    input_tensor = torch.randn(batch_size, input_dim)
    model_name = f"MLP_{input_dim}_{hidden_dim}_{output_dim}"
    return model, input_tensor, model_name


def create_standard_model(model_name, batch_size):
    """Create standard torchvision model"""
    try:
        model = getattr(models, model_name)(pretrained=False)
        model.eval()
    except AttributeError:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    # Standard ImageNet input
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    return model, input_tensor, model_name


def main():
    parser = argparse.ArgumentParser(
        description="EDP-focused multi-architecture comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MLP comparison
  %(prog)s --model mlp --input-dim 128 --hidden-dim 256 --output-dim 64

  # Standard models
  %(prog)s --model resnet18 --batch-size 1
  %(prog)s --model mobilenet_v2 --batch-size 8 --output report.html

  # EDP explanation
  %(prog)s --model resnet18 --explain GPU TPU
        """
    )

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (mlp, resnet18, mobilenet_v2, etc.)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')

    # MLP-specific parameters
    parser.add_argument('--input-dim', type=int,
                        help='MLP input dimension (required if --model mlp)')
    parser.add_argument('--hidden-dim', type=int,
                        help='MLP hidden dimension (required if --model mlp)')
    parser.add_argument('--output-dim', type=int,
                        help='MLP output dimension (required if --model mlp)')

    # Architecture selection
    parser.add_argument('--architectures', type=str, nargs='+',
                        choices=['CPU', 'GPU', 'TPU', 'KPU'],
                        help='Architectures to compare (default: all)')

    # Precision
    parser.add_argument('--precision', type=str, default='FP32',
                        choices=['FP32', 'FP16', 'BF16', 'INT8'],
                        help='Numerical precision (default: FP32)')

    # Output
    parser.add_argument('--output', type=str,
                        help='Output file (auto-detect format from extension: .html, .json, .csv, .md)')

    # Explanation mode
    parser.add_argument('--explain', type=str, nargs=2, metavar=('ARCH1', 'ARCH2'),
                        help='Explain EDP difference between two architectures')

    args = parser.parse_args()

    # Validate MLP parameters
    if args.model == 'mlp':
        if not all([args.input_dim, args.hidden_dim, args.output_dim]):
            parser.error("--model mlp requires --input-dim, --hidden-dim, --output-dim")

    # Create model
    print(f"Creating model: {args.model}")
    if args.model == 'mlp':
        model, input_tensor, model_name = create_mlp_model(
            args.input_dim, args.hidden_dim, args.output_dim, args.batch_size
        )
    else:
        model, input_tensor, model_name = create_standard_model(args.model, args.batch_size)

    print(f"Batch size: {args.batch_size}")
    print(f"Precision: {args.precision}")
    print()

    # Get architectures
    architectures = get_architecture_mappers(args.architectures)
    print(f"Comparing architectures: {', '.join(architectures.keys())}")
    print()

    # Run comparison
    precision = Precision[args.precision]
    comparator = ArchitectureComparator(
        model_name=model_name,
        architectures=architectures,
        batch_size=args.batch_size,
        precision=precision
    )

    comparator.analyze_all()

    # Generate output
    if args.explain:
        # Explanation mode
        arch1, arch2 = args.explain
        if arch1 not in architectures or arch2 not in architectures:
            print(f"Error: Both architectures must be in comparison set")
            sys.exit(1)

        print(comparator.explain_edp_difference(arch1, arch2))

    elif args.output:
        # File output
        output_path = Path(args.output)
        ext = output_path.suffix.lower()

        print(f"Generating report: {args.output}")

        if ext == '.html':
            content = comparator.export_html()
        elif ext == '.json':
            content = comparator.export_json()
        elif ext == '.csv':
            content = comparator.export_csv()
        elif ext in ['.md', '.markdown']:
            content = comparator.generate_summary()
        else:
            print(f"Error: Unknown output format '{ext}'")
            sys.exit(1)

        output_path.write_text(content)
        print(f"Report saved to: {args.output}")

    else:
        # Console output (summary)
        print(comparator.generate_summary())
        print()

        # EDP-specific insights
        print("=" * 80)
        print("EDP-FOCUSED RECOMMENDATIONS")
        print("=" * 80)
        print()

        edp_winner = comparator.summary.edp_winner
        energy_winner = comparator.summary.energy_winner
        latency_winner = comparator.summary.latency_winner

        if edp_winner == energy_winner == latency_winner:
            print(f"✓ {edp_winner} is the CLEAR WINNER - best energy, latency, and EDP!")
        elif edp_winner == energy_winner:
            print(f"✓ {edp_winner} wins EDP through superior energy efficiency")
            print(f"  Trade-off: Slower than {latency_winner}, but energy savings dominate")
        elif edp_winner == latency_winner:
            print(f"✓ {edp_winner} wins EDP through superior latency")
            print(f"  Trade-off: Uses more energy than {energy_winner}, but speed compensates")
        else:
            print(f"✓ {edp_winner} wins EDP through optimal BALANCE of energy and latency")
            print(f"  Not the fastest ({latency_winner}) or most efficient ({energy_winner}),")
            print(f"  but achieves the best energy-latency product")

        print()
        print(f"Use --explain {edp_winner} <other> to understand why {edp_winner} wins")
        print(f"Use --output report.html for interactive visualizations")
        print()


if __name__ == '__main__':
    main()
```

**Deliverables:**
- ✅ CLI tool for EDP comparison
- ✅ MLP parameterization support
- ✅ Standard model support
- ✅ Multi-format output (HTML, JSON, CSV, Markdown)
- ✅ EDP explanation mode
- ✅ Recommendation engine

---

### Phase 5: Enhanced Architectural Energy Features

**Goal:** Improve explanatory power

#### 5.1 Add EDP Breakdown to ArchitecturalEnergyBreakdown

Location: `src/graphs/hardware/architectural_energy.py`

```python
@dataclass
class ArchitecturalEnergyBreakdown:
    """
    Result of architectural energy calculation.

    Contains energy overheads (positive = cost, negative = savings)
    and human-readable explanation.
    """
    compute_overhead: float  # Additional compute energy (Joules)
    data_movement_overhead: float   # Additional memory energy (Joules)
    control_overhead: float  # Control/coordination energy (Joules)

    # Additional details for specific architectures
    extra_details: Dict[str, float] = field(default_factory=dict)

    # Human-readable explanation
    explanation: str = ""

    # NEW: EDP-aware metrics (optional, computed externally)
    latency_s: Optional[float] = None

    @property
    def total_overhead(self) -> float:
        """Total architectural overhead"""
        return self.compute_overhead + self.data_movement_overhead + self.control_overhead

    @property
    def edp_contribution(self) -> float:
        """
        EDP contribution from architectural overhead.

        Returns:
            Overhead × latency (J·s) if latency available, else 0
        """
        if self.latency_s is not None:
            return self.total_overhead * self.latency_s
        return 0.0

    @property
    def compute_edp(self) -> float:
        """Compute overhead EDP component"""
        if self.latency_s is not None:
            return self.compute_overhead * self.latency_s
        return 0.0

    @property
    def memory_edp(self) -> float:
        """Memory overhead EDP component"""
        if self.latency_s is not None:
            return self.data_movement_overhead * self.latency_s
        return 0.0

    @property
    def control_edp(self) -> float:
        """Control overhead EDP component"""
        if self.latency_s is not None:
            return self.control_overhead * self.latency_s
        return 0.0
```

#### 5.2 Enhance Explanation with EDP Context

For each architectural energy model, add EDP-aware explanations to the `explanation` field:

**Example for DataParallelEnergyModel:**

```python
# Existing explanation
explanation = (
    f"Data Parallel (GPU SIMT) Architecture Energy Events:\n"
    # ... existing content ...
)

# NEW: Add EDP context if latency provided
if execution_context.get('latency_s'):
    latency_s = execution_context['latency_s']
    edp_contribution = (control_overhead + total_data_movement_overhead) * latency_s

    explanation += f"\n"
    explanation += f"EDP IMPACT (Energy × Latency):\n"
    explanation += f"  Latency: {latency_s*1e3:.2f} ms\n"
    explanation += f"  Energy overhead: {(control_overhead + total_data_movement_overhead)*1e9:.2f} nJ\n"
    explanation += f"  EDP contribution: {edp_contribution*1e9:.2f} nJ·s\n"
    explanation += f"\n"
    explanation += f"  This architecture's energy events contribute {edp_contribution*1e9:.2f} nJ·s to EDP.\n"
    explanation += f"  For workloads where both energy and latency matter, this is the key metric.\n"
    explanation += f"  At small batch sizes, this overhead dominates the EDP calculation.\n"
```

**Apply to all models:**
- StoredProgramEnergyModel
- DataParallelEnergyModel
- SystolicArrayEnergyModel
- DomainFlowEnergyModel
- DataFlowMachineEnergyModel
- SpatialPartitionEnergyModel
- AdaptiveDatapathEnergyModel

**Deliverables:**
- ✅ EDP properties in ArchitecturalEnergyBreakdown
- ✅ EDP-aware explanations in all architectural models
- ✅ Latency-aware energy reporting

---

### Phase 6: Softmax Special Case Analysis

**Goal:** Handle complex operations separately

#### 6.1 Create Softmax Energy Analyzer

Location: `src/graphs/analysis/softmax_energy.py`

```python
"""
Softmax Energy Analysis

Softmax presents unique architectural challenges:
- CPU: Good (sequential, complex math units available)
- GPU: Moderate (exp() causes divergence, reduction overhead)
- TPU: Poor (no exp() in systolic array, falls back to scalar)
- KPU: Moderate (wavefront scheduling helps with reduction)

Components:
1. Max reduction: O(n) comparisons
2. Exp: O(n) transcendental operations
3. Sum reduction: O(n) additions
4. Division: O(n) divisions

This module quantifies WHY softmax behaves differently across architectures.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

from graphs.hardware.resource_model import HardwareResourceModel
from graphs.hardware.architectural_energy import ArchitecturalEnergyModel


@dataclass
class SoftmaxEnergyBreakdown:
    """Energy breakdown for softmax operation"""

    # Component energies (Joules)
    max_reduction_energy: float
    exp_energy: float
    sum_reduction_energy: float
    division_energy: float
    total_energy: float

    # Component latencies (seconds)
    max_reduction_latency: float
    exp_latency: float
    sum_reduction_latency: float
    division_latency: float
    total_latency: float

    # Component EDPs (J·s)
    max_reduction_edp: float
    exp_edp: float
    sum_reduction_edp: float
    division_edp: float
    total_edp: float

    # Bottleneck
    bottleneck_operation: str
    bottleneck_reason: str

    # Architecture-specific challenges
    architecture_challenges: str


class SoftmaxEnergyAnalyzer:
    """
    Analyzes softmax energy across different hardware architectures.

    Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Steps:
    1. Max reduction: max_val = max(x)
    2. Shift and exp: y = exp(x - max_val)
    3. Sum reduction: sum_val = sum(y)
    4. Normalization: output = y / sum_val
    """

    def __init__(self, hardware: HardwareResourceModel, arch_energy_model: ArchitecturalEnergyModel):
        self.hardware = hardware
        self.arch_energy_model = arch_energy_model

    def analyze(self, batch_size: int, num_classes: int) -> SoftmaxEnergyBreakdown:
        """
        Analyze softmax energy for given shape.

        Args:
            batch_size: Batch size
            num_classes: Number of classes (softmax dimension)

        Returns:
            SoftmaxEnergyBreakdown with component-level analysis
        """
        total_elements = batch_size * num_classes

        # 1. Max reduction
        max_ops = total_elements  # Comparisons
        max_energy, max_latency = self._estimate_reduction_energy_latency(
            ops=max_ops,
            op_type='comparison'
        )

        # 2. Exp operations (THE BOTTLENECK on most architectures!)
        exp_ops = total_elements
        exp_energy, exp_latency = self._estimate_transcendental_energy_latency(
            ops=exp_ops,
            op_type='exp'
        )

        # 3. Sum reduction
        sum_ops = total_elements
        sum_energy, sum_latency = self._estimate_reduction_energy_latency(
            ops=sum_ops,
            op_type='addition'
        )

        # 4. Division
        div_ops = total_elements
        div_energy, div_latency = self._estimate_arithmetic_energy_latency(
            ops=div_ops,
            op_type='division'
        )

        # Total
        total_energy = max_energy + exp_energy + sum_energy + div_energy
        total_latency = max_latency + exp_latency + sum_latency + div_latency

        # EDPs
        max_edp = max_energy * max_latency
        exp_edp = exp_energy * exp_latency
        sum_edp = sum_energy * sum_latency
        div_edp = div_energy * div_latency
        total_edp = total_energy * total_latency

        # Identify bottleneck
        component_edps = {
            'max_reduction': max_edp,
            'exp': exp_edp,
            'sum_reduction': sum_edp,
            'division': div_edp,
        }
        bottleneck_op = max(component_edps.items(), key=lambda x: x[1])[0]

        # Architecture-specific analysis
        challenges = self._analyze_architecture_challenges(bottleneck_op)

        return SoftmaxEnergyBreakdown(
            max_reduction_energy=max_energy,
            exp_energy=exp_energy,
            sum_reduction_energy=sum_energy,
            division_energy=div_energy,
            total_energy=total_energy,
            max_reduction_latency=max_latency,
            exp_latency=exp_latency,
            sum_reduction_latency=sum_latency,
            division_latency=div_latency,
            total_latency=total_latency,
            max_reduction_edp=max_edp,
            exp_edp=exp_edp,
            sum_reduction_edp=sum_edp,
            division_edp=div_edp,
            total_edp=total_edp,
            bottleneck_operation=bottleneck_op,
            bottleneck_reason=self._explain_bottleneck(bottleneck_op),
            architecture_challenges=challenges,
        )

    def _estimate_reduction_energy_latency(self, ops: int, op_type: str) -> Tuple[float, float]:
        """Estimate energy and latency for reduction operations (max, sum)"""
        # Simplified model - customize per architecture
        energy_per_op = 1.0e-12  # 1 pJ per operation
        latency_per_op = 1.0e-9  # 1 ns per operation

        # Reduction has log(n) depth in parallel
        # But different architectures handle this differently

        energy = ops * energy_per_op
        latency = ops * latency_per_op  # Sequential estimate

        return energy, latency

    def _estimate_transcendental_energy_latency(self, ops: int, op_type: str) -> Tuple[float, float]:
        """Estimate energy and latency for transcendental operations (exp, log)"""
        # Transcendentals are 10-100× more expensive than arithmetic

        if self.hardware.name.startswith('TPU'):
            # TPU doesn't have exp() in systolic array - must use scalar units
            energy_per_op = 50.0e-12  # 50 pJ per exp (very expensive!)
            latency_per_op = 50.0e-9  # 50 ns (slow scalar path)
        elif self.hardware.name.startswith('GPU'):
            # GPU has SFU (Special Function Units), but causes divergence
            energy_per_op = 10.0e-12  # 10 pJ per exp
            latency_per_op = 10.0e-9  # 10 ns
        else:
            # CPU has good transcendental support
            energy_per_op = 5.0e-12  # 5 pJ per exp
            latency_per_op = 5.0e-9  # 5 ns

        energy = ops * energy_per_op
        latency = ops * latency_per_op

        return energy, latency

    def _estimate_arithmetic_energy_latency(self, ops: int, op_type: str) -> Tuple[float, float]:
        """Estimate energy and latency for arithmetic operations (add, mul, div)"""
        energy_per_op = 1.0e-12  # 1 pJ
        latency_per_op = 1.0e-9  # 1 ns

        if op_type == 'division':
            # Division is 2-3× more expensive than multiplication
            energy_per_op *= 2.5
            latency_per_op *= 2.5

        energy = ops * energy_per_op
        latency = ops * latency_per_op

        return energy, latency

    def _explain_bottleneck(self, bottleneck_op: str) -> str:
        """Explain why this operation is the bottleneck"""
        explanations = {
            'max_reduction': "Max reduction requires sequential comparison or tree reduction",
            'exp': "Exp() is a transcendental operation, 10-100× more expensive than arithmetic",
            'sum_reduction': "Sum reduction requires sequential accumulation or tree reduction",
            'division': "Division is 2-3× more expensive than multiplication",
        }
        return explanations.get(bottleneck_op, "Unknown bottleneck")

    def _analyze_architecture_challenges(self, bottleneck_op: str) -> str:
        """Analyze architecture-specific challenges for softmax"""

        if self.hardware.name.startswith('CPU'):
            return (
                "CPU Advantages:\n"
                "  ✓ Excellent transcendental function support (hardware exp units)\n"
                "  ✓ Sequential reduction is natural fit\n"
                "  ✓ No synchronization overhead\n"
                "CPU Challenges:\n"
                "  ✗ Limited parallelism (8-16 cores)\n"
                "  ✗ Cannot hide latency effectively"
            )

        elif self.hardware.name.startswith('GPU'):
            return (
                "GPU Advantages:\n"
                "  ✓ SFU (Special Function Units) for transcendentals\n"
                "  ✓ Massive parallelism for element-wise ops\n"
                "GPU Challenges:\n"
                "  ✗ Exp() causes warp divergence (threads run at different speeds)\n"
                "  ✗ Reduction requires synchronization (expensive!)\n"
                "  ✗ Coherence machinery overhead at small batch sizes"
            )

        elif self.hardware.name.startswith('TPU'):
            return (
                "TPU Advantages:\n"
                "  ✓ Excellent for matrix multiply (systolic array)\n"
                "TPU Challenges:\n"
                "  ✗✗ NO exp() in systolic array!\n"
                "  ✗✗ Falls back to scalar units (very slow)\n"
                "  ✗✗ This is why TPUs struggle with attention/softmax!\n"
                "  ✗ Reduction not natural for 2D systolic flow"
            )

        elif self.hardware.name.startswith('KPU'):
            return (
                "KPU Advantages:\n"
                "  ✓ Wavefront scheduling can pipeline reductions\n"
                "  ✓ Domain tracking manages dependencies efficiently\n"
                "KPU Challenges:\n"
                "  ✗ Transcendental functions still expensive\n"
                "  ✗ Must configure datapath for exp() operation"
            )

        else:
            return "Architecture-specific analysis not available"


def compare_softmax_edp(
    batch_size: int,
    num_classes: int,
    architectures: Dict[str, Tuple[HardwareResourceModel, ArchitecturalEnergyModel]]
) -> str:
    """
    Compare softmax EDP across multiple architectures.

    Args:
        batch_size: Batch size
        num_classes: Number of classes
        architectures: Dict mapping name to (HardwareResourceModel, ArchitecturalEnergyModel)

    Returns:
        Human-readable comparison report
    """
    results = {}

    for name, (hardware, arch_model) in architectures.items():
        analyzer = SoftmaxEnergyAnalyzer(hardware, arch_model)
        results[name] = analyzer.analyze(batch_size, num_classes)

    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append(f"SOFTMAX EDP COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Shape: (batch={batch_size}, classes={num_classes})")
    lines.append(f"Total elements: {batch_size * num_classes:,}")
    lines.append("")

    lines.append(f"{'Architecture':<12} {'Total EDP':<15} {'Bottleneck':<20} {'EDP Ratio'}")
    lines.append("-" * 80)

    # Find baseline (GPU)
    baseline_name = 'GPU' if 'GPU' in results else list(results.keys())[0]
    baseline_edp = results[baseline_name].total_edp

    for name in sorted(results.keys()):
        breakdown = results[name]
        edp_ratio = breakdown.total_edp / baseline_edp

        lines.append(
            f"{name:<12} {breakdown.total_edp*1e9:<15.2f} nJ·s "
            f"{breakdown.bottleneck_operation:<20} {edp_ratio:.2f}×"
        )

    lines.append("")
    lines.append("Key Insights:")
    lines.append("  • Softmax is dominated by exp() transcendental operations")
    lines.append("  • TPU struggles with softmax (no exp() in systolic array)")
    lines.append("  • CPU has good transcendental support (hardware exp units)")
    lines.append("  • GPU reduction overhead hurts at small batch sizes")
    lines.append("")

    # Show per-component breakdown for each architecture
    for name in sorted(results.keys()):
        breakdown = results[name]
        lines.append(f"\n{name} Component Breakdown:")
        lines.append(f"  Max reduction: {breakdown.max_reduction_edp*1e9:.2f} nJ·s")
        lines.append(f"  Exp:           {breakdown.exp_edp*1e9:.2f} nJ·s ← {breakdown.bottleneck_operation == 'exp' and 'BOTTLENECK' or ''}")
        lines.append(f"  Sum reduction: {breakdown.sum_reduction_edp*1e9:.2f} nJ·s")
        lines.append(f"  Division:      {breakdown.division_edp*1e9:.2f} nJ·s")
        lines.append("")
        lines.append(breakdown.architecture_challenges)

    return "\n".join(lines)
```

**Deliverables:**
- ✅ Softmax energy analyzer with component breakdown
- ✅ Architecture-specific challenge analysis
- ✅ EDP comparison across architectures
- ✅ Explanation of WHY softmax behaves differently

---

## Implementation Priority

### Immediate (Week 1):
1. ✅ **Phase 1.1-1.4**: Add EDP calculations to ArchitectureComparator
2. ✅ **Phase 2.1**: Add EDP to summary report
3. ✅ **Phase 3.1**: Create basic MLP test harness

### Short-term (Week 2):
4. ✅ **Phase 2.2**: EDP explanation method
5. ✅ **Phase 3.2**: Baseline validation dataset
6. ✅ **Phase 4.1**: CLI tool for EDP comparison

### Medium-term (Week 3-4):
7. ✅ **Phase 2.3**: HTML visualization with EDP charts
8. ✅ **Phase 5.1-5.2**: Enhanced architectural energy with EDP context
9. ✅ **Phase 6.1**: Softmax special case analyzer

---

## Key Design Principles

1. **Preserve Existing Models**: All changes are additive, no breaking changes
2. **Two-Tier Approach**: Keep simple energy model for quick level-setting, use architectural model for deep analysis
3. **Validation-First**: Start with MLP (linear + bias + ReLU) as trusted baseline
4. **Educational Focus**: Every metric includes explanation of WHY
5. **Progressive Complexity**: Start simple (MLP), then add softmax, then full models

---

## Success Criteria

### Functional:
- ✅ EDP calculated and reported for all architectures
- ✅ EDP-based winner identified in comparisons
- ✅ MLP validation passes with expected EDP ordering

### Educational:
- ✅ Clear explanations of WHY EDP differs
- ✅ Architectural energy events linked to EDP impact
- ✅ Trade-offs between energy and latency explained

### Practical:
- ✅ CLI tool makes EDP comparison accessible
- ✅ HTML reports visualize EDP effectively
- ✅ Recommendations guide architecture selection

---

## Notes

- **EDP Units**: J·s (joule-seconds) or nJ·s (nanojoule-seconds)
- **EDP Interpretation**: Lower is better. EDP captures the trade-off between energy and delay.
- **Architectural Context**: EDP reveals WHY one architecture is better - not just that it is better.
- **Validation**: MLP provides trusted baseline before moving to complex models.
- **Softmax**: Special case analysis required due to transcendental operations.

---

## References

- Existing: `src/graphs/hardware/architectural_energy.py`
- Existing: `src/graphs/analysis/architecture_comparator.py`
- Existing: `src/graphs/analysis/energy.py`
- Existing: `examples/demo_architectural_energy.py`
- New: `cli/compare_edp.py`
- New: `validation/energy/test_mlp_edp.py`
- New: `src/graphs/analysis/softmax_energy.py`
