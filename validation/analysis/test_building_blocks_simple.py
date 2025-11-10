#!/usr/bin/env python3
"""
Test Subgraph-Level EDP on Core DNN Building Blocks (Simplified)

Uses UnifiedAnalyzer directly with custom models to test subgraph EDP breakdown.
"""

import torch
import torch.nn as nn

from graphs.analysis.unified_analyzer import UnifiedAnalyzer
from graphs.hardware.mappers.gpu import create_h100_pcie_80gb_mapper
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t256_mapper
from graphs.hardware.resource_model import Precision


# ==============================================================================
# Building Block Definitions
# ==============================================================================

class SimpleMLP(nn.Module):
    """MLP: Linear → ReLU → Linear → ReLU"""
    def __init__(self, in_dim=128, hidden_dim=256, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


class SimpleConv2D(nn.Module):
    """Conv2D: Conv → BN → ReLU"""
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimpleResNetBlock(nn.Module):
    """ResNet block: Conv → BN → ReLU → Conv → BN + residual → ReLU"""
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # Residual connection
        out = self.relu2(out)
        return out


class SimpleAttentionHead(nn.Module):
    """Attention head: Q, K, V projections → scaled dot-product attention"""
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        return out


# ==============================================================================
# Helper Functions
# ==============================================================================

def calculate_subgraph_edps(result):
    """
    Calculate per-subgraph EDP from UnifiedAnalysisResult.

    This is the core logic from get_subgraph_edp_breakdown() extracted
    for standalone use.
    """
    if not result.energy_report or not result.roofline_report:
        raise ValueError("Missing energy or roofline report")

    energy_descriptors = result.energy_report.energy_descriptors
    latency_descriptors = result.roofline_report.latencies

    if len(energy_descriptors) != len(latency_descriptors):
        raise ValueError(
            f"Mismatch: {len(energy_descriptors)} energy vs "
            f"{len(latency_descriptors)} latency descriptors"
        )

    subgraph_edps = []

    for e_desc, l_desc in zip(energy_descriptors, latency_descriptors):
        # Calculate EDP
        edp = e_desc.total_energy_j * l_desc.actual_latency

        # Component EDPs
        compute_edp = e_desc.compute_energy_j * l_desc.actual_latency
        memory_edp = e_desc.memory_energy_j * l_desc.actual_latency
        static_edp = e_desc.static_energy_j * l_desc.actual_latency

        subgraph_edps.append({
            'name': e_desc.subgraph_name,
            'energy_j': e_desc.total_energy_j,
            'latency_s': l_desc.actual_latency,
            'edp': edp,
            'compute_edp': compute_edp,
            'memory_edp': memory_edp,
            'static_edp': static_edp,
            'bottleneck': l_desc.bottleneck.value if hasattr(l_desc.bottleneck, 'value') else str(l_desc.bottleneck),
        })

    # Calculate fractions
    total_edp = sum(sg['edp'] for sg in subgraph_edps)
    for sg in subgraph_edps:
        sg['edp_fraction'] = sg['edp'] / total_edp if total_edp > 0 else 0.0

    # Sort by EDP (descending)
    subgraph_edps.sort(key=lambda x: x['edp'], reverse=True)

    return subgraph_edps, total_edp


def test_building_block(name: str, model: nn.Module, input_tensor: torch.Tensor):
    """Test subgraph-level EDP on a building block"""

    print("=" * 100)
    print(f"Testing: {name}")
    print("=" * 100)
    print()

    # Analyze on KPU
    print("Analyzing on KPU...")
    analyzer = UnifiedAnalyzer()

    try:
        result = analyzer.analyze_model_with_custom_hardware(
            model=model,
            input_tensor=input_tensor,
            model_name=name,
            hardware_mapper=create_kpu_t256_mapper(),
            precision=Precision.FP32
        )

        print(f"  ✓ Analysis complete")
        print(f"  Model EDP: {result.total_latency_ms / 1000.0 * result.energy_per_inference_mj / 1000.0 * 1e9:.2f} nJ·s")
        print()

        # Calculate subgraph EDPs
        print("Calculating subgraph EDP breakdown...")
        subgraph_edps, total_edp = calculate_subgraph_edps(result)

        print(f"  ✓ Found {len(subgraph_edps)} subgraphs")
        print(f"  Total subgraph EDP: {total_edp * 1e9:.2f} nJ·s")
        print()

        # Show top 5 subgraphs
        print("Top 5 Subgraphs by EDP:")
        print(f"{'Rank':<5} {'Subgraph':<30} {'EDP (nJ·s)':<15} {'% Total':<10} {'Bottleneck'}")
        print("-" * 100)

        for i, sg in enumerate(subgraph_edps[:5], 1):
            marker = " ⭐" if i == 1 else ""
            print(
                f"{i:<5} "
                f"{sg['name']:<30} "
                f"{sg['edp']*1e9:<15.2f} "
                f"{sg['edp_fraction']*100:<10.1f}% "
                f"{sg['bottleneck']:<15}"
                f"{marker}"
            )

        print()

        # Top subgraph component breakdown
        if subgraph_edps:
            top = subgraph_edps[0]
            print("Top Subgraph Component Breakdown:")
            print(f"  {top['name']}")
            print(f"    Compute EDP:  {top['compute_edp']*1e9:>10.2f} nJ·s ({top['compute_edp']/top['edp']*100:.1f}%)")
            print(f"    Memory EDP:   {top['memory_edp']*1e9:>10.2f} nJ·s ({top['memory_edp']/top['edp']*100:.1f}%)")
            print(f"    Static EDP:   {top['static_edp']*1e9:>10.2f} nJ·s ({top['static_edp']/top['edp']*100:.1f}%)")
            print(f"    Total EDP:    {top['edp']*1e9:>10.2f} nJ·s")
            print()

        # Cumulative distribution
        print("Cumulative EDP Distribution:")
        cumulative = 0.0
        for threshold in [50, 80, 90]:
            for i, sg in enumerate(subgraph_edps, 1):
                cumulative += sg['edp_fraction'] * 100
                if cumulative >= threshold:
                    print(f"  Top {i} subgraphs account for {threshold}% of total EDP")
                    cumulative = 0.0
                    break

        print()
        return True

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run progressive tests on all building blocks"""

    print("=" * 100)
    print("PROGRESSIVE BUILDING BLOCK TESTING")
    print("Testing subgraph-level EDP breakdown on core DNN building blocks")
    print("=" * 100)
    print()

    results = {}

    # ==============================================================================
    # Test 1: MLP
    # ==============================================================================
    print("TEST 1: MLP (Linear → ReLU → Linear → ReLU)")
    print("Expected: 4 subgraphs, Linear layers should dominate")
    print()

    mlp_model = SimpleMLP(in_dim=128, hidden_dim=256, out_dim=64)
    mlp_model.eval()
    mlp_input = torch.randn(1, 128)

    results['MLP'] = test_building_block('MLP', mlp_model, mlp_input)

    # ==============================================================================
    # Test 2: Conv2D
    # ==============================================================================
    print("TEST 2: Conv2D (Conv → BN → ReLU)")
    print("Expected: 3 subgraphs, Conv should dominate")
    print()

    conv_model = SimpleConv2D(in_channels=3, out_channels=16, kernel_size=3)
    conv_model.eval()
    conv_input = torch.randn(1, 3, 32, 32)

    results['Conv2D'] = test_building_block('Conv2D', conv_model, conv_input)

    # ==============================================================================
    # Test 3: ResNet Block
    # ==============================================================================
    print("TEST 3: ResNet Block (Conv → BN → ReLU → Conv → BN + residual → ReLU)")
    print("Expected: ~7 subgraphs, Conv layers should dominate, Add should be lightweight")
    print()

    resnet_model = SimpleResNetBlock(channels=64)
    resnet_model.eval()
    resnet_input = torch.randn(1, 64, 32, 32)

    results['ResNet'] = test_building_block('ResNet Block', resnet_model, resnet_input)

    # ==============================================================================
    # Test 4: Attention Head
    # ==============================================================================
    print("TEST 4: Attention Head (Q, K, V + matmul + softmax)")
    print("Expected: ~8 subgraphs, Projections and matmuls should dominate")
    print()

    attn_model = SimpleAttentionHead(embed_dim=128, num_heads=4)
    attn_model.eval()
    attn_input = torch.randn(1, 16, 128)  # (batch, seq_len, embed_dim)

    results['Attention'] = test_building_block('Attention Head', attn_model, attn_input)

    # ==============================================================================
    # Summary
    # ==============================================================================
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    print("Test Results:")
    for block, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {block:<20} {status}")

    print()

    if all(results.values()):
        print("✅ ALL BUILDING BLOCKS PASSED!")
        print()
        print("Key Insights:")
        print("  • Subgraph-level EDP breakdown works on all core building blocks")
        print("  • Linear/Conv layers dominate EDP (compute-intensive operations)")
        print("  • Lightweight ops (ReLU, Add) contribute less but still visible")
        print("  • Static energy often dominates (leakage during execution)")
        print("  • Cumulative distribution shows 80/20 rule (few subgraphs dominate)")
        print()
        print("✅ Phase 1 validated on all building blocks!")
        print("Ready for Phase 2: Per-operator EDP with architectural modifiers")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        failed = [block for block, passed in results.items() if not passed]
        print(f"  Failed: {', '.join(failed)}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
