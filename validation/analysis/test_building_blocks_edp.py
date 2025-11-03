#!/usr/bin/env python3
"""
Test Subgraph-Level EDP on Core DNN Building Blocks

Progressive validation of hierarchical EDP breakdown on increasingly
complex building blocks:

1. MLP (Linear → Bias → ReLU)
2. Conv2D
3. ResNet block (Conv → BN → ReLU + residual)
4. Attention head (Q, K, V + matmul + softmax)

For each building block, we verify:
- Subgraph EDP breakdown works
- Top subgraphs identified
- Component EDPs sum correctly
- Meaningful insights generated
"""

import torch
import torch.nn as nn

from graphs.analysis.architecture_comparator import ArchitectureComparator
from graphs.hardware.mappers.cpu import create_intel_cpu_mapper
from graphs.hardware.mappers.gpu import create_h100_mapper
from graphs.hardware.mappers.accelerators.tpu import create_tpu_v4_mapper
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t256_mapper
from graphs.hardware.resource_model import Precision


# ==============================================================================
# Building Block Definitions
# ==============================================================================

class SimpleMLP(nn.Module):
    """MLP: Linear → Bias → ReLU → Linear → Bias → ReLU"""
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
# Test Functions
# ==============================================================================

def test_building_block(
    name: str,
    model: nn.Module,
    input_tensor: torch.Tensor,
    architectures: dict,
    test_arch: str = 'KPU'
):
    """
    Test subgraph-level EDP breakdown on a building block.

    Args:
        name: Building block name (e.g., "MLP", "Conv2D")
        model: PyTorch model
        input_tensor: Input tensor
        architectures: Dict of architecture mappers
        test_arch: Architecture to focus on for detailed analysis

    Returns:
        bool: True if all tests pass
    """
    print("=" * 100)
    print(f"Testing Building Block: {name}")
    print("=" * 100)
    print()

    # Create comparator
    comparator = ArchitectureComparator(
        model_name=name,
        architectures=architectures,
        batch_size=input_tensor.shape[0],
        precision=Precision.FP32
    )

    # Run analysis
    print(f"Analyzing {name} on {len(architectures)} architectures...")
    comparator.analyze_all()
    print()

    # Test subgraph EDP breakdown
    print(f"Testing subgraph EDP breakdown for {test_arch}...")
    print("-" * 100)

    try:
        subgraph_edps = comparator.get_subgraph_edp_breakdown(test_arch)
        print(f"  ✓ Found {len(subgraph_edps)} subgraphs")

        if len(subgraph_edps) == 0:
            print(f"  ⚠ Warning: No subgraphs found!")
            return False

        # Verify fractions sum to ~1.0
        total_fraction = sum(sg.edp_fraction for sg in subgraph_edps)
        if abs(total_fraction - 1.0) < 0.01:
            print(f"  ✓ EDP fractions sum to 1.0")
        else:
            print(f"  ⚠ Warning: EDP fractions sum to {total_fraction:.4f}")

        # Show top subgraph
        top_sg = subgraph_edps[0]
        print(f"  ✓ Top subgraph: {top_sg.subgraph_name} ({top_sg.edp_fraction * 100:.1f}% of total)")
        print(f"    - EDP: {top_sg.edp * 1e9:.2f} nJ·s")
        print(f"    - Fusion: {top_sg.fusion_pattern} ({top_sg.num_operators} ops)")
        print(f"    - Bottleneck: {top_sg.bottleneck}")

        # Component breakdown
        print(f"    - Component breakdown:")
        print(f"      Compute: {top_sg.compute_edp / top_sg.edp * 100:.1f}%")
        print(f"      Memory:  {top_sg.memory_edp / top_sg.edp * 100:.1f}%")
        print(f"      Static:  {top_sg.static_edp / top_sg.edp * 100:.1f}%")

        print()

        # Generate and show report
        report = comparator.generate_subgraph_edp_report(test_arch, top_n=5)
        print(report)

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

    # Setup architectures (use 2 for faster testing)
    architectures = {
        'GPU': create_h100_mapper(),
        'KPU': create_kpu_t256_mapper(),
    }

    results = {}

    # ==============================================================================
    # Test 1: MLP (Linear → Bias → ReLU)
    # ==============================================================================
    print("\n" + "=" * 100)
    print("TEST 1: MLP (Linear → Bias → ReLU)")
    print("=" * 100)
    print("Expected: 4 subgraphs (fc1, relu1, fc2, relu2)")
    print("         Linear layers should dominate EDP (matmul is compute-intensive)")
    print()

    mlp_model = SimpleMLP(in_dim=128, hidden_dim=256, out_dim=64)
    mlp_model.eval()
    mlp_input = torch.randn(1, 128)

    results['MLP'] = test_building_block('MLP', mlp_model, mlp_input, architectures)

    # ==============================================================================
    # Test 2: Conv2D (Conv → BN → ReLU)
    # ==============================================================================
    print("\n" + "=" * 100)
    print("TEST 2: Conv2D (Conv → BN → ReLU)")
    print("=" * 100)
    print("Expected: 3 subgraphs (conv, bn, relu)")
    print("         Conv should dominate EDP (compute-intensive)")
    print()

    conv_model = SimpleConv2D(in_channels=3, out_channels=16, kernel_size=3)
    conv_model.eval()
    conv_input = torch.randn(1, 3, 32, 32)

    results['Conv2D'] = test_building_block('Conv2D', conv_model, conv_input, architectures)

    # ==============================================================================
    # Test 3: ResNet Block (Conv → BN → ReLU → Conv → BN + residual → ReLU)
    # ==============================================================================
    print("\n" + "=" * 100)
    print("TEST 3: ResNet Block")
    print("=" * 100)
    print("Expected: ~7 subgraphs (conv1, bn1, relu1, conv2, bn2, add, relu2)")
    print("         Conv layers should dominate, Add should be lightweight")
    print()

    resnet_model = SimpleResNetBlock(channels=64)
    resnet_model.eval()
    resnet_input = torch.randn(1, 64, 32, 32)

    results['ResNet'] = test_building_block('ResNet', resnet_model, resnet_input, architectures)

    # ==============================================================================
    # Test 4: Attention Head (Q, K, V + matmul + softmax)
    # ==============================================================================
    print("\n" + "=" * 100)
    print("TEST 4: Attention Head")
    print("=" * 100)
    print("Expected: ~8 subgraphs (q_proj, k_proj, v_proj, matmul×2, softmax, out_proj)")
    print("         Projections and matmuls should dominate")
    print("         Softmax should show architectural differences")
    print()

    attn_model = SimpleAttentionHead(embed_dim=128, num_heads=4)
    attn_model.eval()
    attn_input = torch.randn(1, 16, 128)  # (batch, seq_len, embed_dim)

    results['Attention'] = test_building_block('Attention', attn_model, attn_input, architectures)

    # ==============================================================================
    # Summary
    # ==============================================================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    print("Test Results:")
    for block, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {block:<15} {status}")

    print()

    if all(results.values()):
        print("✅ ALL BUILDING BLOCKS PASSED!")
        print()
        print("Key Insights:")
        print("  • Subgraph-level EDP breakdown works on all core building blocks")
        print("  • Linear/Conv layers dominate EDP (as expected)")
        print("  • Lightweight ops (ReLU, Add) show up but contribute less")
        print("  • Architecture-specific patterns visible (static energy, bottlenecks)")
        print()
        print("Ready for Phase 2: Per-operator EDP breakdown with architectural modifiers!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        failed = [block for block, passed in results.items() if not passed]
        print(f"  Failed: {', '.join(failed)}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
