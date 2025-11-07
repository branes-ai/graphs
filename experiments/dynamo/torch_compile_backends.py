"""
torch.compile Backend Comparison

This demonstrates different torch.compile backends and explains when to use each.
For graph characterization, we only need the CUSTOM BACKEND approach (already shown
in other examples).

This file is for EDUCATIONAL purposes - to understand what torch.compile does
with different backends.

Usage:
    python experiments/dynamo/torch_compile_backends.py
"""

import torch
import torch._dynamo as dynamo
import time
from typing import List


class SimpleModel(torch.nn.Module):
    """Simple CNN for demonstration."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc = torch.nn.Linear(32 * 56 * 56, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================================
# Backend Demonstrations
# ============================================================================

def demo_eager_mode():
    """
    Eager mode: No compilation at all (baseline PyTorch execution).

    Use when: Debugging, development, or when you want PyTorch's original behavior.
    """
    print("\n" + "="*80)
    print("1. EAGER MODE (No Compilation)")
    print("="*80)
    print("Purpose: Normal PyTorch execution (baseline)")
    print("Use case: Development, debugging")

    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 224, 224)

    # No compilation
    with torch.no_grad():
        output = model(input_tensor)

    print(f"✓ Output shape: {output.shape}")
    print("→ No optimization, just normal PyTorch")


def demo_inductor_backend():
    """
    Inductor backend: TorchInductor optimization (default for torch.compile).

    Use when: Production inference, want automatic speedup.
    This is what torch.compile does by DEFAULT.
    """
    print("\n" + "="*80)
    print("2. INDUCTOR BACKEND (Default torch.compile)")
    print("="*80)
    print("Purpose: Automatic optimization for production")
    print("Use case: Inference speedup, production deployment")

    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 224, 224)

    # Compile with inductor (default)
    compiled_model = torch.compile(model, backend="inductor")

    # First run (compilation happens here)
    print("\nFirst run (compilation)...")
    with torch.no_grad():
        output = compiled_model(input_tensor)

    print(f"✓ Output shape: {output.shape}")
    print("→ Model is now optimized with fused kernels, etc.")
    print("→ Subsequent runs will be faster")


def demo_aot_eager_backend():
    """
    AOT Eager backend: Ahead-of-time graph capture with eager execution.

    Use when: Debugging torch.compile issues, seeing what graphs are captured.
    """
    print("\n" + "="*80)
    print("3. AOT_EAGER BACKEND (Debugging)")
    print("="*80)
    print("Purpose: Debug torch.compile without actual compilation")
    print("Use case: Debugging, seeing graph breaks")

    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 224, 224)

    # Compile with aot_eager
    compiled_model = torch.compile(model, backend="aot_eager")

    with torch.no_grad():
        output = compiled_model(input_tensor)

    print(f"✓ Output shape: {output.shape}")
    print("→ Graphs were captured but executed eagerly (not compiled)")
    print("→ Useful for finding graph breaks without compilation overhead")


def demo_custom_backend_for_extraction():
    """
    Custom backend: Our graph extraction approach (RECOMMENDED for characterization).

    Use when: Graph analysis, characterization, research.
    This is what we do in the graphs package!
    """
    print("\n" + "="*80)
    print("4. CUSTOM BACKEND (Graph Extraction) ← WE USE THIS")
    print("="*80)
    print("Purpose: Extract graph for analysis (no optimization)")
    print("Use case: Workload characterization, FLOP counting, hardware mapping")

    # Our custom backend
    class GraphExtractor:
        def __init__(self):
            self.graphs = []
            self.call_count = 0

        def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
            self.call_count += 1
            self.graphs.append(gm.graph)

            print(f"\n  Graph partition {self.call_count} captured:")
            print(f"    Nodes: {len(list(gm.graph.nodes))}")

            # Count operations
            op_count = sum(1 for n in gm.graph.nodes if n.op == 'call_function')
            print(f"    Operations: {op_count}")

            # Return original (no optimization)
            return gm.forward

    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 224, 224)

    # Use our custom backend
    extractor = GraphExtractor()
    compiled_model = torch.compile(model, backend=extractor)

    with torch.no_grad():
        output = compiled_model(input_tensor)

    print(f"\n✓ Output shape: {output.shape}")
    print(f"✓ Captured {len(extractor.graphs)} graph(s)")
    print("→ We extracted the graph structure WITHOUT optimizing")
    print("→ This is EXACTLY what we do for characterization!")


def benchmark_backends():
    """
    Benchmark different backends to show performance differences.

    NOTE: For characterization/analysis, performance doesn't matter.
    We only care about graph extraction accuracy.
    """
    print("\n" + "="*80)
    print("5. PERFORMANCE COMPARISON (Just for Context)")
    print("="*80)
    print("Note: For characterization, we don't care about speed!")
    print("      We only care about accurate graph extraction.")

    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 224, 224)

    # Warmup
    with torch.no_grad():
        _ = model(input_tensor)

    # Benchmark eager
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)
    eager_time = time.time() - start

    # Benchmark compiled (inductor)
    compiled_model = torch.compile(model, backend="inductor")
    with torch.no_grad():
        _ = compiled_model(input_tensor)  # Warmup

    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = compiled_model(input_tensor)
    compiled_time = time.time() - start

    print(f"\nEager mode: {eager_time*1000:.2f} ms (100 runs)")
    print(f"Inductor:   {compiled_time*1000:.2f} ms (100 runs)")
    print(f"Speedup:    {eager_time/compiled_time:.2f}x")

    print("\n→ Inductor makes inference faster (production use)")
    print("→ For characterization, we use custom backend (doesn't need speed)")


# ============================================================================
# Summary and Recommendations
# ============================================================================

def print_summary():
    """Print summary of when to use each backend."""
    print("\n" + "="*80)
    print("SUMMARY: When to Use Each Backend")
    print("="*80)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ Backend          │ Purpose              │ graphs Package?     │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│ 'inductor'       │ Production speedup   │ ❌ No (we analyze) │")
    print("│ 'aot_eager'      │ Debugging            │ ❌ No (we analyze) │")
    print("│ 'cudagraphs'     │ GPU optimization     │ ❌ No (we analyze) │")
    print("│ CUSTOM BACKEND   │ Graph extraction     │ ✅ YES! (we use)   │")
    print("│ (no backend)     │ Eager mode           │ ❌ No (baseline)   │")
    print("└─────────────────────────────────────────────────────────────────┘")

    print("\n📌 For graphs package workload characterization:")
    print("   → Use CUSTOM BACKEND (what we built in other examples)")
    print("   → Don't need inductor, aot_eager, or other backends")
    print("   → torch.compile is just the API; we provide the backend")

    print("\n📚 Existing examples already show the right approach:")
    print("   - basic_dynamo_tracing.py")
    print("   - huggingface_complex_models.py")
    print("   - trace_yolo.py")
    print("   - integrate_with_graphs.py")


# ============================================================================
# Key Insight
# ============================================================================

def explain_relationship():
    """Explain the torch.compile ↔ Dynamo relationship."""
    print("\n" + "="*80)
    print("KEY INSIGHT: torch.compile vs Dynamo")
    print("="*80)

    print("""
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  torch.compile(model, backend=...)                            │
│         │                                                      │
│         ├─ Uses TorchDynamo to capture graph                  │
│         │                                                      │
│         └─ Passes graph to BACKEND for processing:            │
│                  │                                             │
│                  ├─ "inductor" → Optimize (default)           │
│                  ├─ "aot_eager" → Debug                       │
│                  └─ CustomBackend → Extract (us!)             │
│                                                                │
└────────────────────────────────────────────────────────────────┘

RELATIONSHIP:
- torch.compile: User-facing API
- TorchDynamo: Graph capture engine (inside torch.compile)
- Backend: What to do with the graph

FOR GRAPHS PACKAGE:
- We use torch.compile with a CUSTOM backend
- Custom backend = our GraphExtractor
- We extract graphs WITHOUT optimizing
- This is what all our examples do!

YOU DON'T NEED SEPARATE torch.compile EXPERIMENTS because:
1. All Dynamo examples already use torch.compile
2. torch.compile is just the entry point
3. Our custom backend does the extraction
4. Other backends (inductor, etc.) are for optimization, not analysis
""")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("torch.compile Backend Demonstrations")
    print("="*80)
    print("\nThis demonstrates different backends to show the relationship")
    print("between torch.compile and Dynamo.")
    print("\n⚠️  For graphs package, you ONLY need the custom backend approach!")
    print("   (Already shown in basic_dynamo_tracing.py, etc.)")

    # Explain relationship first
    explain_relationship()

    # Run demonstrations
    demo_eager_mode()
    demo_inductor_backend()
    demo_aot_eager_backend()
    demo_custom_backend_for_extraction()

    # Benchmark (just for context)
    # benchmark_backends()  # Uncomment to run benchmark

    # Summary
    print_summary()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
For DNN workload characterization in the graphs package:

✅ USE: Custom backend with torch.compile
   → This is what we already do in all examples
   → No need for separate torch.compile experiments

❌ DON'T USE: inductor, aot_eager, cudagraphs, etc.
   → These are for optimization/debugging
   → Not useful for analysis/characterization

📖 REFER TO: Existing examples in experiments/dynamo/
   - basic_dynamo_tracing.py
   - huggingface_complex_models.py
   - trace_yolo.py
   - integrate_with_graphs.py

These already show the correct approach!
""")


if __name__ == "__main__":
    main()
