#!/usr/bin/env python
"""
Datacenter CPU Comparison Tool

Compares ARM and x86 datacenter server processors across current and next-gen:

Current Generation (Shipping Now):
- Ampere AmpereOne 192-core / 128-core (ARM v8.6+)
- Intel Xeon Platinum 8490H / 8592+ (Sapphire Rapids)
- AMD EPYC 9654 / 9754 (Genoa, Zen 4)

Next Generation (2024-2025):
- Intel Xeon Granite Rapids (128-core, Intel 3)
- AMD EPYC Turin (192-core, Zen 5, 3nm)

Test Models:
- ResNet-50: Vision backbone (25M params)
- DeepLabV3+: Semantic segmentation (42M params)
- ViT-Base: Vision Transformer (86M params)
- ConvNeXt-Large: Modernized ConvNet (198M params)
- ViT-Large: Large Vision Transformer (304M params)

Metrics:
- Latency and throughput (FPS)
- Power efficiency (FPS/W)
- Core utilization
- Bottleneck analysis
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
import sys
from pathlib import Path
from typing import List
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graphs.transform.partitioning import FusionBasedPartitioner
from src.graphs.hardware.mappers.cpu import (
    create_ampere_ampereone_192_mapper,
    create_ampere_ampereone_128_mapper,
    create_intel_xeon_platinum_8490h_mapper,
    create_intel_xeon_platinum_8592plus_mapper,
    create_intel_granite_rapids_mapper,
    create_amd_epyc_9654_mapper,
    create_amd_epyc_9754_mapper,
    create_amd_epyc_turin_mapper,
)
from src.graphs.hardware.resource_model import Precision


@dataclass
class CPUBenchmarkResult:
    """Results from CPU benchmark"""
    model_name: str
    cpu_name: str
    vendor: str
    cores: int
    tdp_watts: float

    # Performance metrics
    latency_ms: float
    throughput_fps: float

    # Efficiency metrics
    fps_per_watt: float
    energy_per_inference_mj: float

    # Utilization
    avg_utilization: float
    peak_utilization: float

    # Architecture
    architecture: str  # "ARM", "x86"
    process_node: str  # "5nm", "10nm"


def extract_execution_stages(fusion_report):
    """Extract execution stages from fusion report"""
    stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]
    return stages


def create_resnet50(batch_size=1):
    """Create ResNet-50 model"""
    model = models.resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    return model, input_tensor, "ResNet-50"


def create_deeplabv3(batch_size=1):
    """Create DeepLabV3+ model"""
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    return model, input_tensor, "DeepLabV3+"


def create_vit_base(batch_size=1):
    """Create Vision Transformer Base model"""
    model = models.vit_b_16(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    return model, input_tensor, "ViT-Base"


def create_bert_large(batch_size=1, seq_length=512):
    """Create BERT-Large model (340M params) - Common datacenter NLP workload"""
    from transformers import BertModel, BertConfig

    config = BertConfig(
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=512,
    )
    model = BertModel(config)
    model.eval()

    # Input: token IDs (integers)
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Wrap in a simple forward function for FX tracing
    class BertWrapper(nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert

        def forward(self, input_ids, attention_mask):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    wrapped_model = BertWrapper(model)
    wrapped_model.eval()

    return wrapped_model, (input_ids, attention_mask), "BERT-Large (340M)"


def create_gpt2_xl(batch_size=1, seq_length=512):
    """Create GPT-2 XL model (1.5B params) - Large language model"""
    from transformers import GPT2LMHeadModel, GPT2Config

    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=1600,
        n_layer=48,
        n_head=25,
    )
    model = GPT2LMHeadModel(config)
    model.eval()

    # Input: token IDs
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))

    class GPT2Wrapper(nn.Module):
        def __init__(self, gpt2):
            super().__init__()
            self.gpt2 = gpt2

        def forward(self, input_ids):
            return self.gpt2(input_ids=input_ids).logits

    wrapped_model = GPT2Wrapper(model)
    wrapped_model.eval()

    return wrapped_model, input_ids, "GPT-2 XL (1.5B)"


def create_convnext_large(batch_size=1):
    """Create ConvNeXt-Large model (198M params) - Modernized ConvNet with Transformer-like performance"""
    model = models.convnext_large(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    return model, input_tensor, "ConvNeXt-Large (198M)"


def create_vit_large(batch_size=1):
    """Create ViT-Large model (304M params) - Large vision transformer"""
    model = models.vit_l_16(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    return model, input_tensor, "ViT-Large (304M)"


def benchmark_cpu(
    model, input_tensor, model_name,
    mapper, cpu_name, vendor, cores, tdp, architecture, process_node,
    precision='int8'
):
    """Run single CPU benchmark"""

    print(f"\n  Tracing {model_name}...")
    traced = symbolic_trace(model)

    # Handle tuple inputs (e.g., BERT's (input_ids, attention_mask))
    if isinstance(input_tensor, tuple):
        ShapeProp(traced).propagate(*input_tensor)
        batch_size = input_tensor[0].shape[0]
    else:
        ShapeProp(traced).propagate(input_tensor)
        batch_size = input_tensor.shape[0]

    print(f"  Partitioning...")
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)

    execution_stages = extract_execution_stages(fusion_report)

    print(f"  Mapping to {cpu_name}...")

    if precision.lower() == 'int8':
        prec = Precision.INT8
    elif precision.lower() == 'fp16':
        prec = Precision.FP16
    else:
        prec = Precision.FP32

    hw_report = mapper.map_graph(
        fusion_report,
        execution_stages,
        batch_size=batch_size,
        precision=prec
    )

    # Calculate metrics
    latency_ms = hw_report.total_latency * 1000
    throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0
    fps_per_watt = throughput_fps / tdp if tdp > 0 else 0
    energy_per_inference_mj = hw_report.total_energy * 1000

    return CPUBenchmarkResult(
        model_name=model_name,
        cpu_name=cpu_name,
        vendor=vendor,
        cores=cores,
        tdp_watts=tdp,
        latency_ms=latency_ms,
        throughput_fps=throughput_fps,
        fps_per_watt=fps_per_watt,
        energy_per_inference_mj=energy_per_inference_mj,
        avg_utilization=hw_report.average_utilization,
        peak_utilization=hw_report.peak_utilization,
        architecture=architecture,
        process_node=process_node,
    )


def print_results(results: List[CPUBenchmarkResult]):
    """Print formatted results"""
    print()
    print("="*140)
    print("DATACENTER CPU COMPARISON RESULTS")
    print("="*140)

    # Group by model
    models = {}
    for r in results:
        if r.model_name not in models:
            models[r.model_name] = []
        models[r.model_name].append(r)

    for model_name, model_results in models.items():
        print()
        print(model_name)
        print("-"*140)
        print(f"{'CPU':<30} {'Vendor':<10} {'Cores':<8} {'TDP':<8} {'Latency':<12} {'FPS':<10} {'FPS/W':<10} {'Util%':<8} {'Arch':<8}")
        print("-"*140)

        for r in model_results:
            print(f"{r.cpu_name:<30} {r.vendor:<10} {r.cores:<8} {r.tdp_watts:<8.0f} "
                  f"{r.latency_ms:<12.2f} {r.throughput_fps:<10.1f} "
                  f"{r.fps_per_watt:<10.2f} {r.avg_utilization*100:<8.1f} {r.architecture:<8}")


def print_summary(results: List[CPUBenchmarkResult]):
    """Print executive summary"""
    print()
    print("="*140)
    print("EXECUTIVE SUMMARY")
    print("="*140)

    # Filter ResNet-50 results for summary
    resnet_results = [r for r in results if r.model_name == "ResNet-50"]

    print()
    print("## Best by Metric (ResNet-50 @ INT8):")
    best_fps_w = max(resnet_results, key=lambda r: r.fps_per_watt)
    best_latency = min(resnet_results, key=lambda r: r.latency_ms)
    best_throughput = max(resnet_results, key=lambda r: r.throughput_fps)

    print(f"  • Best FPS/W: {best_fps_w.cpu_name} ({best_fps_w.fps_per_watt:.2f} FPS/W)")
    print(f"  • Best Latency: {best_latency.cpu_name} ({best_latency.latency_ms:.2f} ms)")
    print(f"  • Best Throughput: {best_throughput.cpu_name} ({best_throughput.throughput_fps:.1f} FPS)")

    print()
    print("## Architecture Comparison:")

    print()
    print("### Ampere AmpereOne (ARM):")
    ampere = next(r for r in resnet_results if "Ampere" in r.cpu_name)
    print(f"  • Cores: {ampere.cores} (single-threaded)")
    print(f"  • Process: TSMC 5nm")
    print(f"  • Memory: 332.8 GB/s (8-channel DDR5)")
    print(f"  • AI: Native FP16/BF16/INT8 in ARM SIMD")
    print(f"  • Performance: {ampere.throughput_fps:.1f} FPS @ {ampere.fps_per_watt:.2f} FPS/W")

    print()
    print("### Intel Xeon Platinum 8490H (x86):")
    xeon = next(r for r in resnet_results if "Xeon" in r.cpu_name)
    print(f"  • Cores: {xeon.cores} (HyperThreading)")
    print(f"  • Process: Intel 7 (10nm Enhanced)")
    print(f"  • Memory: 307 GB/s (8-channel DDR5)")
    print(f"  • AI: AMX (Advanced Matrix Extensions)")
    print(f"  • Performance: {xeon.throughput_fps:.1f} FPS @ {xeon.fps_per_watt:.2f} FPS/W")

    print()
    print("### AMD EPYC 9654 (x86):")
    epyc = next(r for r in resnet_results if "EPYC" in r.cpu_name)
    print(f"  • Cores: {epyc.cores} (SMT)")
    print(f"  • Process: TSMC 5nm")
    print(f"  • Memory: 460.8 GB/s (12-channel DDR5)")
    print(f"  • AI: AVX-512 (double-pumped)")
    print(f"  • Performance: {epyc.throughput_fps:.1f} FPS @ {epyc.fps_per_watt:.2f} FPS/W")

    print()
    print("## Key Insights:")

    print()
    print("### Core Count:")
    print(f"  • Ampere AmpereOne: 192 cores (3.2× Intel, 2× AMD)")
    print(f"  • Intel Xeon: 60 cores (baseline)")
    print(f"  • AMD EPYC: 96 cores (1.6× Intel)")

    print()
    print("### Memory Bandwidth:")
    print(f"  • AMD EPYC: 460.8 GB/s (highest, 12-channel)")
    print(f"  • Ampere AmpereOne: 332.8 GB/s (8-channel)")
    print(f"  • Intel Xeon: 307.2 GB/s (8-channel)")

    print()
    print("### AI Acceleration:")
    print(f"  • Intel Xeon: AMX (best for INT8/BF16 matrix ops)")
    print(f"  • Ampere AmpereOne: Native ARM SIMD (competitive)")
    print(f"  • AMD EPYC: AVX-512 only (weakest for AI)")

    print()
    print("### Use Case Recommendations:")
    print(f"  ┌────────────────────────────────┬──────────────────────────┬────────────────────────┐")
    print(f"  │ Use Case                       │ Recommended CPU          │ Reason                 │")
    print(f"  ├────────────────────────────────┼──────────────────────────┼────────────────────────┤")
    print(f"  │ CNN Inference (ResNet, YOLO)   │ Intel Xeon (AMX)         │ 4-10× faster with AMX  │")
    print(f"  │ Vision Transformers (ViT)      │ AMD EPYC                 │ High bandwidth helps   │")
    print(f"  │ Cloud-Native Microservices     │ Ampere AmpereOne         │ Most cores, best $/FPS │")
    print(f"  │ Virtualization (many VMs)      │ AMD EPYC                 │ Most threads (192)     │")
    print(f"  │ General-Purpose Compute        │ Ampere AmpereOne         │ Best FPS/W             │")
    print(f"  │ HPC / Scientific Computing     │ Intel Xeon / AMD EPYC    │ AVX-512 support        │")
    print(f"  └────────────────────────────────┴──────────────────────────┴────────────────────────┘")


def main():
    """Run full datacenter CPU comparison"""
    print("="*140)
    print("DATACENTER CPU COMPARISON: Current Generation + Next Generation")
    print("Testing 8 CPUs: 3 ARM (Ampere), 5 x86 (Intel + AMD)")
    print("="*140)

    # CPU configurations - Current Generation + Next Generation
    cpu_configs = [
        # === CURRENT GENERATION (Shipping Now) ===
        {
            "name": "Ampere AmpereOne 192-core",
            "vendor": "Ampere",
            "cores": 192,
            "tdp": 283,
            "mapper": create_ampere_ampereone_192_mapper(),
            "architecture": "ARM",
            "process": "5nm",
            "generation": "Current",
        },
        {
            "name": "Ampere AmpereOne 128-core",
            "vendor": "Ampere",
            "cores": 128,
            "tdp": 210,
            "mapper": create_ampere_ampereone_128_mapper(),
            "architecture": "ARM",
            "process": "5nm",
            "generation": "Current",
        },
        {
            "name": "Intel Xeon Platinum 8490H",
            "vendor": "Intel",
            "cores": 60,
            "tdp": 350,
            "mapper": create_intel_xeon_platinum_8490h_mapper(),
            "architecture": "x86",
            "process": "10nm",
            "generation": "Current",
        },
        {
            "name": "Intel Xeon Platinum 8592+",
            "vendor": "Intel",
            "cores": 64,
            "tdp": 350,
            "mapper": create_intel_xeon_platinum_8592plus_mapper(),
            "architecture": "x86",
            "process": "10nm",
            "generation": "Current",
        },
        {
            "name": "AMD EPYC 9654",
            "vendor": "AMD",
            "cores": 96,
            "tdp": 360,
            "mapper": create_amd_epyc_9654_mapper(),
            "architecture": "x86",
            "process": "5nm",
            "generation": "Current",
        },
        {
            "name": "AMD EPYC 9754",
            "vendor": "AMD",
            "cores": 128,
            "tdp": 360,
            "mapper": create_amd_epyc_9754_mapper(),
            "architecture": "x86",
            "process": "5nm",
            "generation": "Current",
        },

        # === NEXT GENERATION (2024-2025) ===
        {
            "name": "Intel Granite Rapids",
            "vendor": "Intel",
            "cores": 128,
            "tdp": 500,
            "mapper": create_intel_granite_rapids_mapper(),
            "architecture": "x86",
            "process": "Intel 3",
            "generation": "Next-Gen",
        },
        {
            "name": "AMD EPYC Turin (Zen 5)",
            "vendor": "AMD",
            "cores": 192,
            "tdp": 500,
            "mapper": create_amd_epyc_turin_mapper(),
            "architecture": "x86",
            "process": "3nm",
            "generation": "Next-Gen",
        },
    ]

    # Models to test - Mix of CNN and Transformer workloads
    models_to_test = [
        create_resnet50(batch_size=1),      # CNN: Image classification (25M params)
        create_deeplabv3(batch_size=1),     # CNN: Semantic segmentation (42M params)
        create_vit_base(batch_size=1),      # Transformer: Vision (86M params)
        create_convnext_large(batch_size=1), # Hybrid: Modernized ConvNet (198M params)
        create_vit_large(batch_size=1),     # Transformer: Large vision (304M params)
    ]

    results = []

    for model, input_tensor, model_name in models_to_test:
        print()
        print("="*100)
        print(f"Testing: {model_name}")
        print("="*100)

        for config in cpu_configs:
            print(f"\n► {config['name']} @ {config['tdp']}W")
            result = benchmark_cpu(
                model, input_tensor, model_name,
                config['mapper'], config['name'], config['vendor'],
                config['cores'], config['tdp'], config['architecture'], config['process'],
                precision='int8'
            )
            results.append(result)

    # Print results
    print_results(results)
    print_summary(results)

    print()
    print("="*140)
    print("Benchmark Complete")
    print("="*140)


if __name__ == "__main__":
    main()
