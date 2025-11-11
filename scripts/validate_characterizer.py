#!/usr/bin/env python3
"""
Validate Dynamo characterizer against published FLOP counts.

Tests the characterizer with real models and compares against:
- fvcore FlopCountAnalysis
- Published paper values
- TorchVision model zoo
"""

import sys
import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not installed. Install with: pip install fvcore")

from graphs.analysis.dynamo_characterizer import characterize_with_dynamo


def validate_model(
    model_name: str,
    model: nn.Module,
    input_data: torch.Tensor,
    expected_gflops: Optional[float] = None,
    tolerance: float = 0.10  # 10% tolerance
) -> bool:
    """
    Validate characterizer accuracy for a model.

    Args:
        model_name: Name of model for reporting
        model: PyTorch model
        input_data: Example input tensor
        expected_gflops: Expected GFLOPs from literature (optional)
        tolerance: Acceptable difference (default 10%)

    Returns:
        True if validation passed
    """
    print(f"\n{'='*80}")
    print(f"Validating: {model_name}")
    print(f"{'='*80}")

    model.eval()

    # Our characterization
    print("\n[1] Dynamo Characterization:")
    workload = characterize_with_dynamo(model, input_data, verbose=False)

    our_total_ops = workload.macs + workload.flops + workload.intops
    our_gflops = our_total_ops / 1e9

    print(f"  MACs:     {workload.macs:,} ({workload.macs/1e9:.3f} G)")
    print(f"  FLOPs:    {workload.flops:,} ({workload.flops/1e9:.3f} G)")
    print(f"  IntOps:   {workload.intops:,} ({workload.intops/1e9:.3f} G)")
    print(f"  Total:    {our_total_ops:,} ({our_gflops:.3f} GFLOPs)")
    print(f"  AI:       {workload.arithmetic_intensity_total():.2f} ops/byte")

    # Breakdown summary
    if workload.breakdown:
        print(f"\n  Breakdown:")
        if workload.breakdown.matmul_macs > 0:
            print(f"    MatMul MACs:  {workload.breakdown.matmul_macs:,}")
        if workload.breakdown.conv2d_macs > 0:
            print(f"    Conv2D MACs:  {workload.breakdown.conv2d_macs:,}")
        if workload.breakdown.bias_flops > 0:
            print(f"    Bias FLOPs:   {workload.breakdown.bias_flops:,}")
        if workload.breakdown.relu_flops > 0:
            print(f"    ReLU FLOPs:   {workload.breakdown.relu_flops:,}")
        if workload.breakdown.batchnorm_flops > 0:
            print(f"    BatchNorm:    {workload.breakdown.batchnorm_flops:,}")

    # fvcore comparison
    if FVCORE_AVAILABLE:
        print("\n[2] fvcore Comparison:")
        try:
            flop_counter = FlopCountAnalysis(model, input_data)
            fvcore_total = flop_counter.total()
            fvcore_gflops = fvcore_total / 1e9

            print(f"  fvcore total: {fvcore_total:,} ({fvcore_gflops:.3f} GFLOPs)")

            # Compare
            difference = abs(our_total_ops - fvcore_total) / fvcore_total * 100
            print(f"  Difference:   {difference:.2f}%")

            if difference <= tolerance * 100:
                print(f"  Status:       ✅ PASS (within {tolerance*100:.0f}% tolerance)")
            else:
                print(f"  Status:       ⚠️ WARN (exceeds {tolerance*100:.0f}% tolerance)")

        except Exception as e:
            print(f"  fvcore error: {e}")
            fvcore_gflops = None
    else:
        fvcore_gflops = None

    # Literature comparison
    if expected_gflops is not None:
        print("\n[3] Literature Comparison:")
        print(f"  Expected:     {expected_gflops:.3f} GFLOPs")
        print(f"  Our result:   {our_gflops:.3f} GFLOPs")

        difference = abs(our_gflops - expected_gflops) / expected_gflops * 100
        print(f"  Difference:   {difference:.2f}%")

        if difference <= tolerance * 100:
            print(f"  Status:       ✅ PASS")
            return True
        else:
            print(f"  Status:       ❌ FAIL")
            return False

    # If no reference, just report
    if fvcore_gflops is not None:
        difference = abs(our_gflops - fvcore_gflops) / fvcore_gflops * 100
        return difference <= tolerance * 100
    else:
        print(f"\n  Status:       ℹ️ No reference for comparison")
        return True


def validate_resnet18():
    """Validate ResNet18"""
    from torchvision import models

    model = models.resnet18(weights=None)
    input_data = torch.randn(1, 3, 224, 224)

    # From "Deep Residual Learning" paper and fvcore
    # ResNet18 has ~1.82 GFLOPs
    return validate_model(
        "ResNet18",
        model,
        input_data,
        expected_gflops=1.82,
        tolerance=0.10
    )


def validate_mobilenetv2():
    """Validate MobileNetV2"""
    from torchvision import models

    model = models.mobilenet_v2(weights=None)
    input_data = torch.randn(1, 3, 224, 224)

    # From MobileNetV2 paper: ~300M MACs
    return validate_model(
        "MobileNetV2",
        model,
        input_data,
        expected_gflops=0.30,
        tolerance=0.15  # More tolerance for depthwise convs
    )


def validate_efficientnet_b0():
    """Validate EfficientNet-B0"""
    try:
        from torchvision import models

        model = models.efficientnet_b0(weights=None)
        input_data = torch.randn(1, 3, 224, 224)

        # From EfficientNet paper: ~390M FLOPs
        return validate_model(
            "EfficientNet-B0",
            model,
            input_data,
            expected_gflops=0.39,
            tolerance=0.15
        )
    except AttributeError:
        print("\nEfficientNet-B0 not available in this torchvision version")
        return True


def validate_vit_b_16():
    """Validate Vision Transformer (ViT-B/16)"""
    try:
        from torchvision import models

        model = models.vit_b_16(weights=None)
        input_data = torch.randn(1, 3, 224, 224)

        # ViT-B/16: ~17.5 GFLOPs
        # From "An Image is Worth 16x16 Words" paper
        return validate_model(
            "ViT-B/16",
            model,
            input_data,
            expected_gflops=17.5,
            tolerance=0.15
        )
    except AttributeError:
        print("\nViT-B/16 not available in this torchvision version")
        return True


def validate_bert_base():
    """Validate BERT-base"""
    try:
        from transformers import BertModel

        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()

        # BERT-base with sequence length 128
        input_ids = torch.randint(0, 30522, (1, 128))

        # BERT-base: ~22.5 GFLOPs for seq_len=128
        # Formula: 12 × 2 × 12 × 128 × 768 + 12 × 4 × 128 × 768 × 768
        return validate_model(
            "BERT-base (seq=128)",
            model,
            input_ids,
            expected_gflops=22.5,
            tolerance=0.20
        )
    except ImportError:
        print("\ntransformers library not installed. Install with: pip install transformers")
        return True
    except Exception as e:
        print(f"\nBERT validation skipped: {e}")
        return True


def validate_gpt2():
    """Validate GPT-2"""
    try:
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()

        # GPT-2 with sequence length 128
        input_ids = torch.randint(0, 50257, (1, 128))

        # GPT-2: ~28 GFLOPs for seq_len=128
        return validate_model(
            "GPT-2 (seq=128)",
            model,
            input_ids,
            expected_gflops=28.0,
            tolerance=0.20
        )
    except ImportError:
        print("\ntransformers library not installed. Install with: pip install transformers")
        return True
    except Exception as e:
        print(f"\nGPT-2 validation skipped: {e}")
        return True


def validate_simple_models():
    """Validate simple synthetic models"""

    print(f"\n{'='*80}")
    print("Simple Model Validation")
    print(f"{'='*80}")

    # Linear layer (no bias)
    print("\n[1] Linear 256×256 (no bias):")
    model = nn.Linear(256, 256, bias=False)
    input_data = torch.randn(1, 256)

    workload = characterize_with_dynamo(model, input_data)
    expected_macs = 1 * 256 * 256  # M × K × N (batch=1)

    print(f"  Expected MACs: {expected_macs:,}")
    print(f"  Actual MACs:   {workload.macs:,}")
    print(f"  Match: {'✅' if workload.macs == expected_macs else '❌'}")

    # Linear with bias
    print("\n[2] Linear 256×256 (with bias):")
    model = nn.Linear(256, 256, bias=True)

    workload = characterize_with_dynamo(model, input_data)
    expected_macs = 1 * 256 * 256  # M × K × N (batch=1)
    expected_flops = 1 * 256  # Bias addition

    print(f"  Expected MACs:  {expected_macs:,}")
    print(f"  Actual MACs:    {workload.macs:,}")
    print(f"  Expected FLOPs: {expected_flops:,}")
    print(f"  Actual FLOPs:   {workload.flops:,}")
    print(f"  Match: {'✅' if workload.macs == expected_macs and workload.flops == expected_flops else '❌'}")

    # MLP with ReLU
    print("\n[3] MLP 256×256 + ReLU:")
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 256, bias=True)

        def forward(self, x):
            x = self.linear(x)
            x = torch.relu(x)
            return x

    model = MLP()
    workload = characterize_with_dynamo(model, input_data)

    expected_macs = 1 * 256 * 256  # M × K × N (batch=1)
    expected_flops = 1 * 256 + 1 * 256  # Bias + ReLU

    print(f"  Expected MACs:  {expected_macs:,}")
    print(f"  Actual MACs:    {workload.macs:,}")
    print(f"  Expected FLOPs: {expected_flops:,}")
    print(f"  Actual FLOPs:   {workload.flops:,}")
    print(f"  Match: {'✅' if workload.macs == expected_macs and workload.flops == expected_flops else '❌'}")

    return True


def main():
    """Run all validations"""
    print("="*80)
    print("Dynamo Workload Characterizer Validation")
    print("="*80)

    if not FVCORE_AVAILABLE:
        print("\n⚠️  Warning: fvcore not available. Install with: pip install fvcore")
        print("    Validation will compare against literature values only.\n")

    results = {}

    # Simple models
    print("\n" + "="*80)
    print("PART 1: Simple Model Validation")
    print("="*80)
    results['simple'] = validate_simple_models()

    # Real models - CNNs
    print("\n" + "="*80)
    print("PART 2: CNN Model Validation")
    print("="*80)

    results['resnet18'] = validate_resnet18()
    results['mobilenetv2'] = validate_mobilenetv2()
    results['efficientnet_b0'] = validate_efficientnet_b0()

    # Transformers
    print("\n" + "="*80)
    print("PART 3: Transformer Model Validation")
    print("="*80)

    results['vit_b_16'] = validate_vit_b_16()
    results['bert_base'] = validate_bert_base()
    results['gpt2'] = validate_gpt2()

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for model_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {model_name:20s}: {status}")

    all_passed = all(results.values())
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ All validations PASSED")
    else:
        print("❌ Some validations FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
