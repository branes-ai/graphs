import argparse
import torch
import pandas as pd

from graphs.models.mlp import make_mlp
from graphs.models.conv2d_stack import make_conv2d
from graphs.models.resnet_block import make_resnet_block

from graphs.characterize.arch_profiles import cpu_profile, gpu_profile, tpu_profile, kpu_profile
from graphs.characterize.fused_ops import default_registry
from graphs.characterize.sweep import SweepHarness

def get_model(name, batch_size):
    if name == "MLP":
        return make_mlp(128, 256, 64), torch.randn(batch_size, 128)
    elif name == "Conv2D":
        return make_conv2d(3, 16, 3), torch.randn(batch_size, 3, 64, 64)
    elif name == "ResNetBlock":
        return make_resnet_block(64, 128), torch.randn(batch_size, 64, 56, 56)
    else:
        raise ValueError(f"Unknown model: {name}")

def get_arch_profile(name):
    profiles = {
        "CPU": cpu_profile,
        "GPU": gpu_profile,
        "TPU": tpu_profile,
        "KPU": kpu_profile
    }
    if name not in profiles:
        raise ValueError(f"Unknown architecture: {name}")
    return profiles[name]

def main():
    parser = argparse.ArgumentParser(description="Run model characterization")
    parser.add_argument("--model", type=str, required=True, choices=["MLP", "Conv2D", "ResNetBlock"])
    parser.add_argument("--arch", type=str, required=True, choices=["CPU", "GPU", "TPU", "KPU"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, help="Path to save results CSV")

    args = parser.parse_args()

    model, input_tensor = get_model(args.model, args.batch_size)
    arch_profile = get_arch_profile(args.arch)
    fused_registry = default_registry()

    harness = SweepHarness(
        models={args.model: model},
        inputs={args.model: input_tensor},
        arch_profiles=[arch_profile],
        fused_registry=fused_registry
    )

    results = harness.run()
    metrics = results[0]["metrics"]
    print(f"\n{args.model} on {args.arch}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.output:
        df = pd.DataFrame([{**{"Model": args.model, "Architecture": args.arch}, **metrics}])
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    main()
