import argparse
import torch
import pandas as pd

#
# pipeline for modeling the latency, resource usage, and energy of the hw/sw stack
#
#   experiment
#               contains programs and utilities to set up estimators for power and energy
#   subgraphs
#               contains nn.Modules that represent a synthetic subgraph of a full DL graph
#   compile
#               contains the tiling algorithms to map subgraph onto hw configuration constraints
#   execute
#               contains workload models of the execution on different hardware configurations
#   hw
#               contains hardware configuration profiles
#

from graphs.experiment.sweep import SweepHarness

from graphs.subgraphs.mlp import make_mlp
from graphs.subgraphs.conv2d_stack import make_conv2d
from graphs.subgraphs.resnet_block import make_resnet_block
from graphs.compile.tiling import CPUTilingStrategy
from graphs.execute.fused_ops import default_registry
from graphs.hw.arch_profiles import cpu_profile, gpu_profile, tpu_profile, kpu_profile

def get_subgraph(name, batch_size):
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
    parser = argparse.ArgumentParser(description="Run subgraph EDP estimation")
    parser.add_argument("--subgraph", type=str, required=True, choices=["MLP", "Conv2D", "ResNetBlock"])
    parser.add_argument("--arch", type=str, required=True, choices=["CPU", "GPU", "TPU", "KPU"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, help="Path to save results CSV")

    args = parser.parse_args()

    subgraph, input_tensor = get_subgraph(args.subgraph, args.batch_size)
    arch_profile = get_arch_profile(args.arch)
    fused_registry = default_registry()

    harness = SweepHarness(
        subgraphs={args.subgraph: subgraph},
        inputs={args.subgraph: input_tensor},
        arch_profiles=[arch_profile],
        fused_registry=fused_registry
    )

    results = harness.run()
    metrics = results[0]["metrics"]
    print(f"\n{args.subgraph} on {args.arch}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.output:
        df = pd.DataFrame([{**{"Model": args.subgraph, "Architecture": args.arch}, **metrics}])
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    main()
