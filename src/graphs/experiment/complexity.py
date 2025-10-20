# scripts/run_characterization.py

import torch
import pandas as pd
from graphs.subgraphs.mlp import make_mlp
from graphs.subgraphs.conv2d_stack import make_conv2d
from graphs.subgraphs.resnet_block import make_resnet_block

from graphs.hw.arch_profiles import cpu_profile, gpu_profile, tpu_profile, kpu_profile
from graphs.execute.fused_ops import default_registry
from graphs.experiment.sweep import SweepHarness

def main():
    # Define subgraphs and inputs
    subgraphs = {
        "MLP": make_mlp(in_dim=128, hidden_dim=256, out_dim=64),
        "Conv2D": make_conv2d(in_channels=3, out_channels=16, kernel_size=3),
        "ResNetBlock": make_resnet_block(in_channels=64, out_channels=128)
    }

    inputs = {
        "MLP": torch.randn(32, 128),
        "Conv2D": torch.randn(32, 3, 64, 64),
        "ResNetBlock": torch.randn(32, 64, 56, 56)
    }

    # Architecture profiles and fusion registry
    arch_profiles = [cpu_profile, gpu_profile, tpu_profile, kpu_profile]
    fused_registry = default_registry()

    # Run sweep
    harness = SweepHarness(subgraphs, inputs, arch_profiles, fused_registry)
    results = harness.run()

    # Print and save
    rows = []
    for entry in results:
        print(f"{entry['subgraph']} on {entry['arch']}: {entry['metrics']}")
        row = {
            "Subgraph": entry["subgraph"],
            "Architecture": entry["arch"],
            **entry["metrics"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("sweep_results.csv", index=False)
    print("\nSaved results to sweep_results.csv")

if __name__ == "__main__":
    main()

