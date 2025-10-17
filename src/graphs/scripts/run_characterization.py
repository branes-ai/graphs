# scripts/run_characterization.py

import torch
import pandas as pd
from graphs.models.mlp import make_mlp
from graphs.models.conv2d_stack import make_conv2d
from graphs.models.resnet_block import make_resnet_block

from graphs.characterize.arch_profiles import (
    intel_i7_profile, amd_ryzen7_profile, h100_pcie_profile,
    tpu_v4_profile, kpu_t2_profile, kpu_t100_profile
)
from graphs.characterize.fused_ops import default_registry
from graphs.characterize.sweep import SweepHarness

def main():
    # Define models and inputs
    models = {
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
    arch_profiles = [
        intel_i7_profile, amd_ryzen7_profile, h100_pcie_profile,
        tpu_v4_profile, kpu_t2_profile, kpu_t100_profile
    ]
    fused_registry = default_registry()

    # Run sweep
    harness = SweepHarness(models, inputs, arch_profiles, fused_registry)
    results = harness.run()

    # Print and save
    rows = []
    for entry in results:
        print(f"{entry['model']} on {entry['arch']}: {entry['metrics']}")
        row = {
            "Model": entry["model"],
            "Architecture": entry["arch"],
            **entry["metrics"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = "results/validation/sweep_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()

