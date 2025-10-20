# run_characterization.py

from graphs.models.mlp import make_mlp
from graphs.characterize.arch_profiles import cpu_profile, gpu_profile
from graphs.characterize.fused_ops import default_registry
from graphs.characterize.sweep import SweepHarness

model = make_mlp(in_dim=128, hidden_dim=256, out_dim=64)
input_tensor = torch.randn(32, 128)

sweep = SweepHarness(
    models={"MLP": model},
    inputs={"MLP": input_tensor},
    arch_profiles=[cpu_profile, gpu_profile],
    fused_registry=default_registry
)

results = sweep.run()
