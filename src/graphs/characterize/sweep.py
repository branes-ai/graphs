# characterize/sweep.py

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from graphs.characterize.walker import FXGraphWalker

class SweepHarness:
    def __init__(self, models, inputs, arch_profiles, fused_registry):
        """
        models: dict[str, nn.Module]
        inputs: dict[str, torch.Tensor]
        arch_profiles: list[ArchitectureProfile]
        fused_registry: FusedOpRegistry
        """
        self.models = models
        self.inputs = inputs
        self.arch_profiles = arch_profiles
        self.fused_registry = fused_registry

    def trace_and_propagate(self, model, input_tensor):
        fx_graph = symbolic_trace(model)
        shape_prop = ShapeProp(fx_graph)
        shape_prop.propagate(input_tensor)
        return fx_graph

    def run(self):
        results = []
        for model_name, model in self.models.items():
            input_tensor = self.inputs[model_name]
            fx_graph = self.trace_and_propagate(model, input_tensor)

            for arch in self.arch_profiles:
                walker = FXGraphWalker(arch, self.fused_registry)
                metrics = walker.walk(fx_graph)

                results.append({
                    "model": model_name,
                    "arch": arch.name,
                    "metrics": metrics
                })

        return results
