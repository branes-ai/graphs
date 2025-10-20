# characterize/sweep.py

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from graphs.execute.walker import FXGraphWalker

class SweepHarness:
    def __init__(self, subgraphs, inputs, arch_profiles, fused_registry):
        """
        subgraphs: dict[str, nn.Module]
        inputs: dict[str, torch.Tensor]
        arch_profiles: list[ArchitectureProfile]
        fused_registry: FusedOpRegistry
        """
        self.subgraphs = subgraphs
        self.inputs = inputs
        self.arch_profiles = arch_profiles
        self.fused_registry = fused_registry

    def trace_and_propagate(self, subgraph, input_tensor):
        fx_graph = symbolic_trace(subgraph)
        shape_prop = ShapeProp(fx_graph)
        shape_prop.propagate(input_tensor)
        return fx_graph

    def run(self):
        results = []
        for name, subgraph in self.subgraphs.items():
            input_tensor = self.inputs[name]
            fx_graph = self.trace_and_propagate(subgraph, input_tensor)

            for arch in self.arch_profiles:
                walker = FXGraphWalker(arch, self.fused_registry)
                metrics = walker.walk(fx_graph)

                results.append({
                    "subgraph" : name,
                    "arch": arch.name,
                    "metrics" : metrics
                })

        return results
