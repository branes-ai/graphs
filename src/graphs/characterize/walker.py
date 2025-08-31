# characterize/walker.py

class FXGraphWalker:
    def __init__(self, arch_profile, fused_registry=None):
        self.arch = arch_profile
        self.fused_registry = fused_registry

    def walk(self, fx_graph):
        module_lookup = dict(fx_graph.named_modules())
        nodes = [n for n in fx_graph.graph.nodes if n.op == 'call_module']
        fused_ops = self.fused_registry.match(nodes, module_lookup) if self.fused_registry else []

        metrics = {
            "FLOPs": 0,
            "Memory": 0,
            "Tiles": 0,
            "Latency": 0.0,
            "Energy": 0.0
        }

        for name, fused_nodes, estimator in fused_ops:
            flops, mem = estimator.estimate(fused_nodes)
            shape = fused_nodes[0].meta['tensor_meta'].shape
            tiles = self.arch.tiling_strategy.compute_tile_count(name, {"input_shape": shape})
            latency = flops / self.arch.peak_flops * self.arch.scheduler_model(None) * (1 + 0.05 * (tiles - 1))
            energy = flops * self.arch.energy_per_flop + mem * self.arch.energy_per_byte
            energy *= (1 + 0.05 * (tiles - 1))

            metrics["FLOPs"] += flops
            metrics["Memory"] += mem
            metrics["Tiles"] += tiles
            metrics["Latency"] += latency
            metrics["Energy"] += energy

        return metrics
