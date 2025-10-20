# execute/fused_ops.py

import torch.nn as nn

class FusedEstimator:
    def estimate(self, fused_nodes):
        raise NotImplementedError("Override in subclass")

class ConvBnReLUEstimator(FusedEstimator):
    def estimate(self, fused_nodes):
        conv_node = fused_nodes[0]
        meta = conv_node.meta.get('tensor_meta')
        mod = conv_node.graph.owning_module.get_submodule(conv_node.target)
        if not meta:
            return 0, 0
        B, C_in, H, W = meta.shape
        C_out = mod.out_channels
        K_h, K_w = mod.kernel_size
        S_h, S_w = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
        P = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
        H_out = (H - K_h + 2 * P) // S_h + 1
        W_out = (W - K_w + 2 * P) // S_w + 1
        flops = B * C_out * H_out * W_out * (2 * C_in * K_h * K_w + 2)
        mem = B * C_in * H * W * 4 + C_out * C_in * K_h * K_w * 4 + B * C_out * H_out * W_out * 4
        return flops, mem

class LinearReLUEstimator(FusedEstimator):
    def estimate(self, fused_nodes):
        linear_node = fused_nodes[0]
        meta = linear_node.meta.get('tensor_meta')
        mod = linear_node.graph.owning_module.get_submodule(linear_node.target)
        if not meta:
            return 0, 0
        B, D_in = meta.shape
        D_out = mod.out_features
        flops = B * D_in * D_out * 2 + B * D_out
        mem = B * D_in * 4 + D_in * D_out * 4 + B * D_out * 4
        return flops, mem

class FusedOpRegistry:
    def __init__(self):
        self.patterns = []

    def register(self, name, sequence, estimator):
        self.patterns.append({
            "name": name,
            "sequence": sequence,
            "estimator": estimator
        })

    def match(self, nodes, module_lookup):
        matches = []
        i = 0
        while i < len(nodes):
            for pattern in self.patterns:
                seq = pattern["sequence"]
                if i + len(seq) <= len(nodes):
                    window = nodes[i:i+len(seq)]
                    mods = [module_lookup[n.target] for n in window]
                    if all(isinstance(mods[j], seq[j]) for j in range(len(seq))):
                        matches.append((pattern["name"], window, pattern["estimator"]))
                        i += len(seq) - 1
                        break
            i += 1
        return matches

def default_registry():
    reg = FusedOpRegistry()
    reg.register("conv_bn_relu", [nn.Conv2d, nn.BatchNorm2d, nn.ReLU], ConvBnReLUEstimator())
    reg.register("linear_relu", [nn.Linear, nn.ReLU], LinearReLUEstimator())
    return reg
