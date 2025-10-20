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
        K_h, K_w = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        S_h, S_w = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
        P = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
        groups = mod.groups

        H_out = (H - K_h + 2 * P) // S_h + 1
        W_out = (W - K_w + 2 * P) // S_w + 1

        # Check if depthwise convolution
        if groups > 1 and groups == C_in == C_out:
            # Depthwise + BN + ReLU: K_h * K_w * 2 MACs + 2 ops per output
            flops = B * C_out * H_out * W_out * (2 * K_h * K_w + 2)
            mem = B * C_in * H * W * 4 + C_out * K_h * K_w * 4 + B * C_out * H_out * W_out * 4
        else:
            # Standard or grouped convolution + BN + ReLU
            C_in_per_group = C_in // groups
            flops = B * C_out * H_out * W_out * (2 * C_in_per_group * K_h * K_w + 2)
            mem = B * C_in * H * W * 4 + C_out * C_in_per_group * K_h * K_w * 4 + B * C_out * H_out * W_out * 4

        return flops, mem

class ConvReLUEstimator(FusedEstimator):
    def estimate(self, fused_nodes):
        conv_node = fused_nodes[0]
        meta = conv_node.meta.get('tensor_meta')
        mod = conv_node.graph.owning_module.get_submodule(conv_node.target)
        if not meta:
            return 0, 0
        B, C_in, H, W = meta.shape
        C_out = mod.out_channels
        K_h, K_w = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        S_h, S_w = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
        P = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
        groups = mod.groups

        H_out = (H - K_h + 2 * P) // S_h + 1
        W_out = (W - K_w + 2 * P) // S_w + 1

        # Check if depthwise convolution (groups == in_channels == out_channels)
        if groups > 1 and groups == C_in == C_out:
            # Depthwise convolution: each channel processed separately
            # FLOPs: K_h * K_w * 2 MACs per output element per channel + 1 ReLU op
            flops = B * C_out * H_out * W_out * (2 * K_h * K_w + 1)
            mem = B * C_in * H * W * 4 + C_out * K_h * K_w * 4 + B * C_out * H_out * W_out * 4
        else:
            # Standard or grouped convolution
            # FLOPs: 2*C_in*K_h*K_w MACs per output + 1 ReLU op
            C_in_per_group = C_in // groups
            flops = B * C_out * H_out * W_out * (2 * C_in_per_group * K_h * K_w + 1)
            mem = B * C_in * H * W * 4 + C_out * C_in_per_group * K_h * K_w * 4 + B * C_out * H_out * W_out * 4

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

class Conv2dEstimator(FusedEstimator):
    """Standalone Conv2d without activation (for final layers, 1x1 pointwise, etc.)"""
    def estimate(self, fused_nodes):
        conv_node = fused_nodes[0]
        meta = conv_node.meta.get('tensor_meta')
        mod = conv_node.graph.owning_module.get_submodule(conv_node.target)
        if not meta:
            return 0, 0
        B, C_in, H, W = meta.shape
        C_out = mod.out_channels
        K_h, K_w = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        S_h, S_w = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
        P = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
        groups = mod.groups

        H_out = (H - K_h + 2 * P) // S_h + 1
        W_out = (W - K_w + 2 * P) // S_w + 1

        # Check if depthwise convolution
        if groups > 1 and groups == C_in == C_out:
            # Depthwise: K_h * K_w * 2 MACs per output element per channel
            flops = B * C_out * H_out * W_out * 2 * K_h * K_w
            mem = B * C_in * H * W * 4 + C_out * K_h * K_w * 4 + B * C_out * H_out * W_out * 4
        else:
            # Standard or grouped convolution
            C_in_per_group = C_in // groups
            flops = B * C_out * H_out * W_out * 2 * C_in_per_group * K_h * K_w
            mem = B * C_in * H * W * 4 + C_out * C_in_per_group * K_h * K_w * 4 + B * C_out * H_out * W_out * 4

        return flops, mem

class ConvReLU6Estimator(FusedEstimator):
    """Conv2d + ReLU6 (used in MobileNet)"""
    def estimate(self, fused_nodes):
        conv_node = fused_nodes[0]
        meta = conv_node.meta.get('tensor_meta')
        mod = conv_node.graph.owning_module.get_submodule(conv_node.target)
        if not meta:
            return 0, 0
        B, C_in, H, W = meta.shape
        C_out = mod.out_channels
        K_h, K_w = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        S_h, S_w = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
        P = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
        groups = mod.groups

        H_out = (H - K_h + 2 * P) // S_h + 1
        W_out = (W - K_w + 2 * P) // S_w + 1

        # Check if depthwise convolution
        if groups > 1 and groups == C_in == C_out:
            # Depthwise + ReLU6: K_h * K_w * 2 MACs + 2 ops (clamp) per output
            flops = B * C_out * H_out * W_out * (2 * K_h * K_w + 2)
            mem = B * C_in * H * W * 4 + C_out * K_h * K_w * 4 + B * C_out * H_out * W_out * 4
        else:
            # Standard + ReLU6
            C_in_per_group = C_in // groups
            flops = B * C_out * H_out * W_out * (2 * C_in_per_group * K_h * K_w + 2)
            mem = B * C_in * H * W * 4 + C_out * C_in_per_group * K_h * K_w * 4 + B * C_out * H_out * W_out * 4

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
    # Longer patterns first (greedy matching)
    reg.register("conv_bn_relu", [nn.Conv2d, nn.BatchNorm2d, nn.ReLU], ConvBnReLUEstimator())
    reg.register("conv_bn_relu6", [nn.Conv2d, nn.BatchNorm2d, nn.ReLU6], ConvReLU6Estimator())
    reg.register("conv_bn_hardswish", [nn.Conv2d, nn.BatchNorm2d, nn.Hardswish], ConvReLU6Estimator())
    # Two-layer patterns
    reg.register("conv_relu6", [nn.Conv2d, nn.ReLU6], ConvReLU6Estimator())
    reg.register("conv_relu", [nn.Conv2d, nn.ReLU], ConvReLUEstimator())
    reg.register("conv_hardswish", [nn.Conv2d, nn.Hardswish], ConvReLU6Estimator())
    reg.register("linear_relu", [nn.Linear, nn.ReLU], LinearReLUEstimator())
    # Standalone operations (fallback)
    reg.register("conv2d", [nn.Conv2d], Conv2dEstimator())
    return reg
