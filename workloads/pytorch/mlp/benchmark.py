import torch
import torch.nn as nn

# This code defines a generic MLP builder, runs benchmarks, 
# and profiles the model using VizTracer.
# It supports multiple configurations for MLPs and can be 
# easily extended for more complex models.

MLP_CONFIGS = {
    "small":  {"hidden_dims": [512], "output_dim": 64},
    "medium": {"hidden_dims": [2048, 2048], "output_dim": 128},
    "large":  {"hidden_dims": [8192, 8192, 8192], "output_dim": 256},
#    "xlarge": {"hidden_dims": [32768, 32768, 32768, 32768], "output_dim": 512}
}

def analyze_mlp_memory(input_dim, hidden_dims, output_dim, batch_size=10, dtype_size=4):
    """
    Returns a memory breakdown for a given MLP configuration.
    """
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]

    total_params = 0
    total_param_mem = 0
    total_act_mem = 0

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]

        # Parameters
        weight_params = in_dim * out_dim
        bias_params = out_dim
        layer_params = weight_params + bias_params
        layer_param_mem = layer_params * dtype_size

        # Activations
        act_mem = batch_size * out_dim * dtype_size

        layers.append({
            "layer": f"fc{i+1}",
            "params": layer_params,
            "param_mem": layer_param_mem,
            "act_mem": act_mem
        })

        total_params += layer_params
        total_param_mem += layer_param_mem
        total_act_mem += act_mem

    # Add total row
    layers.append({
        "layer": "Total",
        "params": total_params,
        "param_mem": total_param_mem,
        "act_mem": total_act_mem
    })

    return layers

def print_memory_table(layers):
    print(f"| {'Layer':<6} | {'Parameters':>10} | {'Param Memory':>12} | {'Activation Memory':>17} |")
    print("|--------|--------------|--------------|-------------------|")
    for layer in layers:
        params = layer["params"]
        param_mem = layer["param_mem"]
        act_mem = layer["act_mem"]

        def fmt_bytes(b):
            if b >= 1 << 30:
                return f"~{b / (1 << 30):.2f} GB"
            elif b >= 1 << 20:
                return f"~{b / (1 << 20):.2f} MB"
            elif b >= 1 << 10:
                return f"~{b / (1 << 10):.2f} KB"
            else:
                return f"{b} B"

        print(f"| {layer['layer']:<6} | {params/1e6:>9.2f}M | {fmt_bytes(param_mem):>12} | {fmt_bytes(act_mem):>17} |")

def print_comparative_table(results):
    print(f"| {'Model':<8} | {'Hidden Dims':<20} | {'Params':>10} | {'Param Mem':>12} | {'Act Mem':>12} |")
    print("|----------|----------------------|------------|--------------|--------------|")

    def fmt_bytes(b):
        if b >= 1 << 30:
            return f"~{b / (1 << 30):.2f} GB"
        elif b >= 1 << 20:
            return f"~{b / (1 << 20):.2f} MB"
        elif b >= 1 << 10:
            return f"~{b / (1 << 10):.2f} KB"
        else:
            return f"{b} B"

    for r in results:
        print(f"| {r['name']:<8} | {str(r['hidden_dims']):<20} | {r['total_params']/1e6:>9.2f}M | {fmt_bytes(r['param_mem']):>12} | {fmt_bytes(r['act_mem']):>12} |")


def batch_analyze_configs(configs, input_dim=1024, batch_size=10):
    results = []

    for name, cfg in configs.items():
        layers = analyze_mlp_memory(
            input_dim=input_dim,
            hidden_dims=cfg["hidden_dims"],
            output_dim=cfg["output_dim"],
            batch_size=batch_size
        )
        total = layers[-1]
        results.append({
            "name": name,
            "hidden_dims": cfg["hidden_dims"],
            "output_dim": cfg["output_dim"],
            "total_params": total["params"],
            "param_mem": total["param_mem"],
            "act_mem": total["act_mem"]
        })

    return results

# generic MLP builder
def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)

# benchmark runner
import time
def run_model(model, input_tensor):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    output = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    return output, end - start

def benchmark_model(model, input_tensor, num_runs=10):
    for _ in range(3):  # Warmup
        _ = model(input_tensor)
    times = []
    for _ in range(num_runs):
        _, t = run_model(model, input_tensor)
        times.append(t)
    return times

# VizTracer for profiling
from viztracer import VizTracer

def profile_model(model, input_tensor, label="mlp_profile"):
    tracer = VizTracer(output_file=f"{label}.json", ignore_c_function=True)
    tracer.start()
    _ = model(input_tensor)
    tracer.stop()
    tracer.save()

def main():
    input_dim = 1024
    batch_size = 10
    input_tensor = torch.randn(batch_size, input_dim)

    for name, cfg in MLP_CONFIGS.items():
        print(f"\nRunning config: {name}")
        model = build_mlp(input_dim, cfg["hidden_dims"], cfg["output_dim"])
        model.eval()

        # Profile with VizTracer
        profile_model(model, input_tensor, label=f"profile_{name}")

        # Benchmark
        times = benchmark_model(model, input_tensor)
        print(f"Avg time: {sum(times)/len(times):.6f}s")

if __name__ == "__main__":
    results = batch_analyze_configs(MLP_CONFIGS)
    print_comparative_table(results)

    main()
