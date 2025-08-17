import torch
import torch.nn as nn
import iree.runtime as ireert
import iree.turbine.aot as aot

from multi_layer_perceptrons import OneLayerMLP, MLP_CONFIGS 
        
if __name__ == "__main__":
    # Define a small 1-layer MLP model.
    input_dim  = 1024
    output_dim = MLP_CONFIGS["small"]["output_dim"]
    model = OneLayerMLP(input_dim, output_dim)

    # Example input
    batch_size = 10
    x = torch.randn(batch_size, input_dim)
    y = model(x)
    print("Output of 1-layer MLP:", y)