import torch
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

    # Compile the program using the TorchInductor backend.
    #opt_1l_mlp_module = torch.compile(model, backend="inductor")

    # Compile the program using the turbine backend.
    opt_1l_mlp_module = torch.compile(model, backend="turbine_cpu")
  
    # Use the compiled program as you would the original program.
    turbine_output = opt_1l_mlp_module(x)
    print("Output of compiled 1-layer MLP:", turbine_output)

    # Export the program using the simple API.
    print("Exporting compiled graph")
    export_output = aot.export(model, x)

    # Compile to a deployable artifact.
    binary = export_output.compile(save_to=None)

    # Use the IREE runtime API to test the compiled program.
    config = ireert.Config("local-task")
    vm_module = ireert.load_vm_module(
       ireert.VmModule.copy_buffer(config.vm_instance, binary.map_memory()),
       config,
    )
    #input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = vm_module.main(x)
    print(result.to_host())
