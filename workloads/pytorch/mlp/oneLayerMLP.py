import torch
import torch.nn as nn
import iree.runtime as ireert
import iree.turbine.aot as aot

# Define a 1-layer MLP
class OneLayerMLP(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OneLayerMLP, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.fc(x)
        return self.softmax(y)
        
if __name__ == "__main__":
    # Example usage
    in_features  = 5
    out_features = 3
    model = OneLayerMLP(in_features, out_features)

    # Example input
    batch_size = 10
    x = torch.randn(batch_size, in_features)
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
