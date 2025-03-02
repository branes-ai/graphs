import torch
import torch.nn as nn

# Define a 1-layer MLP
class OneLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLayerMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.fc(x)
        return self.softmax(y)
        
# Example usage
input_dim = 5
output_dim = 3
model = OneLayerMLP(input_dim, output_dim)

# Example input
batch_size = 10
x = torch.randn(batch_size, input_dim)
y = model(x)
print("Output of 1-layer MLP:", y)

# Compile the program using the turbine backend.
opt_1l_mlp_module = torch.compile(OneLayerMLP, backend="turbine_cpu")

# Use the compiled program as you would the original program.
turbine_output = opt_1l_mlp_module(x)
print("Output of compiled 1-layer MLP:", y)


# Export the program using the simple API.
#export_output = aot.export(OneLayerMLP, x)

# Compile to a deployable artifact.
#binary = export_output.compile(save_to=None)

# Use the IREE runtime API to test the compiled program.
#config = ireert.Config("local-task")
#vm_module = ireert.load_vm_module(
#    ireert.VmModule.copy_buffer(config.vm_instance, binary.map_memory()),
#    config,
#)
#input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
#result = vm_module.main(input)
#print(result.to_host())
