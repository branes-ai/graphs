import torch
import iree.compiler as ireec
import iree.runtime as ireert

def convert_pytorch_to_iree(model, example_input, output_path="model.vmfb"):
    """
    Converts a PyTorch model to an IREE executable.

    Args:
        model: The PyTorch model (torch.nn.Module).
        example_input: An example input tensor for tracing.
        output_path: The path to save the IREE executable.
    """
    try:
        # Trace the PyTorch model
        module = torch.jit.trace(model, example_input)

        # Convert to IREE's MLIR format
        mlir_module = ireec.torch_mlir.compile_module(module, example_input)

        # Compile to IREE executable
        flatbuffer_blob = ireec.compile_str(
            str(mlir_module),
            target_backends=["llvm-cpu"],  # Specify target backend (e.g., llvm-cpu, vulkan-spirv)
            input_type="tm_tensor",
            output_format="vmfb",
        )

        with open(output_path, "wb") as f:
            f.write(flatbuffer_blob)

        print(f"IREE executable saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def run_iree_model(vmfb_path, input_tensor):
    """
    Runs an IREE model with a given input tensor.

    Args:
        vmfb_path: The path to the IREE executable.
        input_tensor: The input tensor as a NumPy array.
    """
    try:
        config = ireert.Config("local-task") #or local-sync, depending on your needs.
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.mmap(vmfb_path, ctx.instance)
        ctx.add_vm_module(vm_module)

        func = vm_module["forward"] # or the function name of your model.
        results = func(input_tensor)

        print("IREE model output:", results)

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage (replace with your model and input)
if __name__ == "__main__":
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 3)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    example_input = torch.randn(1, 5)  # Example input tensor
    vmfb_file = "simple_model.vmfb"

    convert_pytorch_to_iree(model, example_input, vmfb_file)

    # Run the compiled model
    import numpy as np
    input_numpy = np.random.rand(1, 5).astype(np.float32)
    run_iree_model(vmfb_file, input_numpy)
