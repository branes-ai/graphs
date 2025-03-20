import torch
import iree.compiler as ireec

def convert_bin_to_mlir(bin_path, example_input, output_mlir_path="model.mlir"):
    """
    Converts a PyTorch .bin file to an MLIR file.

    Args:
        bin_path: Path to the PyTorch .bin file.
        example_input: An example input tensor for tracing.
        output_mlir_path: Path to save the MLIR file.
    """
    try:
        # Load the PyTorch model from the .bin file
        model = torch.jit.load(bin_path)

        # Convert the loaded model to MLIR
        mlir_module = ireec.torch_mlir.compile_module(model, example_input)

        # Save the MLIR module to a file
        with open(output_mlir_path, "w") as f:
            f.write(str(mlir_module))

        print(f"MLIR file saved to: {output_mlir_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Create a simple example .bin file (replace with your actual .bin file)
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 3)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    example_input = torch.randn(1, 5)
    bin_file = "simple_model.bin"
    torch.jit.save(torch.jit.trace(model, example_input), bin_file) # create example bin file.

    mlir_file = "simple_model.mlir"
    convert_bin_to_mlir(bin_file, example_input, mlir_file)
