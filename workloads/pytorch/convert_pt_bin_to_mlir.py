import sys
import torch
import iree.compiler as ireec

def convert_bin_to_mlir(input_pt_bin_file, example_input, output_mlir_file="model.mlir"):
    """
    Converts a PyTorch .bin file to an MLIR file.

    Args:
        bin_path: Path to the PyTorch .bin file.
        example_input: An example input tensor for tracing.
        output_mlir_path: Path to save the MLIR file.
    """
    try:
        # Load the PyTorch model from the .bin file
        try:
            model = torch.jit.load(input_pt_bin_file)
        except FileNotFoundError:
            print("error: pt bin file not found.")
        except RuntimeError:
            print("error loading model: {e}")

        # Convert the loaded model to MLIR
        mlir_module = ireec.torch_mlir.compile_module(model, example_input)

        # Save the MLIR module to a file
        with open(output_mlir_file, "w") as f:
            f.write(str(mlir_module))

        print(f"MLIR file saved to: {output_mlir_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_pt_bin_to_mlir.py <input_file> <output_file>")
        sys.exit(1)

    input_pt_bin_file = sys.argv[1]
    output_mlir_file = sys.argv[2]

    # how do you adapt the example input to the model contained in the bin file?
    # where does that information come from?
    example_input = torch.rand(10)
    convert_bin_to_mlir(input_pt_bin_file, example_input, output_mlir_file)
