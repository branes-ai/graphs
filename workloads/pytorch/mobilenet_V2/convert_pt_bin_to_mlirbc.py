import sys
import torch
from transformers import MobileNetV2ForImageClassification
from iree.turbine import aot  # Ahead-of-time compilation module

class MobileNetV2Wrapper(torch.nn.Module):
    """Wrapper to return only logits from MobileNetV2ForImageClassification."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits  # Return only the logits tensor

def convert_bin_to_mlir(input_pt_bin_file, output_mlir_file="model.mlir"):
    """
    Converts a PyTorch .bin weights file to an MLIR file using IREE Turbine Python API.

    Args:
        input_pt_bin_file: Path to the PyTorch .bin weights file.
        output_mlir_file: Path to save the MLIR file.
    """
    try:
        # Load the MobileNetV2 model from transformers
        model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224", local_files_only=True)
        model.eval()  # Set to evaluation mode

        # Load the weights from the .bin file into the model
        try:
            state_dict = torch.load(input_pt_bin_file, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {input_pt_bin_file} into MobileNetV2 (transformers) model.")
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {input_pt_bin_file}: {e}")

        # Wrap the model to return only logits
        wrapped_model = MobileNetV2Wrapper(model)
        wrapped_model.eval()

        # Define an example input (batch_size=1, channels=3, height=224, width=224)
        example_input = torch.rand(1, 3, 224, 224)

        # Trace the wrapped model with the example input
        traced_model = torch.jit.trace(wrapped_model, example_input, strict=False)
        print("Model traced successfully.")

        # Use iree-turbine AOT to export to MLIR
        try:
            # Create an output module for export
            output_module = aot.export(wrapped_model, args=(example_input,))
            
            # Compile to MLIR and save directly
            output_module.compile(
                target_backends=["llvm-cpu"],  # Adjust backend as needed (e.g., "vulkan", "cuda")
                save_to=output_mlir_file       # Specify the output MLIR file
            )
            print(f"MLIR file saved to: {output_mlir_file}")

        except Exception as e:
            raise RuntimeError(f"Failed to compile model to MLIR with iree-turbine: {e}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_pt_bin_file}' not found.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python KK_convert_ptbin_to_mlir.py <input_file> <output_file>")
        sys.exit(1)

    input_pt_bin_file = sys.argv[1]
    output_mlir_file = sys.argv[2]
    convert_bin_to_mlir(input_pt_bin_file, output_mlir_file)