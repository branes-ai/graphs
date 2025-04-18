import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import iree.turbine.aot as aot
import iree.compiler as ireec
import iree.runtime as rt
import os
import time
import numpy as np
from PIL import Image
import logging
import gc
from transformers.models.detr.modeling_detr import DetrAttention
import torch.nn.functional as F

# Suppress all transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure mlir/ directory exists
os.makedirs("mlir", exist_ok=True)

# Clear temporary files (only if re-exporting is needed)
# for file in ["mlir/detr_resnet50.vmfb"]:
#     if os.path.exists(file):
#         os.remove(file)

batch_size = 8
N = 10

# Load and preprocess image
image = Image.open("dog.jpg").convert("RGB")
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_model.eval()

# Prepare input for DETR with explicit resizing and padding to (800, 1333)
detr_inputs = detr_processor(images=image, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 1333})
detr_pixel_values = detr_inputs["pixel_values"]  # Shape: [1, 3, 800, ~1066]
detr_pixel_mask = detr_inputs["pixel_mask"]      # Shape: [1, 800, ~1066]

# Pad pixel_values to ensure width is 1333
current_height, current_width = detr_pixel_values.shape[2:4]
pad_width = 1333 - current_width
if pad_width > 0:
    # Pad right side (width dimension) with zeros
    detr_pixel_values = F.pad(detr_pixel_values, (0, pad_width, 0, 0), mode='constant', value=0)
    detr_pixel_mask = F.pad(detr_pixel_mask, (0, pad_width, 0, 0), mode='constant', value=0)

# Expand to batch size
detr_pixel_values = detr_pixel_values.expand(batch_size, -1, -1, -1)  # [8, 3, 800, 1333]
detr_pixel_mask = detr_pixel_mask.expand(batch_size, -1, -1)          # [8, 800, 1333]
iree_input = detr_pixel_values  # [8, 3, 800, 1333]

# Verify input shape
print(f"Preprocessed input shape: {detr_pixel_values.shape}")

# Patch DetrAttention to match successful export
class PatchedDetrAttention(DetrAttention):
    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, object_queries=None, key_value_states=None, spatial_position_embeddings=None, output_attentions=False):
        # Handle variable hidden_states shape
        if len(hidden_states.size()) == 3:
            batch_size, seq_len, embed_dim = hidden_states.size()
        else:
            batch_size, seq_len = hidden_states.size()[:2]
            embed_dim = hidden_states.size(-1)

        # Use key_value_states if provided, else fall back to hidden_states
        input_states = key_value_states if key_value_states is not None else hidden_states

        # Use existing query, key, value projections from DetrAttention
        query = self.q_linear(hidden_states) if hasattr(self, 'q_linear') else hidden_states
        key = self.k_linear(input_states) if hasattr(self, 'k_linear') else input_states
        value = self.v_linear(input_states) if hasattr(self, 'v_linear') else input_states

        # Simplified attention without explicit head splitting
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / (embed_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = self.out_proj(attn_output)

        return attn_output, None  # Avoid returning attn_weights

# Replace attention layers in both encoder and decoder
for layer in detr_model.model.encoder.layers:
    layer.self_attn = PatchedDetrAttention(
        embed_dim=detr_model.config.d_model,
        num_heads=detr_model.config.encoder_attention_heads,
        dropout=detr_model.config.attention_dropout
    )
for layer in detr_model.model.decoder.layers:
    layer.self_attn = PatchedDetrAttention(
        embed_dim=detr_model.config.d_model,
        num_heads=detr_model.config.decoder_attention_heads,
        dropout=detr_model.config.attention_dropout
    )
    layer.encoder_attn = PatchedDetrAttention(
        embed_dim=detr_model.config.d_model,
        num_heads=detr_model.config.decoder_attention_heads,
        dropout=detr_model.config.attention_dropout
    )

# Simplified model to avoid post-processing
class DetrCore(torch.nn.Module):
    def __init__(self, detr_model):
        super().__init__()
        self.model = detr_model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits  # Return only logits [8, 100, 92]

# Export core model to MLIR
core_model = DetrCore(detr_model)
core_model.eval()
dummy_input = torch.randn(batch_size, 3, 800, 1333)  # Match benchmarking input
try:
    export_output = aot.export(core_model, args=(dummy_input,))
    export_output.save_mlir("mlir/detr_resnet50_pytorch.mlir")
    print("PyTorch MLIR saved to detr_resnet50_pytorch.mlir")
except Exception as e:
    print(f"Export failed: {e}")
    print("Check the log output above for Dynamo-related errors.")
    exit(1)

@torch.no_grad()
def torch_detr_forward(x: torch.Tensor, pixel_mask: torch.Tensor = None):
    outputs = core_model(x)
    return outputs  # Return logits [8, 100, 92]

# IREE module compilation and loading
def load_iree_module():
    try:
        # Verify runtime is functional
        rt.Config("local-task")
        # Use the pre-generated MLIR file
        mlir_file = "mlir/detr_resnet50_pytorch.mlir"
        if not os.path.exists(mlir_file):
            raise ValueError(f"MLIR file {mlir_file} not found")
        # Compile MLIR to IREE VMFB
        vmfb_file = "mlir/detr_resnet50.vmfb"
        compiled_binary = ireec.compile_file(
            mlir_file,
            target_backends=["llvm-cpu"],
            output_file=vmfb_file,
            extra_args=["--iree-llvmcpu-target-cpu=generic"]
        )
        if not os.path.exists(vmfb_file):
            raise ValueError("VMFB file was not generated")
        # Load VMFB
        config = rt.Config("local-task")
        vmm = rt.load_vm_module(
            rt.VmModule.from_flatbuffer(config.vm_instance, open(vmfb_file, "rb").read(), warn_if_copy=False),
            config
        )
        return vmm
    except Exception as e:
        print(f"IREE module loading failed: {str(e)}")
        return None

# Benchmark PyTorch loading and inference
start_load = time.perf_counter()
core_model.eval()
pytorch_load_time = (time.perf_counter() - start_load) * 1000

start = time.perf_counter()
pytorch_outputs = []
for _ in range(N):
    output = torch_detr_forward(detr_pixel_values, detr_pixel_mask)  # [8, 100, 92]
    pytorch_outputs.append(output.numpy())
pytorch_detr_time = (time.perf_counter() - start) / N * 1000
pytorch_output = np.mean(pytorch_outputs, axis=0)  # Average output for MAE

# Benchmark IREE loading and inference
start_load = time.perf_counter()
detr_vmm = load_iree_module()
iree_load_time = (time.perf_counter() - start_load) * 1000
iree_success = detr_vmm is not None

iree_detr_time = None
iree_output = None
if iree_success:
    # Input is already [8, 3, 800, 1333]
    iree_input_np = iree_input.numpy().astype(np.float32)  # [8, 3, 800, 1333]
    start = time.perf_counter()
    iree_outputs = []
    for _ in range(N):
        output = detr_vmm.main(iree_input_np)  # Returns logits
        # Convert to NumPy
        logits = np.array(output.to_host())  # Convert logits to NumPy
        print(f"IREE output size: {logits.size}, shape: {logits.shape}")  # Debug output
        logits = logits.reshape(batch_size, 8, 100, 92)  # [8, 8, 100, 92]
        iree_outputs.append(logits)
    iree_detr_time = (time.perf_counter() - start) / N * 1000
    iree_output = np.mean(iree_outputs, axis=0)  # Average output for MAE

# Compute Mean Absolute Error (MAE) if both outputs are available
mae = None
if iree_output is not None:
    mae = np.mean(np.abs(pytorch_output - iree_output))

# Print results
print(f"🏁 DETR BENCHMARK RESULTS ({N} runs)")
print("----------------------------------------")
print(f"PyTorch DETR load time: {pytorch_load_time:.2f} ms")
print(f"PyTorch DETR avg inference time: {pytorch_detr_time:.2f} ms")
print(f"IREE DETR load time:    {iree_load_time:.2f} ms")
print(f"IREE DETR avg inference time:    {iree_detr_time:.2f} ms" if iree_detr_time else "IREE DETR: Not executed")
if mae is not None:
    print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Cleanup
detr_vmm = None
core_model = None
detr_processor = None
gc.collect()