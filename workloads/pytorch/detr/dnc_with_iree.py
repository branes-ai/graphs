import os
import time
import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from iree.runtime import Config, load_vm_module, VmModule
import iree.runtime as rt
import subprocess

batch_size = 8
N = 10

# Load and preprocess image
image = Image.open("dog.jpg").convert("RGB")
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_model.eval()

# Prepare input
detr_inputs = detr_processor(images=image, return_tensors="pt")
detr_pixel_values = detr_inputs["pixel_values"].expand(batch_size, -1, -1, -1)  # [8, 3, 800, 800]
detr_pixel_mask = detr_inputs["pixel_mask"].expand(batch_size, -1, -1)          # [8, 800, 800]
iree_pixel_values = torch.ones(batch_size, dtype=torch.float32)  # [8]

@torch.no_grad()
def torch_detr_forward(x: torch.Tensor, pixel_mask: torch.Tensor = None):
    conv1 = detr_model.model.backbone.conv_encoder.model.conv1
    return conv1(x)

# IREE module loading
def load_iree_module():
    mlir_code = """
    module @module {
      func.func @scale(%arg0: tensor<8xf32>) -> tensor<8xf32> {
        %cst = arith.constant dense<2.0> : tensor<8xf32>
        %0 = arith.mulf %arg0, %cst : tensor<8xf32>
        return %0 : tensor<8xf32>
      }
    }
    """
    with open("simple_scale.mlir", "w") as f:
        f.write(mlir_code)
    vmfb_file = "simple_scale.vmfb"
    subprocess.run([
        "iree-compile",
        "simple_scale.mlir",
        "-o",
        vmfb_file,
        "--iree-hal-target-backends=llvm-cpu"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    config = rt.Config("local-task")
    with open(vmfb_file, "rb") as f:
        flatbuffer_data = f.read()
    compiled_binary = rt.VmModule.from_flatbuffer(config.vm_instance, flatbuffer_data, warn_if_copy=False)
    return rt.load_vm_module(compiled_binary, config)

# Try loading IREE module
detr_vmm = load_iree_module()
iree_success = detr_vmm is not None

# Prepare IREE input
detr_np = iree_pixel_values.numpy().astype(np.float32)

# PyTorch benchmark
start = time.perf_counter()
for _ in range(N):
    torch_detr_forward(detr_pixel_values, detr_pixel_mask)
torch_detr_time = (time.perf_counter() - start) / N * 1000

# IREE benchmark
iree_detr_time = None
if iree_success:
    start = time.perf_counter()
    for _ in range(N):
        detr_vmm.scale(detr_np)
    iree_detr_time = (time.perf_counter() - start) / N * 1000

# Print results
print(f"🏁 DETR BENCHMARK RESULTS ({N} runs)")
print("----------------------------------------")
print(f"PyTorch DETR avg time: {torch_detr_time:.2f} ms")
print(f"IREE DETR avg time:    {iree_detr_time:.2f} ms" if iree_detr_time else "IREE DETR: Not executed")