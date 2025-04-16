import mmap

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import numpy as np
import time

from iree.turbine.aot import *
import iree.runtime as rt

image = Image.open("dog.jpg").convert("RGB")
# === LOAD DETR ===
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_model.eval()

# Same image
detr_inputs = detr_processor(images=image, return_tensors="pt")
detr_pixel_values = detr_inputs["pixel_values"]  # shape: [1, 3, 800, 800]
detr_pixel_values = detr_pixel_values.expand(32, -1, -1, -1)  # [32, 3, 800, 800]

@torch.no_grad()
def torch_detr_forward(x: torch.Tensor):
    outputs = detr_model(x)
    # Let's just count objects above a threshold as a simplified output
    probs = outputs.logits.softmax(-1)
    keep = probs[..., :-1].max(-1).values > 0.9  # remove "no-object" class
    return keep.sum(dim=1)  # Number of detections per image


class DETRModule(CompiledModule):
    params = export_parameters(detr_model)

    def forward(self, x=AbstractTensor(32, 3, 800, 800, dtype=torch.float32)):
        return jittable(torch_detr_forward)(x)


exported = export(DETRModule)
exported.save_mlir("detr.mlir")
compiled_binary = exported.compile(target_backends=["llvm-cpu"], save_to=None)  # keep in memory


# === IREE RUNTIME SETUP ===

def create_vm_module():
    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config
    )
    return vmm


detr_np = detr_pixel_values.numpy().astype(np.float32)

# Load IREE module for DETR
def create_detr_vm_module():
    config = rt.Config("local-task")
    with open("detr.vmfb", "rb") as f:
        mmapped_file = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    return rt.load_vm_module(rt.VmModule.wrap_buffer(config.vm_instance, mmapped_file), config)

detr_vmm = create_detr_vm_module()

# Warm-up
detr_vmm.forward(detr_np)
torch_detr_forward(detr_pixel_values)

N=1
# Benchmark
start = time.perf_counter()
for _ in range(N):
    torch_detr_forward(detr_pixel_values)
torch_detr_time = (time.perf_counter() - start) / N * 1000
print(f"PyTorch DETR avg time: {torch_detr_time:.2f} ms")

start = time.perf_counter()
for _ in range(N):
    detr_vmm.forward(detr_np)
iree_detr_time = (time.perf_counter() - start) / N * 1000


print("\n🏁 DETR BENCHMARK RESULTS ({} runs)".format(N))
print("----------------------------------------")
print(f"PyTorch DETR avg time: {torch_detr_time:.2f} ms")
print(f"IREE DETR avg time:    {iree_detr_time:.2f} ms")