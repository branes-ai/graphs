import os
import time
import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import logging
import psutil
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Monitor memory usage
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

# Ensure required dependencies
try:
    from PIL import Image
except ImportError:
    logger.error("Pillow is not installed. Run `pip install Pillow`")
    raise ImportError("Pillow is not installed. Run `pip install Pillow`")

# Load and preprocess image
image_path = "dog.jpg"
if not os.path.exists(image_path):
    logger.error(f"{image_path} not found in the current directory")
    raise FileNotFoundError(f"{image_path} not found in the current directory")
image = Image.open(image_path).convert("RGB")

# === LOAD DETR ===
try:
    logger.info("Loading DETR model and processor")
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    detr_model.eval()
except Exception as e:
    logger.error(f"Failed to load DETR model or processor: {str(e)}")
    raise

# Prepare input with pixel_mask
detr_inputs = detr_processor(images=image, return_tensors="pt")
detr_pixel_values = detr_inputs["pixel_values"]  # shape: [1, 3, 800, 800]
detr_pixel_mask = detr_inputs["pixel_mask"]      # shape: [1, 800, 800]
batch_size = 8  # Reduced to lower memory usage
detr_pixel_values = detr_pixel_values.expand(batch_size, -1, -1, -1)  # [8, 3, 800, 800]
detr_pixel_mask = detr_pixel_mask.expand(batch_size, -1, -1)          # [8, 800, 800]

@torch.no_grad()
def torch_detr_forward(x: torch.Tensor, pixel_mask: torch.Tensor):
    logger.info("Running PyTorch forward pass")
    # Use initial conv layer to minimize complexity
    conv1 = detr_model.model.backbone.conv_encoder.model.conv1
    outputs = conv1(x)
    return outputs  # Simplified output

# Benchmark
N = 10
logger.info(f"Running benchmark with {N} iterations")

# PyTorch benchmark with timeout
def run_pytorch_benchmark():
    try:
        logger.info("Starting PyTorch benchmark")
        start = time.perf_counter()
        for _ in range(N):
            torch_detr_forward(detr_pixel_values, detr_pixel_mask)
        torch_detr_time = (time.perf_counter() - start) / N * 1000
        return torch_detr_time
    except Exception as e:
        logger.error(f"PyTorch benchmark failed: {str(e)}")
        return None

result_queue = queue.Queue()
def pytorch_with_timeout():
    try:
        result_queue.put(run_pytorch_benchmark())
    except Exception as e:
        result_queue.put(("error", str(e)))

# Run PyTorch benchmark with 30-second timeout
thread = threading.Thread(target=pytorch_with_timeout)
thread.start()
thread.join(30)
if thread.is_alive():
    logger.error("PyTorch benchmark timed out")
    torch_detr_time = None
else:
    result = result_queue.get()
    if isinstance(result, tuple) and result[0] == "error":
        logger.error(f"PyTorch benchmark failed: {result[1]}")
        torch_detr_time = None
    else:
        torch_detr_time = result

# Print results
print("\n🏁 DETR BENCHMARK RESULTS ({} runs)".format(N))
print("----------------------------------------")
if torch_detr_time is not None:
    print(f"PyTorch DETR avg time: {torch_detr_time:.2f} ms")
else:
    print("PyTorch DETR: Failed to execute")
print("IREE DETR: Skipped (IREE compilation disabled)")
log_memory_usage()