import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import kagglehub
import numpy as np
import os
import urllib.request

from PIL import Image

workdir = "/tmp/workdir"
os.makedirs(workdir, exist_ok=True)

# Download a model.
download_path = kagglehub.model_download(
    "tensorflow/posenet-mobilenet/tfLite/float-075"
)
tflite_file = os.path.join(download_path, "1.tflite")

# Once downloaded we can compile the model for the selected backends. Both the
# TFLite and TOSA representations of the model are saved for debugging purposes.
# This is optional and can be omitted.
tosa_ir = os.path.join(workdir, "tosa.mlirbc")
bytecode_module = os.path.join(workdir, "iree.vmfb")
backends = ["llvm-cpu"]
backend_extra_args = ["--iree-llvmcpu-target-cpu=host"]

iree_tflite_compile.compile_file(
    tflite_file,
    input_type="tosa",
    extra_args=backend_extra_args,
    output_file=bytecode_module,
    save_temp_iree_input=tosa_ir,
    target_backends=backends,
    import_only=False,
)

# After compilation is completed we configure the VmModule using the local-task
# configuration and compiled IREE module.
config = iree_rt.Config("local-task")
context = iree_rt.SystemContext(config=config)
with open(bytecode_module, "rb") as f:
    vm_module = iree_rt.VmModule.from_flatbuffer(config.vm_instance, f.read())
    context.add_vm_module(vm_module)

# Finally, the IREE module is loaded and ready for execution. Here we load the
# sample image, manipulate to the expected input size, and execute the module.
# By default TFLite models include a single function named 'main'. The final
# results are printed.

jpg_file = "/".join([workdir, "input.jpg"])
jpg_url = "https://raw.githubusercontent.com/tensorflow/tfjs-models/refs/heads/master/pose-detection/test_data/pose.jpg"
urllib.request.urlretrieve(jpg_url, jpg_file)

im = (
    np.array(Image.open(jpg_file).resize((353, 257)))
    .astype(np.float32)
    .reshape((1, 353, 257, 3))
)
args = [im]

invoke = context.modules.module["main"]
iree_results = invoke(*args)
print(iree_results[0].to_host())
