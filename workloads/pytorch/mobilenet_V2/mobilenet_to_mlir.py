from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import torch
import torch.nn as nn
import iree.runtime as ireert
import iree.turbine.aot as aot

if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

    # Example input
    inputs = preprocessor(images=image, return_tensors="pt")

    # Compile the program using the turbine backend.
    iree_compiled_module = torch.compile(model, backend="turbine_cpu")

    # Use the compiled program as you would the original program.
    turbine_output = iree_compiled_module(**inputs)
    print("Output of compiled MobileNet_V2:", turbine_output)

    # Export the program using the simple API.
    print("Exporting compiled graph")
    export_output = aot.export(model, x)

    # Compile to a deployable artifact.
    binary = export_output.compile(save_to=None)

    # Use the IREE runtime API to test the compiled program.
    config = ireert.Config("local-task")
    vm_module = ireert.load_vm_module(
       ireert.VmModule.copy_buffer(config.vm_instance, binary.map_memory()),
       config,
    )
    outputs = vm_module.main(**inputs)
    logits = outputs.to_host().logits

    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
