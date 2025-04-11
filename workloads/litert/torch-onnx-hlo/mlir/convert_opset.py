import onnx
from onnx import version_converter

# Load the model
model = onnx.load("efficientnet_b0.onnx")

# Convert to Opset 13
converted_model = version_converter.convert_version(model, 13)

# Force opset on the main graph and subgraphs
converted_model.opset_import.clear()
converted_model.opset_import.append(onnx.helper.make_opsetid("", 13))

# Save the updated model
onnx.checker.check_model(converted_model)  # Verify model integrity
onnx.save(converted_model, "efficientnet_b0_opset13_fixed.onnx")