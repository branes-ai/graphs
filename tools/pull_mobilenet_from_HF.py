from transformers import AutoImageProcessor, AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
model.save_pretrained("saved_model")
import tensorflow as tf

# Load the saved model
saved_model_dir = 'saved_model'
model = tf.saved_model.load(saved_model_dir)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the converted TFLite model
with open('mobilenetV2.tflite', 'wb') as f:
    f.write(tflite_model)