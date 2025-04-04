# MobileNet V2
MobileNet V2 model pre-trained on ImageNet-1k at resolution 224x224. It was introduced in MobileNetV2: Inverted Residuals and Linear Bottlenecks by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. It was first released in this repository.

Disclaimer: The team releasing MobileNet V2 did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description
From the original README:

MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used. MobileNets can be run efficiently on mobile devices [...] MobileNets trade off between latency, size and accuracy while comparing favorably with popular models from the literature.

The checkpoints are named mobilenet_v2_depth_size, for example mobilenet_v2_1.0_224, where 1.0 is the depth multiplier and 224 is the resolution of the input images the model was trained on.

## Intended uses & limitations
You can use the raw model for image classification. See the model hub to look for fine-tuned versions on a task that interests you.

## How to use
Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

inputs = preprocessor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

Note: This model actually predicts 1001 classes, the 1000 classes from ImageNet plus an extra “background” class (index 0).

Currently, both the feature extractor and model support PyTorch.

## BibTeX entry and citation info
```
@inproceedings{mobilenetv22018,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
  booktitle={CVPR},
  year={2018}
}
```
