# Model Complexity


| Deep Learning Model Architecture | Approximate Parameter Count | Relative Compute Cost | Introduction |
|----------------------------------|-----------------------------|----------------------|-----------|
| MobileNetV1 | ~4M | Very Low | 2017 |
| MobileNetV2 | ~3.5M | Very Low | 2018 |
| EfficientNetV2-B0 | 6.6M | Low | 2019 |
| EfficientNetV2-B1 | 7.8M | Medium | 2021 |
| ResNet-18 | 12.5M | Medium | 2015 |
| ResNet-34 | 18.0M | Medium | 2015 |
| ResNet-50 | 25.6M | Medium | 2015 |
| EfficientNetV2-L | 66M | High | 2021 |
| ViT-Base-16 | 86M | High | 2020 |
| Swin Transformer-Base | 86M | High | 2021 |
| DINO-v2 | 147M | Very High | 2022 |
| ViT-Large-Patch16 | 304M | Very High | 2020 |
| DINO-v2 (Large) | 354M | Very High | 2022 |
| GPT-4 Vision | Billions | Extremely High | 2023 |

## MobileNet

MobileNet is a critical family of models to consider when discussing inference compute costs, especially when focusing on arithmetic operation counts. 

**MobileNet's Focus**:

MobileNet models are specifically designed for efficient inference on mobile and embedded devices with limited computational resources.
Their architecture emphasizes reducing the number of arithmetic operations (FLOPs) and the model size.
MobileNet v1 and v2 in the Table:

To accurately place them, we need to consider their FLOPs and parameter counts:

* **MobileNet v1:**

    * This model uses depthwise separable convolutions to significantly reduce computation. 
    * It has a relatively small parameter count (typically in the few million range, depending on the width multiplier). 
    * Its FLOPs are significantly lower than ResNet-50.
    * Therefore, it would fall into the "Very Low" relative compute cost category.

* **MobileNet v2:**

    * This version builds upon v1 by introducing inverted residual blocks with linear bottlenecks. 
    * It further improves efficiency while maintaining or improving accuracy. 
    * Its FLOPs are generally lower than v1 for similar accuracy. 
    * It also falls into the "Very Low" relative compute cost category.

## Key Takeaways:

* MobileNet models are essential for applications where computational efficiency is paramount.
* Their design choices demonstrate how architectural innovations can dramatically reduce compute costs.
* They are the go to models when needing to run vision models on low power devices.

## EfficientNet

The history and key features of EfficientNet v1 and v2, and how they fit into the landscape of efficient deep learning models.

**EfficientNet v1: Scaling with Compound Coefficients**

* **Motivation:**
    * Traditional CNN scaling involved increasing depth, width, or resolution independently. However, these methods often led to diminishing returns.
    * EfficientNet aimed to find a more principled and efficient way to scale CNNs.
* **Key Innovation: Compound Scaling:**
    * EfficientNet introduced a compound scaling method that uniformly scales all three dimensions (depth, width, and resolution) using a set of compound coefficients.
    * This approach allows for a more balanced allocation of resources, leading to better performance and efficiency.
* **Architecture: MBConv Blocks:**
    * EfficientNet models are based on Mobile Inverted Bottleneck Convolution (MBConv) blocks, which are also used in MobileNet v2.
    * These blocks are highly efficient and reduce the number of parameters and FLOPs.
* **AutoML Search:**
    * The base model, EfficientNet-B0, was designed using Neural Architecture Search (NAS).
    * The compound scaling method was then applied to scale up the base model, resulting in EfficientNet-B1 to B7.
* **Impact:**
    * EfficientNet achieved state-of-the-art accuracy on ImageNet while being significantly more efficient than previous models.

**EfficientNet v2: Faster Training and Improved Efficiency**

* **Motivation:**
    * EfficientNet v1, while efficient, could still be computationally expensive to train, especially for larger models.
    * EfficientNet v2 aimed to address this by improving training speed and overall efficiency.
* **Key Improvements:**
    * **Faster Training:**
        * EfficientNet v2 introduced a combination of techniques to speed up training, including progressive learning and adaptive learning rates.
    * **Improved Scaling:**
        * The scaling strategy was refined to prioritize smaller image sizes in the early stages of training and larger image sizes in the later stages.
    * **Fused MBConv Blocks:**
        * EfficientNet v2 uses a combination of MBConv and Fused-MBConv blocks. The fused blocks replace depthwise and pointwise convolutions with a single dense convolution when the convolution layers are near each other. This increases speed.
    * **Regularization:**
        * Added regularization to prevent overfitting.
* **Impact:**
    * EfficientNet v2 achieved even better accuracy and efficiency than v1, with faster training times.

**How They Fit Into the Table:**

* Both EfficientNet v1 and v2 are designed for efficiency, so they generally fall into the "Low" to "Medium" relative compute cost categories, depending on the specific variant.
* Their focus on efficient architectures and scaling makes them valuable for applications where resources are limited.

**In summary:**

* EfficientNet v1 introduced compound scaling, revolutionizing CNN scaling.
* EfficientNet v2 built upon v1 by improving training speed and overall efficiency.
* Both model families are crucial for efficient deep learning.

## ResNet Architecture

The history and significance of the ResNet (Residual Network) architecture:

**ResNet: Addressing the Degradation Problem**

* **Motivation:**
    * As deep neural networks (DNNs) became deeper, a phenomenon known as the "degradation problem" emerged.
    * This problem wasn't simply overfitting; it was an actual increase in training error as the network depth increased beyond a certain point.
    * Researchers realized that deeper networks were not necessarily learning more complex representations, but rather struggling to optimize the increasingly complex function mappings.
* **Key Innovation: Residual Blocks:**
    * ResNet introduced the concept of "residual blocks," which revolutionized deep learning.
    * Instead of directly learning the underlying mapping, residual blocks learn a "residual mapping" by adding a "shortcut connection" or "skip connection" to the network.
    * This shortcut connection allows the network to learn an identity mapping, effectively bypassing certain layers if they don't contribute to the learning process.
    * Mathematically, if the desired mapping is H(x), ResNet learns F(x) = H(x) - x, where x is the input. The original mapping is then H(x) = F(x) + x.
* **Architecture:**
    * ResNet architectures are built by stacking multiple residual blocks.
    * The depth of ResNet models can vary significantly, ranging from ResNet-18 to ResNet-152 and beyond.
* **Impact:**
    * ResNet effectively solved the degradation problem, enabling the training of extremely deep neural networks.
    * It achieved state-of-the-art accuracy on ImageNet and other benchmark datasets.
    * ResNet's residual learning concept has become a fundamental building block in many modern DNN architectures.

**Key Contributions of ResNet:**

* **Enabling Deep Networks:** ResNet made it possible to train very deep neural networks, unlocking the potential for improved performance.
* **Simplifying Optimization:** Residual connections made it easier to optimize deep networks, as they provided a direct path for gradients to flow through the network.
* **Generalizability:** The residual learning concept proved to be highly generalizable and has been applied to various tasks and domains beyond image classification.

**Evolution of ResNet:**

* Subsequent research has explored variations of ResNet, such as:
    * **ResNeXt:** Introduced cardinality (number of parallel paths) to improve performance.
    * **Wide ResNet:** Increased the width of residual blocks to improve performance.
    * Various other optimizations and alterations to the original architecture.

ResNet's impact on deep learning is undeniable, and its residual learning concept continues to be a cornerstone of modern DNNs.

### HF location

https://huggingface.co/microsoft/resnet-18

https://huggingface.co/microsoft/resnet-34

https://huggingface.co/microsoft/resnet-50

We would like to get these links to a branes-ai organization.

## CNNs versus Transformers

From (Vision Transformers vs CNNs at the Edge)[https://www.edge-ai-vision.com/2024/03/vision-transformers-vs-cnns-at-the-edge/]

```quote
CNNs have an inductive spatial bias baked into them with convolutional kernels whereas vision transformers are based on a much more general architecture. In fact, the first vision transformers used an architecture from NLP tasks without change and simply chopped up the input image into a sequence of patches in the most naïve way possible. Nevertheless, they beat CNNs by overcoming the spatial bias given enough data. This may be another example of Rich Sutton’s famous “bitter lesson” of AI: “building in how we think we think does not work in the long run … breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning.”

It turns out that vision transformers “see” very differently from CNNs. A team from Google Brain studied the representations produced by the two architectures very carefully. While it is folklore that CNNs start with very low-level local information and gradually build up more global structures in the deeper layers, ViTs already have global information at the earliest layer thanks to global self-attention. As pithily summarized in a Quanta article, “If a CNN’s approach is like starting at a single pixel and zooming out, a transformer slowly brings the whole fuzzy image into focus.”. Another interesting observation to emerge from that study is that skip connections are very important for ViTs.
```

The question of whether convolutional architectures (CNNs) will be replaced by Transformers in edge AI and embodied AI is a complex one, and the answer isn't a simple yes or no. Here's a breakdown of the current situation and future trends:

**CNNs: Strengths and Weaknesses**

* **Strengths:**
    * **Computational Efficiency:** CNNs are generally more computationally efficient than Transformers, especially for tasks involving local patterns and spatial hierarchies. This is crucial for edge AI devices with limited resources.
    * **Mature Ecosystem:** CNNs have a long history and a mature ecosystem of tools, libraries, and hardware optimizations.
    * **Spatial Feature Extraction:** CNNs excel at extracting spatial features, making them well-suited for tasks like image classification, object detection, and semantic segmentation.
* **Weaknesses:**
    * **Limited Global Context:** CNNs struggle to capture long-range dependencies and global context, which can limit their performance on tasks requiring a broader understanding of the input.
    * **Generalization:** While CNNs can generalize well within their training domain, they can exhibit catastrophic failures when faced with out-of-distribution data or adversarial attacks.
    * **Lack of Invariance:** CNNs can be sensitive to variations in input data, such as changes in viewpoint or lighting conditions.

**Transformers: Strengths and Weaknesses**

* **Strengths:**
    * **Global Context:** Transformers excel at capturing long-range dependencies and global context, enabling them to handle complex tasks with greater accuracy.
    * **Generalization:** Transformers tend to generalize better than CNNs, especially when trained on large datasets.
    * **Flexibility:** Transformers can be applied to a wide range of tasks beyond vision, including natural language processing, audio processing, and reinforcement learning.
* **Weaknesses:**
    * **Computational Cost:** Transformers are computationally expensive, especially for high-resolution images or long sequences.
    * **Data Requirements:** Transformers typically require large datasets to achieve optimal performance.
    * **Memory Footprint:** The attention mechanism in Transformers can have a large memory footprint.

**Edge AI and Embodied AI Considerations:**

* **Edge AI:**
    * In edge AI, computational efficiency is paramount. CNNs remain a strong contender due to their low computational cost.
    * However, as hardware becomes more powerful and optimization techniques improve, Transformers may become more viable.
    * Hybrid architectures that combine CNNs and Transformers may offer the best of both worlds.
* **Embodied AI:**
    * Embodied AI systems often require a combination of perception, reasoning, and action.
    * Transformers' ability to capture global context and handle diverse data modalities makes them well-suited for embodied AI.
    * However, real-time performance is crucial in embodied AI, so computational efficiency remains a key consideration.
    * Robots, and other embodied agents, will need to efficiently handle data streams, and low power consumption.
* **The future:**
    * It is unlikely that CNNs will disappear entirely. They will likely continue to be used in applications where computational efficiency is critical.
    * Transformers will likely become more prevalent in edge AI and embodied AI as hardware improves and optimization techniques advance.
    * Hybrid architectures that combine CNNs and Transformers may become the dominant approach.
    * There will be a drive to create smaller, more efficient transformer architectures.
    * Improvements in model quantization, and other forms of model compression will be needed.

**In summary:**

While Transformers offer significant advantages in terms of generalization and global context, CNNs remain relevant due to their computational efficiency. The future likely holds a blend of both architectures, with the optimal choice depending on the specific application and resource constraints.

